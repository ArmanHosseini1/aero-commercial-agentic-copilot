import csv
import re
from collections import deque
from contextlib import closing
from pathlib import Path
from time import monotonic, sleep
from typing import TypeAlias

from google import genai
from qdrant_client import QdrantClient, models

from aero_agentic_copilot.config import settings


ROOT_DIR = Path(__file__).resolve().parents[3]
SURVEYS_PATH = ROOT_DIR / "data" / "customer_surveys.csv"
COMMERCIAL_DOCS_DIR = ROOT_DIR / "data" / "commercial_docs"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_ITEMS_PER_MINUTE = 80
EMBEDDING_WINDOW_SECONDS = 60.0

Payload: TypeAlias = dict[str, str | int]
IngestionRecord: TypeAlias = tuple[int, str, Payload]
SECTION_HEADER = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class EmbeddingRateLimiter:
    def __init__(
        self,
        items_per_minute: int = EMBEDDING_ITEMS_PER_MINUTE,
        window_seconds: float = EMBEDDING_WINDOW_SECONDS,
    ) -> None:
        self.items_per_minute = items_per_minute
        self.window_seconds = window_seconds
        self._submitted_at: deque[float] = deque()

    def wait_for_capacity(self, item_count: int) -> None:
        if not 1 <= item_count <= self.items_per_minute:
            raise ValueError(
                f"Embedding batch size must be between 1 and {self.items_per_minute}."
            )

        while True:
            now = monotonic()
            while (
                self._submitted_at
                and now - self._submitted_at[0] >= self.window_seconds
            ):
                self._submitted_at.popleft()

            if len(self._submitted_at) + item_count <= self.items_per_minute:
                self._submitted_at.extend([now] * item_count)
                return

            sleep(self._submitted_at[0] + self.window_seconds - now)


def load_surveys(csv_path: Path = SURVEYS_PATH) -> list[IngestionRecord]:
    records: list[IngestionRecord] = []
    required_fields = {
        "survey_id",
        "airline_name",
        "date",
        "domain",
        "rating",
        "verbatim_text",
    }

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path} is missing one or more required survey columns.")

        for point_id, row in enumerate(reader, start=1):
            if any(row[field] is None for field in required_fields):
                raise ValueError(f"Survey row {point_id} has a missing required value.")
            rating = int(row["rating"])
            if not 1 <= rating <= 5:
                raise ValueError(f"Survey row {point_id} has rating outside 1-5.")
            payload: Payload = {
                "survey_id": row["survey_id"],
                "airline_name": row["airline_name"],
                "domain": row["domain"],
                "rating": rating,
                "date": row["date"],
                "verbatim_text": row["verbatim_text"],
            }
            records.append((point_id, row["verbatim_text"], payload))

    if len(records) != 80:
        raise ValueError(f"Expected 80 survey rows in {csv_path}; found {len(records)}.")
    return records


def split_markdown_sections(docs_dir: Path = COMMERCIAL_DOCS_DIR) -> list[IngestionRecord]:
    records: list[IngestionRecord] = []
    point_id = 1

    for source_path in sorted(docs_dir.glob("*.md")):
        document = source_path.read_text(encoding="utf-8")
        section_headers = list(SECTION_HEADER.finditer(document))
        if not section_headers:
            raise ValueError(f"{source_path} contains no level-two Markdown sections.")

        for index, header in enumerate(section_headers):
            section_end = (
                section_headers[index + 1].start()
                if index + 1 < len(section_headers)
                else len(document)
            )
            content = document[header.end() : section_end].strip()
            if not content:
                raise ValueError(
                    f"Section {header.group(1)!r} in {source_path} has no content."
                )
            payload: Payload = {
                "source_file": source_path.name,
                "section_title": header.group(1).strip(),
                "content": content,
            }
            embedding_text = f"{payload['section_title']}\n{content}"
            records.append((point_id, embedding_text, payload))
            point_id += 1

    if not records:
        raise ValueError(f"No Markdown sections found in {docs_dir}.")
    return records


def create_qdrant_client() -> QdrantClient:
    if settings.QDRANT_MODE == "local":
        storage_path = settings.qdrant_storage_path
        if storage_path is None:
            raise RuntimeError("Local Qdrant mode requires a storage path.")
        return QdrantClient(path=storage_path)

    server_url = settings.qdrant_server_url
    if server_url is None:
        raise RuntimeError("Server Qdrant mode requires a server URL.")
    return QdrantClient(url=server_url)


def embed_texts(
    client: genai.Client,
    texts: list[str],
    rate_limiter: EmbeddingRateLimiter,
) -> list[list[float]]:
    if not texts:
        raise ValueError("Cannot create embeddings for an empty text collection.")

    vectors: list[list[float]] = []
    for batch_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        rate_limiter.wait_for_capacity(len(batch))
        response = client.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=batch,
        )
        embeddings = response.embeddings
        if embeddings is None or len(embeddings) != len(batch):
            raise RuntimeError(
                "Gemini returned an unexpected number of embeddings "
                f"for batch starting at item {batch_start}."
            )
        for embedding in embeddings:
            if embedding.values is None:
                raise RuntimeError("Gemini returned an embedding without vector values.")
            vectors.append([float(value) for value in embedding.values])

    return vectors


def recreate_collections(
    client: QdrantClient,
    collection_names: tuple[str, ...],
) -> None:
    for collection_name in collection_names:
        if client.collection_exists(collection_name):
            local_client = getattr(client, "_client", None)
            local_collections = getattr(local_client, "collections", None)
            if isinstance(local_collections, dict):
                local_collection = local_collections.get(collection_name)
                if local_collection is not None:
                    local_collection.close()
            client.delete_collection(collection_name)


def create_and_populate_collection(
    client: QdrantClient,
    collection_name: str,
    records: list[IngestionRecord],
    vectors: list[list[float]],
) -> None:
    if len(records) != len(vectors):
        raise ValueError(
            f"{collection_name} has {len(records)} records but {len(vectors)} vectors."
        )
    if not vectors or not vectors[0]:
        raise ValueError(f"{collection_name} requires at least one non-empty embedding.")

    vector_size = len(vectors[0])
    if any(len(vector) != vector_size for vector in vectors):
        raise ValueError(f"{collection_name} contains inconsistent embedding dimensions.")

    if client.collection_exists(collection_name):
        raise RuntimeError(
            f"{collection_name} must be deleted before its replacement is created."
        )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=len(vectors[0]),
            distance=models.Distance.COSINE,
        ),
    )
    points = [
        models.PointStruct(id=point_id, vector=vector, payload=payload)
        for (point_id, _, payload), vector in zip(records, vectors, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)


def ingest() -> dict[str, int]:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required to run the ingestion pipeline.")

    survey_records = load_surveys()
    document_records = split_markdown_sections()

    qdrant_client = create_qdrant_client()
    try:
        with closing(genai.Client(api_key=settings.GEMINI_API_KEY)) as gemini_client:
            rate_limiter = EmbeddingRateLimiter()
            survey_vectors = embed_texts(
                gemini_client,
                [text for _, text, _ in survey_records],
                rate_limiter,
            )
            document_vectors = embed_texts(
                gemini_client,
                [text for _, text, _ in document_records],
                rate_limiter,
            )

            recreate_collections(
                qdrant_client,
                ("surveys", "commercial_docs"),
            )
            create_and_populate_collection(
                qdrant_client,
                "surveys",
                survey_records,
                survey_vectors,
            )
            create_and_populate_collection(
                qdrant_client,
                "commercial_docs",
                document_records,
                document_vectors,
            )

            return {
                name: qdrant_client.count(collection_name=name, exact=True).count
                for name in ("surveys", "commercial_docs")
            }
    finally:
        qdrant_client.close()
