from aero_agentic_copilot.rag.ingestion import ingest


def main() -> None:
    counts = ingest()
    print(f"surveys: {counts['surveys']} points")
    print(f"commercial_docs: {counts['commercial_docs']} points")


if __name__ == "__main__":
    main()
