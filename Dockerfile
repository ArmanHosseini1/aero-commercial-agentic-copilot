FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

RUN uv sync --frozen --no-dev

ENV PORT=8501
ENV QDRANT_MODE=local
ENV UV_NO_SYNC=1

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]