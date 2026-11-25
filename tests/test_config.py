from pathlib import Path

import pytest

from aero_agentic_copilot.config import Settings


@pytest.fixture
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "QDRANT_MODE",
        "QDRANT_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_settings_default_to_local_qdrant(
    clean_settings_environment: None,
) -> None:
    settings = Settings(_env_file=None)

    assert settings.GEMINI_API_KEY == ""
    assert settings.GEMINI_MODEL == "gemini-3.5-flash"
    assert settings.GEMINI_EMBEDDING_MODEL == "gemini-embedding-001"
    assert settings.QDRANT_MODE == "local"
    assert settings.qdrant_storage_path == "./qdrant_storage"
    assert settings.qdrant_server_url is None


def test_settings_select_server_qdrant(
    clean_settings_environment: None,
) -> None:
    settings = Settings(_env_file=None, QDRANT_MODE="server")

    assert settings.qdrant_storage_path is None
    assert settings.qdrant_server_url == "http://localhost:6333"


def test_settings_load_dotenv_values(
    clean_settings_environment: None,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_MODEL=test-model\nQDRANT_MODE=server\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.GEMINI_MODEL == "test-model"
    assert settings.qdrant_server_url == "http://localhost:6333"
