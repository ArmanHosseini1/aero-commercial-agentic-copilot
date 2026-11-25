import pytest
from google.genai import errors

import aero_agentic_copilot.llm_utils as llm_utils
from aero_agentic_copilot.llm_utils import call_with_retry


@pytest.mark.parametrize(
    "error_type,status_code",
    [
        (errors.ServerError, 500),
        (errors.ServerError, 503),
        (errors.ClientError, 429),
    ],
)
def test_call_with_retry_retries_transient_gemini_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[errors.APIError],
    status_code: int,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(llm_utils.time, "sleep", delays.append)
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 4:
            raise error_type(code=status_code, response_json={"message": "retry"})
        return "success"

    assert call_with_retry(operation) == "success"
    assert calls == 4
    assert delays == [2.0, 4.0, 8.0]


def test_call_with_retry_does_not_retry_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(llm_utils.time, "sleep", delays.append)
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise errors.ClientError(code=400, response_json={"message": "invalid"})

    with pytest.raises(errors.ClientError):
        call_with_retry(operation)

    assert calls == 1
    assert delays == []


def test_call_with_retry_stops_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(llm_utils.time, "sleep", delays.append)
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1
        raise errors.ServerError(code=503, response_json={"message": "unavailable"})

    with pytest.raises(errors.ServerError):
        call_with_retry(operation, max_retries=2, base_delay=0.5)

    assert calls == 3
    assert delays == [0.5, 1.0]
