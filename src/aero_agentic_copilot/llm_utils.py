import time
from collections.abc import Callable
from typing import TypeVar

from google.genai import errors


T = TypeVar("T")
RETRYABLE_STATUS_CODES = {429, 500, 503}


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> T:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")
    if base_delay < 0:
        raise ValueError("base_delay cannot be negative.")

    retries = 0
    while True:
        try:
            return fn()
        except (errors.ClientError, errors.ServerError) as error:
            if error.code not in RETRYABLE_STATUS_CODES or retries >= max_retries:
                raise
            time.sleep(base_delay * (2**retries))
            retries += 1
