import random

import anthropic


def is_retryable(exc: BaseException) -> bool:
    """429, 5xx and overloaded errors are retried; 400 and other 4xx are client bugs."""
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True  # network blip, the request may not have reached the API
    if not isinstance(exc, anthropic.APIStatusError):
        return False
    status = exc.status_code
    if status == 429 or status >= 500:  # includes 529 overloaded
        return True
    # An error in the middle of a stream arrives with HTTP status 200;
    # its type is only in the body, e.g. {"error": {"type": "overloaded_error"}}.
    return _error_type(exc) in ("overloaded_error", "rate_limit_error", "api_error")


def _error_type(exc: anthropic.APIStatusError) -> str | None:
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return err.get("type")
    return None


def _retry_after(exc: BaseException) -> float | None:
    """Seconds the server asked us to wait, from the retry-after header."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return float(response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def backoff_delay(attempt: int, exc: BaseException, base: float, cap: float) -> float:
    """Exponential backoff with full jitter: random in [0, min(cap, base * 2**attempt)].

    If the server sent retry-after, wait at least that long.
    """
    delay = random.uniform(0, min(cap, base * 2 ** attempt))
    retry_after = _retry_after(exc)
    if retry_after is not None:
        delay = max(delay, min(retry_after, cap))
    return delay


def describe(exc: BaseException) -> str:
    if isinstance(exc, anthropic.APIStatusError):
        return _error_type(exc) or f"HTTP {exc.status_code}"
    return type(exc).__name__
