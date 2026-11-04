import random

from chat.error import LLMError


def is_retryable(ex: BaseException) -> bool:
    if isinstance(ex, LLMError):
        return ex.retryable
    else:
        return False


def backoff_delay(attempt: int, exc: BaseException, base: float, cap: float) -> float:
    """Exponential backoff with full jitter: random in [0, min(cap, base * 2**attempt)].

    If the server sent retry-after, wait at least that long.
    """
    delay = random.uniform(0, min(cap, base * 2 ** attempt))
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        delay = max(delay, min(retry_after, cap))
    return delay


def describe(exc: BaseException) -> str:
    return type(exc).__name__
