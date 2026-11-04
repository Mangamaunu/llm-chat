class LLMError(Exception):
    retryable: bool = False

    def __init__(self, message, retry_after=None):
        self.retry_after = retry_after
        super().__init__(message)


class RateLimited(LLMError):
    retryable: bool = True


class Overloaded(LLMError):
    retryable: bool = True


class ServerError(LLMError):
    retryable: bool = True


class ConnectionFailed(LLMError):
    retryable: bool = True


class BadRequest(LLMError):
    retryable: bool = False


def retry_after(exc: BaseException) -> float | None:
    """Seconds the server asked us to wait, from the retry-after header."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        return float(response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None