"""SDK errors and usage objects mapped to our own types (no requests are sent)."""
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest

from chat import anthropic_client, openai_client
from chat.error import BadRequest, ConnectionFailed, Overloaded, RateLimited, ServerError
from usage import RequestUsage

REQUEST = httpx.Request("POST", "https://example.invalid")


def response(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, request=REQUEST, headers=headers or {})


def anthropic_status(status: int, error_type: str | None = None, headers: dict | None = None):
    body = {"type": "error", "error": {"type": error_type}} if error_type else None
    return anthropic.APIStatusError("msg", response=response(status, headers), body=body)


def openai_status(cls, status: int, code: str | None = None):
    body = {"code": code} if code else None
    return cls("msg", response=response(status), body=body)


# ---- Anthropic errors ----

@pytest.mark.parametrize("exc, expected", [
    (anthropic.APIConnectionError(request=REQUEST), ConnectionFailed),
    (anthropic_status(429), RateLimited),
    (anthropic_status(529), Overloaded),
    (anthropic_status(500), ServerError),
    (anthropic_status(503), ServerError),
    (anthropic_status(400), BadRequest),
    (anthropic_status(401), BadRequest),
    (anthropic_status(404), BadRequest),
])
def test_anthropic_status_codes(exc, expected):
    assert type(anthropic_client._to_llm_error(exc)) is expected


@pytest.mark.parametrize("error_type, expected", [
    ("rate_limit_error", RateLimited),
    ("overloaded_error", Overloaded),
    ("api_error", ServerError),
    ("invalid_request_error", BadRequest),
])
def test_anthropic_error_inside_a_stream(error_type, expected):
    # Errors sent as stream events are plain APIErrors with no status code.
    exc = anthropic.APIError("msg", request=REQUEST, body={"type": "error", "error": {"type": error_type}})
    assert type(anthropic_client._to_llm_error(exc)) is expected


def test_anthropic_keeps_retry_after():
    err = anthropic_client._to_llm_error(anthropic_status(429, headers={"retry-after": "12"}))
    assert err.retry_after == 12.0


def test_anthropic_usage_copies_only_present_fields():
    usage = RequestUsage("m", input_tokens=10, output_tokens=1)
    # message_delta usage: output is set, input may be missing
    anthropic_client._apply_usage(usage, SimpleNamespace(output_tokens=50, input_tokens=None))
    assert (usage.input_tokens, usage.output_tokens) == (10, 50)


def test_anthropic_usage_cache_fields():
    usage = RequestUsage("m")
    anthropic_client._apply_usage(usage, SimpleNamespace(
        input_tokens=5, output_tokens=1, cache_creation_input_tokens=100, cache_read_input_tokens=200))
    assert usage.prompt_tokens == 305


# ---- OpenAI errors ----

@pytest.mark.parametrize("exc, expected", [
    (openai.APIConnectionError(request=REQUEST), ConnectionFailed),
    (openai_status(openai.RateLimitError, 429), RateLimited),
    (openai_status(openai.RateLimitError, 429, code="insufficient_quota"), BadRequest),
    (openai_status(openai.InternalServerError, 500), ServerError),
    (openai_status(openai.APIStatusError, 503), ServerError),
    (openai_status(openai.BadRequestError, 400), BadRequest),
    (openai_status(openai.AuthenticationError, 401), BadRequest),
    (openai_status(openai.NotFoundError, 404), BadRequest),
])
def test_openai_errors(exc, expected):
    assert type(openai_client._to_llm_error(exc)) is expected


def test_openai_plain_api_error_is_bad_request():
    exc = openai.APIError("msg", request=REQUEST, body=None)
    assert type(openai_client._to_llm_error(exc)) is BadRequest


# ---- OpenAI usage ----

def openai_usage(prompt, completion, cached=None, written=None, details=True):
    d = SimpleNamespace(cached_tokens=cached, cache_write_tokens=written) if details else None
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=d)


def test_openai_usage_without_details():
    usage = RequestUsage("m")
    openai_client._apply_usage(usage, openai_usage(100, 20, details=False))
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_input_tokens) == (100, 20, 0)


def test_openai_usage_splits_cached_tokens_out_of_prompt():
    usage = RequestUsage("m")
    openai_client._apply_usage(usage, openai_usage(1000, 20, cached=600, written=300))
    assert usage.input_tokens == 100
    assert usage.cache_read_input_tokens == 600
    assert usage.cache_creation_input_tokens == 300
    assert usage.prompt_tokens == 1000


def test_openai_usage_never_negative():
    usage = RequestUsage("m")
    openai_client._apply_usage(usage, openai_usage(10, 0, cached=50))
    assert usage.input_tokens == 0
