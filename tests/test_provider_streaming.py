"""_stream_once of the real clients, with the SDK replaced by a stub (no network)."""
from types import SimpleNamespace as NS

import anthropic
import httpx2
import openai
import pytest

from llm_chat.core.error import BadRequest, ConnectionFailed, RateLimited
from llm_chat.providers.anthropic_client import AnthropicClient
from llm_chat.providers.openai_client import OpenAiClient


class StubCreate:
    """Stands in for `messages.create` / `chat.completions.create`.

    Returns the scripted items as a stream; an exception in the script is raised at that point.
    """

    def __init__(self, *items):
        self.items = items
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        # Copy the messages: the client passes its own list, which changes after the turn.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._stream()

    def _stream(self):
        for item in self.items:
            if isinstance(item, BaseException):
                raise item
            yield item


def collect(client, content="hi"):
    """Run one turn; return (yielded texts, raised exception or None)."""
    texts = []
    try:
        for text in client.stream_chat(content):
            texts.append(text)
    except Exception as ex:
        return texts, ex
    return texts, None


# ---- Anthropic ----

def a_usage(**kw):
    fields = dict(input_tokens=None, output_tokens=None, cache_creation_input_tokens=None, cache_read_input_tokens=None)
    fields.update(kw)
    return NS(**fields)


def message_start(input_tokens=10, cache_read=0, cache_write=0):
    usage = a_usage(input_tokens=input_tokens, output_tokens=1,
                    cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write)
    return NS(type="message_start", message=NS(usage=usage))


def text_delta(text):
    return NS(type="content_block_delta", delta=NS(type="text_delta", text=text))


def message_delta(output_tokens, stop_reason="end_turn"):
    return NS(type="message_delta", usage=a_usage(output_tokens=output_tokens), delta=NS(stop_reason=stop_reason))


@pytest.fixture
def anthropic_client():
    client = AnthropicClient()
    client.sleep = lambda seconds: None
    return client


def use_stub(client, stub):
    if isinstance(client, AnthropicClient):
        client._client = NS(messages=NS(create=stub))
    else:
        client._client = NS(chat=NS(completions=NS(create=stub)))
    return stub


def test_anthropic_streams_text_and_usage(anthropic_client):
    stub = use_stub(anthropic_client, StubCreate(
        message_start(input_tokens=12, cache_read=100, cache_write=20),
        NS(type="content_block_start"),
        text_delta("Hel"),
        NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking="hmm")),
        text_delta("lo"),
        message_delta(output_tokens=7),
        NS(type="message_stop"),
    ))
    texts, error = collect(anthropic_client)
    assert error is None
    assert texts == ["Hel", "lo"]
    u = anthropic_client.last_usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens) == (12, 7, 100, 20)
    assert anthropic_client.messages[-1] == {"role": "assistant", "content": "Hello"}
    assert stub.calls[0]["stream"] is True


def test_anthropic_sends_system_and_history_separately(anthropic_client):
    stub = use_stub(anthropic_client, StubCreate(message_start(), text_delta("ok"), message_delta(1)))
    anthropic_client.set_system("be brief")
    collect(anthropic_client, "hi")
    call = stub.calls[0]
    assert call["system"] == "be brief"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["model"] == anthropic_client.model


@pytest.mark.parametrize("network_error", [
    httpx2.ReadError("connection reset"),
    httpx2.RemoteProtocolError("peer closed connection"),
    httpx2.ReadTimeout("timed out"),
])
def test_anthropic_connection_lost_mid_stream_becomes_connection_failed(anthropic_client, network_error):
    use_stub(anthropic_client, StubCreate(message_start(), text_delta("Hel"), network_error))
    texts, error = collect(anthropic_client)
    assert texts == ["Hel"]
    assert isinstance(error, ConnectionFailed)
    assert anthropic_client.messages == []


def test_anthropic_connection_lost_before_text_is_retried(anthropic_client):
    calls = 0

    def create(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return StubCreate(message_start(), httpx2.ReadError("reset"))(**kwargs)
        return StubCreate(message_start(), text_delta("ok"), message_delta(1))(**kwargs)

    use_stub(anthropic_client, create)
    texts, error = collect(anthropic_client)
    assert error is None
    assert texts == ["ok"]
    assert calls == 2


def test_anthropic_sdk_error_is_mapped(anthropic_client):
    response = httpx2.Response(429, request=httpx2.Request("POST", "https://example.invalid"))
    error = anthropic.APIStatusError("slow down", response=response, body=None)
    use_stub(anthropic_client, StubCreate(error))
    anthropic_client.max_retries = 0
    _, raised = collect(anthropic_client)
    assert isinstance(raised, RateLimited)


# ---- OpenAI ----

def chunk(text=None, usage=None, finish_reason=None):
    choices = [NS(delta=NS(content=text), finish_reason=finish_reason)] if usage is None else []
    return NS(choices=choices, usage=usage)


def o_usage(prompt, completion, cached=None):
    details = NS(cached_tokens=cached, cache_write_tokens=None) if cached is not None else None
    return NS(prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=details)


@pytest.fixture
def openai_client():
    client = OpenAiClient()
    client.sleep = lambda seconds: None
    return client


def test_openai_streams_text_and_usage(openai_client):
    stub = use_stub(openai_client, StubCreate(
        chunk("Hel"), chunk(None), chunk(""), chunk("lo"),
        chunk(usage=o_usage(100, 7, cached=60)),
    ))
    texts, error = collect(openai_client)
    assert error is None
    assert texts == ["Hel", "lo"]
    u = openai_client.last_usage
    assert (u.input_tokens, u.cache_read_input_tokens, u.output_tokens) == (40, 60, 7)
    assert stub.calls[0]["stream_options"] == {"include_usage": True}


def test_openai_puts_system_prompt_first_without_storing_it(openai_client):
    stub = use_stub(openai_client, StubCreate(chunk("ok"), chunk(usage=o_usage(5, 1))))
    openai_client.set_system("be brief")
    collect(openai_client, "hi")
    assert stub.calls[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert all(m["role"] != "system" for m in openai_client.messages)


def test_openai_without_system_prompt_sends_only_history(openai_client):
    stub = use_stub(openai_client, StubCreate(chunk("ok"), chunk(usage=o_usage(5, 1))))
    collect(openai_client, "hi")
    assert stub.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


# ---- reply limit and stop reason ----

def test_anthropic_sends_reply_limit(anthropic_client):
    stub = use_stub(anthropic_client, StubCreate(message_start(), text_delta("ok"), message_delta(1)))
    anthropic_client.max_tokens = 777
    collect(anthropic_client)
    assert stub.calls[0]["max_tokens"] == 777


@pytest.mark.parametrize("api_value, expected", [
    ("end_turn", None),
    ("stop_sequence", None),
    ("max_tokens", "max_tokens"),
    ("model_context_window_exceeded", "context_full"),
    ("refusal", "refusal"),
    (None, None),
])
def test_anthropic_stop_reason(anthropic_client, api_value, expected):
    use_stub(anthropic_client, StubCreate(message_start(), text_delta("ok"), message_delta(5, stop_reason=api_value)))
    collect(anthropic_client)
    assert anthropic_client.last_stop_reason == expected


def test_anthropic_message_delta_still_updates_usage(anthropic_client):
    use_stub(anthropic_client, StubCreate(message_start(), text_delta("ok"), message_delta(42, stop_reason="max_tokens")))
    collect(anthropic_client)
    assert anthropic_client.last_usage.output_tokens == 42


@pytest.mark.parametrize("api_value, expected", [
    ("stop", None),
    ("length", "max_tokens"),
    ("content_filter", "refusal"),
    ("tool_calls", None),
])
def test_openai_finish_reason(openai_client, api_value, expected):
    use_stub(openai_client, StubCreate(chunk("ok"), chunk(None, finish_reason=api_value), chunk(usage=o_usage(5, 1))))
    collect(openai_client)
    assert openai_client.last_stop_reason == expected


def test_openai_finish_reason_in_same_chunk_as_text(openai_client):
    use_stub(openai_client, StubCreate(chunk("cut", finish_reason="length"), chunk(usage=o_usage(5, 1))))
    texts, _ = collect(openai_client)
    assert texts == ["cut"]
    assert openai_client.last_stop_reason == "max_tokens"


@pytest.mark.parametrize("base_url, param", [
    ("http://localhost:11434/v1/", "max_tokens"),
    ("http://127.0.0.1:11434/v1/", "max_tokens"),
    ("http://[::1]:11434/v1/", "max_tokens"),
    ("https://api.openai.com/v1", "max_completion_tokens"),
    ("https://my-proxy.example.com/v1", "max_completion_tokens"),
])
def test_openai_limit_parameter_name(base_url, param):
    client = OpenAiClient(base_url=base_url)
    stub = use_stub(client, StubCreate(chunk("ok"), chunk(usage=o_usage(5, 1))))
    client.max_tokens = 321
    collect(client)
    call = stub.calls[0]
    assert call[param] == 321
    other = "max_completion_tokens" if param == "max_tokens" else "max_tokens"
    assert other not in call


# ---- OpenAI SDK errors inside _stream_once ----

OPENAI_REQUEST = httpx2.Request("POST", "https://example.invalid")


def openai_status_error(cls, status):
    return cls("msg", response=httpx2.Response(status, request=OPENAI_REQUEST), body=None)


def test_openai_rate_limit_when_starting_the_stream(openai_client):
    use_stub(openai_client, StubCreate(openai_status_error(openai.RateLimitError, 429)))
    openai_client.max_retries = 0
    _, error = collect(openai_client)
    assert isinstance(error, RateLimited)
    assert openai_client.messages == []


def test_openai_rate_limit_is_retried(openai_client):
    calls = 0

    def create(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise openai_status_error(openai.RateLimitError, 429)
        return StubCreate(chunk("ok"), chunk(usage=o_usage(5, 1)))(**kwargs)

    use_stub(openai_client, create)
    texts, error = collect(openai_client)
    assert error is None
    assert texts == ["ok"]
    assert calls == 2


def test_openai_connection_lost_mid_stream(openai_client):
    # The OpenAI SDK wraps network errors while streaming in APIConnectionError.
    use_stub(openai_client, StubCreate(chunk("Hel"), openai.APIConnectionError(request=OPENAI_REQUEST)))
    texts, error = collect(openai_client)
    assert texts == ["Hel"]
    assert isinstance(error, ConnectionFailed)
    assert openai_client.messages == []


def test_openai_error_event_mid_stream(openai_client):
    # An error sent inside the stream arrives as a plain APIError.
    use_stub(openai_client, StubCreate(chunk("Hel"), openai.APIError("server said no", request=OPENAI_REQUEST, body=None)))
    _, error = collect(openai_client)
    assert isinstance(error, BadRequest)
    assert "server said no" in str(error)
    assert isinstance(error.__cause__, openai.APIError)


def test_openai_broken_stream_reports_no_tokens(openai_client):
    # Usage comes in the last chunk, so a stream that breaks records nothing (review L6).
    use_stub(openai_client, StubCreate(chunk("Hel"), openai.APIConnectionError(request=OPENAI_REQUEST)))
    collect(openai_client)
    assert openai_client.usage.by_model == {}


# ---- prompt caching (review M7) ----

def test_anthropic_requests_prompt_caching(anthropic_client):
    stub = use_stub(anthropic_client, StubCreate(message_start(), text_delta("ok"), message_delta(1)))
    collect(anthropic_client)
    assert stub.calls[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_caching_on_every_request(anthropic_client):
    stub = use_stub(anthropic_client, StubCreate(message_start(), text_delta("ok"), message_delta(1)))
    collect(anthropic_client, "q1")
    stub.items = (message_start(cache_read=500), text_delta("ok"), message_delta(1))
    collect(anthropic_client, "q2")
    assert [c["cache_control"] for c in stub.calls] == [{"type": "ephemeral"}] * 2
    assert anthropic_client.last_usage.cache_read_input_tokens == 500


def test_openai_does_not_send_cache_control(openai_client):
    # OpenAI caches automatically; the parameter doesn't exist there.
    stub = use_stub(openai_client, StubCreate(chunk("ok"), chunk(usage=o_usage(5, 1))))
    collect(openai_client)
    assert "cache_control" not in stub.calls[0]
