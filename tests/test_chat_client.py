"""stream_chat: history, usage and retries, tested through FakeClient."""
import pytest

from chat.error import BadRequest, ConnectionFailed, LLMError, Overloaded, RateLimited, ServerError
from fake_client import FAKE_MODEL, FakeClient, Reply, chat


# ---- normal turns ----

def test_yields_chunks_in_order():
    client = FakeClient(Reply(["Hel", "lo", "!"]))
    assert list(client.stream_chat("hi")) == ["Hel", "lo", "!"]


def test_turn_adds_user_and_joined_assistant_message():
    client = FakeClient(Reply(["Hel", "lo"]))
    chat(client, "hi")
    assert client.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_next_request_includes_previous_turns():
    client = FakeClient("first answer", "second answer")
    chat(client, "q1")
    chat(client, "q2")
    assert client.requests[1]["messages"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "q2"},
    ]


def test_system_prompt_is_kept_out_of_messages():
    client = FakeClient("arr")
    client.set_system("You are a pirate.")
    chat(client, "hi")
    assert client.requests[0]["system"] == "You are a pirate."
    assert all(m["role"] != "system" for m in client.messages)


def test_set_model_is_used_for_the_next_request():
    client = FakeClient("ok")
    client.set_model("other-model")
    chat(client, "hi")
    assert client.requests[0]["model"] == "other-model"
    assert client.last_usage.model == "other-model"


def test_reset_clears_history_but_keeps_system_prompt():
    client = FakeClient("ok")
    client.set_system("sys")
    chat(client, "hi")
    client.reset()
    assert client.messages == []
    assert client.history_tokens == 0
    assert client.system == "sys"


def test_empty_reply_is_not_stored():
    # Anthropic rejects empty assistant messages, so one would break every later request.
    client = FakeClient(Reply([]))
    assert chat(client, "hi") == ""
    assert client.messages == []


def test_empty_reply_keeps_earlier_history():
    client = FakeClient("first", Reply([]))
    chat(client, "q1")
    before = list(client.messages)
    chat(client, "q2")
    assert client.messages == before


def test_chat_continues_normally_after_empty_reply():
    client = FakeClient(Reply([]), "ok")
    chat(client, "q1")
    assert chat(client, "q2") == "ok"
    assert client.requests[1]["messages"] == [{"role": "user", "content": "q2"}]


def test_empty_reply_tokens_are_still_recorded():
    # The request was answered (and billed) even though the text is empty.
    client = FakeClient(Reply([], input_tokens=40, output_tokens=3))
    chat(client, "hi")
    mu = client.usage.by_model[FAKE_MODEL]
    assert (mu.requests, mu.input_tokens, mu.output_tokens) == (1, 40, 3)


# ---- usage ----

def test_successful_turn_sets_last_usage_and_records_it():
    client = FakeClient(Reply(["ok"], input_tokens=100, output_tokens=20))
    chat(client, "hi")
    assert client.last_usage.input_tokens == 100
    assert client.last_usage.output_tokens == 20
    mu = client.usage.by_model[FAKE_MODEL]
    assert (mu.requests, mu.input_tokens, mu.output_tokens) == (1, 100, 20)


def test_history_tokens_is_whole_prompt_plus_reply():
    client = FakeClient(Reply(["ok"], input_tokens=100, output_tokens=20, cache_read=300, cache_write=50))
    chat(client, "hi")
    assert client.history_tokens == 100 + 300 + 50 + 20


def test_usage_accumulates_over_turns():
    client = FakeClient(
        Reply(input_tokens=10, output_tokens=1),
        Reply(input_tokens=30, output_tokens=2),
    )
    chat(client, "a")
    chat(client, "b")
    mu = client.usage.by_model[FAKE_MODEL]
    assert (mu.requests, mu.input_tokens, mu.output_tokens) == (2, 40, 3)
    assert client.last_usage.input_tokens == 30


# ---- retries ----

@pytest.mark.parametrize("error", [RateLimited, Overloaded, ServerError, ConnectionFailed])
def test_retryable_error_is_retried_then_succeeds(error, sleeps):
    client = FakeClient(error("temporary"), "ok")
    assert chat(client, "hi") == "ok"
    assert len(client.requests) == 2
    assert len(sleeps) == 1
    assert client.messages[-1] == {"role": "assistant", "content": "ok"}


def test_retry_resends_the_same_messages(sleeps):
    client = FakeClient(ServerError("boom"), "ok")
    chat(client, "hi")
    assert client.requests[0]["messages"] == client.requests[1]["messages"] == [{"role": "user", "content": "hi"}]


def test_retry_prints_notice_to_stderr(sleeps, capsys):
    client = FakeClient(RateLimited("slow down"), "ok")
    chat(client, "hi")
    err = capsys.readouterr().err
    assert "[RateLimited, retry 1/5 in" in err


def test_gives_up_after_max_retries(sleeps):
    client = FakeClient(*[ServerError("down")] * 3, max_retries=2)
    with pytest.raises(ServerError):
        chat(client, "hi")
    assert len(client.requests) == 3  # first try + 2 retries
    assert len(sleeps) == 2
    assert client.messages == []


def test_zero_max_retries_means_one_attempt(sleeps):
    client = FakeClient(ServerError("down"), max_retries=0)
    with pytest.raises(ServerError):
        chat(client, "hi")
    assert len(client.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("error", [BadRequest("bad"), LLMError("base"), ValueError("bug")])
def test_non_retryable_error_is_raised_at_once(error, sleeps):
    client = FakeClient(error)
    with pytest.raises(type(error)):
        chat(client, "hi")
    assert len(client.requests) == 1
    assert sleeps == []


def test_no_retry_once_text_was_shown(sleeps):
    client = FakeClient(Reply(["Hel"], error=ServerError("cut")), "never used")
    received = []
    with pytest.raises(ServerError):
        for text in client.stream_chat("hi"):
            received.append(text)
    assert received == ["Hel"]
    assert len(client.requests) == 1
    assert sleeps == []
    assert client.messages == []


def test_server_retry_after_sets_minimum_delay(sleeps):
    client = FakeClient(RateLimited("wait", retry_after=3), "ok")
    chat(client, "hi")
    assert sleeps == [3]


def test_retry_after_is_capped_by_max_delay(sleeps):
    client = FakeClient(RateLimited("wait", retry_after=120), "ok", retry_max_delay=10)
    chat(client, "hi")
    assert sleeps == [10]


def test_backoff_delay_stays_within_exponential_bounds(sleeps):
    client = FakeClient(*[ServerError("down")] * 4, "ok", retry_base_delay=1.0, retry_max_delay=5.0)
    chat(client, "hi")
    for attempt, delay in enumerate(sleeps):
        assert 0 <= delay <= min(5.0, 2 ** attempt)


def test_failed_attempt_with_tokens_is_still_billed(sleeps):
    client = FakeClient(Reply([], input_tokens=50, error=ServerError("cut")), Reply(input_tokens=50, output_tokens=5))
    chat(client, "hi")
    mu = client.usage.by_model[FAKE_MODEL]
    assert mu.requests == 2
    assert mu.input_tokens == 100


def test_failed_attempt_without_tokens_is_not_recorded(sleeps):
    client = FakeClient(ServerError("refused"), Reply(input_tokens=50, output_tokens=5))
    chat(client, "hi")
    assert client.usage.by_model[FAKE_MODEL].requests == 1


# ---- failures keep history valid ----

def test_failure_removes_only_the_unanswered_user_message():
    client = FakeClient("first", BadRequest("bad"))
    chat(client, "q1")
    before = list(client.messages)
    with pytest.raises(BadRequest):
        chat(client, "q2")
    assert client.messages == before


def test_failure_keeps_previous_last_usage_and_history_tokens():
    client = FakeClient(Reply(input_tokens=40, output_tokens=4), BadRequest("bad"))
    chat(client, "q1")
    last_usage, history_tokens = client.last_usage, client.history_tokens
    with pytest.raises(BadRequest):
        chat(client, "q2")
    assert client.last_usage is last_usage
    assert client.history_tokens == history_tokens


def test_stopping_the_stream_early_removes_the_user_message():
    client = FakeClient(Reply(["a", "b", "c"]))
    stream = client.stream_chat("hi")
    next(stream)
    stream.close()
    assert client.messages == []


def test_keyboard_interrupt_is_not_retried(sleeps):
    client = FakeClient(KeyboardInterrupt(), "never used")
    with pytest.raises(KeyboardInterrupt):
        chat(client, "hi")
    assert len(client.requests) == 1
    assert client.messages == []


def test_can_chat_again_after_a_failure():
    client = FakeClient(BadRequest("bad"), "ok")
    with pytest.raises(BadRequest):
        chat(client, "q1")
    assert chat(client, "q2") == "ok"
    assert client.messages == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.parametrize("chunks", [[""], ["", ""], [" "], ["\n", "  "]])
def test_reply_without_visible_text_is_not_stored(chunks):
    # Anthropic rejects assistant messages that are empty or only whitespace.
    client = FakeClient(Reply(chunks))
    chat(client, "hi")
    assert client.messages == []


# ---- stop reason and reply limit ----

def test_default_reply_limit():
    assert FakeClient().max_tokens == 4096


def test_stop_reason_of_finished_turn():
    client = FakeClient(Reply(["cut"], stop_reason="max_tokens"))
    chat(client, "hi")
    assert client.last_stop_reason == "max_tokens"


def test_stop_reason_resets_on_the_next_turn():
    client = FakeClient(Reply(["cut"], stop_reason="max_tokens"), "full answer")
    chat(client, "q1")
    chat(client, "q2")
    assert client.last_stop_reason is None


def test_stop_reason_comes_from_the_successful_attempt(sleeps):
    # The failed attempt reported max_tokens; the retry that succeeded did not.
    client = FakeClient(Reply([], stop_reason="max_tokens", error=ServerError("cut")), "ok")
    chat(client, "hi")
    assert client.last_stop_reason is None


def test_reply_cut_at_limit_is_kept_in_history():
    client = FakeClient(Reply(["partial"], stop_reason="max_tokens"))
    chat(client, "hi")
    assert client.messages[-1] == {"role": "assistant", "content": "partial"}
