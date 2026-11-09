"""History trimming: drop the oldest turns when history grows past max_history_tokens."""
import pytest

from chat.chat_client import estimate_tokens
from chat.error import BadRequest
from fake_client import FakeClient, Reply, chat

# 400 characters ≈ 104 estimated tokens per message, so one turn ≈ 208 tokens.
LONG = "x" * 400


def run_turns(client: FakeClient, n: int) -> None:
    for i in range(n):
        client.queue(Reply([f"answer {i} " + LONG]))
        chat(client, f"question {i} " + LONG)


def assert_valid_history(messages: list[dict]) -> None:
    assert len(messages) % 2 == 0
    for i, m in enumerate(messages):
        assert m["role"] == ("user" if i % 2 == 0 else "assistant")


# ---- estimate_tokens ----

@pytest.mark.parametrize("content, expected", [
    ("", 4),
    ("abc", 4),
    ("abcd", 5),
    ("x" * 400, 104),
])
def test_estimate_tokens(content, expected):
    assert estimate_tokens({"role": "user", "content": content}) == expected


# ---- when trimming happens ----

def test_no_limit_never_trims():
    client = FakeClient()
    run_turns(client, 10)
    assert len(client.messages) == 20
    assert client.last_trimmed == 0


def test_below_limit_nothing_is_dropped():
    client = FakeClient()
    client.max_history_tokens = 10_000
    run_turns(client, 3)
    assert len(client.messages) == 6
    assert client.last_trimmed == 0


def test_exactly_at_limit_nothing_is_dropped():
    client = FakeClient(Reply(input_tokens=900, output_tokens=100), Reply(input_tokens=900, output_tokens=100))
    client.max_history_tokens = 1000
    chat(client, "q1")
    chat(client, "q2")
    assert client.last_trimmed == 0
    assert len(client.messages) == 4


def test_above_limit_trims_below_target():
    client = FakeClient()
    client.max_history_tokens = 1000
    run_turns(client, 5)  # ~1040 tokens after the 5th turn
    assert client.last_trimmed > 0
    assert client.history_tokens <= 1000 * client.trim_target_ratio


def test_trimming_drops_whole_turns_and_keeps_roles_alternating():
    client = FakeClient()
    client.max_history_tokens = 1000
    run_turns(client, 12)
    assert client.last_trimmed % 2 == 0
    assert_valid_history(client.messages)


def test_oldest_turns_are_dropped_first():
    client = FakeClient()
    client.max_history_tokens = 1000
    run_turns(client, 5)
    first_kept = client.messages[0]["content"]
    assert first_kept.startswith("question 3 ")
    assert client.messages[-1]["content"].startswith("answer 4 ")


def test_trimmed_messages_are_not_sent_again():
    client = FakeClient()
    client.max_history_tokens = 1000
    run_turns(client, 5)
    client.queue("ok")
    chat(client, "next")
    sent = [m["content"] for m in client.requests[-1]["messages"]]
    assert not any(c.startswith(("question 0 ", "answer 0 ")) for c in sent)


def test_last_trimmed_resets_on_the_next_turn():
    client = FakeClient()
    client.max_history_tokens = 1000
    run_turns(client, 5)
    assert client.last_trimmed > 0
    run_turns(client, 1)
    assert client.last_trimmed == 0


def test_trimming_happens_again_as_history_grows():
    client = FakeClient()
    client.max_history_tokens = 1000
    trims = 0
    for _ in range(15):
        run_turns(client, 1)
        trims += client.last_trimmed > 0
    assert trims >= 3
    assert client.history_tokens <= 1000


def test_higher_target_ratio_drops_fewer_messages():
    low, high = FakeClient(), FakeClient()
    for client, ratio in ((low, 0.3), (high, 0.9)):
        client.max_history_tokens = 1000
        client.trim_target_ratio = ratio
        run_turns(client, 5)
    assert low.last_trimmed > high.last_trimmed > 0


# ---- what is always kept ----

def test_newest_turn_is_kept_even_if_larger_than_limit():
    client = FakeClient(Reply(["huge answer"], input_tokens=50_000, output_tokens=5_000))
    client.max_history_tokens = 1000
    chat(client, "huge question")
    assert client.last_trimmed == 0
    assert [m["content"] for m in client.messages] == ["huge question", "huge answer"]


def test_newest_turn_is_kept_when_older_turns_are_not_enough():
    client = FakeClient(
        Reply(["small"], input_tokens=10, output_tokens=10),
        Reply(["huge"], input_tokens=50_000, output_tokens=5_000),
    )
    client.max_history_tokens = 1000
    chat(client, "q1")
    chat(client, "q2")
    assert client.last_trimmed == 2
    assert [m["content"] for m in client.messages] == ["q2", "huge"]


def test_system_prompt_survives_trimming():
    client = FakeClient()
    client.set_system("You are a pirate.")
    client.max_history_tokens = 1000
    run_turns(client, 6)
    client.queue("arr")
    chat(client, "next")
    assert client.system == "You are a pirate."
    assert client.requests[-1]["system"] == "You are a pirate."


def test_history_tokens_never_goes_negative():
    # The API reports fewer tokens than the rough estimate of what is dropped.
    # Here the API says 150 tokens, but the dropped turn alone is estimated at ~208.
    client = FakeClient(Reply([LONG], input_tokens=5, output_tokens=5), Reply([LONG], input_tokens=150, output_tokens=0))
    client.max_history_tokens = 100
    chat(client, LONG)
    chat(client, LONG)
    assert client.last_trimmed == 2
    assert client.history_tokens == 0


# ---- interaction with failures and reset ----

def test_failed_request_does_not_trim():
    client = FakeClient()
    client.max_history_tokens = 1000
    run_turns(client, 4)  # just under the limit
    before = list(client.messages)
    client.queue(BadRequest("bad"))
    with pytest.raises(BadRequest):
        chat(client, "q " + LONG * 10)
    assert client.messages == before


def test_reset_after_trimming_starts_fresh():
    client = FakeClient()
    client.max_history_tokens = 1000
    run_turns(client, 5)
    client.reset()
    assert client.messages == []
    assert client.history_tokens == 0
    run_turns(client, 1)
    assert client.last_trimmed == 0
    assert len(client.messages) == 2
