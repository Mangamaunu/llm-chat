"""Conversation: history bookkeeping and trimming, without a client or fake replies."""
import pytest

from llm_chat.core.conversation import Conversation


def turn(conv: Conversation, i: int, prompt_tokens: int, output_tokens: int = 0, size: int = 100) -> int:
    """One finished turn with messages of `size` characters; returns the messages dropped."""
    conv.add_user(f"q{i} ".ljust(size, "x"))
    return conv.add_reply(f"a{i} ".ljust(size, "x"), prompt_tokens, output_tokens)


def contents(conv: Conversation) -> list[str]:
    return [m["content"].split()[0] for m in conv.messages]


# ---- bookkeeping ----

def test_add_user_and_reply():
    conv = Conversation()
    conv.add_user("hi")
    assert conv.add_reply("hello", prompt_tokens=10, output_tokens=3) == 0
    assert conv.messages == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert conv.history_tokens == 13


def test_drop_last_user_rolls_back_the_turn():
    conv = Conversation()
    conv.add_user("hi")
    conv.add_reply("hello", 10, 3)
    conv.add_user("unanswered")
    conv.drop_last_user()
    assert [m["content"] for m in conv.messages] == ["hi", "hello"]
    assert conv.history_tokens == 13


def test_drop_last_user_refuses_to_drop_a_reply():
    conv = Conversation()
    conv.add_user("hi")
    conv.add_reply("hello", 10, 3)
    with pytest.raises(AssertionError):
        conv.drop_last_user()


def test_reset_keeps_system_prompt():
    conv = Conversation(system="be brief", max_history_tokens=500)
    turn(conv, 0, 50)
    conv.reset()
    assert conv.messages == []
    assert conv.history_tokens == 0
    assert conv.system == "be brief"
    assert conv.max_history_tokens == 500


def test_each_conversation_has_its_own_list():
    a, b = Conversation(), Conversation()
    a.add_user("hi")
    assert b.messages == []


# ---- when trimming happens ----

def test_no_limit_never_trims():
    conv = Conversation()
    assert sum(turn(conv, i, 100_000) for i in range(5)) == 0
    assert len(conv.messages) == 10


@pytest.mark.parametrize("tokens, dropped", [(999, 0), (1000, 0), (1001, 2)])
def test_trims_only_above_the_limit(tokens, dropped):
    conv = Conversation(max_history_tokens=1000)
    turn(conv, 0, 10)
    assert turn(conv, 1, tokens) == dropped


def test_trims_oldest_turns_down_to_the_target():
    # 4 turns of equal size, 1200 tokens: 150 per message, target 500.
    conv = Conversation(max_history_tokens=1000)
    for i in range(3):
        turn(conv, i, 10)
    assert turn(conv, 3, 1200) == 6
    assert contents(conv) == ["q3", "a3"]
    assert conv.history_tokens == 300


def test_stops_as_soon_as_below_the_target():
    conv = Conversation(max_history_tokens=1000, trim_target_ratio=0.8)
    for i in range(3):
        turn(conv, i, 10)
    assert turn(conv, 3, 1200) == 4  # 1200 -> 900 -> 600 <= 800
    assert contents(conv) == ["q2", "a2", "q3", "a3"]
    assert conv.history_tokens == 600


def test_higher_target_ratio_drops_fewer_messages():
    def dropped(ratio: float) -> int:
        conv = Conversation(max_history_tokens=1000, trim_target_ratio=ratio)
        for i in range(5):
            turn(conv, i, 10)
        return turn(conv, 5, 1200)
    assert dropped(0.3) > dropped(0.9) > 0


# ---- what is always kept ----

def test_newest_turn_is_kept_even_if_larger_than_limit():
    conv = Conversation(max_history_tokens=1000)
    assert turn(conv, 0, 50_000, 5_000) == 0
    assert contents(conv) == ["q0", "a0"]


def test_newest_turn_is_kept_when_older_turns_are_not_enough():
    conv = Conversation(max_history_tokens=1000)
    turn(conv, 0, 10)
    assert turn(conv, 1, 50_000) == 2
    assert contents(conv) == ["q1", "a1"]


def test_system_prompt_is_never_dropped_but_counts_in_the_ratio():
    # 400 system chars + 4 x 100 message chars = 800 chars for 800 tokens: 1 per char.
    conv = Conversation(system="s" * 400, max_history_tokens=500)
    turn(conv, 0, 10)
    assert turn(conv, 1, 800) == 2
    assert conv.system == "s" * 400
    assert conv.history_tokens == 600


def test_history_stays_valid_after_trimming():
    conv = Conversation(max_history_tokens=1000)
    for i in range(20):
        turn(conv, i, 300 * (i % 5))
        assert len(conv.messages) % 2 == 0
        assert [m["role"] for m in conv.messages] == ["user", "assistant"] * (len(conv.messages) // 2)


# ---- size of dropped messages comes from the real total (review M5) ----

@pytest.mark.parametrize("tokens_per_char", [0.25, 0.6, 2.0])
def test_dropped_size_follows_this_conversations_ratio(tokens_per_char):
    # 4 messages of 100 chars; whatever the language, dropping half leaves half.
    conv = Conversation(max_history_tokens=10)
    turn(conv, 0, 1)
    total = int(400 * tokens_per_char)
    assert turn(conv, 1, total) == 2
    assert conv.history_tokens == total // 2


def test_messages_of_different_length_are_weighted_by_length():
    conv = Conversation(max_history_tokens=100)
    turn(conv, 0, 1, size=300)
    turn_dropped = turn(conv, 1, 400, size=100)  # 800 chars, 0.5 tokens per char
    assert turn_dropped == 2
    assert conv.history_tokens == 100
