"""Slash commands, run against a FakeClient."""
import pytest

from llm_chat.commands import COMMANDS, Context, dispatch
from fake_client import FAKE_MODEL, FakeClient, Reply, chat


@pytest.fixture
def ctx() -> Context:
    return Context(client=FakeClient())


@pytest.mark.parametrize("line", ["/exit", "/quit", "/q"])
def test_exit_and_aliases_stop_the_loop(ctx, line):
    assert dispatch(ctx, line) is False


def test_unknown_command_keeps_running(ctx, capsys):
    assert dispatch(ctx, "/nope") is True
    assert "Unknown command: /nope" in capsys.readouterr().out


def test_reset_clears_history(ctx, capsys):
    ctx.client.queue("ok")
    chat(ctx.client, "hi")
    assert dispatch(ctx, "/reset") is True
    assert ctx.client.messages == []
    assert "cleared" in capsys.readouterr().out


def test_system_sets_prompt_with_spaces(ctx):
    dispatch(ctx, "/system   You are a pirate.  ")
    assert ctx.client.system == "You are a pirate."


def test_model_switches_model(ctx):
    assert dispatch(ctx, "/model other-model") is True
    assert ctx.client.model == "other-model"


def test_help_lists_each_command_once(ctx, capsys):
    dispatch(ctx, "/help")
    out = capsys.readouterr().out
    listed = [line.split()[0] for line in out.splitlines() if line.startswith("  /")]
    assert sorted(listed) == sorted({f"/{c.name}" for c in COMMANDS.values()})
    assert "/quit" not in out  # aliases are not listed separately


def test_help_explains_multiline_input(ctx, capsys):
    dispatch(ctx, "/help")
    assert 'type """ on its own line' in capsys.readouterr().out


# ---- /model and /system without and with arguments (review M8) ----

def test_model_without_argument_shows_current_model(ctx, capsys):
    assert dispatch(ctx, "/model") is True
    assert capsys.readouterr().out == "Model: fake-model\n"
    assert ctx.client.model == "fake-model"


def test_model_with_argument_confirms(ctx, capsys):
    dispatch(ctx, "/model other-model")
    assert "Model set to other-model." in capsys.readouterr().out


def test_model_with_only_spaces_does_not_clear_the_model(ctx):
    dispatch(ctx, "/model    ")
    assert ctx.client.model == "fake-model"


def test_system_without_argument_when_none_is_set(ctx, capsys):
    dispatch(ctx, "/system")
    assert "No system prompt" in capsys.readouterr().out
    assert ctx.client.system == ""


def test_system_without_argument_does_not_clear_it(ctx, capsys):
    ctx.client.set_system("You are a pirate.")
    dispatch(ctx, "/system")
    assert "You are a pirate." in capsys.readouterr().out
    assert ctx.client.system == "You are a pirate."


def test_system_with_argument_confirms(ctx, capsys):
    dispatch(ctx, "/system be brief")
    assert "System prompt set." in capsys.readouterr().out


def test_system_off_clears_it(ctx, capsys):
    ctx.client.set_system("You are a pirate.")
    dispatch(ctx, "/system off")
    assert ctx.client.system == ""
    assert "System prompt cleared." in capsys.readouterr().out


def test_system_prompt_may_contain_off(ctx):
    dispatch(ctx, "/system switch the lights off")
    assert ctx.client.system == "switch the lights off"


def test_multiline_argument(ctx):
    dispatch(ctx, "/system\nline one\n  line two")
    assert ctx.client.system == "line one\n  line two"


def test_bare_slash_is_unknown_command(ctx, capsys):
    assert dispatch(ctx, "/") is True
    assert "Unknown command: /." in capsys.readouterr().out


def test_cost_without_requests(ctx, capsys):
    dispatch(ctx, "/cost")
    assert "No API requests yet" in capsys.readouterr().out


def test_cost_shows_tokens_and_price(ctx, capsys):
    # fake-model: $1 per 1M input, $2 per 1M output
    ctx.client.queue(Reply(input_tokens=16_820, output_tokens=1_000))
    chat(ctx.client, "hi")
    dispatch(ctx, "/cost")
    out = capsys.readouterr().out
    assert "Session cost (1 requests)" in out
    assert FAKE_MODEL in out
    assert "16 820" in out
    assert "$0.0188" in out  # 16820 * 1/1M + 1000 * 2/1M
    assert "* excludes" not in out


def test_cost_marks_unpriced_models(ctx, capsys):
    dispatch(ctx, "/model unpriced-model")
    ctx.client.queue(Reply(input_tokens=10, output_tokens=1))
    chat(ctx.client, "hi")
    dispatch(ctx, "/cost")
    out = capsys.readouterr().out
    assert "unpriced-model" in out
    assert "n/a" in out
    assert "* excludes models with no price in pricing.toml" in out


def test_cost_mixed_priced_and_unpriced(ctx, capsys):
    ctx.client.queue(Reply(input_tokens=1_000_000, output_tokens=0))
    chat(ctx.client, "hi")
    dispatch(ctx, "/model unpriced-model")
    ctx.client.queue(Reply(input_tokens=10, output_tokens=1))
    chat(ctx.client, "hi")
    dispatch(ctx, "/cost")
    out = capsys.readouterr().out
    assert "$1.0000*" in out
    assert "* excludes models with no price in pricing.toml" in out


def test_cost_shows_cache_line_when_cache_was_used(ctx, capsys):
    ctx.client.queue(Reply(input_tokens=10, output_tokens=1, cache_read=1_000_000, cache_write=0))
    chat(ctx.client, "hi")
    dispatch(ctx, "/cost")
    out = capsys.readouterr().out
    assert "cache_write: 0 tokens" in out
    assert "saved by caching: $0.9000" in out  # 1M * (1.0 - 0.1) / 1M
