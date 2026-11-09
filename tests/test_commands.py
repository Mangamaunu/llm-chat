"""Slash commands, run against a FakeClient."""
import pytest

from commands import COMMANDS, Context, dispatch
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
    names = {c.name for c in COMMANDS.values()}
    for name in names:
        assert out.count(f"/{name} ") == 1
    assert "/quit" not in out  # aliases are not listed separately


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
