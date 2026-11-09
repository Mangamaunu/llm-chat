"""Command-line options and the chat loop, with FakeClient in place of a real API client."""
import builtins

import pytest

import main
from chat.error import BadRequest
from fake_client import FakeClient, Reply


def run(monkeypatch, client: FakeClient, lines: list[str], argv: list[str] | None = None) -> None:
    """Run main() with `client` and the given input lines; the input ends with EOF."""
    monkeypatch.setitem(main.CLIENTS, "anthropic", lambda: client)
    inputs = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)
    main.main(argv or [])


# ---- parse_args ----

def test_defaults():
    args = main.parse_args([])
    assert (args.client, args.model, args.max_history_tokens) == ("anthropic", None, None)


def test_all_options():
    args = main.parse_args(["--client", "openai", "--model", "llama3.2", "--max-history-tokens", "400"])
    assert (args.client, args.model, args.max_history_tokens) == ("openai", "llama3.2", 400)


@pytest.mark.parametrize("argv", [
    ["--client", "gemini"],
    ["--max-history-tokens", "abc"],
    ["--max-history-tokens", "0"],
    ["--max-history-tokens", "-5"],
])
def test_invalid_options_exit(argv):
    with pytest.raises(SystemExit) as exc:
        main.parse_args(argv)
    assert exc.value.code == 2


# ---- the chat loop ----

def test_prints_client_and_model_on_start(monkeypatch, capsys):
    run(monkeypatch, FakeClient(), [])
    assert capsys.readouterr().out.startswith("[anthropic: fake-model]")


def test_options_are_applied_to_client(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, [], ["--model", "m2", "--max-history-tokens", "500"])
    assert client.model == "m2"
    assert client.max_history_tokens == 500


def test_reply_and_usage_line(monkeypatch, capsys):
    client = FakeClient(Reply(["Hel", "lo"], input_tokens=12, output_tokens=3))
    run(monkeypatch, client, ["hi"])
    out = capsys.readouterr().out
    assert "Hello\n" in out
    assert "[fake-model: 12 in / 3 out, $0.000018]" in out


def test_unpriced_model_shows_na(monkeypatch, capsys):
    client = FakeClient(Reply(input_tokens=1, output_tokens=1))
    run(monkeypatch, client, ["hi"], ["--model", "unpriced-model"])
    assert "[unpriced-model: 1 in / 1 out, n/a]" in capsys.readouterr().out


def test_empty_lines_are_skipped(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, ["", "   ", "hi"])
    assert len(client.requests) == 1


def test_commands_are_dispatched(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, ["/system be brief"])
    assert client.system == "be brief"
    assert client.requests == []


def test_exit_command_stops_before_remaining_input(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, ["/exit", "hi"])
    assert client.requests == []


def test_llm_error_is_shown_and_loop_continues(monkeypatch, capsys):
    client = FakeClient(BadRequest("model not found"), "ok")
    run(monkeypatch, client, ["first", "second"])
    out = capsys.readouterr().out
    assert "[error: model not found]" in out
    assert "ok\n" in out


def test_trim_notice_is_printed(monkeypatch, capsys):
    long = "x" * 4000
    # Each message is ~1004 tokens: 2008 after turn 1, 4016 after turn 2, so turn 2 trims.
    client = FakeClient(*[Reply([long])] * 2)
    run(monkeypatch, client, ["a" + long, "b" + long], ["--max-history-tokens", "3000"])
    err = capsys.readouterr().err
    assert "[history trimmed: dropped 2 old messages, ~2 008 tokens left]" in err
    assert err.count("[history trimmed") == 1


def test_no_trim_notice_without_limit(monkeypatch, capsys):
    long = "x" * 4000
    client = FakeClient(*[Reply([long])] * 3)
    run(monkeypatch, client, ["a" + long, "b" + long, "c" + long])
    captured = capsys.readouterr()
    assert "[history trimmed" not in captured.out + captured.err
