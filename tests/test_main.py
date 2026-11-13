"""Command-line options and the chat loop, with FakeClient in place of a real API client."""
import builtins

import pytest

import chat.chat_client
import main
from chat.error import BadRequest, ServerError
from fake_client import FakeClient, Reply


def run(monkeypatch, client: FakeClient, lines: list[str], argv: list[str] | None = None) -> None:
    """Run main() with `client` and the given input lines; the input ends with EOF."""
    monkeypatch.setitem(main.CLIENTS, "anthropic", lambda: client)
    monkeypatch.setitem(main.CLIENTS, "openai", lambda: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
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


# ---- startup checks ----

def test_missing_anthropic_key_exits_with_message(monkeypatch, capsys):
    # Uses the real factory: the key check lives in _anthropic_client().
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main.main([])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.out + captured.err


def test_openai_client_does_not_need_anthropic_key(monkeypatch, capsys):
    client = FakeClient("ok")
    inputs = iter(["hi"])
    monkeypatch.setitem(main.CLIENTS, "openai", lambda: client)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)
    main.main(["--client", "openai"])
    assert "ok\n" in capsys.readouterr().out


# ---- the loop survives interruptions and unexpected errors ----

def test_ctrl_c_during_reply_keeps_chat_running(monkeypatch, capsys):
    client = FakeClient(Reply(["Hel"], error=KeyboardInterrupt()), "ok")
    run(monkeypatch, client, ["first", "second"])
    out = capsys.readouterr().out
    assert "[interrupted]" in out
    assert "ok\n" in out
    assert client.messages == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok"},
    ]


def test_ctrl_c_while_waiting_to_retry_keeps_chat_running(monkeypatch, capsys):
    def interrupted_sleep(delay):
        raise KeyboardInterrupt

    monkeypatch.setattr(chat.chat_client.time, "sleep", interrupted_sleep)
    client = FakeClient(ServerError("down"), "ok")
    run(monkeypatch, client, ["first", "second"])
    out = capsys.readouterr().out
    assert "[interrupted]" in out
    assert "ok\n" in out


def test_unexpected_error_is_shown_and_loop_continues(monkeypatch, capsys):
    client = FakeClient(ValueError("bug"), "ok")
    run(monkeypatch, client, ["first", "second"])
    out = capsys.readouterr().out
    assert "[unexpected error: ValueError('bug')]" in out
    assert "ok\n" in out
    assert client.messages[0] == {"role": "user", "content": "second"}


def test_unexpected_error_does_not_repeat_previous_usage_line(monkeypatch, capsys):
    client = FakeClient(Reply(input_tokens=7, output_tokens=1), ValueError("bug"))
    run(monkeypatch, client, ["first", "second"])
    out = capsys.readouterr().out
    assert out.count("[fake-model: 7 in / 1 out") == 1


def test_interrupt_does_not_repeat_previous_usage_line(monkeypatch, capsys):
    client = FakeClient(Reply(input_tokens=7, output_tokens=1), Reply(["Hel"], error=KeyboardInterrupt()))
    run(monkeypatch, client, ["first", "second"])
    out = capsys.readouterr().out
    assert out.count("[fake-model: 7 in / 1 out") == 1


def test_empty_reply_does_not_repeat_previous_usage_or_trim_lines(monkeypatch, capsys):
    long = "x" * 4000
    # Turn 2 trims; turn 3 is empty and must not print turn 2's lines again.
    client = FakeClient(Reply([long]), Reply([long], input_tokens=3000, output_tokens=1004), Reply([]))
    run(monkeypatch, client, ["a" + long, "b" + long, "c"], ["--max-history-tokens", "3000"])
    captured = capsys.readouterr()
    assert captured.out.count("[fake-model: 3000 in / 1004 out") == 1
    assert captured.err.count("[history trimmed") == 1


def test_empty_reply_prints_notice(monkeypatch, capsys):
    client = FakeClient(Reply([]))
    run(monkeypatch, client, ["hi"])
    assert "[empty reply]" in capsys.readouterr().out


def test_empty_chunk_inside_reply_is_not_an_empty_reply(monkeypatch, capsys):
    client = FakeClient(Reply(["Hel", "", "lo"]))
    run(monkeypatch, client, ["hi"])
    out = capsys.readouterr().out
    assert "Hello\n" in out
    assert "[empty reply]" not in out


def test_reply_of_only_empty_chunks_prints_notice(monkeypatch, capsys):
    client = FakeClient(Reply(["", ""]))
    run(monkeypatch, client, ["hi"])
    assert "[empty reply]" in capsys.readouterr().out


def test_whitespace_chunks_are_printed(monkeypatch, capsys):
    # Models often stream spaces and line breaks as separate chunks.
    client = FakeClient(Reply(["Hello", " ", "world", "\n\n", "Bye"]))
    run(monkeypatch, client, ["hi"])
    assert "Hello world\n\nBye\n" in capsys.readouterr().out


# ---- --max-tokens and stop-reason notices ----

def test_max_tokens_option_sets_client_limit(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, [], ["--max-tokens", "500"])
    assert client.max_tokens == 500


def test_without_max_tokens_option_default_is_kept(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, [])
    assert client.max_tokens == 4096


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_invalid_max_tokens_exits(value):
    with pytest.raises(SystemExit) as exc:
        main.parse_args(["--max-tokens", value])
    assert exc.value.code == 2


@pytest.mark.parametrize("stop_reason, notice", [
    ("max_tokens", "[reply cut: reached the 300-token limit, use --max-tokens to raise it]"),
    ("context_full", "[reply cut: the model's context window is full, use /reset or --max-history-tokens]"),
    ("refusal", "[the model declined to answer]"),
])
def test_stop_reason_notice(monkeypatch, capsys, stop_reason, notice):
    client = FakeClient(Reply(["partial"], stop_reason=stop_reason))
    run(monkeypatch, client, ["hi"], ["--max-tokens", "300"])
    assert notice in capsys.readouterr().err


def test_normal_end_prints_no_notice(monkeypatch, capsys):
    run(monkeypatch, FakeClient("done"), ["hi"])
    err = capsys.readouterr().err
    assert "[reply cut" not in err
    assert "declined" not in err


def test_empty_reply_cut_at_limit_shows_both_notices(monkeypatch, capsys):
    # A reasoning model can spend the whole limit thinking and return no text.
    client = FakeClient(Reply([], stop_reason="max_tokens"))
    run(monkeypatch, client, ["hi"])
    captured = capsys.readouterr()
    assert "[empty reply]" in captured.out
    assert "[reply cut: reached the 4096-token limit" in captured.err


def test_cut_notice_is_not_repeated_after_error(monkeypatch, capsys):
    client = FakeClient(Reply(["partial"], stop_reason="max_tokens"), BadRequest("bad"))
    run(monkeypatch, client, ["first", "second"])
    assert capsys.readouterr().err.count("[reply cut") == 1


def test_unknown_stop_reason_does_not_crash(monkeypatch, capsys):
    client = FakeClient(Reply(["ok"], stop_reason="something_new"), "second")
    run(monkeypatch, client, ["first", "again"])
    assert "second\n" in capsys.readouterr().out
