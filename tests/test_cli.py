"""Command-line options and the chat loop, with FakeClient in place of a real API client."""
import builtins

import pytest

from llm_chat import cli
from llm_chat.core.error import BadRequest, ServerError
from fake_client import FakeClient, Reply


def fake_inputs(monkeypatch, lines: list) -> list[str]:
    """Replace input() with the given lines. An exception in the list is raised
    at that point (e.g. KeyboardInterrupt for Ctrl+C). After the last line: EOF.
    Returns the list of prompts that were shown."""
    inputs = iter(lines)
    prompts: list[str] = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        try:
            item = next(inputs)
        except StopIteration:
            raise EOFError
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(builtins, "input", fake_input)
    return prompts


def factory(client: FakeClient):
    """A CLIENTS entry that returns `client`, applying --model like the real factories."""
    def make(model=None):
        if model:
            client.model = model
        return client
    return make


def run(monkeypatch, client: FakeClient, lines: list, argv: list[str] | None = None) -> list[str]:
    """Run main() with `client` and the given input lines; the input ends with EOF."""
    monkeypatch.setitem(cli.CLIENTS, "anthropic", factory(client))
    monkeypatch.setitem(cli.CLIENTS, "openai", factory(client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    prompts = fake_inputs(monkeypatch, lines)
    cli.main(argv or [])
    return prompts


# ---- parse_args ----

def test_defaults():
    args = cli.parse_args([])
    assert (args.client, args.model, args.max_history_tokens) == ("anthropic", None, None)


def test_all_options():
    args = cli.parse_args(["--client", "openai", "--model", "llama3.2", "--max-history-tokens", "400"])
    assert (args.client, args.model, args.max_history_tokens) == ("openai", "llama3.2", 400)


@pytest.mark.parametrize("argv", [
    ["--client", "gemini"],
    ["--max-history-tokens", "abc"],
    ["--max-history-tokens", "0"],
    ["--max-history-tokens", "-5"],
])
def test_invalid_options_exit(argv):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(argv)
    assert exc.value.code == 2


# ---- the chat loop ----

def test_prints_client_and_model_on_start(monkeypatch, capsys):
    run(monkeypatch, FakeClient(), [])
    assert capsys.readouterr().out.startswith("[anthropic: fake-model]")


def test_options_are_applied_to_client(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, [], ["--model", "m2", "--max-history-tokens", "500"])
    assert client.model == "m2"
    assert client.conversation.max_history_tokens == 500


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
    assert client.conversation.system == "be brief"
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
        cli.main([])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "ANTHROPIC_API_KEY" in captured.out + captured.err


def test_openai_client_does_not_need_anthropic_key(monkeypatch, capsys):
    client = FakeClient("ok")
    inputs = iter(["hi"])
    monkeypatch.setitem(cli.CLIENTS, "openai", factory(client))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)
    cli.main(["--client", "openai"])
    assert "ok\n" in capsys.readouterr().out


# ---- the loop survives interruptions and unexpected errors ----

def test_ctrl_c_during_reply_keeps_chat_running(monkeypatch, capsys):
    client = FakeClient(Reply(["Hel"], error=KeyboardInterrupt()), "ok")
    run(monkeypatch, client, ["first", "second"])
    out = capsys.readouterr().out
    assert "[interrupted]" in out
    assert "ok\n" in out
    assert client.conversation.messages == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok"},
    ]


def test_ctrl_c_while_waiting_to_retry_keeps_chat_running(monkeypatch, capsys):
    def interrupted_sleep(delay):
        raise KeyboardInterrupt

    client = FakeClient(ServerError("down"), "ok", sleep=interrupted_sleep)
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
    assert client.conversation.messages[0] == {"role": "user", "content": "second"}


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
        cli.parse_args(["--max-tokens", value])
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


# ---- prices file problems (H3) ----

def test_prices_problem_prints_warning_and_chat_works(monkeypatch, capsys, tmp_path):
    from llm_chat.core.usage import PRICING_ENV, UsageTracker

    monkeypatch.setenv(PRICING_ENV, str(tmp_path / "nope.toml"))
    # UsageTracker() loads from the (missing) file, like a real client.
    client = FakeClient(Reply(input_tokens=5, output_tokens=1), usage=UsageTracker())
    run(monkeypatch, client, ["hi"])
    captured = capsys.readouterr()
    assert "[warning: could not load prices from" in captured.err
    assert "costs will show n/a]" in captured.err
    assert "[fake-model: 5 in / 1 out, n/a]" in captured.out


def test_no_warning_when_prices_load(monkeypatch, capsys):
    run(monkeypatch, FakeClient(), [])
    assert "[warning" not in capsys.readouterr().err


def test_python_dash_m_runs_the_cli():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "llm_chat", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage: llm-chat" in result.stdout


# ---- client factories and `python -m llm_chat` ----

def test_anthropic_factory_creates_anthropic_client(monkeypatch):
    from llm_chat.providers.anthropic_client import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert isinstance(cli.CLIENTS["anthropic"](), AnthropicClient)


def test_openai_factory_creates_openai_client():
    from llm_chat.providers.openai_client import OpenAiClient

    assert isinstance(cli.CLIENTS["openai"](), OpenAiClient)


# Set after the modules were imported: the environment is read when the client is created.

def test_anthropic_model_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-env")
    assert cli.CLIENTS["anthropic"]().model == "claude-env"


def test_anthropic_default_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert cli.CLIENTS["anthropic"]().model == cli.ANTHROPIC_DEFAULT_MODEL


def test_openai_settings_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-env")
    client = cli.CLIENTS["openai"]()
    assert client.model == "gpt-env"
    assert str(client._client.base_url) == "https://api.example.com/v1/"
    assert client._client.api_key == "sk-env"
    assert client._limit_param == "max_completion_tokens"


def test_openai_defaults_point_at_ollama(monkeypatch):
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    client = cli.CLIENTS["openai"]()
    assert client.model == cli.OPENAI_DEFAULT_MODEL
    assert str(client._client.base_url) == cli.OPENAI_DEFAULT_BASE_URL
    assert client._limit_param == "max_tokens"


@pytest.mark.parametrize("name, env", [("anthropic", "ANTHROPIC_MODEL"), ("openai", "OPENAI_MODEL")])
def test_model_option_wins_over_environment(monkeypatch, name, env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv(env, "from-env")
    assert cli.CLIENTS[name]("from-option").model == "from-option"


# ---- retry notice ----

def test_retry_prints_notice_to_stderr(monkeypatch, capsys):
    from llm_chat.core.error import RateLimited

    client = FakeClient(RateLimited("slow down"), "ok", max_retries=3)
    run(monkeypatch, client, ["hi"])
    captured = capsys.readouterr()
    assert "[RateLimited, retry 1/3 in" in captured.err
    assert "ok\n" in captured.out


def test_package_main_module_calls_cli(monkeypatch, capsys):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["llm-chat", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("llm_chat", run_name="__main__")
    assert exc.value.code == 0
    assert "usage: llm-chat" in capsys.readouterr().out


# ---- multi-line input and line editing (review L4) ----

Q = '"""'


def sent(client: FakeClient) -> list[str]:
    """The user messages the client received, in order."""
    return [r["messages"][-1]["content"] for r in client.requests]


def test_multiline_block_is_sent_as_one_message(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, [Q, "line one", "line two", Q])
    assert sent(client) == ["line one\nline two"]


def test_multiline_block_keeps_indentation_and_inner_blank_lines(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, [Q, "", "def f():", "    return 1", "", "print(f())", "", Q])
    assert sent(client) == ["def f():\n    return 1\n\nprint(f())"]


def test_text_before_opening_quotes_is_kept(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, [f"Explain this code: {Q}", "x = 1", Q])
    assert sent(client) == ["Explain this code:\nx = 1"]


def test_text_before_closing_quotes_is_kept(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, [Q, "first", f"last{Q}"])
    assert sent(client) == ["first\nlast"]


def test_continuation_lines_get_their_own_prompt(monkeypatch):
    prompts = run(monkeypatch, FakeClient("ok"), [Q, "a", Q])
    assert prompts[1:3] == ["... ", "... "]


def test_quotes_closed_on_the_same_line_are_plain_text(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, [f"{Q}hi{Q}"])
    assert sent(client) == [f"{Q}hi{Q}"]


def test_quotes_in_the_middle_of_a_line_are_plain_text(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, [f"what does {Q} mean in python?"])
    assert sent(client) == [f"what does {Q} mean in python?"]


def test_block_starting_with_slash_is_a_message_not_a_command(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, [Q, "/usr/bin/python crashes, why?", Q])
    assert sent(client) == ["/usr/bin/python crashes, why?"]


def test_command_with_multiline_argument(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, [f"/system {Q}", "You are a pirate.", "Answer briefly.", Q])
    assert client.conversation.system == "You are a pirate.\nAnswer briefly."
    assert client.requests == []


def test_empty_block_sends_nothing(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, [Q, "", "   ", Q])
    assert client.requests == []


def test_ctrl_c_inside_block_cancels_only_the_block(monkeypatch, capsys):
    client = FakeClient("ok")
    run(monkeypatch, client, [Q, "half a thought", KeyboardInterrupt(), "next question"])
    assert "[cancelled]" in capsys.readouterr().out
    assert sent(client) == ["next question"]


def test_ctrl_d_inside_block_quits(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, [Q, "unfinished"])  # EOF follows
    assert client.requests == []


def test_ctrl_c_at_the_prompt_still_quits(monkeypatch):
    client = FakeClient()
    run(monkeypatch, client, [KeyboardInterrupt(), "never read"])
    assert client.requests == []


def test_single_line_input_is_still_stripped(monkeypatch):
    client = FakeClient("ok")
    run(monkeypatch, client, ["   hello   "])
    assert sent(client) == ["hello"]


def test_line_editing_is_enabled(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "readline", raising=False)
    run(monkeypatch, FakeClient(), [])
    assert "readline" in sys.modules


def test_missing_readline_is_fine(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "readline", None)  # makes `import readline` fail
    cli.enable_line_editing()


# ---- cached tokens in the usage line (review M7) ----

def test_usage_line_without_cache_is_unchanged(monkeypatch, capsys):
    run(monkeypatch, FakeClient(Reply(input_tokens=120, output_tokens=50)), ["hi"])
    out = capsys.readouterr().out
    assert "[fake-model: 120 in / 50 out, $0.000220]" in out
    assert "cache" not in out


def test_usage_line_shows_cache_reads(monkeypatch, capsys):
    # fake-model: input $1, cache read $0.1, output $2 per 1M tokens
    client = FakeClient(Reply(input_tokens=120, output_tokens=50, cache_read=15_300))
    run(monkeypatch, client, ["hi"])
    assert "[fake-model: 120 in + 15 300 cached / 50 out, $0.001750]" in capsys.readouterr().out


def test_usage_line_shows_cache_writes(monkeypatch, capsys):
    # cache write $1.25 per 1M tokens
    client = FakeClient(Reply(input_tokens=120, output_tokens=50, cache_write=900))
    run(monkeypatch, client, ["hi"])
    assert "[fake-model: 120 in + 900 cache write / 50 out, $0.001345]" in capsys.readouterr().out


def test_usage_line_shows_cache_reads_and_writes(monkeypatch, capsys):
    client = FakeClient(Reply(input_tokens=120, output_tokens=50, cache_read=15_300, cache_write=900))
    run(monkeypatch, client, ["hi"])
    assert "[fake-model: 120 in + 15 300 cached + 900 cache write / 50 out, $0.002875]" in capsys.readouterr().out


def test_usage_line_shows_cache_tokens_for_unpriced_model(monkeypatch, capsys):
    # Token counts are known even when the price isn't.
    client = FakeClient(Reply(input_tokens=120, output_tokens=50, cache_read=15_300))
    run(monkeypatch, client, ["hi"], ["--model", "unpriced-model"])
    assert "[unpriced-model: 120 in + 15 300 cached / 50 out, n/a]" in capsys.readouterr().out


# ---- TTFT, speed and total time ----

def fake_clock(monkeypatch, *values: float) -> None:
    """Make time.perf_counter() return `values` one by one."""
    times = iter(values)
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(times))


def test_prints_ttft_speed_and_total(monkeypatch, capsys):
    # perf_counter() is called three times: start, first text, end.
    fake_clock(monkeypatch, 10.0, 10.5, 12.5)
    client = FakeClient(Reply(["Hello"], output_tokens=100))
    run(monkeypatch, client, ["hi"])
    assert "[TTFT 0.50s, 50 tok/s, total 2.50s]" in capsys.readouterr().err


def test_no_timing_line_without_text(monkeypatch, capsys):
    fake_clock(monkeypatch, 10.0, 12.5)  # only start and end: no first text
    client = FakeClient(Reply([""]))
    run(monkeypatch, client, ["hi"])
    assert "[TTFT" not in capsys.readouterr().err


def test_speed_is_na_when_reply_ends_with_first_text(monkeypatch, capsys):
    # The whole reply arrived at once: no time between first text and end.
    fake_clock(monkeypatch, 10.0, 10.5, 10.5)
    client = FakeClient(Reply(["Hello"], output_tokens=100))
    run(monkeypatch, client, ["hi"])
    assert "[TTFT 0.50s, n/a, total 0.50s]" in capsys.readouterr().err
