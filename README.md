# llm-chat

A terminal chat app for large language models. It streams replies as they are
generated, keeps the conversation history, retries temporary API errors, trims
old messages when the history gets too long, and shows token usage and cost.

```
$ llm-chat --client openai
[openai: llama3.2]
How can I help you today?
What is TCP in one sentence?
TCP is a protocol that delivers data between programs reliably and in order.
[llama3.2: 34 in / 17 out, $0.000000]
```

## Supported APIs

| `--client` | SDK | Works with |
|---|---|---|
| `anthropic` (default) | [`anthropic`](https://github.com/anthropics/anthropic-sdk-python) Python SDK | Claude models through the Anthropic API |
| `openai` | [`openai`](https://github.com/openai/openai-python) Python SDK | The OpenAI API, or any OpenAI-compatible server. By default it connects to a local [Ollama](https://ollama.com) server |

Both clients share the same chat loop, retries, history trimming and cost
tracking (see [How it works](#how-it-works)).

## Requirements

- Python 3.12 or newer
- For `--client anthropic`: an Anthropic API key
- For `--client openai`: either Ollama running locally (free) or an OpenAI API key

## Installation

```bash
git clone <repo-url> llm-chat
cd llm-chat
python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # or: pip install -e ".[dev]" to also get the test tools
```

This installs the `llm-chat` command into the virtual environment.

## Configuration

Settings are read from environment variables. The app does not load a `.env`
file, so export them in your shell.

| Variable | Default | Used by |
|---|---|---|
| `ANTHROPIC_API_KEY` | none (required) | `anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | `anthropic` |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1/` (Ollama) | `openai` |
| `OPENAI_API_KEY` | `ollama` (Ollama ignores it) | `openai` |
| `OPENAI_MODEL` | `llama3.2` | `openai` |

`--model` on the command line takes precedence over `ANTHROPIC_MODEL` and `OPENAI_MODEL`.

## Running

### Claude (Anthropic API)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
llm-chat
llm-chat --model claude-sonnet-5-5
```

### Local model with Ollama (free)

```bash
# once: install Ollama from https://ollama.com, then download a model
ollama pull llama3.2

ollama serve                # skip if the Ollama app is already running
llm-chat --client openai
llm-chat --client openai --model qwen3:8b
```

### OpenAI API

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
llm-chat --client openai --model gpt-5.4-mini
```

### Command-line options

| Option | Description |
|---|---|
| `--client {anthropic,openai}` | Which API to use (default: `anthropic`) |
| `--model MODEL` | Model to use instead of the client's default |
| `--max-history-tokens N` | Trim old messages when the history exceeds N tokens (default: no limit) |
| `-h`, `--help` | Show help |

## Chat commands

Type these instead of a message:

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/system <text>` | Set the system prompt |
| `/model <name>` | Switch model for the next requests |
| `/reset` | Clear the conversation history (the system prompt is kept) |
| `/cost` | Show token usage and cost for this session, per model |
| `/exit`, `/quit`, `/q` | Quit (Ctrl+D also works) |

## How it works

### Streaming and history

Reply text is printed as it arrives. Each finished turn (your message and the
reply) is added to the history and sent with the next request. If a request
fails, your unanswered message is removed so the history stays valid. Replies
are limited to 1024 tokens.

### Retries

Temporary errors are retried up to 5 times with exponential backoff and jitter
(1 s base, 30 s maximum). If the server sends a `retry-after` header, the app
waits at least that long. Each retry is reported on stderr, for example
`[RateLimited, retry 1/5 in 2.3s]`.

| Retried | Not retried |
|---|---|
| Rate limits (429), overloaded (529), server errors (5xx), connection failures | Invalid requests, authentication errors, unknown models, exhausted OpenAI quota |

A reply that breaks after text was already shown is not retried, because that
would print the text twice.

Each SDK's errors are mapped to the app's own error types in `chat/error.py`,
so the retry logic doesn't depend on either SDK.

### History trimming

With `--max-history-tokens N`, the app checks the history size after each turn.
It takes the size from the token counts the API reports, so it costs nothing extra.
When the size is over N, the oldest turns are dropped until it is under N × 0.5.
Two limits mean trimming happens every few turns rather than on every turn.

- The system prompt is never removed; it is stored separately from the messages.
- The newest turn is always kept, even if it alone is larger than N.
- Messages are dropped in user/assistant pairs, as both APIs expect.

After trimming, the app prints a message such as
`[history trimmed: dropped 6 old messages, ~9 800 tokens left]` to stderr.

The size of each dropped message is estimated at about 4 characters per token,
which undercounts non-English text. With Ollama, keep N below the model's
context window (`num_ctx`): Ollama silently cuts longer prompts itself.

### Usage and cost

After each reply the app prints the input and output tokens and the cost, for example
`[claude-opus-5-5: 1204 in / 310 out, $0.011016]`. `/cost` shows session totals,
including prompt-cache reads and writes and the money saved by caching.

Prices come from [`pricing.toml`](pricing.toml) in USD per 1M tokens. It also
supports a higher price tier for long prompts. Models that aren't listed show
`n/a`; local Ollama models are listed at $0. Prices change, so update the file when they do.

## Project structure

```
src/
  main.py               command-line options and the chat loop
  commands.py           /commands
  retry.py              retry decisions and backoff delays
  usage.py              token usage, pricing.toml, cost calculation
  chat/
    chat_client.py      ChatClient base class: history, retries, trimming
    anthropic_client.py Anthropic SDK client
    openai_client.py    OpenAI SDK client (also Ollama)
    error.py            SDK-independent error types
tests/                  unit tests (no network)
pricing.toml            model prices
```

### Adding another API

1. Subclass `ChatClient` and implement `_stream_once(usage, reply_parts)`. It
   sends `self.system` and `self.messages` and yields reply text. It also
   appends that text to `reply_parts` and fills in the token counts in `usage`.
2. Convert the SDK's errors to the types in `chat/error.py`. The retry logic
   decides what to retry from those types.
3. Register the client in `CLIENTS` in `main.py`.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
python -m pytest --cov=src --cov-report=term-missing   # with coverage
```

The tests use `FakeClient` (`tests/fake_client.py`), a `ChatClient` that plays
back scripted replies and errors, so no API key or network is needed.
