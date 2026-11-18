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

This installs the `llm-chat` command into the virtual environment. Editable
mode (`-e`) is for working on the code; to just use the app, a normal install
works too, for example `pipx install .` puts `llm-chat` on your PATH in its own
environment. `python -m llm_chat` starts the app as well.

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
| `LLM_CHAT_PRICING` | the bundled `pricing.toml` | both: path to your own prices file |

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
| `--max-tokens N` | Maximum length of a reply in tokens (default: 4096) |
| `--max-history-tokens N` | Trim old messages when the history exceeds N tokens (default: no limit) |
| `-h`, `--help` | Show help |

## Chat commands

Type these instead of a message:

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/system` | Show the current system prompt |
| `/system <text>` | Set the system prompt |
| `/system off` | Remove the system prompt |
| `/model` | Show the current model |
| `/model <name>` | Switch model for the next requests |
| `/reset` | Clear the conversation history (the system prompt is kept) |
| `/cost` | Show token usage and cost for this session, per model |
| `/exit`, `/quit`, `/q` | Quit (Ctrl+D also works) |

## Typing messages

Arrow keys move the cursor, and up/down go through earlier inputs
(this uses Python's `readline`; on Windows plain input is used).

To send several lines at once, for example code, type `"""` on its own line,
then the text, then `"""` again:

```
How can I help you today?
"""
... def add(a, b):
...     return a + b
... Why is this slow?
... """
```

- Indentation and blank lines inside the block are kept.
- Text before the opening `"""` is kept too: `Explain this: """` starts the
  message with "Explain this:". The same works for commands:
  `/system """` sets a multi-line system prompt.
- A block is always sent as a message, even if its first line starts with `/`.
- Ctrl+C inside a block cancels it; Ctrl+C at the normal prompt quits.

## How it works

### Streaming and history

Reply text is printed as it arrives. Each finished turn (your message and the
reply) is added to the history and sent with the next request. If a request
fails, your unanswered message is removed so the history stays valid. If you
press Ctrl+C during a reply, the reply is dropped (`[interrupted]`) and the chat
goes on. A reply with no text is not saved, and the app prints `[empty reply]`.

### Reply length

Replies are limited to 4096 tokens; change this with `--max-tokens N`. With
reasoning models the limit also covers the model's reasoning, so a low limit
can leave no room for the answer itself.

When the model stops early, the app says why (on stderr):

| Notice | Meaning |
|---|---|
| `[reply cut: reached the N-token limit, use --max-tokens to raise it]` | The reply hit the limit. The partial reply stays in the history, so you can ask the model to continue. |
| `[reply cut: the model's context window is full, use /reset or --max-history-tokens]` | The conversation is too long for the model. |
| `[the model declined to answer]` | The model refused, or the provider's content filter stopped the reply. |

The OpenAI API calls the limit `max_completion_tokens`, while Ollama expects
the older `max_tokens`. The app sends `max_tokens` when `OPENAI_BASE_URL`
points at `localhost`, `127.0.0.1` or `::1`, and `max_completion_tokens` everywhere else.

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

Each SDK's errors are mapped to the app's own error types in `core/error.py`,
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

The API only reports the size of the whole prompt, not of each message. To know
how much a dropped message frees up, the app works out how many tokens per
character this conversation uses (the real total divided by the number of
characters) and applies that to the message. This works for any language:
Russian text, for example, has two to three times more tokens per character
than English.

With Ollama, keep N below the model's context window (`num_ctx`): Ollama
silently cuts longer prompts itself.

### Usage and cost

After each reply the app prints the input and output tokens and the cost, for example
`[claude-opus-5-5: 1204 in / 310 out, $0.011016]`. `/cost` shows session totals,
including prompt-cache reads and writes and the money saved by caching.

### Prompt caching

Each request sends the whole conversation again. To avoid paying the full price
for it every turn, requests to Claude ask Anthropic to cache the prompt
(`cache_control`): the part that was already sent in the last 5 minutes is read
from the cache at about a tenth of the input price, and only the new part is
charged in full. Writing to the cache costs 1.25× the input price once. OpenAI
caches long prompts automatically, so nothing is needed there.

When part of the prompt came from the cache, the usage line says so:

```
[claude-opus-5-5: 120 in + 15 300 cached / 310 out, $0.009740]
```

- `in` is the new, uncached part of the prompt, at the full price.
- `cached` was read from the cache, at the cache-read price.
- `cache write` (when shown) was written to the cache for the next turns.

Short prompts are not cached; each model has a minimum length. The cache only
helps while the beginning of the prompt stays the same, so these start a new
cache on the next request: changing the system prompt (`/system`), switching
model (`/model`), `/reset`, and history trimming (`--max-history-tokens`), which
removes the oldest messages. A pause of more than 5 minutes also ends the cache.

Prices come from [`src/llm_chat/pricing.toml`](src/llm_chat/pricing.toml) in
USD per 1M tokens; the file is installed together with the app. It also supports
a higher price tier for long prompts. Models that aren't listed show `n/a`;
local Ollama models are listed at $0.

Prices change. To use your own prices without editing the installed file,
point `LLM_CHAT_PRICING` at a copy:

```bash
cp src/llm_chat/pricing.toml ~/my-prices.toml   # then edit it
export LLM_CHAT_PRICING=~/my-prices.toml
```

If the file is missing or broken, the app prints a warning at startup and keeps
working; costs then show `n/a`.

## Project structure

```
src/llm_chat/
  cli.py                  command-line options and the chat loop
  commands.py             /commands
  __main__.py             lets `python -m llm_chat` start the app
  pricing.toml            model prices
  core/                   provider-independent logic
    chat_client.py        ChatClient base class: one turn, retries, usage bookkeeping
    conversation.py       Conversation: system prompt, history, history trimming
    error.py              SDK-independent error types
    retry.py              retry decisions and backoff delays
    usage.py              token usage, loading prices, cost calculation
  providers/              one ChatClient per API SDK
    anthropic_client.py   Anthropic SDK client
    openai_client.py      OpenAI SDK client (also Ollama)
tests/                    unit tests (no network)
.github/workflows/        CI: runs the tests on every push
```

`core/` doesn't import anything from `providers/` or `cli.py`, so the chat logic
can be reused without the terminal or a particular SDK. It prints nothing and
reads no environment variables: `cli.py` reads the settings, passes them to the
client's constructor and sets `client.on_retry` to print the retry notices.

### Adding another API

1. Subclass `ChatClient` and implement `_stream_once(usage, reply_parts)`. It
   sends `self.conversation.system` and `self.conversation.messages` and yields
   reply text. It also appends that text to `reply_parts` and fills in the token
   counts in `usage`. Take the API key, model and other settings as constructor
   arguments and pass `**kwargs` on to `ChatClient.__init__`.
2. Convert the SDK's errors to the types in `core/error.py`. The retry logic
   decides what to retry from those types.
3. Put it in `providers/`. In `cli.py`, add a factory that reads the
   environment variables and creates the client (it gets the `--model` value or
   `None`), and register it in `CLIENTS`.
4. Add the SDK to `dependencies` in `pyproject.toml`, with an upper bound on
   the major version.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
python -m pytest --cov=llm_chat --cov-report=term-missing   # with coverage
```

The tests use `FakeClient` (`tests/fake_client.py`), a `ChatClient` that plays
back scripted replies and errors, so no API key or network is needed. A reply
can deliver its token counts before the text (like Anthropic) or after it (like
OpenAI, `usage_at_end=True`). Retries don't really wait: the delays are
recorded in `client.sleeps`. It counts tokens like a real API, about
`chars_per_token` characters per token (4 by default; 1.5 imitates Russian
text). The trimming rules are tested on `Conversation` directly
(`tests/test_conversation.py`), without a client. The real clients are tested
with a stub in place of the SDK (`tests/test_provider_streaming.py`).

GitHub Actions runs the tests on every push and pull request, on Python 3.12
and 3.13 (`.github/workflows/tests.yml`). It installs the package without `-e`
and fails if coverage drops below 95%. To run the same check locally:

```bash
pip install ".[dev]"
python -m pytest -o pythonpath= --cov=llm_chat --cov-fail-under=95
```
