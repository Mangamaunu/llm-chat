import argparse
import os
import sys

from chat.chat_client import ChatClient
from chat.error import LLMError
from commands import Context, dispatch


def _anthropic_client() -> ChatClient:
    from chat.anthropic_client import AnthropicClient

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[error: No ANTHROPIC_API_KEY found]", file=sys.stderr)
        sys.exit(1)

    return AnthropicClient()


def _openai_client() -> ChatClient:
    from chat.openai_client import OpenAiClient
    return OpenAiClient()


# --client value -> function that creates that client
CLIENTS = {
    "anthropic": _anthropic_client,
    "openai": _openai_client,
}


_STOP_REASONS = {
    "max_tokens":   lambda x: f"[reply cut: reached the {x}-token limit, use --max-tokens to raise it]",
    "context_full": lambda _: "[reply cut: the model's context window is full, use /reset or --max-history-tokens]",
    "refusal":      lambda _: "[the model declined to answer]",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="llm-chat", description="Chat with an LLM in the terminal.")
    parser.add_argument("--client", choices=CLIENTS, default="anthropic", help="which API to use (default: %(default)s)")
    parser.add_argument("--model", help="model to use instead of the client's default")
    parser.add_argument("--max-tokens", type=int, metavar="N", help="maximum length of a reply in tokens (default: 4096)")
    parser.add_argument("--max-history-tokens", type=int, default=None, metavar="N", help="trim old messages when history exceeds N tokens")

    args = parser.parse_args(argv)

    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    if args.max_history_tokens is not None and args.max_history_tokens <= 0:
        parser.error("--max-history-tokens must be positive")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    client = CLIENTS[args.client]()

    if args.model:
        client.set_model(args.model)

    if args.max_tokens:
        client.max_tokens = args.max_tokens

    if args.max_history_tokens:
        client.max_history_tokens = args.max_history_tokens

    ctx = Context(client=client)
    print(f"[{args.client}: {client.model}]")

    while True:
        try:
            line = input("How can I help you today?\n").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.startswith("/"):
            if not dispatch(ctx, line):
                break
        else:
            try:
                got_text = False
                for text in ctx.client.stream_chat(line):
                    if text.strip():
                        got_text = True
                    print(text, end="", flush=True)

            except LLMError as ex:
                print(f"\n[error: {ex}]")
                continue
            except KeyboardInterrupt:
                print("\n[interrupted]")
                continue
            except Exception as ex:
                print(f"\n[unexpected error: {ex!r}]")
                continue

            if not got_text:
                print("[empty reply]", end="")

            print()
            u = ctx.client.last_usage
            if u is not None:
                priced = ctx.client.usage.cost_of(u)
                cost = f"${priced[0]:.6f}" if priced else "n/a"
                print(f"[{u.model}: {u.input_tokens} in / {u.output_tokens} out, {cost}]")

            if ctx.client.last_stop_reason:
                formatter = _STOP_REASONS.get(ctx.client.last_stop_reason)
                if formatter:
                    print(formatter(ctx.client.max_tokens), file=sys.stderr, flush=True)

            if ctx.client.last_trimmed:
                tokens = f"{ctx.client.history_tokens:,}".replace(",", " ")
                print(f"[history trimmed: dropped {ctx.client.last_trimmed} old messages, ~{tokens} tokens left]", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
