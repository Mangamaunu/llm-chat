import argparse

from chat.chat_client import ChatClient
from chat.error import LLMError
from commands import Context, dispatch


def _anthropic_client() -> ChatClient:
    from chat.anthropic_client import AnthropicClient
    return AnthropicClient()


def _openai_client() -> ChatClient:
    from chat.openai_client import OpenAiClient
    return OpenAiClient()


# --client value -> function that creates that client
CLIENTS = {
    "anthropic": _anthropic_client,
    "openai": _openai_client,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="llm-chat", description="Chat with an LLM in the terminal.")
    parser.add_argument(
        "--client",
        choices=CLIENTS,
        default="anthropic",
        help="which API to use (default: %(default)s)",
    )
    parser.add_argument("--model", help="model to use instead of the client's default")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    client = CLIENTS[args.client]()
    if args.model:
        client.set_model(args.model)
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
                for text in ctx.client.stream_chat(line):
                    print(text, end="", flush=True)
            except LLMError as ex:
                print(f"\n[error: {ex}]")
                continue
            print()
            u = ctx.client.last_usage
            if u is not None:
                priced = ctx.client.usage.cost_of(u)
                cost = f"${priced[0]:.6f}" if priced else "n/a"
                print(f"[{u.model}: {u.input_tokens} in / {u.output_tokens} out, {cost}]")


if __name__ == "__main__":
    main()
