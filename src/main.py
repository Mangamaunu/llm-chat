from anthropic_client import AnthropicClient
from commands import Context, dispatch


def main() -> None:
    ctx = Context(client=AnthropicClient())
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
            for text in ctx.client.send(line):
                print(text, end="", flush=True)
            print()
            u = ctx.client.last_usage
            if u is not None:
                priced = ctx.client.usage.cost_of(u)
                cost = f"${priced[0]:.6f}" if priced else "n/a"
                print(f"[{u.model}: {u.input_tokens} in / {u.output_tokens} out, {cost}]")


if __name__ == "__main__":
    main()
