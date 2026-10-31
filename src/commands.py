from dataclasses import dataclass
from typing import Callable

from anthropic_client import AnthropicClient


@dataclass
class Context:
    """Shared state available to every command."""
    client: AnthropicClient


@dataclass
class Command:
    name: str
    help: str
    handler: Callable[[Context, str], bool]  # returns False to quit the loop


COMMANDS: dict[str, Command] = {}


def command(name: str, help: str, aliases: tuple[str, ...] = ()):
    """Register the decorated function as a /command."""
    def register(fn: Callable[[Context, str], bool]):
        cmd = Command(name, help, fn)
        for n in (name, *aliases):
            COMMANDS[n] = cmd
        return fn
    return register


def dispatch(ctx: Context, line: str) -> bool:
    """Run the command in `line` (which starts with '/'). Returns False to quit."""
    name, _, args = line[1:].partition(" ")
    cmd = COMMANDS.get(name)
    if cmd is None:
        print(f"Unknown command: /{name}. Type /help for a list.")
        return True
    return cmd.handler(ctx, args.strip())


# ---- commands ----

@command("exit", "Quit the chat", aliases=("quit", "q"))
def _exit(ctx: Context, args: str) -> bool:
    return False


@command("reset", "Clear conversation history")
def _reset(ctx: Context, args: str) -> bool:
    ctx.client.reset()
    print("Conversation history cleared.")
    return True


@command("help", "Show available commands")
def _help(ctx: Context, args: str) -> bool:
    unique = {c.name: c for c in COMMANDS.values()}
    for cmd in unique.values():
        print(f"  /{cmd.name:<10} {cmd.help}")
    return True


@command("model", "Set model for request")
def _model(ctx: Context, args: str) -> bool:
    ctx.client.set_model(args)
    return True

@command("system", "Set system prompt")
def _system(ctx: Context, args: str) -> bool:
    ctx.client.set_system(args)
    return True

@command("cost", "Show token usage and cost for this session")
def _cost(ctx: Context, args: str) -> bool:
    tracker = ctx.client.usage
    if not tracker.by_model:
        print("No API requests yet in this session.")
        return True

    total = tracker.total()
    name_width = max(20, *(len(m) + 2 for m in tracker.by_model))

    def num(n: int) -> str:
        return f"{n:,}".replace(",", " ")  # 16820 -> "16 820"

    def money(mu) -> str:
        if not mu.priced and not mu.cost:
            return "n/a"
        return f"${mu.cost:.4f}" + ("" if mu.priced else "*")

    def row(name, inp, out, cache_read, cost) -> str:
        return f"{name:<{name_width}}{inp:>7}{out:>10}{cache_read:>13}   {cost}"

    print(f"Session cost ({total.requests} requests)")
    print(row("model", "input", "output", "cache_read", "cost"))
    for model, mu in tracker.by_model.items():
        print(row(model, num(mu.input_tokens), num(mu.output_tokens),
                  num(mu.cache_read_input_tokens), money(mu)))
    print("-" * (name_width + 37))
    print(row("total", num(total.input_tokens), num(total.output_tokens),
              num(total.cache_read_input_tokens), money(total)))

    if not total.priced:
        print("* excludes models with no price in pricing.toml")
    if total.cache_creation_input_tokens or total.cache_read_input_tokens:
        print(f"cache_write: {num(total.cache_creation_input_tokens)} tokens, "
              f"saved by caching: ${total.cache_savings:.4f}")
    return True
