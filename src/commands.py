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
