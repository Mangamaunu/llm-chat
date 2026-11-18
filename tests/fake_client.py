"""A ChatClient that needs no network: it plays back a script of replies and errors."""
from dataclasses import dataclass, field

from llm_chat.core.chat_client import ChatClient
from llm_chat.core.usage import ModelPricing, Price, RequestUsage, UsageTracker

FAKE_MODEL = "fake-model"

# USD per 1M tokens, chosen so costs are easy to check by hand.
FAKE_PRICING = {
    FAKE_MODEL: ModelPricing(base=Price(input=1.0, output=2.0, cache_write=1.25, cache_read=0.1)),
}


@dataclass
class Reply:
    """One scripted answer from the fake API.

    chunks        text pieces yielded one by one
    input_tokens  None means "count it like a real API": system prompt + history
    error         raised after the chunks are yielded (a stream that breaks midway)
    stop_reason   why the model stopped: None, "max_tokens", "context_full" or "refusal"
    usage_at_end  when the token counts arrive:
                  False - before the text, like Anthropic (message_start)
                  True  - after the text, like OpenAI (the last chunk). A stream that
                          breaks midway then reports no tokens at all.
    """
    chunks: list[str] = field(default_factory=lambda: ["ok"])
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read: int = 0
    cache_write: int = 0
    error: BaseException | None = None
    stop_reason: str | None = None
    usage_at_end: bool = False


class FakeClient(ChatClient):
    """chars_per_token: how the fake API counts tokens. 4 is typical for English;
    Russian text has 2-3 times more tokens per character (about 1.5-2)."""

    def __init__(self, *script, chars_per_token: float = 4.0, **kwargs):
        self.chars_per_token = chars_per_token
        # Retries don't really wait: each requested delay is recorded in self.sleeps.
        self.sleeps: list[float] = []
        kwargs.setdefault("sleep", self.sleeps.append)
        kwargs.setdefault("usage", UsageTracker(pricing=FAKE_PRICING))
        super().__init__(**kwargs)
        self.model = FAKE_MODEL
        self.script: list = list(script)
        self.requests: list[dict] = []  # what each _stream_once call was sent

    def queue(self, *items) -> None:
        self.script.extend(items)

    def _stream_once(self, usage: RequestUsage, reply_parts: list[str]):
        self.requests.append({
            "model": self.model,
            "system": self.conversation.system,
            "messages": [dict(m) for m in self.conversation.messages],
        })
        if not self.script:
            raise AssertionError("FakeClient script is empty: the test made more requests than expected")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            item = Reply([item])

        if not item.usage_at_end:
            self._set_input_usage(usage, item)

        for chunk in item.chunks:
            reply_parts.append(chunk)
            yield chunk

        self.last_stop_reason = item.stop_reason
        if item.error is not None:
            raise item.error
        if item.usage_at_end:
            self._set_input_usage(usage, item)
        text = "".join(item.chunks)
        usage.output_tokens = self.count(text) if item.output_tokens is None else item.output_tokens

    def _set_input_usage(self, usage: RequestUsage, item: Reply) -> None:
        prompt = sum(self.count(m["content"]) for m in self.conversation.messages)
        if self.conversation.system:
            prompt += self.count(self.conversation.system)
        usage.input_tokens = prompt if item.input_tokens is None else item.input_tokens
        usage.cache_read_input_tokens = item.cache_read
        usage.cache_creation_input_tokens = item.cache_write

    def count(self, text: str) -> int:
        """Tokens the fake API reports for one message: its text plus ~4 tokens of overhead."""
        return estimate_tokens({"content": text}, self.chars_per_token)


def chat(client: ChatClient, content: str) -> str:
    """Run one turn to the end and return the full reply."""
    return "".join(client.stream_chat(content))


def estimate_tokens(message: dict, chars_per_token: float = 4.0) -> int:
    """A stand-in for a real tokenizer: chars_per_token characters per token,
    plus 4 tokens for the role and message framing."""
    return int(len(message["content"]) / chars_per_token) + 4
