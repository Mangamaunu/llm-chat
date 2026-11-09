"""A ChatClient that needs no network: it plays back a script of replies and errors."""
from dataclasses import dataclass, field

from chat.chat_client import ChatClient, estimate_tokens
from usage import ModelPricing, Price, RequestUsage, UsageTracker

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
    """
    chunks: list[str] = field(default_factory=lambda: ["ok"])
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read: int = 0
    cache_write: int = 0
    error: BaseException | None = None


class FakeClient(ChatClient):
    def __init__(self, *script, **kwargs):
        # No real waiting between retries unless a test asks for it.
        kwargs.setdefault("retry_base_delay", 0.0)
        super().__init__(**kwargs)
        self.model = FAKE_MODEL
        self.usage = UsageTracker(pricing=FAKE_PRICING)
        self.script: list = list(script)
        self.requests: list[dict] = []  # what each _stream_once call was sent

    def queue(self, *items) -> None:
        self.script.extend(items)

    def _stream_once(self, usage: RequestUsage, reply_parts: list[str]):
        self.requests.append({
            "model": self.model,
            "system": self.system,
            "messages": [dict(m) for m in self.messages],
        })
        if not self.script:
            raise AssertionError("FakeClient script is empty: the test made more requests than expected")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            item = Reply([item])

        prompt = sum(estimate_tokens(m) for m in self.messages)
        if self.system:
            prompt += estimate_tokens({"content": self.system})
        usage.input_tokens = prompt if item.input_tokens is None else item.input_tokens
        usage.cache_read_input_tokens = item.cache_read
        usage.cache_creation_input_tokens = item.cache_write

        for chunk in item.chunks:
            reply_parts.append(chunk)
            yield chunk

        if item.error is not None:
            raise item.error
        text = "".join(item.chunks)
        usage.output_tokens = estimate_tokens({"content": text}) if item.output_tokens is None else item.output_tokens


def chat(client: ChatClient, content: str) -> str:
    """Run one turn to the end and return the full reply."""
    return "".join(client.stream_chat(content))
