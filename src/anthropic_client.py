import os

from anthropic import Anthropic

from chat_client import ChatClient
from usage import RequestUsage

_API_KEY=os.environ.get("ANTHROPIC_API_KEY")
_DEFAULT_MODEL="claude-opus-5-5"


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _apply_usage(target: RequestUsage, source) -> None:
    """Copy the token counts that are present in an API usage object."""
    for field in _USAGE_FIELDS:
        value = getattr(source, field, None)
        if value is not None:
            setattr(target, field, value)


class AnthropicClient(ChatClient):
    def __init__(self):
        super().__init__()
        self._client = Anthropic(api_key=_API_KEY, max_retries=0)
        self.model = _DEFAULT_MODEL

    def _stream_once(self, usage: RequestUsage, reply_parts: list[str]):
        stream = self._client.messages.create(
            max_tokens=1024,
            system=self.system,
            messages=self.messages,
            model=self.model,
            stream=True,
        )
        for event in stream:
            if event.type == "message_start":
                _apply_usage(usage, event.message.usage)
            elif event.type == "message_delta":
                # Counts here are cumulative, so they replace earlier values.
                _apply_usage(usage, event.usage)
            elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                reply_parts.append(event.delta.text)
                yield event.delta.text


