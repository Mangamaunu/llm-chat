import os
from anthropic import Anthropic

from usage import RequestUsage, UsageTracker


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


class AnthropicClient:
    def __init__(self):
        self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._messages = []
        self._model = "claude-opus-5-5"
        self._system = ""
        self.usage = UsageTracker()
        self.last_usage: RequestUsage | None = None

    def send(self, content):
        """Send a user message and yield the reply text as it streams in.

        The user message and the full reply are kept in the conversation
        history, so later calls include all earlier turns.
        """
        self._messages.append({"role": "user", "content": content})
        reply_parts = []
        usage = RequestUsage(model=self._model)
        try:
            stream = self._client.messages.create(
                max_tokens=1024,
                system=self._system,
                messages=self._messages,
                model=self._model,
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
        except BaseException:
            # Drop the unanswered user message so the history stays valid.
            self._messages.pop()
            if usage.input_tokens or usage.output_tokens:
                # The request was billed even though it didn't finish.
                self.usage.record(usage)
            raise
        self._messages.append({"role": "assistant", "content": "".join(reply_parts)})
        self.usage.record(usage)
        self.last_usage = usage

    def reset(self):
        self._messages.clear()

    def set_model(self, model):
        self._model = model

    def set_system(self, system):
        self._system = system
