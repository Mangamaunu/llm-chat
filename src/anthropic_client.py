import os
import sys
import time

from anthropic import Anthropic

from retry import backoff_delay, describe, is_retryable
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
    def __init__(self, max_retries: int = 5, retry_base_delay: float = 1.0, retry_max_delay: float = 30.0):
        # SDK retries are off: retries are handled in send() instead.
        self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=0)
        self._messages = []
        self._model = "claude-opus-5-5"
        self._system = ""
        self.usage = UsageTracker()
        self.last_usage: RequestUsage | None = None
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def send(self, content):
        """Send a user message and yield the reply text as it streams in.

        The user message and the full reply are kept in the conversation
        history, so later calls include all earlier turns.

        429, 5xx and overloaded errors are retried with exponential backoff
        and jitter, as long as no reply text has been yielded yet.
        """
        self._messages.append({"role": "user", "content": content})
        reply_parts = []
        attempt = 0
        try:
            while True:
                usage = RequestUsage(model=self._model)
                try:
                    yield from self._stream_once(usage, reply_parts)
                    break
                except Exception as exc:
                    if usage.input_tokens or usage.output_tokens:
                        # The attempt was billed even though it didn't finish.
                        self.usage.record(usage)
                    # Once text has been shown, a retry would print the reply twice.
                    if reply_parts or attempt >= self.max_retries or not is_retryable(exc):
                        raise
                    delay = backoff_delay(attempt, exc, self.retry_base_delay, self.retry_max_delay)
                    attempt += 1
                    print(f"[{describe(exc)}, retry {attempt}/{self.max_retries} in {delay:.1f}s]",
                          file=sys.stderr, flush=True)
                    time.sleep(delay)
        except BaseException:
            # Drop the unanswered user message so the history stays valid.
            self._messages.pop()
            raise
        self._messages.append({"role": "assistant", "content": "".join(reply_parts)})
        self.usage.record(usage)
        self.last_usage = usage

    def _stream_once(self, usage: RequestUsage, reply_parts: list[str]):
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

    def reset(self):
        self._messages.clear()

    def set_model(self, model):
        self._model = model

    def set_system(self, system):
        self._system = system
