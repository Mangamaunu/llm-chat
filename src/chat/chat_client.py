import sys
import time
from abc import ABC, abstractmethod

from retry import backoff_delay, is_retryable, describe
from usage import RequestUsage, UsageTracker


class ChatClient(ABC):
    def __init__(self, max_retries: int = 5, retry_base_delay: float = 1.0, retry_max_delay: float = 30.0):
        self.model: str | None = None
        self.system = ""
        self.usage = UsageTracker()
        self.last_usage: RequestUsage | None = None
        self.messages = []
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.max_history_tokens: int | None = None
        self.trim_target_ratio: float = 0.5
        self.history_tokens: int = 0
        self.last_trimmed: int = 0

    def reset(self):
        self.messages.clear()
        self.history_tokens = 0

    def set_model(self, model):
        self.model = model

    def set_system(self, system):
        self.system = system

    def stream_chat(self, content):
        self.messages.append({"role": "user", "content": content})
        reply_parts = []
        attempt = 0
        try:
            while True:
                usage = RequestUsage(model=self.model)
                try:
                    yield from self._stream_once(usage, reply_parts)
                    break
                except Exception as exc:
                    if usage.input_tokens or usage.output_tokens:
                        # The attempt used tokens (and may be billed) even though it didn't finish.
                        self.usage.record(usage)
                    # Once text has been shown, a retry would print the reply twice.
                    if reply_parts or attempt >= self.max_retries or not is_retryable(exc):
                        raise
                    delay = backoff_delay(attempt, exc, self.retry_base_delay, self.retry_max_delay)
                    attempt += 1
                    print(f"[{describe(exc)}, retry {attempt}/{self.max_retries} in {delay:.1f}s]", file=sys.stderr, flush=True)
                    time.sleep(delay)
        except BaseException:
            # Drop the unanswered user message so the history stays valid.
            self.messages.pop()
            raise
        self.messages.append({"role": "assistant", "content": "".join(reply_parts)})
        self.usage.record(usage)
        self.last_usage = usage
        self.history_tokens = usage.prompt_tokens + usage.output_tokens
        self.last_trimmed = self._trim_history()

    def _trim_history(self) -> int:
        """Removes the oldest turns if the history exceeds `max_history_tokens`. Returns the number of messages removed."""
        if not self.max_history_tokens or self.history_tokens <= self.max_history_tokens:
            return 0

        target = int(self.max_history_tokens * self.trim_target_ratio)
        dropped = 0

        while self.history_tokens > target and len(self.messages) > 2:
            assert self.messages[0]["role"] == "user"
            user_message = self.messages.pop(0)
            assistant_message = self.messages.pop(0)
            self.history_tokens = max(0, self.history_tokens - estimate_tokens(user_message) - estimate_tokens(assistant_message))
            dropped += 2

        return dropped

    @abstractmethod
    def _stream_once(self, usage: RequestUsage, reply_parts: list[str]):
        ...


def estimate_tokens(message: dict) -> int:
    return len(message["content"]) // 4 + 4 # ~4 chars per token in English; underestimates other languages.
