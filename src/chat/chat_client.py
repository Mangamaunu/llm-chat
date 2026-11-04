import sys
import time
from abc import ABC, abstractmethod

from retry import backoff_delay, is_retryable, describe
from usage import RequestUsage, UsageTracker


class ChatClient(ABC):
    def __init__(self, max_retries: int = 5, retry_base_delay: float = 1.0, retry_max_delay: float = 30.0):
        self.model = ""
        self.system = ""
        self.usage = UsageTracker()
        self.last_usage: RequestUsage | None = None
        self.messages = []
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def reset(self):
        self.messages.clear()

    def set_model(self, model):
        self.model = model

    def set_system(self, system):
        self.system = system

    def stream_chat(self, content):
        """Send a user message and yield the reply text as it streams in.

        The user message and the full reply are kept in the conversation
        history, so later calls include all earlier turns.

        429, 5xx and overloaded errors are retried with exponential backoff
        and jitter, as long as no reply text has been yielded yet.
        """
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
                        # The attempt was billed even though it didn't finish.
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

    @abstractmethod
    def _stream_once(self, usage: RequestUsage, reply_parts: list[str]):
        ...