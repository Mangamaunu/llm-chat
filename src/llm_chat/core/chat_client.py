import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from .retry import backoff_delay, is_retryable
from .conversation import Conversation
from .usage import RequestUsage, UsageTracker


class ChatClient(ABC):
    def __init__(
        self,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        usage: UsageTracker | None = None,
        on_retry: Callable[[int, float, Exception], None] | None = None,
        conversation: Conversation | None = None,
    ):
        self.model: str | None = None
        self.max_tokens = 4096
        self.conversation = conversation if conversation is not None else Conversation()
        self.usage = usage if usage is not None else UsageTracker()
        self.last_usage: RequestUsage | None = None
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        # Called before each retry with (attempt, delay, error); the CLI prints a notice.
        self.on_retry = on_retry
        # Waits between retries; tests pass their own function instead of really sleeping.
        self.sleep = sleep
        self.last_trimmed: int = 0
        self.last_stop_reason: str | None = None

    def stream_chat(self, content):
        self.conversation.add_user(content)
        reply_parts = []
        attempt = 0
        try:
            while True:
                usage = RequestUsage(model=self.model)
                self.last_stop_reason = None
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

                    if self.on_retry:
                        self.on_retry(attempt, delay, exc)

                    self.sleep(delay)
        except BaseException:
            self.conversation.drop_last_user()
            raise
        self.usage.record(usage)
        self.last_usage = usage
        reply = "".join(reply_parts)
        if reply.strip():
            self.last_trimmed = self.conversation.add_reply(reply, usage.prompt_tokens, usage.output_tokens)
        else:
            # Anthropic rejects empty assistant messages; drop the whole turn.
            self.conversation.drop_last_user()
            self.last_trimmed = 0

    @abstractmethod
    def _stream_once(self, usage: RequestUsage, reply_parts: list[str]):
        ...
