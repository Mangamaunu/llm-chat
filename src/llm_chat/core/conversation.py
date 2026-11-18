from dataclasses import dataclass, field


@dataclass
class Conversation:
    """The system prompt and the message history, and trimming of that history.

    history_tokens is the size of system prompt + history as the API last reported
    it (prompt + reply of the last turn), reduced by an estimate after trimming.
    """
    system: str = ""
    messages: list[dict] = field(default_factory=list)
    max_history_tokens: int | None = None
    trim_target_ratio: float = 0.5
    history_tokens: int = 0

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def drop_last_user(self) -> None:
        """Remove the unanswered user message, so the history stays valid after a failed turn."""
        assert self.messages and self.messages[-1]["role"] == "user"
        self.messages.pop()

    def add_reply(self, text: str, prompt_tokens: int, output_tokens: int) -> int:
        """Store the reply, update history_tokens, trim. Returns the number of messages dropped."""
        self.messages.append({"role": "assistant", "content": text})
        self.history_tokens = prompt_tokens + output_tokens
        return self._trim()

    def reset(self) -> None:
        """Clear the history; the system prompt is kept."""
        self.messages.clear()
        self.history_tokens = 0

    def _trim(self) -> int:
        """Remove the oldest turns if the history exceeds `max_history_tokens`. Returns the number of messages removed."""
        if not self.max_history_tokens or self.history_tokens <= self.max_history_tokens:
            return 0

        # The API reports only the total, so estimate each message from this
        # conversation's tokens per character. That works for any language.
        chars = len(self.system) + sum(len(m["content"]) for m in self.messages)
        tokens_per_char = self.history_tokens / chars if chars else 0.25

        def size(message: dict) -> int:
            return round(len(message["content"]) * tokens_per_char)

        target = int(self.max_history_tokens * self.trim_target_ratio)
        dropped = 0

        # The newest turn is always kept, even if it alone is over the limit.
        while self.history_tokens > target and len(self.messages) > 2:
            assert self.messages[0]["role"] == "user"
            user_message = self.messages.pop(0)
            assistant_message = self.messages.pop(0)
            self.history_tokens = max(0, self.history_tokens - size(user_message) - size(assistant_message))
            dropped += 2

        return dropped
