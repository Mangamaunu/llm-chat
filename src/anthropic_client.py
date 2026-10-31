import os
from anthropic import Anthropic


class AnthropicClient:
    def __init__(self):
        self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._messages = []

    def send(self, content):
        """Send a user message and yield the reply text as it streams in.

        The user message and the full reply are kept in the conversation
        history, so later calls include all earlier turns.
        """
        self._messages.append({"role": "user", "content": content})
        reply_parts = []
        try:
            stream = self._client.messages.create(
                max_tokens=1024,
                messages=self._messages,
                model="claude-opus-5-5",
                stream=True,
            )
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    reply_parts.append(event.delta.text)
                    yield event.delta.text
        except BaseException:
            # Drop the unanswered user message so the history stays valid.
            self._messages.pop()
            raise
        self._messages.append({"role": "assistant", "content": "".join(reply_parts)})

    def reset(self):
        self._messages.clear()
