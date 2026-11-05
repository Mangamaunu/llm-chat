import os

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APIStatusError

from .chat_client import ChatClient
from .error import LLMError, ConnectionFailed, RateLimited, BadRequest, ServerError, retry_after
from usage import RequestUsage

# Defaults point at a local Ollama server, which serves an OpenAI-compatible API.
_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1/")
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "llama3.2")


class OpenAiClient(ChatClient):
    # Ollama ignores the API key, but the OpenAI SDK requires one to be set.
    def __init__(self, base_url: str = _OPENAI_BASE_URL, api_key: str = _OPENAI_API_KEY):
        super().__init__()
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.model = _OPENAI_MODEL

    def _stream_once(self, usage: RequestUsage, reply_parts: list[str]):
        # The OpenAI API takes the system prompt as the first message.
        messages = self.messages
        if self.system:
            messages = [{"role": "system", "content": self.system}, *messages]

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                # With include_usage, the last chunk has the token counts and no choices.
                if chunk.usage is not None:
                    _apply_usage(usage, chunk.usage)
                if chunk.choices:
                    text = chunk.choices[0].delta.content
                    if text:
                        reply_parts.append(text)
                        yield text
        except APIError as ex:
            raise _to_llm_error(ex) from ex



def _to_llm_error(ex: APIError) -> LLMError:
    if isinstance (ex, APIConnectionError):
        return ConnectionFailed(ex.message, retry_after=retry_after(ex))

    elif isinstance (ex, RateLimitError) and ex.code == "insufficient_quota":
        return BadRequest(ex.message, retry_after=retry_after(ex))

    elif isinstance (ex, RateLimitError):
        return RateLimited(ex.message, retry_after=retry_after(ex))

    elif isinstance(ex, APIStatusError) and ex.status_code >= 500:
        return ServerError(ex.message, retry_after=retry_after(ex))

    elif isinstance (ex, APIStatusError):
        return BadRequest(ex.message, retry_after=retry_after(ex))

    else:
        return BadRequest(ex.message, retry_after=retry_after(ex))


def _apply_usage(target: RequestUsage, source) -> None:
    """Split OpenAI's prompt_tokens into uncached input, cache read and cache write."""
    details = getattr(source, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or 0
    written = getattr(details, "cache_write_tokens", None) or 0

    target.cache_read_input_tokens = cached
    target.cache_creation_input_tokens = written
    target.input_tokens = max(0, source.prompt_tokens - cached - written)
    target.output_tokens = source.completion_tokens
