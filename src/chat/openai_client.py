from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APIStatusError

from .chat_client import ChatClient
from .error import LLMError, ConnectionFailed, RateLimited, BadRequest, ServerError, retry_after
from usage import RequestUsage

# Ollama serves an OpenAI-compatible API on this address; the key is ignored but required.
_BASE_URL = "http://localhost:11434/v1/"
_DEFAULT_MODEL = "llama3.2"


class OpenAiClient(ChatClient):
    def __init__(self, base_url: str = _BASE_URL, api_key: str = "ollama"):
        super().__init__()
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.model = _DEFAULT_MODEL

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
                    usage.input_tokens = chunk.usage.prompt_tokens
                    usage.output_tokens = chunk.usage.completion_tokens
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

