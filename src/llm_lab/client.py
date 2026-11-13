import time
from dataclasses import dataclass

from ollama import Client

from llm_lab.config import MODEL, OLLAMA_HOST


@dataclass
class LLMResult:
    text: str
    latency_seconds: float


class OllamaClient:
    """Thin adapter around the Ollama chat endpoint.

    The rest of the lab depends on this small interface, so tests can
    substitute a fake client without a running model.
    """

    def __init__(self, model: str = MODEL, host: str = OLLAMA_HOST) -> None:
        self.model = model
        self._client = Client(host=host)

    def chat(
        self,
        prompt: str,
        temperature: float = 0.0,
        json_mode: bool = False,
        system: str | None = None,
        max_tokens: int = 256,
    ) -> LLMResult:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if json_mode:
            kwargs["format"] = "json"

        start = time.perf_counter()
        response = self._client.chat(**kwargs)
        latency = time.perf_counter() - start

        return LLMResult(
            text=response.message.content.strip(),
            latency_seconds=latency,
        )
