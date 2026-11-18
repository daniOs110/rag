from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

from app.config import Settings, resolved_vendor
from app.errors import ProviderError

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+", re.UNICODE)
_LOCAL_DIMENSIONS = 256


class EmbeddingProvider(Protocol):
    name: str
    model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class ChatProvider(Protocol):
    name: str
    model: str | None

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class LocalEmbeddingProvider:
    """Hash de tokens. Sirve para desarrollo y pruebas, no para similitud semántica."""

    name = "local"
    model = "local-hash"

    def __init__(self, dimensions: int = _LOCAL_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN.findall(text.lower()) or ["empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), 64):
                batch = texts[start : start + 64]
                response = self._client.embeddings.create(model=self.model, input=batch)
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors.extend(item.embedding for item in ordered)
        except Exception as exc:
            logger.exception("Fallo al pedir embeddings a OpenAI")
            raise ProviderError("No se pudieron generar embeddings") from exc
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIChatProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
        except Exception as exc:
            logger.exception("Fallo al pedir una respuesta a OpenAI")
            raise ProviderError("No se pudo obtener la respuesta del modelo") from exc
        if not content or not content.strip():
            raise ProviderError("El modelo no devolvió texto")
        return content.strip()


class GeminiEmbeddingProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import types

        self.model = model
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        contents = [
            self._types.Content(
                parts=[self._types.Part.from_text(text=f"title: none | text: {text}")]
            )
            for text in texts
        ]
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(contents), 32):
                batch = contents[start : start + 32]
                response = self._client.models.embed_content(model=self.model, contents=batch)
                embeddings = response.embeddings or []
                if len(embeddings) != len(batch):
                    raise ProviderError("El proveedor devolvió embeddings incompletos")
                for item in embeddings:
                    if not item.values:
                        raise ProviderError("El proveedor devolvió un embedding vacío")
                    vectors.append(list(item.values))
        except ProviderError:
            raise
        except Exception as exc:
            logger.exception("Fallo al pedir embeddings a Gemini")
            raise ProviderError("No se pudieron generar embeddings") from exc
        return vectors

    def embed_query(self, text: str) -> list[float]:
        content = self._types.Content(
            parts=[
                self._types.Part.from_text(text=f"task: question answering | query: {text}")
            ]
        )
        try:
            response = self._client.models.embed_content(model=self.model, contents=content)
            embeddings = response.embeddings or []
            if not embeddings or not embeddings[0].values:
                raise ProviderError("El proveedor devolvió un embedding vacío")
            return list(embeddings[0].values)
        except ProviderError:
            raise
        except Exception as exc:
            logger.exception("Fallo al pedir el embedding de la pregunta a Gemini")
            raise ProviderError("No se pudieron generar embeddings") from exc


class GeminiChatProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import types

        self.model = model
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                ),
            )
            text = response.text
        except Exception as exc:
            logger.exception("Fallo al pedir una respuesta a Gemini")
            raise ProviderError("No se pudo obtener la respuesta del modelo") from exc
        if not text or not text.strip():
            raise ProviderError("El modelo no devolvió texto")
        return text.strip()


class LocalChatProvider:
    """Doble de pruebas. No llama a un modelo."""

    name = "local"
    model = "local"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "respuesta-local"


class UnavailableChatProvider:
    name = "none"
    model = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise ProviderError(
            "No hay proveedor de chat configurado. Define OPENAI_API_KEY o GEMINI_API_KEY."
        )


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    choice = settings.embedding_provider
    if choice == "auto":
        choice = resolved_vendor(settings) or "local"
    if choice == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requiere OPENAI_API_KEY")
        return OpenAIEmbeddingProvider(settings.openai_api_key, settings.openai_embedding_model)
    if choice == "gemini":
        key = settings.gemini_key()
        if not key:
            raise RuntimeError("EMBEDDING_PROVIDER=gemini requiere GEMINI_API_KEY o GOOGLE_API_KEY")
        return GeminiEmbeddingProvider(key, settings.gemini_embedding_model)
    return LocalEmbeddingProvider()


def build_chat(settings: Settings) -> ChatProvider:
    choice = settings.llm_provider
    if choice == "auto":
        vendor = resolved_vendor(settings)
        choice = vendor or "none"
    if choice == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requiere OPENAI_API_KEY")
        return OpenAIChatProvider(settings.openai_api_key, settings.openai_chat_model)
    if choice == "gemini":
        key = settings.gemini_key()
        if not key:
            raise RuntimeError("LLM_PROVIDER=gemini requiere GEMINI_API_KEY o GOOGLE_API_KEY")
        return GeminiChatProvider(key, settings.gemini_chat_model)
    if choice == "local":
        return LocalChatProvider()
    return UnavailableChatProvider()
