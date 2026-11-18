from __future__ import annotations

import os
from typing import Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_JWT_SECRET = "dev-only-change-me-please-now!!!"

DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente del negocio. Ayudas a consultar la información de un cliente "
    "a partir de sus documentos. Respondes en el idioma de la pregunta, con claridad y sin inventar."
)

SYSTEM_RULES = (
    "Reglas fijas: responde solo con el contexto del cliente indicado. "
    "Si el dato no consta en ese contexto, dilo con claridad. "
    "No mezcles información de otros clientes ni inventes hechos."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_file=None)

    environment: str = "development"
    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24

    storage_backend: str = "auto"
    firestore_project_id: str | None = None

    chroma_backend: str = "persistent"
    chroma_path: str = "data/chroma"

    embedding_provider: str = "auto"
    llm_provider: str = "auto"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_api_key: str | None = None
    google_api_key: str | None = None
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2"

    chunk_size: int = 800
    chunk_overlap: int = 120
    max_chunks: int = 300
    max_upload_bytes: int = 10 * 1024 * 1024
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @field_validator("environment")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"development", "test", "production"}:
            raise ValueError("environment debe ser development, test o production")
        return value

    @field_validator("storage_backend")
    @classmethod
    def normalize_storage(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "memory", "firestore"}:
            raise ValueError("storage_backend debe ser auto, memory o firestore")
        return value

    @field_validator("chroma_backend")
    @classmethod
    def normalize_chroma(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"persistent", "memory"}:
            raise ValueError("chroma_backend debe ser persistent o memory")
        return value

    @field_validator("embedding_provider")
    @classmethod
    def normalize_embedding_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "openai", "gemini", "local"}:
            raise ValueError("embedding_provider debe ser auto, openai, gemini o local")
        return value

    @field_validator("llm_provider")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "openai", "gemini", "local", "none"}:
            raise ValueError("llm_provider debe ser auto, openai, gemini, local o none")
        return value

    @field_validator("jwt_algorithm")
    @classmethod
    def only_hs256(cls, value: str) -> str:
        if value != "HS256":
            raise ValueError("Solo se admite HS256")
        return value

    @model_validator(mode="after")
    def check_runtime(self) -> Self:
        if len(self.jwt_secret) < 32:
            raise ValueError("JWT_SECRET debe tener al menos 32 caracteres")
        if self.environment == "production" and self.jwt_secret == DEFAULT_JWT_SECRET:
            raise ValueError("En production hay que definir JWT_SECRET")
        if self.chunk_size < 200:
            raise ValueError("chunk_size debe ser al menos 200")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        if self.max_chunks < 1:
            raise ValueError("max_chunks debe ser positivo")
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes debe ser positivo")
        if self.jwt_expire_hours <= 0:
            raise ValueError("jwt_expire_hours debe ser positivo")
        return self

    def gemini_key(self) -> str | None:
        return self.gemini_api_key or self.google_api_key


def load_settings() -> Settings:
    return Settings()


def resolved_storage_backend(settings: Settings) -> str:
    if settings.storage_backend != "auto":
        return settings.storage_backend
    if os.environ.get("FIRESTORE_EMULATOR_HOST") or settings.firestore_project_id:
        return "firestore"
    return "memory"


def resolved_vendor(settings: Settings) -> str | None:
    """Proveedor concreto cuando la selección es automática.

    Si hay clave de OpenAI y de Gemini, se usa OpenAI para que embeddings y chat
    compartan proveedor. Si no hay ninguna clave, no hay proveedor remoto.
    """

    if settings.openai_api_key:
        return "openai"
    if settings.gemini_key():
        return "gemini"
    return None
