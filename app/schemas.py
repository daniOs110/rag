from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_INTL = re.compile(r"^\+[0-9][0-9\s().-]{6,22}$")
_PHONE_LOCAL = re.compile(r"^[0-9][0-9\s().-]{6,22}$")


def normalize_contact(value: str) -> str:
    value = value.strip()
    if _EMAIL.fullmatch(value):
        return value
    digits = re.sub(r"\D", "", value)
    if 7 <= len(digits) <= 15 and (_PHONE_INTL.fullmatch(value) or _PHONE_LOCAL.fullmatch(value)):
        return re.sub(r"\s+", " ", value)
    raise ValueError("contact debe ser un correo o un teléfono")


def validate_metadata(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        raise ValueError("metadata demasiado anidada")
    if isinstance(value, dict):
        if not all(isinstance(key, str) and key.strip() for key in value):
            raise ValueError("las claves de metadata deben ser texto")
        cleaned = {key: validate_metadata(item, depth + 1) for key, item in value.items()}
    elif isinstance(value, list):
        cleaned = [validate_metadata(item, depth + 1) for item in value]
    elif isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        cleaned = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata no admite NaN ni infinito")
        cleaned = value
    else:
        raise ValueError("metadata solo admite JSON (texto, número, bool, lista u objeto)")

    if depth == 0:
        encoded = json.dumps(cleaned, ensure_ascii=False)
        if len(encoded) > 16_000:
            raise ValueError("metadata demasiado grande")
    return cleaned


class ClientCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    contact: str = Field(min_length=1, max_length=320)
    notes: str | None = Field(default=None, max_length=4000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("el nombre está vacío")
        return value

    @field_validator("contact")
    @classmethod
    def check_contact(cls, value: str) -> str:
        return normalize_contact(value)

    @field_validator("notes")
    @classmethod
    def strip_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("metadata")
    @classmethod
    def check_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_metadata(value)


class ClientOut(BaseModel):
    client_id: str
    business_id: str
    name: str
    contact: str
    notes: str | None
    metadata: dict[str, Any]
    created_at: datetime


class BusinessUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    system_prompt: str = Field(min_length=1, max_length=8000)

    @field_validator("name", "system_prompt")
    @classmethod
    def strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("el campo está vacío")
        return value


class BusinessOut(BaseModel):
    business_id: str
    name: str
    system_prompt: str
    prompt_configured: bool


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("la pregunta está vacía")
        return value


class SourceOut(BaseModel):
    document_id: str
    chunk_index: int
    source: str
    excerpt: str
    distance: float | None = None


class ChatOut(BaseModel):
    answer: str
    client_id: str
    sources: list[SourceOut]


class UploadOut(BaseModel):
    document_id: str
    client_id: str
    chunks: int


class HealthOut(BaseModel):
    status: str
    storage: str
    embeddings: str
    embedding_model: str
    llm: str
    llm_model: str | None = None
