from __future__ import annotations

import logging

from fastapi import FastAPI

from app.config import DEFAULT_JWT_SECRET, Settings, load_settings, resolved_storage_backend
from app.services.firestore_store import FirestoreStore
from app.services.ingestion import IngestionService
from app.services.providers import build_chat, build_embeddings
from app.services.rag import RagService
from app.services.store import MemoryStore
from app.services.vector_store import build_chroma_store, collection_name

logger = logging.getLogger(__name__)


def wire(app: FastAPI, settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    storage = resolved_storage_backend(settings)
    if storage == "firestore":
        store = FirestoreStore(settings.firestore_project_id)
    else:
        store = MemoryStore()
        if settings.storage_backend == "auto":
            logger.warning(
                "Firestore no está configurado. Los clientes viven solo en memoria de este proceso."
            )

    embeddings = build_embeddings(settings)
    chat = build_chat(settings)
    if embeddings.name == "local" and settings.embedding_provider == "auto":
        logger.warning(
            "Embeddings locales por hash. No recuperan por significado. "
            "Define OPENAI_API_KEY o GEMINI_API_KEY antes de indexar documentos reales."
        )
    if settings.jwt_secret == DEFAULT_JWT_SECRET:
        logger.warning("JWT_SECRET es el valor de desarrollo. Cámbialo antes de exponer la API.")

    vectors = build_chroma_store(
        settings.chroma_backend,
        settings.chroma_path,
        collection_name(embeddings.name, embeddings.model),
    )
    app.state.settings = settings
    app.state.storage_name = storage
    app.state.store = store
    app.state.embeddings = embeddings
    app.state.chat = chat
    app.state.vectors = vectors
    app.state.ingestion = IngestionService(
        embeddings,
        vectors,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_chunks=settings.max_chunks,
    )
    app.state.rag = RagService(
        embeddings,
        vectors,
        chat,
        store,
        default_prompt=settings.default_system_prompt,
    )
