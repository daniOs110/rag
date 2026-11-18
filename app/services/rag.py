from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import SYSTEM_RULES
from app.services.providers import ChatProvider, EmbeddingProvider
from app.services.store import Store
from app.services.vector_store import ChunkHit, VectorStore

logger = logging.getLogger(__name__)

_EXCERPT_CHARS = 500


@dataclass
class RagSource:
    document_id: str
    chunk_index: int
    source: str
    excerpt: str
    distance: float | None


@dataclass
class RagAnswer:
    answer: str
    sources: list[RagSource]


class RagService:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vectors: VectorStore,
        chat: ChatProvider,
        store: Store,
        *,
        default_prompt: str,
    ) -> None:
        self._embeddings = embeddings
        self._vectors = vectors
        self._chat = chat
        self._store = store
        self._default_prompt = default_prompt

    def answer(
        self,
        *,
        business_id: str,
        client_id: str,
        question: str,
        top_k: int,
    ) -> RagAnswer:
        business = self._store.get_business(business_id)
        configured = business.system_prompt if business and business.system_prompt else None
        system_prompt = f"{configured or self._default_prompt}\n\n{SYSTEM_RULES}"
        embedding = self._embeddings.embed_query(question)
        hits = self._vectors.query(
            business_id=business_id,
            client_id=client_id,
            embedding=embedding,
            top_k=top_k,
        )
        user_prompt = _user_prompt(question, hits)
        answer = self._chat.complete(system_prompt, user_prompt)
        logger.info(
            "Chat business=%s client=%s hits=%s question_chars=%s",
            business_id,
            client_id,
            len(hits),
            len(question),
        )
        return RagAnswer(answer=answer, sources=[_source(hit) for hit in hits])


def _user_prompt(question: str, hits: list[ChunkHit]) -> str:
    if not hits:
        context = "(sin documentos para este cliente)"
    else:
        blocks = [f"[{index}] {hit.text}" for index, hit in enumerate(hits, start=1)]
        context = "\n\n".join(blocks)
    return (
        "Usa únicamente el contexto de este cliente para responder.\n\n"
        f"Contexto del cliente:\n{context}\n\n"
        f"Pregunta: {question}"
    )


def _source(hit: ChunkHit) -> RagSource:
    excerpt = hit.text if len(hit.text) <= _EXCERPT_CHARS else hit.text[:_EXCERPT_CHARS].rstrip() + "…"
    return RagSource(
        document_id=hit.document_id,
        chunk_index=hit.chunk_index,
        source=hit.source,
        excerpt=excerpt,
        distance=hit.distance,
    )
