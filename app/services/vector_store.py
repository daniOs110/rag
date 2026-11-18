from __future__ import annotations

import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol


@dataclass
class ChunkHit:
    document_id: str
    chunk_index: int
    source: str
    text: str
    distance: float | None


class VectorStore(Protocol):
    def add(
        self,
        *,
        business_id: str,
        client_id: str,
        document_id: str,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None: ...

    def query(
        self,
        *,
        business_id: str,
        client_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[ChunkHit]: ...


def collection_name(provider: str, model: str) -> str:
    raw = f"chunks_{provider}_{model}".lower()
    slug = re.sub(r"[^a-z0-9_-]+", "_", raw).strip("_")
    slug = slug[:63].strip("_")
    return slug if len(slug) >= 3 else "chunks_local"


class ChromaVectorStore:
    """Chunks con metadata obligatoria business_id + client_id.

    No hay un método de búsqueda sin ambos filtros: un negocio no puede
    recuperar vectores de otro, ni un cliente los de otro cliente.
    """

    def __init__(self, collection: Any) -> None:
        self._collection = collection
        self._lock = Lock()

    def add(
        self,
        *,
        business_id: str,
        client_id: str,
        document_id: str,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Cada chunk necesita su embedding")
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {
                "business_id": business_id,
                "client_id": client_id,
                "document_id": document_id,
                "chunk_index": index,
                "source": source,
            }
            for index in range(len(chunks))
        ]
        with self._lock:
            batch = 100
            for start in range(0, len(chunks), batch):
                end = start + batch
                self._collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=chunks[start:end],
                    metadatas=metadatas[start:end],
                )

    def query(
        self,
        *,
        business_id: str,
        client_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[ChunkHit]:
        where = {
            "$and": [
                {"business_id": business_id},
                {"client_id": client_id},
            ]
        }
        with self._lock:
            total = self._collection.count()
            if total == 0:
                return []
            raw = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, total),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        return _hits(raw)


class PrecomputedEmbeddings:
    """Chroma no calcula vectores: los proveedores de la API ya los entregan.

    Se declara legacy para que Chroma no intente reconstruir un modelo al abrir
    la colección. add() y query() siempre reciben los vectores.
    """

    def name(self) -> str:
        return "precomputed"

    def is_legacy(self) -> bool:
        return True

    def __call__(self, input):  # noqa: A002 - firma que espera Chroma
        raise RuntimeError("Los embeddings se calculan fuera de Chroma")

    def embed_query(self, input=None, **kwargs):  # noqa: A002
        raise RuntimeError("Los embeddings se calculan fuera de Chroma")


def build_chroma_store(backend: str, path: str, name: str) -> ChromaVectorStore:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    chroma_settings = ChromaSettings(anonymized_telemetry=False)
    if backend == "memory":
        client = chromadb.EphemeralClient(settings=chroma_settings)
    else:
        os.makedirs(path, exist_ok=True)
        client = chromadb.PersistentClient(path=path, settings=chroma_settings)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=PrecomputedEmbeddings(),
    )
    return ChromaVectorStore(collection)


def _hits(raw: dict[str, Any]) -> list[ChunkHit]:
    documents = (raw.get("documents") or [[]])[0] or []
    metadatas = (raw.get("metadatas") or [[]])[0] or []
    distances = (raw.get("distances") or [[]])[0] or []
    hits: list[ChunkHit] = []
    for index, text in enumerate(documents):
        meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        if not text or "document_id" not in meta or "client_id" not in meta:
            continue
        distance = distances[index] if index < len(distances) else None
        hits.append(
            ChunkHit(
                document_id=str(meta["document_id"]),
                chunk_index=int(meta.get("chunk_index") or 0),
                source=str(meta.get("source") or ""),
                text=str(text),
                distance=float(distance) if distance is not None else None,
            )
        )
    return hits
