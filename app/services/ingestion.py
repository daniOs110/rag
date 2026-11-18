from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.errors import DocumentError
from app.ids import new_document_id
from app.services.providers import EmbeddingProvider
from app.services.text import chunk_text, clean_text
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)

_TEXT_SUFFIXES = {".txt", ".md", ".text", ".markdown"}
_SOURCE_CLEAN = re.compile(r"[\r\n\t]+")


@dataclass
class IngestResult:
    document_id: str
    chunks: int


class IngestionService:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vectors: VectorStore,
        *,
        chunk_size: int,
        chunk_overlap: int,
        max_chunks: int,
    ) -> None:
        self._embeddings = embeddings
        self._vectors = vectors
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_chunks = max_chunks

    def ingest(
        self,
        *,
        data: bytes,
        filename: str,
        business_id: str,
        client_id: str,
    ) -> IngestResult:
        text = clean_text(extract_text(data, filename))
        if not text:
            raise DocumentError(
                "El archivo no tiene texto digital. Los escaneos requieren OCR y quedan fuera de esta versión."
            )
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not chunks:
            raise DocumentError(
                "El archivo no tiene texto digital. Los escaneos requieren OCR y quedan fuera de esta versión."
            )
        if len(chunks) > self._max_chunks:
            raise DocumentError(
                f"El documento genera {len(chunks)} fragmentos y el máximo es {self._max_chunks}. Divídelo en partes más cortas."
            )
        document_id = new_document_id()
        embeddings = self._embeddings.embed_documents(chunks)
        self._vectors.add(
            business_id=business_id,
            client_id=client_id,
            document_id=document_id,
            source=safe_source(filename),
            chunks=chunks,
            embeddings=embeddings,
        )
        logger.info(
            "Documento indexado business=%s client=%s document=%s chunks=%s",
            business_id,
            client_id,
            document_id,
            len(chunks),
        )
        return IngestResult(document_id=document_id, chunks=len(chunks))


def extract_text(data: bytes, filename: str) -> str:
    if not data:
        raise DocumentError("El archivo está vacío")
    suffix = Path(filename).suffix.lower()
    if data.startswith(b"%PDF") or suffix == ".pdf":
        return _extract_pdf(data)
    if b"\x00" in data[:4096]:
        raise DocumentError("El archivo no parece una nota de texto ni un PDF con texto.")
    if suffix in _TEXT_SUFFIXES or suffix == "":
        return _decode_text(data)
    raise DocumentError("Formato no soportado. Usa un PDF con texto o una nota .txt o .md.")


def safe_source(filename: str) -> str:
    name = Path(filename or "nota.txt").name.strip() or "nota.txt"
    return _SOURCE_CLEAN.sub(" ", name)[:180]


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError("El PDF está protegido. Esta versión no lee PDFs cifrados.")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except DocumentError:
        raise
    except Exception as exc:
        logger.info("PDF ilegible: %s", type(exc).__name__)
        raise DocumentError("No se pudo leer el PDF.") from exc
    return "\n\n".join(pages)
