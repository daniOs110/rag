from app.services.ingestion import extract_text
from app.services.text import chunk_text, clean_text
from tests.pdf_bytes import empty_pdf, text_pdf


def test_clean_text_keeps_spanish_and_collapses_space() -> None:
    assert clean_text("El niño   pidió\t tortilla\n\n\n\nde patatas") == (
        "El niño pidió tortilla\n\nde patatas"
    )


def test_chunk_text_covers_start_and_end() -> None:
    words = [f"pieza{i:04d}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, size=200, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].startswith("pieza0000")
    assert chunks[-1].endswith("pieza0299")
    assert all(chunks)


def test_extract_pdf_text() -> None:
    assert "MESA-PDF-99" in extract_text(text_pdf("MESA-PDF-99"), "nota.pdf")


def test_extract_empty_pdf_has_no_text() -> None:
    assert extract_text(empty_pdf(), "vacio.pdf").strip() == ""


def test_extract_note_utf8() -> None:
    assert extract_text("El niño pidió tortilla.".encode(), "nota.txt") == "El niño pidió tortilla."


def test_reject_unknown_extension() -> None:
    from app.errors import DocumentError
    import pytest

    with pytest.raises(DocumentError):
        extract_text(b"hola", "archivo.docx")
