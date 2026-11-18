from __future__ import annotations

import re

_SPACES = re.compile(r"[ ]{2,}")
_SPACE_AROUND_NL = re.compile(r" *\n *")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normaliza espacios sin tocar letras acentuadas ni eñes."""

    cleaned = []
    for char in text.replace("\x00", ""):
        if char.isprintable() or char in "\n\t":
            cleaned.append(" " if char == "\t" else char)
        else:
            cleaned.append(" ")
    collapsed = "".join(cleaned)
    collapsed = _SPACES.sub(" ", collapsed)
    collapsed = _SPACE_AROUND_NL.sub("\n", collapsed)
    collapsed = _BLANK_LINES.sub("\n\n", collapsed)
    return collapsed.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = text[start:end]
            break_at = max(window.rfind("\n"), window.rfind(" "))
            if break_at >= size // 2:
                end = start + break_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks
