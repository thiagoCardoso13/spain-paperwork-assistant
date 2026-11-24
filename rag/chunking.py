"""Split a cleaned page into overlapping chunks, preferring paragraph boundaries."""
from __future__ import annotations


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 300) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # a paragraph longer than the limit is cut on sentence-ish boundaries
        pieces = [para] if len(para) <= max_chars else _split_long(para, max_chars)
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > max_chars:
                chunks.append(current)
                current = (_tail(current, overlap) + " " + piece).strip() if overlap else piece
            else:
                current = f"{current}\n{piece}".strip() if current else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(para: str, max_chars: int) -> list[str]:
    out, cur = [], ""
    for sentence in para.replace("; ", ". ").split(". "):
        s = sentence.strip()
        if not s:
            continue
        if cur and len(cur) + 2 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur}. {s}" if cur else s
    if cur:
        out.append(cur)
    # hard cut if a single sentence is still too long
    final: list[str] = []
    for piece in out:
        while len(piece) > max_chars:
            final.append(piece[:max_chars])
            piece = piece[max_chars:]
        final.append(piece)
    return final


def _tail(text: str, size: int) -> str:
    """Last `size` characters, moved forward to the next word boundary so a chunk never starts mid-word."""
    tail = text[-size:]
    if len(text) > size and not text[-size - 1].isspace():
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if cut != -1 else ""
    return tail.strip()
