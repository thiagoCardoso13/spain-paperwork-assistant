import json
import hashlib

import chromadb

from rag.chunking import chunk_text
from rag.ingest import COLLECTION, build_index, build_records


def fake_embed(texts):
    """Deterministic 8-dim vectors from a hash, so tests need no model and no network."""
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        out.append([b / 255 for b in h[:8]])
    return out


def test_chunks_respect_limit_and_keep_all_text():
    text = "\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert all(len(c) <= 600 for c in chunks)  # limit plus overlap
    assert len(chunks) > 3
    assert "Paragraph 0" in chunks[0] and "Paragraph 19" in chunks[-1]


def test_long_paragraph_is_split():
    chunks = chunk_text("x" * 3000, max_chars=900, overlap=0)
    assert len(chunks) == 4 and all(len(c) <= 900 for c in chunks)


def test_records_carry_source_metadata(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"id": "a", "topic": "NIE", "title": "T", "url": "https://x", "fetched_at": "2026-01-01", "text": "Hola.\n" * 50}),
        encoding="utf-8",
    )
    recs = build_records(tmp_path)
    assert recs and recs[0]["meta"]["url"] == "https://x" and recs[0]["id"] == "a-000"


def test_build_index_stores_vectors_and_metadata(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.json").write_text(
        json.dumps({"id": "a", "topic": "NIE", "title": "T", "url": "https://x", "fetched_at": "2026-01-01", "text": "Texto de prueba.\n" * 80}),
        encoding="utf-8",
    )
    n = build_index(fake_embed, index_dir=tmp_path / "idx", raw_dir=raw)
    col = chromadb.PersistentClient(path=str(tmp_path / "idx")).get_collection(COLLECTION)
    assert col.count() == n > 0
    res = col.get(limit=1, include=["metadatas"])
    assert res["metadatas"][0]["topic"] == "NIE"


def test_overlap_never_starts_mid_word():
    words = [f"palabra{i}" for i in range(400)]
    text = "\n".join(" ".join(words[i : i + 10]) for i in range(0, 400, 10))
    chunks = chunk_text(text, max_chars=300, overlap=60)
    assert len(chunks) > 3
    for c in chunks:
        assert c.split()[0].startswith("palabra"), c[:30]
