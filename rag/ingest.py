"""Build the vector index from data/raw/*.json.

Run:  python -m rag.ingest
Output: data/index/ (ChromaDB, committed to the repo so the app starts fast)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import chromadb

from rag.chunking import chunk_text

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INDEX = ROOT / "data" / "index"
COLLECTION = "paperwork"


def build_records(raw_dir: Path = RAW) -> list[dict]:
    records = []
    for path in sorted(raw_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for i, chunk in enumerate(chunk_text(doc["text"])):
            records.append(
                {
                    "id": f"{doc['id']}-{i:03d}",
                    "text": chunk,
                    "meta": {
                        "source_id": doc["id"],
                        "title": doc["title"],
                        "url": doc["url"],
                        "topic": doc["topic"],
                        "fetched_at": doc["fetched_at"],
                    },
                }
            )
    return records


def build_index(embed: Callable[[list[str]], list[list[float]]], index_dir: Path = INDEX, raw_dir: Path = RAW) -> int:
    records = build_records(raw_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    client = chromadb.PersistentClient(path=str(index_dir))
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    batch = 32
    for i in range(0, len(records), batch):
        part = records[i : i + batch]
        col.add(
            ids=[r["id"] for r in part],
            documents=[r["text"] for r in part],
            metadatas=[r["meta"] for r in part],
            embeddings=embed([r["text"] for r in part]),
        )
    return len(records)


if __name__ == "__main__":
    from rag.embeddings import embed as real_embed

    print(f"Indexed {build_index(real_embed)} chunks into {INDEX}")
