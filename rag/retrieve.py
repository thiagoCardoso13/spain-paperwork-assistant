"""Semantic search over the committed Chroma index."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import chromadb

from rag.ingest import COLLECTION, INDEX
from rag.lexical import BM25


@dataclass
class Hit:
    id: str
    text: str
    title: str
    url: str
    topic: str
    fetched_at: str
    distance: float  # cosine distance: 0 = identical, larger = less similar


@lru_cache(maxsize=2)
def _collection(index_dir: str):
    return chromadb.PersistentClient(path=index_dir).get_collection(COLLECTION)


KW_WEIGHT = 0.5  # keywords help with names and exact terms, but are useless across languages (EN question, ES page)


@lru_cache(maxsize=2)
def _lexical(index_dir: str):
    col = _collection(index_dir)
    got = col.get(include=["documents", "metadatas"])
    # the source title is searchable too ("... (dentro de España)" tells apart pages that share vocabulary)
    return got["ids"], BM25([f"{m['title']} {d}" for d, m in zip(got["documents"], got["metadatas"])])


def _fused_top(col, index_dir: str, question: str, embed, k: int) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion of semantic (cosine) and keyword (BM25) rankings: [(id, vector distance)]."""
    res = col.query(query_embeddings=embed([question]), n_results=col.count(), include=["distances"])
    dist = dict(zip(res["ids"][0], res["distances"][0]))
    vec_rank = {id_: r for r, id_ in enumerate(res["ids"][0])}
    ids, bm25 = _lexical(index_dir)
    scores = bm25.scores(question)
    order = sorted(range(len(ids)), key=lambda i: -scores[i])
    kw_rank = {ids[i]: r for r, i in enumerate(order) if scores[i] > 0}
    fused = {i: 1 / (60 + vec_rank[i]) + (KW_WEIGHT / (60 + kw_rank[i]) if i in kw_rank else 0) for i in vec_rank}
    return [(i, dist[i]) for i in sorted(fused, key=fused.get, reverse=True)[:k]]


def retrieve(
    question: str,
    embed: Callable[[list[str]], list[list[float]]],
    k: int = 5,
    index_dir: Path = INDEX,
    neighbors: int = 1,
    page_limit: int = 8,
) -> list[Hit]:
    """Top-k chunks (semantic + keyword fusion), their neighbours (chunk N-1 and N+1), and the whole best page if it is short.

    Official pages are forms ("Plazo de resolución" ... "Tres meses"), so the fact the user needs often sits in the
    chunk next to the best match. Neighbours inherit the distance of the hit that pulled them in.
    """
    col = _collection(str(index_dir))
    top = _fused_top(col, str(index_dir), question, embed, k)
    got = col.get(ids=[i for i, _ in top], include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    hits: dict[str, Hit] = {}
    for id_, dist in top:
        doc, meta = by_id[id_]
        hits[id_] = Hit(id_, doc, meta["title"], meta["url"], meta["topic"], meta["fetched_at"], dist)
    if neighbors:
        wanted = {}
        for id_, hit in list(hits.items()):
            base, _, num = id_.rpartition("-")
            for d in range(1, neighbors + 1):
                for n in (int(num) - d, int(num) + d):
                    nid = f"{base}-{n:03d}"
                    if n >= 0 and nid not in hits:
                        wanted.setdefault(nid, hit.distance)
        if wanted:
            got = col.get(ids=list(wanted), include=["documents", "metadatas"])
            for id_, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                hits[id_] = Hit(id_, doc, meta["title"], meta["url"], meta["topic"], meta["fetched_at"], wanted[id_])
    # the best page is usually a short form: give the model all of it (up to `page_limit` chunks)
    best = min(hits.values(), key=lambda h: h.distance)
    source_id = best.id.rpartition("-")[0]
    page = col.get(where={"source_id": source_id}, include=["documents", "metadatas"])
    if len(page["ids"]) <= page_limit:
        for id_, doc, meta in zip(page["ids"], page["documents"], page["metadatas"]):
            hits.setdefault(id_, Hit(id_, doc, meta["title"], meta["url"], meta["topic"], meta["fetched_at"], best.distance))
    # best matches first; within a page keep reading order so the context reads naturally
    return sorted(hits.values(), key=lambda h: (round(h.distance, 3), h.id))
