"""Retrieval hit rate on eval/questions.yml (no LLM calls, so it costs no quota).

Run:  python -m rag.evaluate
A question is a hit if any of its expected sources appears in the top-k retrieved chunks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from rag.embeddings import embed
from rag.retrieve import retrieve

ROOT = Path(__file__).resolve().parent.parent


def source_id(url: str, title: str) -> str:
    return url  # compared against sources.yml ids through the url map below


def main(k: int = 4) -> int:
    sources = yaml.safe_load((ROOT / "sources" / "sources.yml").read_text(encoding="utf-8"))
    by_url = {s["url"]: s["id"] for s in sources}
    questions = yaml.safe_load((ROOT / "eval" / "questions.yml").read_text(encoding="utf-8"))
    hits_n = 0
    for item in questions:
        hits = retrieve(item["q"], embed, k=k)
        got = [by_url.get(h.url, h.url) for h in hits]
        ok = any(e in got for e in item["expect"])
        if ok and item.get("contains"):
            ok = any(by_url.get(h.url) in item["expect"] and any(c.lower() in h.text.lower() for c in item["contains"]) for h in hits)
        hits_n += ok
        print(("HIT  " if ok else "MISS ") + item["q"][:75])
        if not ok:
            print("       got:", ", ".join(dict.fromkeys(got)))
    rate = hits_n / len(questions)
    print(f"\nhit@{k}: {hits_n}/{len(questions)} = {rate:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4))
