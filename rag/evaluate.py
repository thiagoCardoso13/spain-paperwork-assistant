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

DEFAULT_K = 5

ROOT = Path(__file__).resolve().parent.parent


def run(path: Path, by_url: dict, k: int) -> tuple[int, int]:
    questions = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    return hits_n, len(questions)


def main(k: int = DEFAULT_K) -> int:
    sources = yaml.safe_load((ROOT / "sources" / "sources.yml").read_text(encoding="utf-8"))
    by_url = {s["url"]: s["id"] for s in sources}
    for label, name in (("tuning set", "questions.yml"), ("held-out set", "held_out.yml")):
        path = ROOT / "eval" / name
        if path.exists():
            print(f"\n== {label} ({name})")
            n, total = run(path, by_url, k)
            print(f"{label}: hit@{k} = {n}/{total} = {n / total:.0%}")
    off = ROOT / "eval" / "off_topic.yml"
    if off.exists():
        from rag.answer import MAX_DISTANCE

        qs = yaml.safe_load(off.read_text(encoding="utf-8"))
        refused = 0
        for q in qs:
            best = min(h.distance for h in retrieve(q, embed, k=k))
            refused += best > MAX_DISTANCE
            print(("REFUSED " if best > MAX_DISTANCE else "ANSWERS ") + f"{best:.3f} {q[:70]}")
        print(f"off-topic: refused {refused}/{len(qs)} (threshold {MAX_DISTANCE})")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_K))
