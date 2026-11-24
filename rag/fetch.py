"""Download the official pages listed in sources/sources.yml and save clean text.

Run:  python -m rag.fetch
Output: data/raw/<id>.json  ({id, topic, title, url, fetched_at, text})
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import requests
import trafilatura
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "spain-paperwork-assistant/0.1 (student portfolio project; low-volume)"}
MIN_CHARS = 600  # shorter than this usually means a JS shell or an error page


def load_sources() -> list[dict]:
    return yaml.safe_load((ROOT / "sources" / "sources.yml").read_text(encoding="utf-8"))


def clean(html: str, url: str) -> str:
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True, favor_recall=True)
    return (text or "").strip()


def fetch_one(src: dict) -> tuple[bool, str]:
    resp = requests.get(src["url"], headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = clean(resp.text, src["url"])
    if len(text) < MIN_CHARS:
        return False, f"only {len(text)} chars (JS page or error?)"
    record = {**src, "fetched_at": date.today().isoformat(), "text": text}
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{src['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, f"{len(text)} chars"


def main() -> int:
    failures = 0
    for src in load_sources():
        try:
            ok, msg = fetch_one(src)
        except Exception as exc:  # network errors, 4xx/5xx
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        print(("OK   " if ok else "FAIL ") + f"{src['id']:22} {msg}")
        failures += not ok
        time.sleep(1.5)  # be polite
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
