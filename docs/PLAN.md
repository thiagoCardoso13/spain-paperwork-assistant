# Spain Paperwork Assistant (project #2 of the portfolio list)

A RAG chat that answers questions about NIE, empadronamiento, health card (SIP) and degree recognition in Spain,
**citing the official page each answer comes from**. Goal: show generative AI + RAG + vector search with a live demo.

## Hard constraint: everything free
No paid service, no credit card. Every choice below has a free tier or is open source.

| Piece | Choice | Cost |
|---|---|---|
| Language / UI | Python 3.12, Streamlit | free |
| Hosting | Streamlit Community Cloud (public repo) | free |
| LLM | Gemini API free tier (`gemini-3.5-flash-lite`, the fast one; see Project_AI_Agent lessons) with a Groq free fallback | free (quota-limited) |
| Embeddings | `fastembed` (ONNX, runs on CPU, multilingual model) | free, no API calls |
| Vector DB | ChromaDB, persisted in the repo (index built offline and committed) | free |
| Scraping | `requests` + `trafilatura` over a short allowlist of **official** pages | free |
| CI | GitHub Actions (public repo) | free |

Free-tier limits are real: keep a per-session question cap, cache answers, and show a friendly message when the quota is out.

## Decisions
- Sources: only official public pages (Policía Nacional, Sede Electrónica, Ayuntamiento de Elche, Conselleria de Sanidad, Ministerio de Universidades/BOE). List and URLs in `sources/sources.yml`.
- The index is built offline (`python -m rag.ingest`) and committed, so the cloud app starts fast and needs no scraping at runtime.
- Every answer must cite sources (title + URL); if retrieval finds nothing relevant, the assistant says so instead of guessing.
- Answers in the language of the question (ES / EN / PT). Documents are mostly Spanish, so the embedding model must be multilingual.
- Clear disclaimer: informational only, not legal advice, rules change; confirm on the official page.
- Evaluation: a small set of questions with the expected source page, measured as "retrieval hit rate", so quality is a number, not a feeling.
- Tests run without any API key (fake embeddings / fake LLM), like Project_AI_Agent.

## Phases
1. [x] Sources list + downloader + text cleaning
2. [x] Chunking + embeddings + Chroma index (`rag/ingest.py`)
3. [x] Retrieval + answer chain with citations (`rag/answer.py`)
4. [x] Streamlit UI (chat, sources panel, disclaimer, question cap)
5. [x] Evaluation set + hit-rate report in the README
6. [ ] GitHub repo, CI, Streamlit Cloud deploy, README with GIF
7. [ ] Add the card to the portfolio site and write the post

## Risks
- Pages change or block scraping: keep the raw snapshot date in the index and show it in the UI.
- Quota: low cap per session, cache repeated questions.
- Legal accuracy: cite, disclaim, never present as official.

## Retrieval improvements (second pass)
- Added the Policía Nacional sede pages (NIE assignment, residence and non-residence certificates).
- Hybrid search: cosine + BM25 with reciprocal-rank fusion (keyword weight 0.5). Largest single gain.
- Refusal threshold moved from 0.58 to 0.75 (best match); the 0.58 cutoff threw away right pages worded differently.
- Evaluation now has a held-out set and an off-topic set. Query expansion with synonyms was tried and removed (it lowered the score).
- Open: cross-language weakness of the small embedding model; a larger model does not fit the free hosting memory.
