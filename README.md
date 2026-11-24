# 📄 Spain Paperwork Assistant

[![CI](https://github.com/thiagoCardoso13/spain-paperwork-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/thiagoCardoso13/spain-paperwork-assistant/actions/workflows/ci.yml)

A chat that answers questions about **NIE, empadronamiento, the health card (SIP) and degree recognition** in Spain, in Spanish,
English or Portuguese, and **cites the official page each answer comes from**.

Built with **Python · Streamlit · fastembed · ChromaDB · Gemini**, and **100% free to run**: no paid service, no credit card.

> 🔗 **Live demo:** [spain-paperwork-assistant-y7ancesxw2rncyukxz8r5s.streamlit.app](https://spain-paperwork-assistant-y7ancesxw2rncyukxz8r5s.streamlit.app/)
>
> Free hosting and free AI quota: the app sleeps when idle, the first question downloads the embedding model (about 30 seconds), and each session is limited to 10 questions.

## The problem

Paperwork information for foreigners is scattered across ministries, city councils and regional health services, written in
administrative Spanish, and it changes. A generic chatbot will confidently invent fees and deadlines. This assistant is
restricted to a small list of **official pages**, shows which ones it used, and says so when it cannot find the answer.

## How it works

```
sources.yml ──fetch──▶ clean text ──chunk──▶ embeddings ──▶ ChromaDB (committed in the repo)
                                                                  │
question ──▶ semantic search + keyword search (BM25) ──▶ fused top chunks + neighbours + best page ──▶ Gemini ──▶ answer with [1] [2] citations
                          │
                  clearly unrelated question? ──▶ "I could not find that" (no model call, no quota spent)
```

| Piece | Choice | Cost |
|---|---|---|
| UI / hosting | Streamlit, Streamlit Community Cloud | free |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` through `fastembed` (ONNX, CPU, runs locally) | free, no API calls |
| Vector database | ChromaDB, index built offline and committed (≈3 MB) | free |
| LLM | Gemini free tier: `gemini-3.5-flash-lite` with automatic fallback to other free models | free (quota-limited) |

### Decisions worth knowing
- **Grounded or silent.** Answers use only the retrieved official text. A clearly unrelated question (best match farther than 0.75 cosine distance) is refused without calling the model; in the grey zone the prompt makes the model say it found nothing, and a "not found" answer lists no sources. A single distance cutoff cannot separate short in-domain questions (0.72) from off-topic ones (0.60), so the cutoff is only the cheap first filter.
- **Hybrid retrieval.** Semantic search (cosine) and a small BM25 keyword search are fused with reciprocal-rank fusion, keywords weighted at 0.5 because they are useless across languages (English question, Spanish page). The source title is part of both indexes.
- **Official pages are forms** ("Plazo de resolución" … "Tres meses"), so the fact a user needs often sits next to the best match. Retrieval returns the top chunks, their neighbours and the whole best page when it is short.
- **Cheap on the free tier.** Repeated questions are cached, each session is limited to 10 questions, and the answer language is pinned in the prompt (the pages are in Spanish, the question may not be).
- **Tests need no key.** 13 tests run with fake embeddings and a fake LLM.

## Does it retrieve the right page? (evaluation)

`python -m rag.evaluate` makes no model calls, so it costs no quota. A question is a hit if the retrieved chunks come from an expected official page **and contain the fact** (for example the document list or the fee).

| Version | Tuning set (19 questions) |
|---|---|
| Plain vector search, 900-character chunks (14 questions at the time) | 79% |
| + source title in the embedding, overlap that never cuts a word, neighbours, best-page expansion (14 questions) | 93% |
| Added 3 police pages and 5 new NIE questions: more pages, same model | **68%** (the new pages crowded each other out) |
| + hybrid search (BM25 fusion), k = 5 | 84% → 89% |
| + dropping a near-duplicate consulate page | **95%** (18/19) |

Questions written **after** tuning and never used for it (`eval/held_out.yml`): **88%** (7/8). Off-topic questions (`eval/off_topic.yml`): 5 of 8 refused outright, the other 3 get a "not found" answer from the model.

What did not work, and was removed: expanding questions with administrative synonyms ("cuesta" → "tasas"). It lowered the tuning score from 95% to 89% because "tasas" also pulls the NIE fee pages.
Known misses: an English question about renting a room (the answer is in Spanish text; the small multilingual model is weak across languages) and one NIE question that the consulate pages still win. The sets are small, so treat the percentages as a regression guard, not as a benchmark.

## Sources and limits

Twelve official pages: Ayuntamiento de Elche (padrón, OMAC), Conselleria de Sanidad (SIP, FAQ, convenio especial, universal healthcare for foreigners), the Sede Electrónica of the Policía Nacional (NIE assignment, certificates of residence and non-residence), the Ministry of Foreign Affairs (NIE pages of two consulates, for applications from abroad) and the BOE (Real Decreto 889/2022 on degree recognition). List in `sources/sources.yml`.

- The Sede Electrónica of the Ministry (`sede.administracionespublicas.gob.es`) answers 503 to automated requests, so it is not used and I did not try to get around it. The NIE procedure inside Spain comes from the Policía Nacional sede (`sede.policia.gob.es`), a different host that works.
- The tax amount of the NIE form (790-012) is not in the pages, so the assistant cannot quote it.
- Empadronamiento covers **Elche** only; the health card covers the **Comunitat Valenciana**.
- General information, **not legal advice**. Rules change: always confirm on the official page. Each source shows the date it was read.

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .env   # or: echo "GOOGLE_API_KEY=..." > .env  (free key: aistudio.google.com)
python -m rag.fetch      # download the official pages (optional: the index is already committed)
python -m rag.ingest     # build the vector index
streamlit run app.py
```

```bash
pytest               # 13 tests, no key needed
python -m rag.evaluate
```

## Structure

```
app.py                 # Streamlit chat UI
rag/fetch.py           # download + clean the official pages
rag/chunking.py        # overlapping chunks that never cut a word
rag/embeddings.py      # local multilingual embeddings (fastembed)
rag/ingest.py          # build the ChromaDB index
rag/lexical.py         # tiny BM25 keyword scorer
rag/retrieve.py        # semantic + keyword fusion, neighbours, best page
rag/answer.py          # prompt, citations, model fallback, refusal
rag/evaluate.py        # retrieval hit rate on eval/questions.yml
sources/sources.yml    # the allowlist of official pages
data/index/            # committed vector index
docs/PLAN.md           # plan, decisions and phases
```
