# 📄 Spain Paperwork Assistant

A chat that answers questions about **NIE, empadronamiento, the health card (SIP) and degree recognition** in Spain, in Spanish,
English or Portuguese, and **cites the official page each answer comes from**.

Built with **Python · Streamlit · fastembed · ChromaDB · Gemini**, and **100% free to run**: no paid service, no credit card.

> 🔗 **Live demo:** _coming soon_

## The problem

Paperwork information for foreigners is scattered across ministries, city councils and regional health services, written in
administrative Spanish, and it changes. A generic chatbot will confidently invent fees and deadlines. This assistant is
restricted to a small list of **official pages**, shows which ones it used, and says so when it cannot find the answer.

## How it works

```
sources.yml ──fetch──▶ clean text ──chunk──▶ embeddings ──▶ ChromaDB (committed in the repo)
                                                                  │
question ──embed──▶ top chunks + neighbours + best page ──▶ Gemini ──▶ answer with [1] [2] citations
                          │
                  too far from every chunk? ──▶ "I could not find that" (no model call, no quota spent)
```

| Piece | Choice | Cost |
|---|---|---|
| UI / hosting | Streamlit, Streamlit Community Cloud | free |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` through `fastembed` (ONNX, CPU, runs locally) | free, no API calls |
| Vector database | ChromaDB, index built offline and committed (≈1 MB) | free |
| LLM | Gemini free tier: `gemini-3.5-flash-lite` with automatic fallback to other free models | free (quota-limited) |

### Decisions worth knowing
- **Grounded or silent.** Answers use only the retrieved official text. If no chunk is close enough (cosine distance above 0.58), the app refuses without calling the model.
- **Good retrieval is more than a vector search.** Official pages are forms ("Plazo de resolución" … "Tres meses"), so the fact a user needs often sits next to the best match. Retrieval returns the top chunks, their neighbours and the whole best page when it is short. The source title is embedded with each chunk.
- **Cheap on the free tier.** Repeated questions are cached, each session is limited to 10 questions, and the answer language is pinned in the prompt (the pages are in Spanish, the question may not be).
- **Tests need no key.** 12 tests run with fake embeddings and a fake LLM.

## Does it retrieve the right page? (evaluation)

`python -m rag.evaluate` runs 14 questions (ES / EN / PT) and checks that the retrieved chunks come from an expected official page **and contain the fact** (for example the document list or the fee). It makes no model calls.

| Version | hit@4 |
|---|---|
| Plain vector search, 900-character chunks | 79% (11/14) |
| + source title in the embedding | 86% (12/14) |
| + overlap that never cuts a word, neighbours and best-page expansion | **93% (13/14)** |

The remaining miss is a cost question that the search confuses with the municipal offices page. The set is small, so treat the percentages as a regression guard, not as a benchmark.

## Sources and limits

Ten official pages: Ayuntamiento de Elche (padrón, OMAC), Conselleria de Sanidad (SIP, FAQ, convenio especial, universal healthcare for foreigners), the Ministry of Foreign Affairs (NIE pages of three consulates) and the BOE (Real Decreto 889/2022 on degree recognition). List in `sources/sources.yml`.

- The Sede Electrónica of the Ministry answers 503 to automated requests, and the Policía Nacional pages are empty shells without JavaScript, so **the NIE coverage is about applying from abroad through a consulate**, not the police procedure in Spain. The app says so in its answers.
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
pytest               # 12 tests, no key needed
python -m rag.evaluate
```

## Structure

```
app.py                 # Streamlit chat UI
rag/fetch.py           # download + clean the official pages
rag/chunking.py        # overlapping chunks that never cut a word
rag/embeddings.py      # local multilingual embeddings (fastembed)
rag/ingest.py          # build the ChromaDB index
rag/retrieve.py        # top-k + neighbours + best page
rag/answer.py          # prompt, citations, model fallback, refusal
rag/evaluate.py        # retrieval hit rate on eval/questions.yml
sources/sources.yml    # the allowlist of official pages
data/index/            # committed vector index
docs/PLAN.md           # plan, decisions and phases
```
