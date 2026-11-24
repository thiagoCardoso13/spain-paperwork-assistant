"""Answer a question from retrieved official pages, with numbered citations."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

from rag.retrieve import Hit, retrieve

MAX_DISTANCE = 0.75  # hard refusal only for clearly unrelated questions; in the grey zone (0.58-0.75) the model decides, see docs/PLAN.md
DEFAULT_MODEL = "gemini-3.5-flash-lite"  # the fast one; the bigger flash took ~2.5 min per question on the free tier

SYSTEM = """You are an assistant that explains paperwork for foreigners in Spain (NIE, empadronamiento, health card, degree recognition).
Answer ONLY with the numbered context below, which comes from official pages. If the context does not contain the answer, say you could not find it in the sources and suggest checking the official page.
Rules:
- Reply in the same language as the question.
- Be concise and practical (short paragraphs or a short list).
- Cite the sources you used as [1], [2] right after the claim.
- Never invent fees, deadlines or requirements. Say when a source is about applying from abroad or about another region.
- This is general information, not legal advice."""

NO_ANSWER = {
    "es": "No encuentro información sobre eso en las fuentes oficiales que tengo (NIE, empadronamiento, tarjeta sanitaria y homologación de títulos). Prueba a reformular la pregunta o consulta la web oficial.",
    "pt": "Não encontrei informação sobre isso nas fontes oficiais que tenho (NIE, empadronamento, cartão de saúde e homologação de diplomas). Tente reformular a pergunta ou consulte o site oficial.",
    "en": "I could not find anything about that in the official sources I have (NIE, empadronamiento, health card and degree recognition). Try rephrasing, or check the official website.",
}


BUSY = {
    "es": "El servicio gratuito de IA está saturado o sin cuota ahora mismo. Estas son las páginas oficiales más relevantes; vuelve a intentarlo en unos minutos.",
    "pt": "O serviço gratuito de IA está sobrecarregado ou sem cota agora. Estas são as páginas oficiais mais relevantes; tente de novo em alguns minutos.",
    "en": "The free AI service is busy or out of quota right now. Here are the most relevant official pages; please try again in a few minutes.",
}


@dataclass
class Source:
    title: str
    url: str
    topic: str
    fetched_at: str
    text: str  # the retrieved chunks of this page, in reading order


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)  # one entry per page, numbered [1], [2]... in the answer
    grounded: bool = True
    busy: bool = False  # the free AI service failed: do not cache this answer


def group_by_page(hits: list[Hit]) -> list[Source]:
    """One Source per page (ordered by its best match), with its chunks in reading order."""
    pages: dict[str, list[Hit]] = {}
    for h in hits:  # hits arrive sorted best-first
        pages.setdefault(h.url, []).append(h)
    return [
        Source(g[0].title, g[0].url, g[0].topic, g[0].fetched_at, "\n".join(c.text for c in sorted(g, key=lambda c: c.id)))
        for g in pages.values()
    ]


def guess_language(q: str) -> str:
    low = f" {q.lower()} "
    if any(c in low for c in "ãõç") or any(w in low for w in (" você", " como ", " meu ", " preciso", " quais ", " não ")):
        return "pt"
    if any(c in low for c in "¿¡ñ") or any(w in low for w in (" qué ", " cómo ", " necesito", " puedo", " dónde ", " para ")):
        return "es"
    return "en"


def cited(text: str, sources: list[Source]) -> list[tuple[int, Source]]:
    """(number, source) for the sources the answer cites as [n]. No citation means "not found": list nothing."""
    import re

    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(sources)})
    return [(n, sources[n - 1]) for n in nums]


LANG_NAME = {"es": "Spanish", "pt": "Portuguese", "en": "English"}


def build_prompt(question: str, sources: list[Source]) -> str:
    context = "\n\n".join(f"[{i}] {s.title}\n{s.text}" for i, s in enumerate(sources, 1))
    lang = LANG_NAME[guess_language(question)]
    return (
        f"{SYSTEM}\n\nContext (the pages are in Spanish; do not answer in Spanish unless the question is in Spanish):\n{context}"
        f"\n\nQuestion: {question}\nWrite the answer in {lang}.\nAnswer:"
    )


def _gemini(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        timeout=25,
        max_retries=0,
    )


FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]  # each model has its own free quota


class FallbackLLM:
    """Try the models in order; a busy (503) or out-of-quota (429) model hands over to the next one."""

    def __init__(self, llms: list, pause: float = 2.0):
        self.llms = llms
        self.pause = pause
        self.last_error: Exception | None = None

    def invoke(self, prompt: str):
        for llm in self.llms:
            for attempt in range(2):
                try:
                    return llm.invoke(prompt)
                except Exception as exc:  # network, 429, 503...
                    self.last_error = exc
                    transient = any(c in str(exc) for c in ("503", "504", "UNAVAILABLE", "DEADLINE"))
                    if not transient:
                        break  # quota or auth problems will not fix themselves in 2 seconds
                    time.sleep(self.pause)
        raise self.last_error  # type: ignore[misc]


def _groq():
    from langchain_groq import ChatGroq

    return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), api_key=os.getenv("GROQ_API_KEY"), temperature=0, timeout=25, max_retries=0)


def default_llm() -> FallbackLLM:
    names = [os.getenv("LLM_MODEL", DEFAULT_MODEL), *FALLBACK_MODELS]
    llms = [_gemini(n) for n in dict.fromkeys(names)]
    if os.getenv("GROQ_API_KEY"):  # optional second free provider, used when every Gemini model is busy
        llms.append(_groq())
    return FallbackLLM(llms)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def answer(question: str, embed: Callable, llm=None, k: int = 5) -> Answer:
    hits = retrieve(question, embed, k=k)
    # refuse only when even the BEST match is far away; judging every chunk dropped right pages that were merely worded differently
    if not hits or min(h.distance for h in hits) > MAX_DISTANCE:
        return Answer(NO_ANSWER[guess_language(question)], [], grounded=False)
    sources = group_by_page(hits)
    llm = llm or default_llm()
    try:
        reply = llm.invoke(build_prompt(question, sources))
    except Exception:
        return Answer(BUSY[guess_language(question)], sources, grounded=False, busy=True)
    return Answer(_text(reply.content).strip(), sources, grounded=True)
