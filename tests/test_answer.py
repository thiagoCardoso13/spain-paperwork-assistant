import json

import pytest

import rag.answer as ans
from rag.ingest import build_index
from rag import retrieve as rt
from tests.test_ingest import fake_embed


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class R:
            content = "Respuesta [1]"

        return R()


@pytest.fixture()
def index(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.json").write_text(
        json.dumps({"id": "a", "topic": "NIE", "title": "Fuente A", "url": "https://a.example", "fetched_at": "2026-01-01", "text": "Texto oficial sobre el NIE.\n" * 40}),
        encoding="utf-8",
    )
    idx = tmp_path / "idx"
    build_index(fake_embed, index_dir=idx, raw_dir=raw)
    rt._collection.cache_clear(); rt._lexical.cache_clear()
    monkeypatch.setattr(ans, "retrieve", lambda q, e, k=5: rt.retrieve(q, e, k=k, index_dir=idx))
    return idx


def test_answer_cites_sources_and_prompts_with_context(index):
    llm = FakeLLM()
    monkeypatch_distance = ans.MAX_DISTANCE
    ans.MAX_DISTANCE = 10  # fake embeddings are not semantic
    try:
        out = ans.answer("¿Qué es el NIE?", fake_embed, llm=llm)
    finally:
        ans.MAX_DISTANCE = monkeypatch_distance
    assert out.grounded and out.sources and out.sources[0].url == "https://a.example"
    assert "[1] Fuente A" in llm.prompts[0] and "¿Qué es el NIE?" in llm.prompts[0]


def test_no_relevant_context_means_no_llm_call(index):
    llm = FakeLLM()
    ans.MAX_DISTANCE = -1  # nothing can be close enough
    try:
        out = ans.answer("What is the best pizza in Valencia?", fake_embed, llm=llm)
    finally:
        ans.MAX_DISTANCE = 0.75
    assert not out.grounded and llm.prompts == []
    assert "could not find" in out.text


def test_language_guess():
    assert ans.guess_language("¿Qué documentos necesito?") == "es"
    assert ans.guess_language("Como pedir o cartão de saúde?") == "pt"
    assert ans.guess_language("What do I need?") == "en"


class Boom:
    def invoke(self, prompt):
        raise RuntimeError("503 UNAVAILABLE")


def test_fallback_llm_uses_next_model():
    out = ans.FallbackLLM([Boom(), FakeLLM()], pause=0).invoke("hola")
    assert out.content == "Respuesta [1]"


def test_all_models_down_gives_friendly_message_with_sources(index):
    ans.MAX_DISTANCE = 10
    try:
        out = ans.answer("¿Qué es el NIE?", fake_embed, llm=ans.FallbackLLM([Boom(), Boom()], pause=0))
    finally:
        ans.MAX_DISTANCE = 0.75
    assert not out.grounded and out.busy and out.sources and "saturado" in out.text


def test_cited_keeps_numbers_and_lists_nothing_when_uncited():
    srcs = [ans.Source(f"S{i}", f"https://s{i}", "T", "d", "x") for i in range(1, 4)]
    assert [n for n, _ in ans.cited("Dato [2] y otro [2], [9]", srcs)] == [2]
    assert ans.cited("No encuentro la respuesta", srcs) == []  # a "not found" answer must not list pages


def test_prompt_pins_the_answer_language():
    srcs = [ans.Source("S", "https://s", "T", "d", "texto")]
    assert "Write the answer in English" in ans.build_prompt("What do I need?", srcs)
    assert "Write the answer in Portuguese" in ans.build_prompt("Como pedir o cartão?", srcs)


def test_transient_error_is_retried_on_the_same_model():
    class Flaky:
        calls = 0

        def invoke(self, prompt):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("503 UNAVAILABLE")
            return FakeLLM().invoke(prompt)

    assert ans.FallbackLLM([Flaky()], pause=0).invoke("x").content == "Respuesta [1]"
    assert Flaky.calls == 2
