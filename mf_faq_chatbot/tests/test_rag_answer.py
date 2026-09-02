"""End-to-end tests for MutualFundRAG.answer(), with the embedding model,
Chroma vectorstore, and Groq LLM fully mocked out so the suite runs fast and
offline in CI (no model downloads, no network, no API key required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import rag as rag_module
from prompts import NOT_FOUND_RESPONSE, SERVICE_ERROR_RESPONSE
from rag import MutualFundRAG, Turn


class FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


class FakeChroma:
    """Class-level `results` lets each test configure retrieval output before
    instantiating MutualFundRAG (which constructs this class internally)."""

    results: list[tuple[Document, float]] = []

    def __init__(self, *args, **kwargs):
        self._collection = SimpleNamespace(count=lambda: 123)

    def similarity_search_with_score(self, query, k=5):
        return FakeChroma.results[:k]


class FakeChatGroq:
    reply_content = "This is a grounded factual answer."
    should_raise = False
    last_messages: list[dict] = []
    call_count = 0

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, messages):
        FakeChatGroq.call_count += 1
        FakeChatGroq.last_messages = messages
        if FakeChatGroq.should_raise:
            raise RuntimeError("simulated Groq outage")
        return SimpleNamespace(
            content=FakeChatGroq.reply_content,
            usage_metadata={"input_tokens": 42, "output_tokens": 8, "total_tokens": 50},
        )


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeChroma.results = []
    FakeChatGroq.reply_content = "This is a grounded factual answer."
    FakeChatGroq.should_raise = False
    FakeChatGroq.last_messages = []
    FakeChatGroq.call_count = 0
    yield


@pytest.fixture
def rag(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag_module, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(rag_module, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(rag_module, "Chroma", FakeChroma)
    monkeypatch.setattr(rag_module, "ChatGroq", FakeChatGroq)
    return MutualFundRAG()


def _doc(source: str, title: str, content: str = "chunk content") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "title": title, "last_updated": "2026-02-01"},
    )


class TestGuardrailsShortCircuitBeforeLLM:
    def test_pii_query_is_blocked_without_calling_llm(self, rag):
        response = rag.answer("My PAN is ABCDE1234F, what is the expense ratio?")
        assert response.blocked is True
        assert response.block_reason == "privacy"
        assert FakeChatGroq.call_count == 0

    def test_advice_query_is_blocked_without_calling_llm(self, rag):
        response = rag.answer("Which fund is better, Bluechip or Contra?")
        assert response.blocked is True
        assert response.block_reason == "investment_advice"
        assert response.sources == ["https://www.sebi.gov.in/"]
        assert FakeChatGroq.call_count == 0

    def test_empty_question_short_circuits(self, rag):
        response = rag.answer("   ")
        assert not response.blocked
        assert FakeChatGroq.call_count == 0


class TestRetrievalThreshold:
    def test_no_docs_returns_not_found_without_calling_llm(self, rag):
        FakeChroma.results = []
        response = rag.answer("What is the expense ratio of SBI Bluechip Fund?")
        assert response.answer == NOT_FOUND_RESPONSE
        assert FakeChatGroq.call_count == 0

    def test_only_far_docs_are_treated_as_not_found(self, rag):
        far_doc = _doc("https://sbimf.com/far", "Far")
        FakeChroma.results = [(far_doc, rag_module.MAX_DISTANCE + 1.0)]
        response = rag.answer("What is the expense ratio of SBI Bluechip Fund?")
        assert response.answer == NOT_FOUND_RESPONSE
        assert FakeChatGroq.call_count == 0

    def test_close_docs_are_used(self, rag):
        close_doc = _doc("https://sbimf.com/close", "Close")
        FakeChroma.results = [(close_doc, 0.1)]
        response = rag.answer("What is the expense ratio of SBI Bluechip Fund?")
        assert response.answer == FakeChatGroq.reply_content
        assert FakeChatGroq.call_count == 1


class TestMultiSourceCitations:
    def test_dedupes_and_orders_citations(self, rag):
        FakeChroma.results = [
            (_doc("https://sbimf.com/a", "Factsheet A"), 0.1),
            (_doc("https://sbimf.com/a", "Factsheet A"), 0.15),
            (_doc("https://sbimf.com/b", "Factsheet B"), 0.2),
        ]
        response = rag.answer("What is the exit load of SBI Small Cap Fund?")
        assert [c.source for c in response.citations] == [
            "https://sbimf.com/a",
            "https://sbimf.com/b",
        ]
        assert response.sources == ["https://sbimf.com/a", "https://sbimf.com/b"]
        assert response.chunks_used == 3

    def test_answer_has_no_embedded_source_footer(self, rag):
        FakeChatGroq.reply_content = "The expense ratio is disclosed on the TER page."
        FakeChroma.results = [(_doc("https://sbimf.com/a", "TER"), 0.1)]
        response = rag.answer("What is the expense ratio?")
        assert "Source:" not in response.answer


class TestLLMFailureHandling:
    def test_llm_exception_returns_friendly_error(self, rag):
        FakeChatGroq.should_raise = True
        FakeChroma.results = [(_doc("https://sbimf.com/a", "Doc"), 0.1)]
        response = rag.answer("What is the expense ratio of SBI Bluechip Fund?")
        assert response.error is True
        assert response.answer == SERVICE_ERROR_RESPONSE


class TestMultiTurnHistory:
    def test_history_is_forwarded_to_llm_messages(self, rag):
        FakeChroma.results = [(_doc("https://sbimf.com/a", "Doc"), 0.1)]
        history = [Turn(question="What is the exit load of SBI Contra Fund?", answer="It is 1%.")]

        rag.answer("What about the minimum SIP?", history=history)

        roles_and_content = [(m["role"], m["content"]) for m in FakeChatGroq.last_messages]
        assert ("user", "What is the exit load of SBI Contra Fund?") in roles_and_content
        assert ("assistant", "It is 1%.") in roles_and_content

    def test_no_history_still_works(self, rag):
        FakeChroma.results = [(_doc("https://sbimf.com/a", "Doc"), 0.1)]
        response = rag.answer("What is the expense ratio of SBI Bluechip Fund?")
        assert response.answer == FakeChatGroq.reply_content
