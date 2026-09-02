from langchain_core.documents import Document

from rag import build_citations


def _doc(source: str, title: str = "Title", last_updated: str = "2026-01-01") -> Document:
    return Document(
        page_content="content",
        metadata={"source": source, "title": title, "last_updated": last_updated},
    )


class TestBuildCitations:
    def test_dedupes_same_source(self):
        docs = [
            _doc("https://a.example/x"),
            _doc("https://a.example/x"),
            _doc("https://b.example/y"),
        ]
        citations = build_citations(docs)
        sources = [c.source for c in citations]
        assert sources == ["https://a.example/x", "https://b.example/y"]

    def test_preserves_retrieval_order(self):
        docs = [_doc("https://c.example"), _doc("https://a.example"), _doc("https://b.example")]
        citations = build_citations(docs)
        assert [c.source for c in citations] == [
            "https://c.example",
            "https://a.example",
            "https://b.example",
        ]

    def test_empty_input_returns_empty_list(self):
        assert build_citations([]) == []

    def test_carries_title_and_last_updated(self):
        docs = [_doc("https://a.example", title="Factsheet", last_updated="2026-03-15")]
        citations = build_citations(docs)
        assert citations[0].title == "Factsheet"
        assert citations[0].last_updated == "2026-03-15"
