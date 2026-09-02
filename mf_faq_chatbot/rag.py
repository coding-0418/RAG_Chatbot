"""
RAG pipeline: retrieval, guardrails (refusal/privacy), and Groq LLM generation.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from prompts import (
    INVESTMENT_ADVICE_REFUSAL,
    NOT_FOUND_RESPONSE,
    PRIVACY_REFUSAL,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    SERVICE_ERROR_RESPONSE,
)

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.getenv("TOP_K", "5"))
# Chroma returns L2 distance over normalized embeddings (0 = identical, 2 = opposite).
# Chunks farther than this are dropped as "not relevant enough" rather than sent to the LLM.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.1"))
# How many prior user/assistant turns to include so the LLM can resolve follow-ups
# like "what about the exit load?". Kept small to bound prompt size and cost.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "2"))

# Cross-cutting sources (regulator/platform docs) aren't a fund house and are
# excluded from the AMC picker — mirrors ingest.py's CROSS_CUTTING_AMC_TAGS.
CROSS_CUTTING_AMC_TAGS = {"Regulatory", "Platform", "General"}


@dataclass
class Citation:
    source: str
    title: str
    last_updated: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    last_updated: str = ""
    blocked: bool = False
    block_reason: str | None = None
    error: bool = False
    latency_seconds: float = 0.0
    chunks_used: int = 0


@dataclass
class Turn:
    """One prior user/assistant exchange, used only to resolve follow-up references."""

    question: str
    answer: str


INVESTMENT_ADVICE_PATTERNS = [
    r"\bshould\s+i\s+invest\b",
    r"\bwhich\s+fund\s+is\s+better\b",
    r"\bwhich\s+.*\s+fund\s+.*\b(better|best)\b",
    r"\bbuy\s+or\s+sell\b",
    r"\bshould\s+i\s+(buy|sell)\b",
    r"\bportfolio\s+allocation\b",
    r"\bhow\s+much\s+should\s+i\s+invest\b",
    r"\bpredict\b.*\breturn",
    r"\breturn\s+prediction\b",
    r"\bbest\s+mutual\s+fund\b",
    r"\brecommend\b.*\b(fund|invest)",
    r"\bcompare\b.*\bfunds?\b",
    r"\bwhere\s+should\s+i\s+invest\b",
    r"\bwhich\s+fund\s+should\s+i\b",
]

PRIVACY_PATTERNS = [
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    r"\b[A-Z]{5}\d{4}[A-Z]\b",
    r"\b\d{10}\b",
    r"\b\+91[\s-]?\d{10}\b",
    r"\b\d{6}\b.*\b(?:otp|one[\s-]?time)\b",
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    r"\b(?:account|a/c)\s*(?:no\.?|number)?\s*[:#]?\s*\d{6,}\b",
    r"\b(?:pan|aadhaar|aadhar)\s*(?:no\.?|number)?\s*[:#]?\s*[\dA-Z]{4,}\b",
]


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_INVESTMENT_PATTERNS = _compile_patterns(INVESTMENT_ADVICE_PATTERNS)
_PRIVACY_PATTERNS = _compile_patterns(PRIVACY_PATTERNS)


def is_investment_advice_query(question: str) -> bool:
    return any(p.search(question) for p in _INVESTMENT_PATTERNS)


def contains_pii(question: str) -> bool:
    return any(p.search(question) for p in _PRIVACY_PATTERNS)


def _today_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "")
        amc = doc.metadata.get("amc", "General")
        last_updated = doc.metadata.get("last_updated", "unknown")
        blocks.append(
            f"[Chunk {idx}]\nAMC: {amc}\nTitle: {title}\nSource: {source}\nLast updated: {last_updated}\nContent:\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def build_citations(docs: list[Document]) -> list[Citation]:
    """Deduplicate retrieved chunks into one citation per distinct source URL,
    preserving retrieval order (most relevant first)."""
    seen: set[str] = set()
    citations: list[Citation] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source in seen:
            continue
        seen.add(source)
        citations.append(
            Citation(
                source=source,
                title=doc.metadata.get("title", ""),
                last_updated=doc.metadata.get("last_updated", "unknown"),
            )
        )
    return citations


def _format_history(history: list[Turn]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in history[-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": turn.answer})
    return messages


def _retrieval_query(question: str, history: list[Turn]) -> str:
    """Expand short/pronoun-heavy follow-ups with the previous question so
    retrieval has enough signal, without a separate LLM rewrite call."""
    if not history:
        return question
    return f"{history[-1].question} {question}"


class MutualFundRAG:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise OSError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at {CHROMA_DIR}. Run `python ingest.py` first."
            )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "convert_to_numpy": True},
        )
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self.embeddings,
            collection_name="mf_faq",
        )
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            groq_api_key=api_key,
        )

    def collection_size(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def list_amcs(self) -> list[str]:
        """Distinct fund houses present in the knowledge base, for a UI picker.
        Excludes cross-cutting regulator/platform sources, which aren't a fund house."""
        try:
            result = self.vectorstore._collection.get(include=["metadatas"])
        except Exception:
            return []
        amcs = {m.get("amc") for m in (result.get("metadatas") or []) if m and m.get("amc")}
        return sorted(a for a in amcs if a not in CROSS_CUTTING_AMC_TAGS)

    def retrieve(self, question: str, k: int | None = None, amc: str | None = None) -> list[Document]:
        """Retrieve top-k chunks, dropping any farther than MAX_DISTANCE
        (i.e. not relevant enough to be trustworthy grounding). When `amc` is
        given, only chunks tagged with that fund house are considered."""
        filter_ = {"amc": amc} if amc else None
        results = self.vectorstore.similarity_search_with_score(question, k=k or TOP_K, filter=filter_)
        kept = [doc for doc, distance in results if distance <= MAX_DISTANCE]
        if not kept and results:
            logger.info(
                "All %d retrieved chunks exceeded MAX_DISTANCE=%.2f (best=%.3f)",
                len(results),
                MAX_DISTANCE,
                min(d for _, d in results),
            )
        return kept

    def answer(
        self,
        question: str,
        history: list[Turn] | None = None,
        amc: str | None = None,
    ) -> RAGResponse:
        start = time.monotonic()
        history = history or []
        question = question.strip()

        if not question:
            return RAGResponse(
                answer="Please enter a factual question about a fund house or scheme covered in the knowledge base.",
                last_updated=_today_iso(),
            )

        if contains_pii(question):
            logger.info("Blocked query: privacy")
            return RAGResponse(
                answer=PRIVACY_REFUSAL,
                last_updated=_today_iso(),
                blocked=True,
                block_reason="privacy",
                latency_seconds=time.monotonic() - start,
            )

        if is_investment_advice_query(question):
            logger.info("Blocked query: investment_advice")
            return RAGResponse(
                answer=INVESTMENT_ADVICE_REFUSAL,
                sources=["https://www.sebi.gov.in/"],
                citations=[
                    Citation(
                        source="https://www.sebi.gov.in/",
                        title="SEBI",
                        last_updated=_today_iso(),
                    )
                ],
                last_updated=_today_iso(),
                blocked=True,
                block_reason="investment_advice",
                latency_seconds=time.monotonic() - start,
            )

        docs = self.retrieve(_retrieval_query(question, history), amc=amc)
        if not docs:
            logger.info("No sufficiently relevant chunks found")
            return RAGResponse(
                answer=NOT_FOUND_RESPONSE,
                last_updated=_today_iso(),
                latency_seconds=time.monotonic() - start,
            )

        context = _format_context(docs)
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)

        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        messages.extend(_format_history(history))
        messages.append({"role": "user", "content": user_prompt})

        try:
            llm_response = self.llm.invoke(messages)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return RAGResponse(
                answer=SERVICE_ERROR_RESPONSE,
                last_updated=_today_iso(),
                error=True,
                latency_seconds=time.monotonic() - start,
            )

        raw_answer = getattr(llm_response, "content", str(llm_response)).strip()
        usage = getattr(llm_response, "usage_metadata", None) or {}
        if usage:
            logger.info(
                "Groq tokens — input=%s output=%s total=%s",
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                usage.get("total_tokens"),
            )

        if NOT_FOUND_RESPONSE.lower() in raw_answer.lower():
            return RAGResponse(
                answer=NOT_FOUND_RESPONSE,
                last_updated=_today_iso(),
                latency_seconds=time.monotonic() - start,
                chunks_used=len(docs),
            )

        citations = build_citations(docs)
        return RAGResponse(
            answer=raw_answer,
            sources=[c.source for c in citations],
            citations=citations,
            last_updated=citations[0].last_updated if citations else "unknown",
            latency_seconds=time.monotonic() - start,
            chunks_used=len(docs),
        )
