"""
RAG pipeline: retrieval, guardrails (refusal/privacy), and Groq LLM generation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
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
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.getenv("TOP_K", "5"))


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]
    last_updated: str
    blocked: bool = False
    block_reason: str | None = None


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


def _format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "")
        last_updated = doc.metadata.get("last_updated", "unknown")
        blocks.append(
            f"[Chunk {idx}]\nTitle: {title}\nSource: {source}\nLast updated: {last_updated}\nContent:\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def _extract_source_and_date(text: str, fallback_docs: list[Document]) -> tuple[list[str], str]:
    source_match = re.search(r"Source:\s*(.+)", text, re.IGNORECASE)
    date_match = re.search(r"Last updated from sources:\s*(.+)", text, re.IGNORECASE)

    sources: list[str] = []
    if source_match:
        sources = [source_match.group(1).strip()]
    elif fallback_docs:
        sources = [fallback_docs[0].metadata.get("source", "unknown")]

    if date_match:
        last_updated = date_match.group(1).strip()
    elif fallback_docs:
        last_updated = fallback_docs[0].metadata.get("last_updated", "unknown")
    else:
        last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return sources, last_updated


def _strip_source_footer(text: str) -> str:
    lines = text.strip().splitlines()
    answer_lines: list[str] = []
    for line in lines:
        if re.match(r"^\s*Source:\s*", line, re.IGNORECASE):
            break
        if re.match(r"^\s*Last updated from sources:\s*", line, re.IGNORECASE):
            break
        answer_lines.append(line)
    return " ".join(" ".join(answer_lines).split())


class MutualFundRAG:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
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

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        return self.vectorstore.similarity_search(question, k=k or TOP_K)

    def answer(self, question: str) -> RAGResponse:
        question = question.strip()
        if not question:
            return RAGResponse(
                answer="Please enter a factual question about the covered SBI Mutual Fund schemes or Kuvera platform.",
                sources=[],
                last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            )

        if contains_pii(question):
            return RAGResponse(
                answer=PRIVACY_REFUSAL,
                sources=[],
                last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                blocked=True,
                block_reason="privacy",
            )

        if is_investment_advice_query(question):
            return RAGResponse(
                answer=INVESTMENT_ADVICE_REFUSAL,
                sources=["https://www.sebi.gov.in/"],
                last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                blocked=True,
                block_reason="investment_advice",
            )

        docs = self.retrieve(question)
        if not docs:
            return RAGResponse(
                answer=NOT_FOUND_RESPONSE,
                sources=[],
                last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            )

        context = _format_context(docs)
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        llm_response = self.llm.invoke(messages)
        raw_answer = getattr(llm_response, "content", str(llm_response)).strip()

        if NOT_FOUND_RESPONSE.lower() in raw_answer.lower():
            return RAGResponse(
                answer=NOT_FOUND_RESPONSE,
                sources=[d.metadata.get("source", "unknown") for d in docs[:1]],
                last_updated=docs[0].metadata.get("last_updated", "unknown"),
            )

        sources, last_updated = _extract_source_and_date(raw_answer, docs)
        factual = _strip_source_footer(raw_answer)

        formatted_answer = (
            f"{factual}\n\nSource: {sources[0] if sources else 'unknown'}\n"
            f"Last updated from sources: {last_updated}"
        )

        return RAGResponse(
            answer=formatted_answer,
            sources=sources,
            last_updated=last_updated,
        )
