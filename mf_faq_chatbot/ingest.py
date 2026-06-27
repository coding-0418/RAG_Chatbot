"""
Ingestion pipeline: downloads URLs, reads local PDFs, chunks, embeds, and stores in ChromaDB.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
URLS_CSV = PROJECT_ROOT / "urls.csv"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

REQUEST_TIMEOUT = 45
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MFFAQBot/1.0; +https://github.com/educational-rag-bot)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
}


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or "pdf" in path


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_html_text(html: bytes, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    raw = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    lines = [_clean_text(line) for line in raw.splitlines() if _clean_text(line)]
    return "\n".join(lines)


def _extract_pdf_bytes(content: bytes, source_url: str) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        cleaned = _clean_text(page_text)
        if cleaned:
            pages.append(cleaned)
    if not pages:
        raise ValueError(f"No extractable text in PDF: {source_url}")
    return "\n".join(pages)


def _fetch_url(url: str) -> tuple[str, str]:
    """Return (text, content_type)."""
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()

    if _is_pdf_url(url) or "pdf" in content_type:
        text = _extract_pdf_bytes(response.content, url)
        return text, "application/pdf"

    response.encoding = response.encoding or "utf-8"
    text = _extract_html_text(response.content, url)
    if len(text) < 80:
        raise ValueError(f"Insufficient text extracted from {url}")
    return text, "text/html"


def _load_local_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        cleaned = _clean_text(page_text)
        if cleaned:
            pages.append(cleaned)
    combined = "\n".join(pages)
    if not combined.strip():
        raise ValueError(f"No text extracted from local PDF: {path}")
    return combined


def _load_local_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty local text file: {path}")
    return text


def _read_urls_csv() -> list[dict[str, str]]:
    if not URLS_CSV.exists():
        raise FileNotFoundError(f"Missing URLs file: {URLS_CSV}")

    rows: list[dict[str, str]] = []
    with URLS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("url") or "").strip()
            if url:
                rows.append(
                    {
                        "url": url,
                        "title": (row.get("title") or url).strip(),
                        "category": (row.get("category") or "General").strip(),
                    }
                )
    return rows


def _build_document(
    text: str,
    source_url: str,
    title: str,
    category: str,
    source_type: str,
) -> Document:
    metadata = {
        "source": source_url,
        "title": title,
        "category": category,
        "source_type": source_type,
        "last_updated": _today_iso(),
    }
    return Document(page_content=text, metadata=metadata)


def load_from_urls() -> list[Document]:
    documents: list[Document] = []
    for row in _read_urls_csv():
        url = row["url"]
        try:
            logger.info("Fetching %s", url)
            text, content_type = _fetch_url(url)
            source_type = "pdf" if "pdf" in content_type else "webpage"
            documents.append(
                _build_document(text, url, row["title"], row["category"], source_type)
            )
            logger.info("Successfully ingested %s (%d chars)", url, len(text))
        except Exception as exc:
            logger.error("Failed to ingest URL %s: %s", url, exc)
    return documents


def load_from_local_data() -> list[Document]:
    documents: list[Document] = []
    if not DATA_DIR.exists():
        return documents

    for path in sorted(DATA_DIR.iterdir()):
        if path.name.startswith("."):
            continue
        try:
            if path.suffix.lower() == ".pdf":
                logger.info("Loading local PDF %s", path.name)
                text = _load_local_pdf(path)
                source_type = "local_pdf"
            elif path.suffix.lower() in {".md", ".txt"}:
                logger.info("Loading local text %s", path.name)
                text = _load_local_text(path)
                source_type = "local_text"
            else:
                continue

            source_url = f"file://{path.name}"
            documents.append(
                _build_document(
                    text=text,
                    source_url=source_url,
                    title=path.stem.replace("_", " ").title(),
                    category="Local Reference",
                    source_type=source_type,
                )
            )
            logger.info("Successfully loaded local file %s", path.name)
        except Exception as exc:
            logger.error("Failed to load local file %s: %s", path.name, exc)
    return documents


def split_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def build_vectorstore(chunks: list[Document]) -> Chroma:
    if not chunks:
        raise RuntimeError(
            "No documents were ingested. Check network access, urls.csv, and data/ folder."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "convert_to_numpy": True},
    )
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if any(CHROMA_DIR.iterdir()):
        logger.info("Removing existing vector store at %s", CHROMA_DIR)
        for item in CHROMA_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil

                shutil.rmtree(item)

    logger.info("Creating ChromaDB with %d chunks", len(chunks))
    batch_size = 128
    vectorstore: Chroma | None = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        logger.info("Embedding batch %d-%d of %d", start + 1, start + len(batch), len(chunks))
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
                collection_name="mf_faq",
            )
        else:
            vectorstore.add_documents(batch)

    if vectorstore is None:
        raise RuntimeError("Failed to create vector store.")
    return vectorstore


def main() -> None:
    logger.info("Starting ingestion pipeline")
    url_docs = load_from_urls()
    local_docs = load_from_local_data()
    all_docs = url_docs + local_docs

    logger.info("Loaded %d source documents (%d URLs, %d local)", len(all_docs), len(url_docs), len(local_docs))
    chunks = split_documents(all_docs)
    logger.info("Split into %d chunks", len(chunks))

    build_vectorstore(chunks)
    logger.info("Ingestion complete. Vector store saved to %s", CHROMA_DIR)


if __name__ == "__main__":
    main()
