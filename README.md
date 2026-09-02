# RAG-Based Mutual Fund FAQ Assistant

[![CI](https://github.com/coding-0418/RAG_Chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/coding-0418/RAG_Chatbot/actions/workflows/ci.yml)

A **facts-only** FAQ chatbot for SBI Mutual Fund schemes accessible on **Kuvera**. It answers factual questions about expense ratio, exit load, minimum SIP, lock-in period, risk-o-meter, benchmark, KIM/SID, and Kuvera statement downloads — and **refuses** investment advice, comparisons, return predictions, and PII.

**Platform:** Kuvera  
**AMC:** SBI Mutual Fund  
**Schemes:** SBI Bluechip Fund, SBI Contra Fund, SBI Long Term Equity Fund (ELSS), SBI Magnum Midcap Fund, SBI Small Cap Fund

> **Disclaimer:** This assistant provides factual information from official mutual fund sources and does not provide investment advice, recommendations, return projections, or portfolio guidance.

---

## Architecture

```mermaid
flowchart TB
    subgraph Ingestion
        CSV[urls.csv] --> ING[ingest.py]
        PDF[data/*.pdf + *.md] --> ING
        ING --> DL[Download & Extract Text]
        DL --> CH[Chunk + Embed<br/>all-MiniLM-L6-v2]
        CH --> VS[(ChromaDB<br/>vectorstore/)]
    end

    subgraph Runtime
        UI[Streamlit app.py] --> GUARD{Guardrails}
        GUARD -->|PII| PRIV[Privacy Refusal]
        GUARD -->|Advice| ADV[Investment Advice Refusal]
        GUARD -->|Factual| RAG[rag.py]
        RAG --> RET[Top-K Retrieval + Score Threshold]
        RET --> VS
        RET --> LLM[Groq Llama 3.3 70B]
        LLM --> ANS[Grounded Answer + Multi-Source Citations]
        ANS --> UI
    end
```

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| Framework | LangChain |
| Config | python-dotenv |

---

## Project Structure

```
mf_faq_chatbot/
├── app.py                # Streamlit UI (sidebar, session metrics, multi-turn chat)
├── ingest.py             # URL/PDF ingestion pipeline
├── rag.py                # Retrieval + guardrails + Groq generation
├── prompts.py            # System prompts and fixed responses
├── requirements.txt
├── requirements-dev.txt  # + pytest, ruff
├── pyproject.toml        # pytest / ruff config
├── Dockerfile
├── .dockerignore
├── README.md
├── urls.csv              # 23 official source URLs
├── sample_qa.md          # Evaluation Q&A pairs
├── .env.example
├── .streamlit/
│   └── config.toml       # Theme
├── data/                 # Local reference documents
├── vectorstore/          # ChromaDB persistence (created by ingest)
├── tests/                # pytest suite (guardrails, citations, end-to-end RAG)
└── docs/
    └── source_list.md    # Full source catalog
```

---

## Setup

### 1. Clone / enter project

```bash
cd mf_faq_chatbot
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB). Ensure internet access.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
```

Obtain a free key at [https://console.groq.com/](https://console.groq.com/).

### 5. Run ingestion

```bash
python ingest.py
```

This will:

1. Read all URLs from `urls.csv`
2. Download and extract HTML/PDF content
3. Load local files from `data/`
4. Split into chunks and embed with `all-MiniLM-L6-v2`
5. Persist vectors to `vectorstore/`

Ingestion logs successes and failures per URL. Local reference files ensure baseline coverage even if some remote URLs are temporarily unavailable.

### 6. Launch Streamlit

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

The suite (`mf_faq_chatbot/tests/`) covers the investment-advice and PII guardrails, multi-source
citation deduplication, the retrieval score threshold, multi-turn history handling, and graceful
Groq-outage fallback — all with the embedding model, ChromaDB, and Groq LLM mocked out, so it runs
in well under a second with no network access or API key. `ruff check .` runs alongside it in CI
(`.github/workflows/ci.yml`) on every push and pull request.

---

## Deployment

A `Dockerfile` is provided for containerized deployment:

```bash
cd mf_faq_chatbot
docker build -t mf-faq-chatbot .

# Build the vector store once (persist it outside the container)
docker run --rm -v "$(pwd)/vectorstore:/app/vectorstore" --env-file .env mf-faq-chatbot python ingest.py

# Run the app against that vector store
docker run -p 8501:8501 -v "$(pwd)/vectorstore:/app/vectorstore" --env-file .env mf-faq-chatbot
```

For a managed environment (Kubernetes, ECS, Cloud Run, etc.), mount `vectorstore/` as a persistent
volume, inject `GROQ_API_KEY` via your platform's secrets manager rather than a plain `.env` file,
and run ingestion as a separate scheduled job so knowledge-base refreshes don't require redeploying
the app container.

---

## Usage

Ask factual questions such as:

- *What is the expense ratio of SBI Bluechip Fund?*
- *What is the lock-in period of SBI Long Term Equity Fund?*
- *How do I download my capital gains statement on Kuvera?*

Follow-up questions within the same session (e.g. *"What about the exit load?"* right after asking
about a specific fund) are resolved using the last couple of turns of conversation history.

The assistant will **not** answer:

- Investment recommendations
- Fund comparisons ("which is better")
- Buy/sell suggestions
- Return predictions
- Portfolio allocation advice

---

## Guardrails

| Trigger | Response |
|---------|----------|
| Investment advice patterns | Fixed SEBI-linked refusal message |
| PAN, Aadhaar, OTP, phone, email, account number | PII privacy warning |
| Missing context / weak retrieval match | "I could not find this information in the approved sources." |
| Groq API failure/timeout | Friendly "temporarily unavailable" message (logged, not a stack trace) |

Guardrails run **before** retrieval and generation — a blocked question never reaches the LLM.

Every factual answer is grounded in retrieved chunks and displayed with:

- The factual answer (max 3 sentences), generated with no free-text citation footer
- A **Citations** panel listing every distinct source URL actually used (deduplicated across
  retrieved chunks, not just the single top match) with its title and last-updated date
- A latency + source-chunk-count caption, for transparency into what was retrieved

Chunks whose retrieval distance exceeds `MAX_DISTANCE` (env-configurable, default `1.1`) are
dropped before generation rather than passed to the LLM as weak grounding — this is what backs the
"I could not find this information" response when nothing relevant enough was retrieved.

---

## Known Limitations

1. **Dynamic TER/NAV:** Expense ratios and NAV change frequently. Answers point to official TER/NAV pages; always verify latest figures on sbimf.com.
2. **Scheme renaming:** SBI Bluechip Fund is now SBI Large Cap Fund; ELSS is branded SBI ELSS Tax Saver Fund. The assistant uses official current names from sources.
3. **Web ingestion dependency:** `ingest.py` requires network access to fetch URLs. Some PDFs are large and may take time.
4. **JavaScript-heavy pages:** Scheme detail pages on sbimf.com may render minimal static HTML; SID/KIM PDFs and local reference files compensate.
5. **No live account integration:** Kuvera answers describe platform steps only; the bot cannot access user accounts.
6. **LLM variability:** Groq responses are temperature-0 but phrasing may differ slightly run-to-run while preserving facts.
7. **Full re-index on ingest:** `ingest.py` rebuilds the entire vector store rather than diffing changes — fine for a periodic scheduled job, but there's no incremental/delta ingestion yet.
8. **No authentication:** the Streamlit app has no login/SSO layer. A lightweight per-session rate limit is built in (throttles rapid repeat questions), but a shared-network deployment should sit behind SSO or a reverse-proxy auth layer before going further than a demo.
9. **Regex-based guardrails:** investment-advice and PII detection are pattern-based, not an LLM classifier — fast and dependency-free, but rephrasing can evade them. Treat as a first line of defense, not a compliance guarantee.

---

## Evaluation Notes

Automated coverage lives in `mf_faq_chatbot/tests/` (run via `pytest`, see [Testing](#testing)) for
the guardrails, citation logic, and RAG response shape — those run on every push via CI.

`sample_qa.md` remains the manual, content-level test set (does the LLM actually get the *facts*
right against live sources) since that requires a real Groq call and live-sourced context:

- Expense ratio, exit load, minimum SIP, benchmark, risk-o-meter
- ELSS 3-year lock-in
- Kuvera statement and capital gains downloads
- KIM/SID explanation
- Investment advice refusal
- PII detection

Suggested manual evaluation checklist:

- [ ] Factual answers cite at least one approved source URL (multiple when multiple sources
      genuinely contributed)
- [ ] Advice questions return exact refusal text
- [ ] PII in input triggers privacy message
- [ ] Out-of-scope / weakly-matched questions return the not-found message
- [ ] Answers stay within 3 factual sentences
- [ ] Follow-up questions in the same session resolve pronouns/references correctly
- [ ] UI shows disclaimer, examples, citations panel, and latency/source-count caption

---

## License

Educational / assignment submission project. Official fund data belongs to respective AMCs, SEBI, AMFI, and Kuvera.
