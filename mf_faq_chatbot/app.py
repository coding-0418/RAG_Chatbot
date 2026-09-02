"""
Streamlit UI for the Mutual Fund FAQ Assistant.
"""

from __future__ import annotations

import time

import streamlit as st

from prompts import DISCLAIMER
from rag import MutualFundRAG, Turn

st.set_page_config(
    page_title="Mutual Fund FAQ Assistant",
    page_icon="📊",
    layout="wide",
)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of SBI Bluechip Fund?",
    "What is the lock-in period of SBI Long Term Equity Fund?",
    "How do I download my capital gains statement on Kuvera?",
]

SCHEMES = [
    "SBI Bluechip Fund",
    "SBI Contra Fund",
    "SBI Long Term Equity Fund (ELSS)",
    "SBI Magnum Midcap Fund",
    "SBI Small Cap Fund",
]

MIN_QUERY_INTERVAL_SECONDS = 3
MAX_HISTORY_MESSAGES = 6  # 3 user/assistant turns


@st.cache_resource(show_spinner="Loading knowledge base and LLM...")
def load_rag() -> MutualFundRAG:
    return MutualFundRAG()


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("total_questions", 0)
    st.session_state.setdefault("blocked_count", 0)
    st.session_state.setdefault("last_query_time", 0.0)
    st.session_state.setdefault("rate_limited", False)


def build_history() -> list[Turn]:
    """Turn the last few clean (non-error, non-blocked) exchanges into
    Turn objects the RAG pipeline can use to resolve follow-up questions."""
    turns: list[Turn] = []
    messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    for i in range(0, len(messages) - 1, 2):
        user_msg, assistant_msg = messages[i], messages[i + 1]
        if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
            continue
        if assistant_msg.get("blocked") or assistant_msg.get("error"):
            continue
        turns.append(Turn(question=user_msg["content"], answer=assistant_msg["content"]))
    return turns


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("blocked"):
            st.warning(message["content"])
        elif message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

        citations = message.get("citations") or []
        if citations:
            with st.expander(f"Citations ({len(citations)})"):
                for c in citations:
                    label = c["title"] or c["source"]
                    if c["source"].startswith("http"):
                        st.markdown(f"- [{label}]({c['source']}) — updated {c['last_updated']}")
                    else:
                        st.markdown(f"- {label} — updated {c['last_updated']}")

        meta_bits = []
        if message.get("latency") is not None:
            meta_bits.append(f"{message['latency']:.1f}s")
        if message.get("chunks_used"):
            meta_bits.append(f"{message['chunks_used']} source chunk(s)")
        if meta_bits:
            st.caption(" · ".join(meta_bits))


def process_question(question: str) -> None:
    now = time.monotonic()
    if now - st.session_state.last_query_time < MIN_QUERY_INTERVAL_SECONDS:
        st.session_state.rate_limited = True
        st.toast("Please wait a couple of seconds between questions.", icon="⏳")
        return
    st.session_state.last_query_time = now
    st.session_state.rate_limited = False

    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        with st.spinner("Searching approved sources..."):
            rag = load_rag()
            history = build_history()
            response = rag.answer(question, history=history)

        if response.blocked:
            st.warning(response.answer)
        elif response.error:
            st.error(response.answer)
        else:
            st.markdown(response.answer)

        if response.citations:
            with st.expander(f"Citations ({len(response.citations)})"):
                for c in response.citations:
                    label = c.title or c.source
                    if c.source.startswith("http"):
                        st.markdown(f"- [{label}]({c.source}) — updated {c.last_updated}")
                    else:
                        st.markdown(f"- {label} — updated {c.last_updated}")

        meta_bits = [f"{response.latency_seconds:.1f}s"]
        if response.chunks_used:
            meta_bits.append(f"{response.chunks_used} source chunk(s)")
        st.caption(" · ".join(meta_bits))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "citations": [
                {"source": c.source, "title": c.title, "last_updated": c.last_updated}
                for c in response.citations
            ],
            "last_updated": response.last_updated,
            "blocked": response.blocked,
            "error": response.error,
            "latency": response.latency_seconds,
            "chunks_used": response.chunks_used,
        }
    )
    st.session_state.total_questions += 1
    if response.blocked:
        st.session_state.blocked_count += 1


def render_sidebar() -> None:
    with st.sidebar:
        st.header("📊 Assistant Info")
        st.caption("Facts-only FAQ assistant — no investment advice.")

        st.subheader("Coverage")
        st.markdown("**Platform:** Kuvera  \n**AMC:** SBI Mutual Fund")
        for scheme in SCHEMES:
            st.markdown(f"- {scheme}")

        st.subheader("Guardrails")
        st.markdown("✅ Investment-advice refusal\n\n✅ PII / privacy filter\n\n✅ Source-grounded answers only")

        try:
            rag = load_rag()
            st.caption(f"Knowledge base: {rag.collection_size()} indexed chunks")
        except Exception:
            pass

        st.subheader("Session metrics")
        col1, col2 = st.columns(2)
        col1.metric("Questions", st.session_state.total_questions)
        col2.metric("Blocked", st.session_state.blocked_count)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.session_state.blocked_count = 0
            st.rerun()


def main() -> None:
    init_session_state()

    st.title("Mutual Fund FAQ Assistant")
    st.caption("Facts-only. No investment advice.")
    st.info(
        "Welcome! Ask factual questions about SBI Bluechip, Contra, Long Term Equity (ELSS), "
        "Magnum Midcap, and Small Cap funds, or about Kuvera statements and tax reports."
    )

    try:
        load_rag()
    except (OSError, FileNotFoundError) as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to initialize the assistant: {exc}")
        st.stop()

    if not st.session_state.messages:
        st.markdown("**Try an example question:**")
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, example in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(example, use_container_width=True):
                process_question(example)
                st.rerun()

    st.divider()

    for message in st.session_state.messages:
        render_message(message)

    if st.session_state.rate_limited:
        st.warning("You're asking questions a bit fast — please wait a couple of seconds and try again.")

    user_input = st.chat_input("Ask a factual mutual fund question...")
    if user_input:
        process_question(user_input)

    render_sidebar()

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
