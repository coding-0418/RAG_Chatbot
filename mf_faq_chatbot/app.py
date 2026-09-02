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
    ("💰", "What is the expense ratio of SBI Bluechip Fund?"),
    ("🔒", "What is the lock-in period of SBI Long Term Equity Fund?"),
    ("📄", "How do I download my capital gains statement on Kuvera?"),
]

SCHEMES = [
    "SBI Bluechip Fund",
    "SBI Contra Fund",
    "SBI Long Term Equity Fund (ELSS)",
    "SBI Magnum Midcap Fund",
    "SBI Small Cap Fund",
]

USER_AVATAR = "🧑‍💼"
ASSISTANT_AVATAR = "📊"

MIN_QUERY_INTERVAL_SECONDS = 3
MAX_HISTORY_MESSAGES = 6  # 3 user/assistant turns


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1100px;
        }

        /* Hero header */
        .mf-hero {
            background: linear-gradient(135deg, #0B5FFF 0%, #2E8BFF 100%);
            border-radius: 16px;
            padding: 1.75rem 2rem;
            color: #FFFFFF;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px rgba(11, 95, 255, 0.18);
        }
        .mf-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.9rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .mf-hero p {
            margin: 0;
            opacity: 0.92;
            font-size: 0.98rem;
        }
        .mf-hero-badges {
            margin-top: 0.9rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .mf-pill {
            display: inline-block;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 500;
        }

        /* Example question cards (buttons) */
        div[data-testid="column"] div[data-testid="stButton"] button {
            border-radius: 12px;
            border: 1px solid rgba(11, 95, 255, 0.25);
            background: #F7FAFF;
            color: #1A1D23;
            text-align: left;
            padding: 0.85rem 1rem;
            height: 100%;
            white-space: normal;
            transition: all 0.15s ease-in-out;
        }
        div[data-testid="column"] div[data-testid="stButton"] button:hover {
            border-color: #0B5FFF;
            background: #EAF1FF;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(11, 95, 255, 0.12);
        }

        /* Citation chips */
        .citation-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.5rem 0 0.25rem 0;
        }
        .citation-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: #F2F5FA;
            border: 1px solid #DCE4F0;
            border-radius: 999px;
            padding: 0.28rem 0.75rem;
            font-size: 0.8rem;
            color: #1A1D23;
            text-decoration: none;
        }
        .citation-chip:hover {
            border-color: #0B5FFF;
            background: #EAF1FF;
        }
        .citation-chip .dot {
            color: #8A93A3;
        }

        .mf-meta-caption {
            color: #8A93A3;
            font-size: 0.78rem;
            margin-top: 0.15rem;
        }

        /* Sidebar cards */
        section[data-testid="stSidebar"] .stContainer,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
        }

        /* Footer badge row */
        .mf-footer-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.4rem;
            margin-bottom: 0.6rem;
        }
        .mf-footer-badge {
            display: inline-block;
            background: #F2F5FA;
            border: 1px solid #E3E8F0;
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            font-size: 0.76rem;
            color: #4A5468;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="mf-hero">
            <h1>📊 Mutual Fund FAQ Assistant</h1>
            <p>Facts-only answers about SBI Mutual Fund schemes on Kuvera — grounded in
            official sources, with no investment advice.</p>
            <div class="mf-hero-badges">
                <span class="mf-pill">✅ Guardrails active</span>
                <span class="mf-pill">📚 Source-cited answers</span>
                <span class="mf-pill">🔒 No PII collected</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _avatar_for(role: str) -> str:
    return USER_AVATAR if role == "user" else ASSISTANT_AVATAR


def _render_citation_chips(citations: list[dict]) -> None:
    if not citations:
        return
    chips = []
    for c in citations:
        label = c["title"] or c["source"]
        updated = c.get("last_updated", "unknown")
        chips.append(f'<span class="citation-chip">📄 {label} <span class="dot">·</span> {updated}</span>')
    st.markdown(f'<div class="citation-wrap">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"], avatar=_avatar_for(message["role"])):
        if message.get("blocked"):
            st.warning(message["content"])
        elif message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

        _render_citation_chips(message.get("citations") or [])

        meta_bits = []
        if message.get("latency") is not None:
            meta_bits.append(f"⏱️ {message['latency']:.1f}s")
        if message.get("chunks_used"):
            meta_bits.append(f"📚 {message['chunks_used']} source chunk(s)")
        if meta_bits:
            st.markdown(f'<div class="mf-meta-caption">{" &nbsp;·&nbsp; ".join(meta_bits)}</div>', unsafe_allow_html=True)


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

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
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

        citation_dicts = [
            {"source": c.source, "title": c.title, "last_updated": c.last_updated}
            for c in response.citations
        ]
        _render_citation_chips(citation_dicts)

        meta_bits = [f"⏱️ {response.latency_seconds:.1f}s"]
        if response.chunks_used:
            meta_bits.append(f"📚 {response.chunks_used} source chunk(s)")
        st.markdown(f'<div class="mf-meta-caption">{" &nbsp;·&nbsp; ".join(meta_bits)}</div>', unsafe_allow_html=True)

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
        st.markdown("### 📊 Assistant Info")
        st.caption("Facts-only FAQ assistant — no investment advice.")

        with st.container(border=True):
            st.markdown("**🗂️ Coverage**")
            st.markdown("**Platform:** Kuvera  \n**AMC:** SBI Mutual Fund")
            for scheme in SCHEMES:
                st.markdown(f"- {scheme}")

        with st.container(border=True):
            st.markdown("**🛡️ Guardrails**")
            st.markdown(
                "✅ Investment-advice refusal\n\n"
                "✅ PII / privacy filter\n\n"
                "✅ Source-grounded answers only"
            )

        try:
            rag = load_rag()
            st.caption(f"🗃️ Knowledge base: {rag.collection_size()} indexed chunks")
        except Exception:
            pass

        with st.container(border=True):
            st.markdown("**📈 Session metrics**")
            col1, col2 = st.columns(2)
            col1.metric("Questions", st.session_state.total_questions)
            col2.metric("Blocked", st.session_state.blocked_count)

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.session_state.blocked_count = 0
            st.rerun()


def render_footer() -> None:
    st.markdown(
        """
        <div class="mf-footer-badges">
            <span class="mf-footer-badge">⚡ Streamlit</span>
            <span class="mf-footer-badge">🧠 Groq · Llama 3.3</span>
            <span class="mf-footer-badge">🗂️ ChromaDB</span>
            <span class="mf-footer-badge">🔗 LangChain</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(DISCLAIMER)


def main() -> None:
    init_session_state()
    inject_custom_css()
    render_hero()

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
        for col, (icon, example) in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(f"{icon}  {example}", use_container_width=True):
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
    render_footer()


if __name__ == "__main__":
    main()
