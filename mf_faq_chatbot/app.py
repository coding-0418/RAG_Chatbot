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
    ("💰", "Expense ratio", "What is the expense ratio of SBI Bluechip Fund?"),
    ("🔒", "Lock-in period", "What is the lock-in period of SBI Long Term Equity Fund?"),
    ("📊", "Exit load", "What is the exit load of HDFC Flexi Cap Fund?"),
    ("📈", "Riskometer", "What is the Risk-o-meter for mutual fund schemes?"),
]

ALL_FUND_HOUSES = "All fund houses"

USER_AVATAR = "🧑‍💼"
ASSISTANT_AVATAR = "📊"

MIN_QUERY_INTERVAL_SECONDS = 3
MAX_HISTORY_MESSAGES = 6  # 3 user/assistant turns


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.75rem;
            max-width: 1140px;
        }

        h1, h2, h3, h4 { letter-spacing: -0.01em; }

        /* ---------- Hero ---------- */
        .mf-hero {
            background: linear-gradient(135deg, #071B3D 0%, #0B5FFF 55%, #2E8BFF 100%);
            border-radius: 20px;
            padding: 2.1rem 2.3rem;
            color: #FFFFFF;
            margin-bottom: 1.1rem;
            box-shadow: 0 16px 40px rgba(11, 40, 95, 0.28);
            position: relative;
            overflow: hidden;
        }
        .mf-hero::after {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 197, 66, 0.28) 0%, rgba(255,255,255,0) 70%);
        }
        .mf-hero-eyebrow {
            display: inline-block;
            background: rgba(255, 197, 66, 0.18);
            border: 1px solid rgba(255, 197, 66, 0.45);
            color: #FFD98A;
            border-radius: 999px;
            padding: 0.2rem 0.75rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .mf-hero h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2.15rem;
            font-weight: 800;
            color: #FFFFFF;
        }
        .mf-hero p {
            margin: 0;
            max-width: 640px;
            opacity: 0.92;
            font-size: 1rem;
            line-height: 1.5;
        }
        .mf-hero-badges {
            margin-top: 1.1rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            position: relative;
            z-index: 1;
        }
        .mf-pill {
            display: inline-block;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.32);
            border-radius: 999px;
            padding: 0.22rem 0.75rem;
            font-size: 0.78rem;
            font-weight: 600;
        }

        /* ---------- Stats strip ---------- */
        .mf-stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.4rem;
        }
        .mf-stat-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFF 100%);
            border: 1px solid #E4EAF5;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 2px 8px rgba(20, 40, 90, 0.05);
        }
        .mf-stat-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0B2860;
            line-height: 1.1;
        }
        .mf-stat-label {
            font-size: 0.78rem;
            color: #6B7688;
            margin-top: 0.15rem;
            font-weight: 500;
        }

        /* ---------- AMC showcase chips ---------- */
        .mf-section-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #1A1D23;
            margin: 0.2rem 0 0.6rem 0;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
            border-radius: 999px !important;
            border: 1px solid #D7E1F5 !important;
            background: #F7FAFF !important;
            color: #0B2860 !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            padding: 0.3rem 0.4rem !important;
            transition: all 0.15s ease-in-out;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
            border-color: #0B5FFF !important;
            background: #0B5FFF !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
        }

        /* Example question cards (buttons) */
        div[data-testid="column"] div[data-testid="stButton"] button {
            border-radius: 14px;
            border: 1px solid rgba(11, 95, 255, 0.18);
            background: #FFFFFF;
            color: #1A1D23;
            text-align: left;
            padding: 0.95rem 1.05rem;
            height: 100%;
            white-space: normal;
            box-shadow: 0 1px 3px rgba(20, 40, 90, 0.04);
            transition: all 0.15s ease-in-out;
        }
        div[data-testid="column"] div[data-testid="stButton"] button:hover {
            border-color: #0B5FFF;
            background: #F0F6FF;
            transform: translateY(-2px);
            box-shadow: 0 8px 18px rgba(11, 95, 255, 0.14);
        }

        .mf-meta-caption {
            color: #8A93A3;
            font-size: 0.78rem;
            margin-top: 0.15rem;
        }

        /* ---------- Chat bubbles ---------- */
        div[data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 0.35rem 0.15rem;
        }
        div[data-testid="stChatMessageAvatarUser"] {
            background: linear-gradient(135deg, #0B5FFF, #2E8BFF) !important;
        }
        div[data-testid="stChatMessageAvatarAssistant"] {
            background: linear-gradient(135deg, #071B3D, #0B5FFF) !important;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FBFCFE 0%, #F2F5FA 100%);
        }
        section[data-testid="stSidebar"] .stContainer,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
        }
        .mf-amc-count-badge {
            display: inline-block;
            background: #0B5FFF;
            color: #FFFFFF;
            border-radius: 999px;
            padding: 0.12rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 700;
            margin-left: 0.4rem;
        }

        /* ---------- Footer ---------- */
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
            <span class="mf-hero-eyebrow">Facts-only · Zero investment advice</span>
            <h1>📊 Mutual Fund FAQ Assistant</h1>
            <p>Instant, source-cited answers about mutual fund schemes across every fund house
            in the knowledge base — grounded in official disclosures, refreshed as sources change,
            and built to refuse anything that isn't a plain fact.</p>
            <div class="mf-hero-badges">
                <span class="mf-pill">✅ Guardrails active</span>
                <span class="mf-pill">📚 Source-cited answers</span>
                <span class="mf-pill">🔒 No PII collected</span>
                <span class="mf-pill">⚡ Sub-second retrieval</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats_bar(amc_count: int, chunk_count: int) -> None:
    stats = [
        (str(amc_count) if amc_count else "—", "Fund houses covered"),
        (f"{chunk_count:,}" if chunk_count else "—", "Verified source chunks"),
        ("100%", "Guardrail-checked answers"),
        ("₹0", "Cost to ask a question"),
    ]
    cards = "".join(
        f'<div class="mf-stat-card"><div class="mf-stat-value">{value}</div>'
        f'<div class="mf-stat-label">{label}</div></div>'
        for value, label in stats
    )
    st.markdown(f'<div class="mf-stats-row">{cards}</div>', unsafe_allow_html=True)


def render_amc_showcase(amcs: list[str]) -> None:
    if not amcs:
        return
    st.markdown('<div class="mf-section-title">🏦 Browse by fund house</div>', unsafe_allow_html=True)
    cols_per_row = 5
    for row_start in range(0, len(amcs), cols_per_row):
        row_amcs = amcs[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, amc_name in zip(cols, row_amcs):
            short_name = amc_name.replace(" Mutual Fund", "")
            if col.button(short_name, key=f"amc-chip-{amc_name}", use_container_width=True):
                st.session_state.amc_filter = amc_name
                st.rerun()
    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading knowledge base and LLM...")
def load_rag() -> MutualFundRAG:
    return MutualFundRAG()


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("total_questions", 0)
    st.session_state.setdefault("blocked_count", 0)
    st.session_state.setdefault("last_query_time", 0.0)
    st.session_state.setdefault("rate_limited", False)
    st.session_state.setdefault("amc_filter", ALL_FUND_HOUSES)


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


def render_message(message: dict) -> None:
    with st.chat_message(message["role"], avatar=_avatar_for(message["role"])):
        if message.get("blocked"):
            st.warning(message["content"])
        elif message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

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
            selected_amc = st.session_state.amc_filter
            amc = None if selected_amc == ALL_FUND_HOUSES else selected_amc
            response = rag.answer(question, history=history, amc=amc)

        if response.blocked:
            st.warning(response.answer)
        elif response.error:
            st.error(response.answer)
        else:
            st.markdown(response.answer)

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

        try:
            rag = load_rag()
            amcs = rag.list_amcs()
            chunk_count = rag.collection_size()
        except Exception:
            amcs, chunk_count = [], 0

        with st.container(border=True):
            st.markdown(f"**🗂️ Coverage** <span class='mf-amc-count-badge'>{len(amcs)}</span>", unsafe_allow_html=True)
            if amcs:
                st.markdown("**Fund houses in knowledge base:**")
                for amc_name in amcs:
                    st.markdown(f"- {amc_name}")
            else:
                st.caption("No fund houses indexed yet — run `python ingest.py`.")

        if amcs:
            with st.container(border=True):
                st.markdown("**🔎 Filter answers**")
                st.selectbox(
                    "Scope questions to one fund house",
                    options=[ALL_FUND_HOUSES, *amcs],
                    key="amc_filter",
                    label_visibility="collapsed",
                )

        with st.container(border=True):
            st.markdown("**🛡️ Guardrails**")
            st.markdown(
                "✅ Investment-advice refusal\n\n"
                "✅ PII / privacy filter\n\n"
                "✅ Source-grounded answers only"
            )

        if chunk_count:
            st.caption(f"🗃️ Knowledge base: {chunk_count} indexed chunks")

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

    rag = load_rag()
    amcs = rag.list_amcs()
    render_stats_bar(len(amcs), rag.collection_size())

    if not st.session_state.messages:
        render_amc_showcase(amcs)

        st.markdown('<div class="mf-section-title">💬 Try a popular question</div>', unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, (icon, label, example) in zip(cols, EXAMPLE_QUESTIONS):
            if col.button(f"{icon}  {label}: {example}", use_container_width=True):
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
