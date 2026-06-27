"""
Streamlit UI for the Mutual Fund FAQ Assistant.
"""

from __future__ import annotations

import streamlit as st

from prompts import DISCLAIMER
from rag import MutualFundRAG

st.set_page_config(
    page_title="Mutual Fund FAQ Assistant",
    page_icon="📊",
    layout="centered",
)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of SBI Bluechip Fund?",
    "What is the lock-in period of SBI Long Term Equity Fund?",
    "How do I download my capital gains statement on Kuvera?",
]


@st.cache_resource(show_spinner="Loading knowledge base and LLM...")
def load_rag() -> MutualFundRAG:
    return MutualFundRAG()


def main() -> None:
    st.title("Mutual Fund FAQ Assistant")
    st.caption("Facts-only. No investment advice.")
    st.info(
        "Welcome! Ask factual questions about SBI Bluechip, Contra, Long Term Equity (ELSS), "
        "Magnum Midcap, and Small Cap funds, or about Kuvera statements and tax reports."
    )

    st.markdown("**Example questions:**")
    for example in EXAMPLE_QUESTIONS:
        st.markdown(f"- {example}")

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Citations"):
                    for src in message["sources"]:
                        st.markdown(f"- [{src}]({src})")
            if message.get("last_updated"):
                st.caption(f"Last updated from sources: {message['last_updated']}")

    user_input = st.chat_input("Ask a factual mutual fund question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching approved sources..."):
                try:
                    rag = load_rag()
                    response = rag.answer(user_input)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.stop()
                except EnvironmentError as exc:
                    st.error(str(exc))
                    st.stop()
                except Exception as exc:
                    st.error(f"An error occurred: {exc}")
                    st.stop()

            st.markdown(response.answer)

            if response.sources:
                with st.expander("Citations"):
                    for src in response.sources:
                        if src.startswith("http"):
                            st.markdown(f"- [{src}]({src})")
                        else:
                            st.markdown(f"- {src}")

            if response.last_updated:
                st.caption(f"Last updated from sources: {response.last_updated}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "last_updated": response.last_updated,
            }
        )

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
