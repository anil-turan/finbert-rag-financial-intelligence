"""
Financial Intelligence Dashboard — Streamlit App

Tab 1: Sentiment Analyser
  - Paste financial news headlines or sentences
  - FinBERT classifies each as positive / negative / neutral
  - Bar chart of sentiment distribution + colour-coded results table

Tab 2: Regulatory Q&A (RAG)
  - Ask questions about FCA conduct rules, Basel III capital requirements, AML/KYC
  - ChromaDB retrieves the most relevant regulatory passages
  - Answer displayed with source citations and retrieved chunks

Run:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Cached model loading ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading FinBERT model…")
def load_sentiment_model():
    from src.sentiment.model import FinBERTSentiment
    return FinBERTSentiment()


@st.cache_resource(show_spinner="Loading regulatory knowledge base…")
def load_rag():
    from src.rag.retriever import RegulatoryRAG
    try:
        return RegulatoryRAG()
    except FileNotFoundError:
        return None


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 Financial Intelligence Dashboard")
st.caption(
    "FinBERT sentiment analysis · Regulatory Q&A (RAG) over FCA / Basel III / AML documents"
)
st.divider()

tab1, tab2 = st.tabs(["💬 Sentiment Analyser", "📜 Regulatory Q&A"])

# ── Tab 1: Sentiment Analyser ─────────────────────────────────────────────────

with tab1:
    st.subheader("Financial News Sentiment — FinBERT")
    st.markdown(
        "Enter one sentence per line. FinBERT classifies each as **positive**, **negative**, or **neutral**."
    )

    default_texts = (
        "The company reported record profits, beating analyst expectations by 15%.\n"
        "Revenues declined 12% year-on-year amid weakening consumer demand.\n"
        "The board approved a £500 million share buyback programme.\n"
        "Credit losses increased sharply as the economic outlook deteriorated.\n"
        "Operating margins remained stable despite inflationary pressures."
    )

    input_text = st.text_area(
        "Financial sentences (one per line):",
        value=default_texts,
        height=180,
    )

    if st.button("Analyse Sentiment", type="primary"):
        sentences = [s.strip() for s in input_text.strip().split("\n") if s.strip()]
        if not sentences:
            st.warning("Please enter at least one sentence.")
        elif len(sentences) > 50:
            st.error("Maximum 50 sentences per request.")
        else:
            with st.spinner("Running FinBERT inference…"):
                model = load_sentiment_model()
                results = model.predict(sentences)

            df = pd.DataFrame([
                {
                    "Text": r.text[:120] + ("…" if len(r.text) > 120 else ""),
                    "Sentiment": r.label.capitalize(),
                    "Confidence": f"{r.score:.1%}",
                    "Positive": f"{r.scores['positive']:.3f}",
                    "Neutral": f"{r.scores['neutral']:.3f}",
                    "Negative": f"{r.scores['negative']:.3f}",
                }
                for r in results
            ])

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Results**")
                sentiment_colours = {
                    "Positive": "background-color: #d1fae5",
                    "Negative": "background-color: #fee2e2",
                    "Neutral": "background-color: #f1f5f9",
                }

                def colour_row(row):
                    colour = sentiment_colours.get(row["Sentiment"], "")
                    return [colour] * len(row)

                st.dataframe(
                    df.style.apply(colour_row, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.markdown("**Distribution**")
                counts = df["Sentiment"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Count"]
                colour_map = {"Positive": "#0f766e", "Negative": "#dc2626", "Neutral": "#64748b"}
                fig = px.bar(
                    counts,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    color_discrete_map=colour_map,
                    text="Count",
                )
                fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Regulatory Q&A ─────────────────────────────────────────────────────

with tab2:
    st.subheader("Regulatory Q&A — RAG over FCA / Basel III / AML Documents")
    st.markdown(
        "Ask any question about financial regulation. "
        "The system retrieves relevant passages from the knowledge base and answers with citations."
    )

    rag = load_rag()
    if rag is None:
        st.error(
            "Vector store not found. Run `python scripts/build_vectorstore.py` first, then restart the app."
        )
    else:
        example_questions = [
            "What are the minimum CET1 capital requirements under Basel III?",
            "What is the Liquidity Coverage Ratio?",
            "When must a firm apply Enhanced Due Diligence (EDD)?",
            "What is the Consumer Duty and when did it come into effect?",
            "What are the record keeping requirements under AML regulations?",
            "What is the countercyclical capital buffer?",
        ]

        selected = st.selectbox("Or choose an example question:", ["— type your own —"] + example_questions)
        question = st.text_input(
            "Your question:",
            value="" if selected == "— type your own —" else selected,
            placeholder="e.g. What are the Basel III leverage ratio requirements?",
        )

        if st.button("Ask", type="primary") and question.strip():
            with st.spinner("Searching regulatory knowledge base…"):
                response = rag.ask(question.strip())

            st.markdown("### Answer")
            st.info(response.answer)

            if response.sources:
                st.markdown(f"**Sources:** {', '.join(response.sources)}")

            with st.expander(f"📄 Retrieved chunks ({response.chunks_used.__len__()} passages)"):
                for i, chunk in enumerate(response.chunks_used, 1):
                    src = chunk.metadata.get("title", chunk.metadata.get("source", "unknown"))
                    st.markdown(f"**[{i}] {src}**")
                    st.text(chunk.page_content[:400] + ("…" if len(chunk.page_content) > 400 else ""))
                    st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with FinBERT (ProsusAI/finbert) · ChromaDB · sentence-transformers · LangChain · Streamlit · "
    "Part of [anil-turan/finbert-rag-financial-intelligence](https://github.com/anil-turan)"
)
