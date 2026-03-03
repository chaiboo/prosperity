"""Comment analysis page: sentiment, topics, and key terms from YouTube comments."""

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis.comments import load_comments, add_sentiment, get_topic_model, get_top_terms, get_bigrams

st.set_page_config(page_title="Comment Analysis", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F5F5F0; }
    h1, h2, h3 { color: #1A1A1A !important; font-weight: 900 !important; letter-spacing: -0.02em !important; }
    .stMetric label { color: #1A1A1A !important; font-weight: 700 !important; text-transform: uppercase !important; font-size: 0.75rem !important; letter-spacing: 0.05em !important; }
    .stMetric [data-testid="stMetricValue"] { color: #3A0CA3 !important; font-weight: 900 !important; }
    hr { border-color: #1A1A1A !important; border-width: 3px !important; }
    [data-testid="stExpander"] { border: 3px solid #1A1A1A !important; border-radius: 0 !important; background-color: #FFFFFF !important; }
    [data-testid="stExpander"] summary { font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.03em !important; }
    .desc-box { border: 3px solid #1A1A1A; padding: 1rem 1.2rem; margin-bottom: 1rem; background: #FFFFFF; font-size: 0.95rem; line-height: 1.5; }
    .desc-box strong { font-weight: 800; }
</style>
""", unsafe_allow_html=True)

VIDEO_ID = "GA6uE2CPo1I"

st.title("Comment Analysis")
st.markdown(
    f'Sentiment, topics, and key terms from 500 YouTube comments on a Joel Osteen video '
    f'([GA6uE2CPo1I](https://www.youtube.com/watch?v={VIDEO_ID})).'
)

with st.expander("Methodology"):
    st.markdown("""
**Sentiment:** VADER (Valence Aware Dictionary and sEntiment Reasoner), designed for social media text. Compound score ranges from −1 (most negative) to +1 (most positive). Thresholds: ≥0.05 = positive, ≤−0.05 = negative, else neutral.

**Topics:** Latent Dirichlet Allocation (LDA) with 6 topics on a bag-of-words matrix (min 3 occurrences, max 85% doc frequency, English stop words removed).

**Key terms:** TF-IDF on individual comments; top terms by mean TF-IDF score across all comments.

**Bigrams:** Two-word phrases by raw count (min 3 occurrences).
    """)


@st.cache_data
def get_comment_analysis(video_id: str):
    cdf = load_comments(video_id)
    if cdf.empty:
        return None, None
    cdf = add_sentiment(cdf)
    topics, cdf = get_topic_model(cdf, n_topics=6)
    return cdf, topics


cdf, topics = get_comment_analysis(VIDEO_ID)

if cdf is None:
    st.warning("No comments found. Run `python fetch_comments.py GA6uE2CPo1I` first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Comments", len(cdf))
with col2:
    st.metric("Avg sentiment", f"{cdf['sentiment_compound'].mean():.2f}")
with col3:
    pos_pct = 100 * (cdf["sentiment_label"] == "positive").mean()
    st.metric("Positive %", f"{pos_pct:.0f}%")
with col4:
    neg_pct = 100 * (cdf["sentiment_label"] == "negative").mean()
    st.metric("Negative %", f"{neg_pct:.0f}%")

st.markdown("---")

st.subheader("Sentiment distribution")
fig = px.histogram(
    cdf, x="sentiment_compound", nbins=40,
    color="sentiment_label",
    color_discrete_map={"positive": "#2D6A4F", "neutral": "#888888", "negative": "#7209B7"},
    labels={"sentiment_compound": "VADER compound score", "count": "Comments"},
)
fig.update_layout(bargap=0.05)
st.plotly_chart(fig, width="stretch")

st.markdown("---")

st.subheader("Topics (LDA)")
for t in topics:
    st.markdown(f"**Topic {t['topic']}:** {', '.join(t['words'])}")

topic_counts = cdf["dominant_topic"].value_counts().reset_index()
topic_counts.columns = ["topic", "comments"]
topic_counts["label"] = topic_counts["topic"].apply(
    lambda i: topics[i]["label"] if i < len(topics) else f"Topic {i}"
)
fig = px.bar(
    topic_counts.sort_values("topic"), x="label", y="comments",
    labels={"label": "Topic", "comments": "Comments"},
    color_discrete_sequence=["#3A0CA3"],
)
fig.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig, width="stretch")

st.markdown("---")

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Top terms (TF-IDF)")
    terms = get_top_terms(cdf, top_n=20)
    term_df = pd.DataFrame(terms, columns=["term", "tfidf"])
    fig = px.bar(
        term_df, y="term", x="tfidf", orientation="h",
        labels={"tfidf": "Mean TF-IDF", "term": ""},
        color_discrete_sequence=["#3A0CA3"],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

with col_r:
    st.subheader("Top bigrams")
    bigrams = get_bigrams(cdf, top_n=20)
    bi_df = pd.DataFrame(bigrams, columns=["bigram", "count"])
    fig = px.bar(
        bi_df, y="bigram", x="count", orientation="h",
        labels={"count": "Count", "bigram": ""},
        color_discrete_sequence=["#7209B7"],
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

st.markdown("---")

st.subheader("Most liked comments by sentiment")
for label, color in [("positive", "#2D6A4F"), ("negative", "#7209B7"), ("neutral", "#888888")]:
    subset = cdf[cdf["sentiment_label"] == label].nlargest(3, "likes")
    if not subset.empty:
        st.markdown(f"**{label.title()}** (top by likes):")
        for _, row in subset.iterrows():
            st.markdown(
                f'<div class="desc-box" style="border-color:{color}; margin-bottom:0.5rem;">'
                f'<strong>{row["likes"]} likes</strong> (score: {row["sentiment_compound"]:.2f})<br>'
                f'{row["text"][:300]}</div>',
                unsafe_allow_html=True,
            )
