"""Comment analysis page: LLM classification, sentiment, prosperity echo,
engagement analysis, and key terms from YouTube comments."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.comments import (
    load_comments, add_sentiment, add_lexicon_scores,
    get_top_terms, get_bigrams, get_prosperity_echo_stats,
    get_engagement_by_sentiment, get_engagement_by_prosperity,
    load_classifications,
)

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


def desc(text: str):
    st.markdown(f'<div class="desc-box">{text}</div>', unsafe_allow_html=True)


VIDEO_ID = "GA6uE2CPo1I"

CATEGORY_COLORS = {
    "prosperity_echo": "#7209B7",
    "affirmation": "#3A0CA3",
    "gratitude": "#2D6A4F",
    "testimony": "#E07A5F",
    "prayer_request": "#457B9D",
    "scripture": "#6D6875",
    "personal_struggle": "#E63946",
    "criticism": "#1A1A1A",
    "other": "#AAAAAA",
}

CATEGORY_LABELS = {
    "prosperity_echo": "Prosperity Echo",
    "affirmation": "Affirmation",
    "gratitude": "Gratitude",
    "testimony": "Testimony",
    "prayer_request": "Prayer Request",
    "scripture": "Scripture",
    "personal_struggle": "Personal Struggle",
    "criticism": "Criticism",
    "other": "Other",
}

st.title("Comment Analysis")
st.markdown(
    f'Analysis of 500 YouTube comments on a Joel Osteen video '
    f'([GA6uE2CPo1I](https://www.youtube.com/watch?v={VIDEO_ID})).'
)

with st.expander("Methodology"):
    st.markdown("""
**LLM Classification:** Each comment is classified by Gemini 2.5 Flash into one of nine categories:
testimony, prayer request, affirmation, gratitude, prosperity echo, scripture, personal struggle, criticism, or other.

**Sentiment:** VADER (Valence Aware Dictionary and sEntiment Reasoner), designed for social media text.
Compound score from −1 (most negative) to +1 (most positive). Thresholds: ≥0.05 positive, ≤−0.05 negative.

**Prosperity Echo:** Lexicon matching using the same prosperity-theology term lists from the main dashboard
(market terms, technique/causality terms). Checks whether commenters mirror the preacher's language.

**Engagement:** Like counts cross-tabulated with sentiment and prosperity-language presence.

**Key terms:** TF-IDF on individual comments; top terms by mean TF-IDF score. **Bigrams:** two-word phrases by count.
    """)


@st.cache_data
def get_analysis(video_id: str):
    cdf = load_comments(video_id)
    if cdf.empty:
        return None
    cdf = add_sentiment(cdf)
    cdf = add_lexicon_scores(cdf)

    # LLM classifications (pre-computed cache)
    cls_df = load_classifications(video_id)
    if not cls_df.empty:
        cat_map = dict(zip(cls_df["id"], cls_df["category"]))
        conf_map = dict(zip(cls_df["id"], cls_df.get("confidence", pd.Series(dtype=float))))
        cdf["llm_category"] = cdf.index.map(lambda i: cat_map.get(i, "other"))
        cdf["llm_confidence"] = cdf.index.map(lambda i: conf_map.get(i, 0.0))
    else:
        cdf["llm_category"] = "unavailable"
        cdf["llm_confidence"] = 0.0

    return cdf


cdf = get_analysis(VIDEO_ID)

if cdf is None:
    st.warning("No comments found. Run `python fetch_comments.py GA6uE2CPo1I` first.")
    st.stop()

# ── Headline metrics ─────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Comments", len(cdf))
with col2:
    st.metric("Avg Sentiment", f"{cdf['sentiment_compound'].mean():.2f}")
with col3:
    pos_pct = 100 * (cdf["sentiment_label"] == "positive").mean()
    st.metric("Positive", f"{pos_pct:.0f}%")
with col4:
    echo_pct = 100 * cdf["has_prosperity_language"].mean()
    st.metric("Prosperity Echo", f"{echo_pct:.0f}%")
with col5:
    if "llm_category" in cdf.columns and cdf["llm_category"].iloc[0] != "unavailable":
        top_cat = cdf["llm_category"].value_counts().idxmax()
        st.metric("Top Category", CATEGORY_LABELS.get(top_cat, top_cat))

st.markdown("---")

# ── Tab layout ────────────────────────────────────────────────────────────

tab_class, tab_sent, tab_echo, tab_engage, tab_terms = st.tabs([
    "LLM Classification", "Sentiment", "Prosperity Echo", "Engagement", "Key Terms"
])

# ── Tab 1: LLM Classification ────────────────────────────────────────────

with tab_class:
    if cdf["llm_category"].iloc[0] == "unavailable":
        st.info("LLM classifications not available. Run classification with a Gemini API key.")
    else:
        desc(
            "<strong>What this shows:</strong> Each comment classified by Gemini into categories like "
            "testimony, affirmation, prosperity echo, criticism, etc. "
            "Prosperity echo means the commenter mirrors the preacher's language about seeds, harvests, "
            "breakthroughs, and claiming blessings."
        )

        cat_counts = cdf["llm_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        cat_counts["label"] = cat_counts["category"].map(CATEGORY_LABELS).fillna(cat_counts["category"])
        cat_counts["pct"] = 100 * cat_counts["count"] / cat_counts["count"].sum()

        col_l, col_r = st.columns([2, 1])

        with col_l:
            fig = px.bar(
                cat_counts.sort_values("count", ascending=True),
                y="label", x="count", orientation="h",
                color="category", color_discrete_map=CATEGORY_COLORS,
                labels={"count": "Comments", "label": ""},
            )
            fig.update_layout(
                showlegend=False,
                yaxis=dict(tickfont=dict(size=13, family="monospace")),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig.update_traces(
                texttemplate="%{x}", textposition="outside",
                marker_line_width=2, marker_line_color="#1A1A1A",
            )
            st.plotly_chart(fig, width="stretch")

        with col_r:
            st.markdown("**Category breakdown**")
            for _, row in cat_counts.sort_values("count", ascending=False).iterrows():
                color = CATEGORY_COLORS.get(row["category"], "#888")
                st.markdown(
                    f'<span style="color:{color}; font-weight:800;">{row["label"]}</span>: '
                    f'{row["count"]} ({row["pct"]:.1f}%)',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.subheader("Sample comments by category")
        selected_cat = st.selectbox(
            "Category", options=cat_counts.sort_values("count", ascending=False)["category"].tolist(),
            format_func=lambda c: CATEGORY_LABELS.get(c, c),
        )
        samples = cdf[cdf["llm_category"] == selected_cat].nlargest(5, "likes")
        for _, row in samples.iterrows():
            color = CATEGORY_COLORS.get(selected_cat, "#888")
            st.markdown(
                f'<div class="desc-box" style="border-color:{color}; margin-bottom:0.5rem;">'
                f'<strong>{row["likes"]} likes</strong> · sentiment: {row["sentiment_compound"]:.2f}<br>'
                f'{row["text"][:400]}</div>',
                unsafe_allow_html=True,
            )


# ── Tab 2: Sentiment ─────────────────────────────────────────────────────

with tab_sent:
    desc(
        "<strong>What this shows:</strong> VADER sentiment scores for each comment. "
        "Compound score from −1 (negative) to +1 (positive). The histogram shows the "
        "distribution of sentiment across all 500 comments."
    )

    fig = px.histogram(
        cdf, x="sentiment_compound", nbins=40,
        color="sentiment_label",
        color_discrete_map={"positive": "#2D6A4F", "neutral": "#888888", "negative": "#7209B7"},
        labels={"sentiment_compound": "VADER compound score", "count": "Comments"},
    )
    fig.update_layout(bargap=0.05, margin=dict(t=10, l=10, r=10, b=10))
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


# ── Tab 3: Prosperity Echo ───────────────────────────────────────────────

with tab_echo:
    desc(
        "<strong>What this shows:</strong> How many commenters mirror the prosperity theology "
        "language used by the preacher — terms like <em>seed, harvest, declare, breakthrough, "
        "abundance</em>. Uses the same lexicon from the main sermon analysis."
    )

    echo = get_prosperity_echo_stats(cdf)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Comments with prosperity language",
                   f"{echo['with_prosperity_language']} / {echo['total']}")
        st.metric("Percentage", f"{echo['pct']:.1f}%")

    with col_b:
        fig = go.Figure(go.Pie(
            labels=["Uses prosperity language", "Does not"],
            values=[echo["with_prosperity_language"], echo["total"] - echo["with_prosperity_language"]],
            marker=dict(colors=["#7209B7", "#E5E5E5"], line=dict(width=3, color="#1A1A1A")),
            textinfo="percent+label", textfont=dict(size=13),
        ))
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    col_m, col_t = st.columns(2)
    with col_m:
        st.subheader("Top market/wealth terms in comments")
        if echo["top_market_terms"]:
            mt_df = pd.DataFrame(
                list(echo["top_market_terms"].items()), columns=["term", "count"]
            ).sort_values("count", ascending=True)
            fig = px.bar(
                mt_df, y="term", x="count", orientation="h",
                color_discrete_sequence=["#7209B7"],
                labels={"count": "Occurrences", "term": ""},
            )
            fig.update_traces(marker_line_width=2, marker_line_color="#1A1A1A")
            fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")

    with col_t:
        st.subheader("Top technique/causality terms")
        if echo["top_technique_terms"]:
            tt_df = pd.DataFrame(
                list(echo["top_technique_terms"].items()), columns=["term", "count"]
            ).sort_values("count", ascending=True)
            fig = px.bar(
                tt_df, y="term", x="count", orientation="h",
                color_discrete_sequence=["#3A0CA3"],
                labels={"count": "Occurrences", "term": ""},
            )
            fig.update_traces(marker_line_width=2, marker_line_color="#1A1A1A")
            fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Sample comments with prosperity language")
    prosperity_comments = cdf[cdf["has_prosperity_language"]].nlargest(5, "likes")
    for _, row in prosperity_comments.iterrows():
        st.markdown(
            f'<div class="desc-box" style="border-color:#7209B7; margin-bottom:0.5rem;">'
            f'<strong>{row["likes"]} likes</strong> · market: {row["market_hits"]} · '
            f'technique: {row["technique_hits"]} · outcome: {row["outcome_hits"]}<br>'
            f'{row["text"][:400]}</div>',
            unsafe_allow_html=True,
        )


# ── Tab 4: Engagement ────────────────────────────────────────────────────

with tab_engage:
    desc(
        "<strong>What this shows:</strong> Like counts cross-tabulated with sentiment and "
        "prosperity-language presence. Do negative comments get more engagement? Do comments "
        "that echo prosperity language get more or fewer likes?"
    )

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.subheader("Likes by sentiment")
        eng_sent = get_engagement_by_sentiment(cdf)
        cat_order = ["positive", "neutral", "negative"]
        eng_sent["sentiment_label"] = pd.Categorical(eng_sent["sentiment_label"], cat_order, ordered=True)
        eng_sent = eng_sent.sort_values("sentiment_label")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=eng_sent["sentiment_label"], y=eng_sent["avg_likes"], name="Avg likes",
            marker_color=["#2D6A4F", "#888888", "#7209B7"],
            marker_line_width=2, marker_line_color="#1A1A1A",
        ))
        fig.update_layout(
            yaxis_title="Average likes per comment",
            margin=dict(t=10, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Negative comments average **{eng_sent[eng_sent['sentiment_label']=='negative']['avg_likes'].values[0]:.0f}** "
            f"likes vs **{eng_sent[eng_sent['sentiment_label']=='positive']['avg_likes'].values[0]:.0f}** for positive."
        )

    with col_e2:
        st.subheader("Likes by prosperity language")
        eng_pros = get_engagement_by_prosperity(cdf)

        fig = go.Figure()
        colors = ["#AAAAAA", "#7209B7"]
        fig.add_trace(go.Bar(
            x=eng_pros["comment_type"], y=eng_pros["avg_likes"], name="Avg likes",
            marker_color=colors,
            marker_line_width=2, marker_line_color="#1A1A1A",
        ))
        fig.update_layout(
            yaxis_title="Average likes per comment",
            margin=dict(t=10, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.subheader("Sentiment × Likes scatter")
    fig = px.scatter(
        cdf, x="sentiment_compound", y="likes",
        color="sentiment_label",
        color_discrete_map={"positive": "#2D6A4F", "neutral": "#888888", "negative": "#7209B7"},
        hover_data=["text"],
        labels={"sentiment_compound": "Sentiment score", "likes": "Likes"},
        opacity=0.6,
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#1A1A1A")))
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig, width="stretch")


# ── Tab 5: Key Terms ─────────────────────────────────────────────────────

with tab_terms:
    desc(
        "<strong>What this shows:</strong> Top individual terms by TF-IDF score and top bigrams "
        "(two-word phrases) by raw count across all 500 comments."
    )

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
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10, l=10, r=10, b=10))
        fig.update_traces(marker_line_width=2, marker_line_color="#1A1A1A")
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
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10, l=10, r=10, b=10))
        fig.update_traces(marker_line_width=2, marker_line_color="#1A1A1A")
        st.plotly_chart(fig, width="stretch")
