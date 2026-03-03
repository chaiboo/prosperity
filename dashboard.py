"""Interactive dashboard for prosperity YouTube corpus analysis."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.features import load_corpus, add_features, compute_market_per_term
from analysis.causality import count_technique_outcome_edges, get_top_causality_templates
from analysis.topics import get_tfidf_keywords, compute_umap_embedding, get_top_keywords_per_doc
from analysis.lexicon import MARKET_TERMS, MARKET_PATTERN_NAMES, MARKET_PATTERNS
from analysis.extract import load_extractions

# Group 1: Prosperity (Purple)
PROSPERITY_COLORS = ["#3A0CA3", "#7209B7", "#CDB4DB"]  # eggplant, medium violet, soft lilac
# Group 2: Control (Green)
CONTROL_COLORS = ["#1B4332", "#2D6A4F", "#95D5B2"]     # forest, emerald, mint

PROSPERITY_CHANNELS = {"joel_osteen", "elevation", "creflo"}
CONTROL_CHANNELS = {"desiring_god", "mclean_bible", "village_church"}


def get_channel_color_map(slugs):
    """Map each slug to its color. Prosperity = Group A, Control = Group B."""
    color_map = {}
    p_idx, c_idx = 0, 0
    for s in sorted(slugs):
        if s in PROSPERITY_CHANNELS:
            color_map[s] = PROSPERITY_COLORS[p_idx % len(PROSPERITY_COLORS)]
            p_idx += 1
        else:
            color_map[s] = CONTROL_COLORS[c_idx % len(CONTROL_COLORS)]
            c_idx += 1
    return color_map


st.set_page_config(page_title="Prosperity YouTube Corpus", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F5F5F0; }
    h1, h2, h3 { color: #1A1A1A !important; font-weight: 900 !important; letter-spacing: -0.02em !important; }
    .stMetric label { color: #1A1A1A !important; font-weight: 700 !important; text-transform: uppercase !important; font-size: 0.75rem !important; letter-spacing: 0.05em !important; }
    .stMetric [data-testid="stMetricValue"] { color: #3A0CA3 !important; font-weight: 900 !important; }
    hr { border-color: #1A1A1A !important; border-width: 3px !important; }
    [data-testid="stExpander"] { border: 3px solid #1A1A1A !important; border-radius: 0 !important; background-color: #FFFFFF !important; }
    [data-testid="stExpander"] summary { font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.03em !important; }
    .stInfo { background-color: #FFFFFF !important; border: 3px solid #1A1A1A !important; border-radius: 0 !important; color: #1A1A1A !important; }
    .stWarning { border: 3px solid #1A1A1A !important; border-radius: 0 !important; }
    .stTabs [data-baseweb="tab"] { font-weight: 700 !important; text-transform: uppercase !important; font-size: 0.8rem !important; letter-spacing: 0.03em !important; }
    .desc-box { border: 3px solid #1A1A1A; padding: 1rem 1.2rem; margin-bottom: 1rem; background: #FFFFFF; font-size: 0.95rem; line-height: 1.5; }
    .desc-box strong { font-weight: 800; }
</style>
""", unsafe_allow_html=True)


def desc(text: str):
    """Render a brutalist description box."""
    st.markdown(f'<div class="desc-box">{text}</div>', unsafe_allow_html=True)

st.title("Prosperity YouTube Corpus — Comparative NLP")
st.caption("Analysis of prosperity vs. non-prosperity evangelical YouTube channels")

with st.expander("**About this project**", expanded=False):
    st.markdown("""
    This dashboard compares **prosperity theology** YouTube channels (Joel Osteen, Elevation, Creflo Dollar) with **non-prosperity evangelical** controls (Desiring God, McLean Bible Church / David Platt, The Village Church / Matt Chandler).
    
    **What I measure:** Word and phrase counts in YouTube video transcripts. Most metrics are normalized per 1,000 words for length fairness. I use word-boundary token counting (exact matches) plus regex for stems like *prosper*.
    
    **How to use it:** Filter channels in the sidebar, then browse tabs. Each tab has a description box and an expandable methodology section.
    """)

# Load data
@st.cache_data
def get_data():
    try:
        df = load_corpus()
        return add_features(df)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run `python build_corpus.py` first to build the corpus.")
        return None

df = get_data()
if df is None:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
channels = sorted(df["slug"].unique().tolist())
selected = st.sidebar.multiselect("Channels", channels, default=channels)
df = df[df["slug"].isin(selected)]

st.sidebar.caption("**Color key:** Purple (eggplant/violet/lilac) = prosperity. Green (forest/emerald/mint) = control.")

if df.empty:
    st.warning("No data for selected channels.")
    st.stop()

CHANNEL_COLORS = get_channel_color_map(df["slug"].unique())

# Tabs
tabs = st.tabs([
    "Overview",
    "Market Lexicon",
    "Individualization vs Structure",
    "Scripture & Emotion",
    "Temporal & Authority",
    "Blessing Linkage Patterns",
    "Causal Claims (LLM)",
    "Agency Attribution (LLM)",
    "Suffering Framing (LLM)",
    "Topic Map",
    "Sentiment Over Time",
    "Data",
])

with tabs[0]:
    st.subheader("Corpus Overview")
    desc("A summary of the YouTube transcripts I analyzed. Each row is a video from a YouTube channel.")
    with st.expander("Methodology"):
        st.markdown("""
**Method:** Raw counts of videos, words, channels.  
**Limitation:** Sample is top-N most-viewed per channel, not random.
        """)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Videos", len(df))
    with col2:
        st.metric("Channels", df["slug"].nunique())
    with col3:
        st.metric("Total words", f"{df['word_count'].sum():,.0f}")
    with col4:
        st.metric("Avg words/video", f"{df['word_count'].mean():.0f}")

    st.subheader("Videos with usable transcripts per channel")
    usable = df[df["word_count"] > 0]
    vc = usable["slug"].value_counts().reset_index()
    vc.columns = ["slug", "count"]
    vc["label"] = vc["slug"] + "  (" + vc["count"].astype(str) + ")"
    fig = px.treemap(
        vc, path=["label"], values="count",
        color="slug", color_discrete_map=CHANNEL_COLORS,
    )
    fig.update_traces(
        textinfo="label",
        textfont=dict(size=15, family="monospace"),
        marker=dict(line=dict(width=3, color="#1A1A1A")),
    )
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Box area = share of corpus. Joel Osteen's count is lower because most of his "
        "videos have subtitles/captions disabled. A few other channels have 1–2 videos "
        "without available captions."
    )

with tabs[1]:
    st.subheader("Market Lexicon Density (per 1,000 words)")
    desc("How often each channel uses words tied to money, wealth, and material blessing. Higher = more emphasis on financial/material outcomes.")
    with st.expander("Methodology"):
        st.markdown("""
**Formula:**  `market_density = 1000 × (matched term count) / (total word count)`

**Exact terms** (word-boundary match — e.g., *seed* matches "seed" but not "seeds"):  
`money, debt, wealth, rich, poor, abundance, seed, harvest, return, business, entrepreneur`

**Stem patterns** (regex — matches any word starting with the stem):  
`prosper*` (prosper, prosperity, prosperous…), `invest*` (invest, investment, investing…)

**Tokenization:** Text is lowercased and split into tokens via `\\b[a-z0-9]+('?[a-z]+)?\\b`. Each exact term is matched against the token list. Stems use regex `\\bprosper\\w*` on the raw text.

**Limitation:** Exact terms miss inflections (*seed* ≠ *seeds*). Stems may over-match. Common words like *return* and *rich* can appear in non-financial contexts.
        """)
    
    if "market_density" in df.columns:
        fig = px.box(df, x="slug", y="market_density", color="slug", points="outliers", color_discrete_map=CHANNEL_COLORS)
        fig.update_layout(showlegend=False, xaxis_title="Channel")
        st.plotly_chart(fig, width="stretch")
    st.caption("**Takeaway:** Boxes higher on the chart = that channel talks about wealth/money more often per video.")

    st.subheader("Top market words by channel")
    # Per-term breakdown
    all_terms = MARKET_TERMS + MARKET_PATTERN_NAMES
    term_data = []
    for _, r in df.iterrows():
        wc = max(1, int(r.get("word_count", 1)))
        counts = compute_market_per_term(str(r.get("transcript", "")), wc)
        for term, val in counts.items():
            term_data.append({"slug": r["slug"], "term": term, "density": val})
    term_df = pd.DataFrame(term_data)
    if not term_df.empty:
        agg = term_df.groupby(["slug", "term"])["density"].sum().reset_index()
        # Order: prosperity channels first, then control (so bars group by type, not alternating)
        all_slugs = agg["slug"].unique()
        slug_order = (
            sorted(PROSPERITY_CHANNELS & set(all_slugs))
            + sorted(CONTROL_CHANNELS & set(all_slugs))
            + [s for s in all_slugs if s not in PROSPERITY_CHANNELS and s not in CONTROL_CHANNELS]
        )
        agg["slug"] = pd.Categorical(agg["slug"], categories=slug_order, ordered=True)
        agg = agg.sort_values(["term", "slug"])
        fig = px.bar(
            agg, x="term", y="density", color="slug", barmode="group",
            labels={"density": "Per 1,000 words", "term": "Market term"},
            color_discrete_map=CHANNEL_COLORS,
            category_orders={"slug": slug_order}
        )
        st.plotly_chart(fig, width="stretch")
    st.caption("**Takeaway:** Which specific wealth-related words each channel favors. Compare *seed* vs *money* across channels.")

with tabs[2]:
    st.subheader("Individualization vs. Structure Index")
    desc("Individual focus vs. structural focus. Each dot = one video. X-axis = individualization index, Y-axis = market density.")
    with st.expander("Methodology"):
        st.markdown("""
**Formula:**  `individualization_index = individual_score − structural_score`  
where each score = `1000 × (matched term count) / (total word count)`

**Individual terms** (word-boundary match):  
`mindset, believe, declare, vision, favor, breakthrough, confidence, discipline, habit, speak, words, purpose`

**Structural terms** (word-boundary match):  
`wages, labor, inequality, class, unions, regulation, policy, housing, healthcare, system, exploitation`  
Plus stem: `corporation*`

**Interpretation:** Positive index = more individual/mindset language. Negative = more structural/systemic language.

**Limitation:** Same tokenization rules as Market Lexicon. *class* and *system* are polysemous.
        """)
    if "individualization_index" in df.columns and "market_density" in df.columns:
        fig = px.scatter(
            df, x="individualization_index", y="market_density",
            color="slug", hover_data=["title", "view_count"],
            labels={"individualization_index": "Individualization index", "market_density": "Market density"},
            color_discrete_map=CHANNEL_COLORS
        )
        st.plotly_chart(fig, width="stretch")
    st.caption("**Takeaway:** Videos in the top-right = high individual focus + high market language (typical prosperity). Bottom-left = more structural, less market-focused.")

with tabs[3]:
    st.subheader("Scripture Citation Density")
    desc("How often videos cite specific Bible passages. Higher = more explicit scripture references.")
    with st.expander("Methodology"):
        st.markdown("""
**Formula:**  `scripture_density = 1000 × (citation count) / (total word count)`

**Detection:** Regex pattern matching `<book name> <chapter>` or `<book name> <chapter>:<verse>`, e.g., *Genesis 39*, *Proverbs 3:5-6*.

**Book list:** All 66 canonical books (genesis, exodus, … revelation).

**Limitation:** Misses paraphrases ("the apostle Paul wrote…"), partial citations, abbreviated book names (e.g., "Gen" for Genesis), or non-standard formats.
        """)
    if "scripture_density" in df.columns:
        fig = px.box(df, x="slug", y="scripture_density", color="slug", points="outliers", color_discrete_map=CHANNEL_COLORS)
        fig.update_layout(showlegend=False, xaxis_title="Channel")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Emotional Lexicon")
    desc("Positive vs. negative emotion word frequency. Each dot = one video.")
    with st.expander("Methodology"):
        st.markdown("""
**Formula:**  `emotion_positive = 1000 × count(positive terms) / word_count`  
`emotion_negative = 1000 × count(negative terms) / word_count`

**Positive terms:** `joy, hope, blessed, grateful, peace, love, happy, praise`  
**Negative terms:** `struggle, battle, enemy, attack, fear, worry, anxious, suffer`

**Limitation:** No sentiment weighting — each word counts equally. Words like *love* can appear in non-emotional contexts.
        """)
    if "emotion_positive" in df.columns and "emotion_negative" in df.columns:
        fig = px.scatter(
            df, x="emotion_positive", y="emotion_negative",
            color="slug", hover_data=["title"],
            labels={"emotion_positive": "Positive emotion (per 1k)", "emotion_negative": "Negative emotion (per 1k)"},
            color_discrete_map=CHANNEL_COLORS
        )
        st.plotly_chart(fig, width="stretch")

with tabs[4]:
    st.subheader("Temporal Framing")
    desc("Future vs. present vs. past language orientation.")
    with st.expander("Methodology"):
        st.markdown("""
**Formula:** Each = `1000 × count(terms) / word_count`

**Future terms:** `will, going to, one day, destiny, breakthrough, future, shall`  
**Present terms:** `now, today, right now, this moment, currently`  
**Past terms:** `was, were, testimony, story, testified, remember`

**Limitation:** *will* is highly ambiguous (future tense vs. volition vs. "God's will"). *was*/*were* are extremely common auxiliary verbs. Multi-word phrases like *going to* and *right now* are matched via regex on the text.
        """)
    if "temporal_future" in df.columns:
        melt = df.melt(
            id_vars=["slug", "title"],
            value_vars=["temporal_future", "temporal_present", "temporal_past"],
            var_name="frame", value_name="density"
        )
        melt["frame"] = melt["frame"].str.replace("temporal_", "")
        fig = px.box(melt, x="slug", y="density", color="frame", points="outliers", color_discrete_sequence=["#7209B7", "#2D6A4F", "#CDB4DB"])
        st.plotly_chart(fig, width="stretch")

    st.subheader("Authority vs Hedging")
    desc("Declarative authority vs. hedging/uncertainty language.")
    with st.expander("Methodology"):
        st.markdown("""
**Formula:** Each = `1000 × count(occurrences) / word_count`

**Authority phrases** (regex with word boundaries — counts every occurrence, not just presence):  
`god said, bible says, scripture, thus saith, the lord said, word says`

**Hedging terms** (word-boundary token match):  
`maybe, perhaps, might, could, possibly, sometimes`

**Limitation:** Authority phrases may appear in narrative context ("and then God said…") rather than as rhetorical authority claims. *might* and *could* have non-hedging uses.
        """)
    if "authority_score" in df.columns and "hedging_score" in df.columns:
        fig = px.scatter(
            df, x="authority_score", y="hedging_score",
            color="slug", hover_data=["title"],
            labels={"authority_score": "Authority (per 1k)", "hedging_score": "Hedging (per 1k)"},
            color_discrete_map=CHANNEL_COLORS
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Call-to-Action Density")
    desc("How often videos use direct action/giving words. Higher = more calls to act.")
    with st.expander("Methodology"):
        st.markdown("""
**Formula:**  `cta_density = 1000 × count(CTA terms) / word_count`

**CTA terms:** `give, sow, seed, offer, tithe, donate, declare, speak, believe, receive`

**Limitation:** Overlaps with market terms (*seed*, *give*) and individual terms (*declare*, *believe*). *give* and *speak* are common verbs that may appear in non-CTA contexts.
        """)
    if "cta_density" in df.columns:
        fig = px.box(df, x="slug", y="cta_density", color="slug", points="outliers", color_discrete_map=CHANNEL_COLORS)
        fig.update_layout(showlegend=False, xaxis_title="Channel")
        st.plotly_chart(fig, width="stretch")

with tabs[5]:
    st.subheader("Blessing Linkage Patterns: Technique → Outcome")
    desc("How often technique words appear near outcome words, suggesting a faith→blessing link.")
    with st.expander("Methodology"):
        st.markdown("""
**Sankey diagram — co-occurrence method:**  
For each technique term found in the transcript, scan the next 50 words for any outcome term. Each (technique, outcome) pair is counted once per occurrence.

**Technique terms:** `faith, words, seed, declare, believe, sow, give`  
**Outcome terms:** `money, blessing, abundance, harvest, return, prosper`

**Template matching** (Top 20 Linkage Templates):  
Regex patterns detect sentences like:
- `if you <verb>… (then) God/he/the Lord will…`
- `when you <verb>… God/he/the Lord will…`
- `God will <verb>… when you <verb>…`

**Limitation:** Co-occurrence within a window ≠ causation. These are proximity heuristics, not semantic analysis. The Sankey counts raw co-occurrences, not unique claims.
        """)

    # Filter: all together, prosperity only, or control only
    sankey_scope = st.radio(
        "**Sankey scope:**",
        ["All selected channels", "Prosperity only (Joel, Elevation, Creflo)", "Control only (Desiring God, McLean, Village)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if sankey_scope == "Prosperity only (Joel, Elevation, Creflo)":
        sankey_df = df[df["slug"].isin(PROSPERITY_CHANNELS)]
    elif sankey_scope == "Control only (Desiring God, McLean, Village)":
        sankey_df = df[df["slug"].isin(CONTROL_CHANNELS)]
    else:
        sankey_df = df
    st.caption(f"Showing: **{sankey_scope}** — {len(sankey_df)} videos")

    edges = count_technique_outcome_edges(sankey_df)
    if edges:
        tech_set = sorted(set(e[0] for e in edges))
        out_set = sorted(set(e[1] for e in edges))
        all_nodes = tech_set + out_set
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        source = [node_idx[t] for t, _, _ in edges]
        target = [node_idx[o] for _, o, _ in edges]
        value = [v for _, _, v in edges]
        labels = all_nodes
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels),
            link=dict(source=source, target=target, value=value),
        )])
        fig.update_layout(title="Technique → Outcome flows", height=400)
        st.plotly_chart(fig, width="stretch")
    st.caption("**How to read it:** Left = technique words (faith, words, seed…). Right = outcome words (money, blessing, abundance…). Thicker bands = that link appears more often. Use the radio buttons above to compare prosperity vs control.")

    st.subheader("Top 20 Linkage Templates")
    templates = get_top_causality_templates(df, 20)
    st.caption("Actual transcript excerpts that match 'if you… then God will…' style patterns. Click to expand.")
    for slug, title, excerpt in templates:
        with st.expander(f"[{slug}] {title[:60]}..."):
            st.caption(excerpt)

with tabs[6]:
    st.subheader("Causal Claims (LLM-Extracted)")
    desc('Structured extraction of conditional theological claims: "if you do X, God will do Y."')
    with st.expander("Methodology"):
        st.markdown("""
**Method:** Gemini extracts condition → action → divine response triples from pre-filtered sentences. Each claim is classified as material outcome (money, wealth, health) and/or spiritual outcome (peace, salvation, growth).

**Limitation:** LLM extraction is approximate; claims are extracted, not verified. Pre-filtering selects sentences with relevant keywords before sending to the model.
        """)

    ext_data = load_extractions()
    if not ext_data:
        st.warning("No LLM extractions found. Run `python run_extraction.py` first.")
    else:
        if len(ext_data) < 20:
            st.caption(f"**Note:** LLM extraction has only processed {len(ext_data)} videos so far (Gemini free-tier rate limited). Results are preliminary.")
        ext_selected = [e for e in ext_data if e.get("slug") in selected]
        all_causal = []
        for e in ext_selected:
            for c in e.get("causality", []):
                all_causal.append({
                    "slug": e["slug"],
                    "title": e.get("title", ""),
                    "condition": c.get("condition", ""),
                    "action_required": c.get("action_required", ""),
                    "divine_response": c.get("divine_response", ""),
                    "material_outcome": c.get("material_outcome", False),
                    "spiritual_outcome": c.get("spiritual_outcome", False),
                })
        causal_df = pd.DataFrame(all_causal)

        if causal_df.empty:
            st.info("No causal claims extracted for the selected channels.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total causal claims", len(causal_df))
            with col2:
                mat_pct = causal_df["material_outcome"].sum() / len(causal_df) * 100
                st.metric("Material outcome %", f"{mat_pct:.1f}%")

            st.subheader("Material vs Spiritual outcomes by channel")
            outcome_agg = causal_df.groupby("slug").agg(
                total=("material_outcome", "count"),
                material=("material_outcome", "sum"),
                spiritual=("spiritual_outcome", "sum"),
            ).reset_index()
            outcome_agg["material_pct"] = 100 * outcome_agg["material"] / outcome_agg["total"]
            outcome_agg["spiritual_pct"] = 100 * outcome_agg["spiritual"] / outcome_agg["total"]
            melt = outcome_agg.melt(
                id_vars=["slug"], value_vars=["material_pct", "spiritual_pct"],
                var_name="outcome_type", value_name="percent",
            )
            melt["outcome_type"] = melt["outcome_type"].str.replace("_pct", "")
            fig = px.bar(
                melt, x="slug", y="percent", color="outcome_type", barmode="group",
                labels={"percent": "% of claims", "slug": "Channel"},
                color_discrete_sequence=["#7209B7", "#2D6A4F"],
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("**Takeaway:** Higher material % = more claims linking faith actions to financial/health outcomes.")

            st.subheader("Causal claims table")
            st.dataframe(
                causal_df[["slug", "condition", "action_required", "divine_response", "material_outcome", "spiritual_outcome"]],
                width="stretch", height=400,
            )

with tabs[7]:
    st.subheader("Agency Attribution (LLM-Extracted)")
    desc("Who causes blessings and provision — God acting freely, human faith activating it, enemy blocking it, or systemic causes.")
    with st.expander("Methodology"):
        st.markdown("""
**Method:** Gemini classifies agent (God / Human / Enemy / System / Unspecified) and verb type (bless, withhold, activate, give, heal, restore, block) from pre-filtered sentences.

**Limitation:** LLM classification is approximate; categories are simplified. Agent attribution is inferred, not stated explicitly in every case.
        """)

    ext_data = load_extractions()
    if not ext_data:
        st.warning("No LLM extractions found. Run `python run_extraction.py` first.")
    else:
        if len(ext_data) < 20:
            st.caption(f"**Note:** LLM extraction has only processed {len(ext_data)} videos so far (Gemini free-tier rate limited). Results are preliminary.")
        ext_selected = [e for e in ext_data if e.get("slug") in selected]
        all_agency = []
        for e in ext_selected:
            for a in e.get("agency", []):
                all_agency.append({
                    "slug": e["slug"],
                    "title": e.get("title", ""),
                    "event": a.get("event", ""),
                    "agent": a.get("agent", "Unspecified"),
                    "verb_type": a.get("verb_type", "other"),
                })
        agency_df = pd.DataFrame(all_agency)

        if agency_df.empty:
            st.info("No agency attributions extracted for the selected channels.")
        else:
            st.subheader("Who causes blessings?")
            agent_counts = agency_df.groupby(["slug", "agent"]).size().reset_index(name="count")
            agent_totals = agent_counts.groupby("slug")["count"].transform("sum")
            agent_counts["pct"] = 100 * agent_counts["count"] / agent_totals
            fig = px.bar(
                agent_counts, x="slug", y="pct", color="agent", barmode="stack",
                labels={"pct": "% of attributions", "slug": "Channel"},
                color_discrete_sequence=["#3A0CA3", "#7209B7", "#CDB4DB", "#2D6A4F", "#95D5B2"],
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("**Takeaway:** Higher 'Human' % = blessings framed as activated by human faith/action. Higher 'God' % = God acts unconditionally.")

            st.subheader("Agency by verb type")
            verb_counts = agency_df.groupby(["slug", "verb_type"]).size().reset_index(name="count")
            fig = px.bar(
                verb_counts, x="slug", y="count", color="verb_type", barmode="group",
                labels={"count": "Count", "slug": "Channel"},
            )
            st.plotly_chart(fig, width="stretch")

            st.subheader("Agency attributions table")
            st.dataframe(
                agency_df[["slug", "event", "agent", "verb_type"]],
                width="stretch", height=400,
            )

with tabs[8]:
    st.subheader("Suffering Framing (LLM-Extracted)")
    desc("How suffering is explained — as a test of faith, enemy attack, lack of faith, inevitable part of life, or systemic cause.")
    with st.expander("Methodology"):
        st.markdown("""
**Method:** Gemini classifies suffering references into framings: test, attack, lack_of_faith, inevitable, systemic, discipline, unspecified.

**Limitation:** LLM classification is approximate; framing categories are simplified. Some references may fit multiple framings.
        """)

    ext_data = load_extractions()
    if not ext_data:
        st.warning("No LLM extractions found. Run `python run_extraction.py` first.")
    else:
        if len(ext_data) < 20:
            st.caption(f"**Note:** LLM extraction has only processed {len(ext_data)} videos so far (Gemini free-tier rate limited). Results are preliminary.")
        ext_selected = [e for e in ext_data if e.get("slug") in selected]
        all_suffering = []
        for e in ext_selected:
            for s in e.get("suffering", []):
                all_suffering.append({
                    "slug": e["slug"],
                    "title": e.get("title", ""),
                    "suffering_reference": s.get("suffering_reference", ""),
                    "framing": s.get("framing", "unspecified"),
                })
        suffering_df = pd.DataFrame(all_suffering)

        if suffering_df.empty:
            st.info("No suffering framings extracted for the selected channels.")
        else:
            st.subheader("How is suffering explained?")
            frame_counts = suffering_df.groupby(["slug", "framing"]).size().reset_index(name="count")
            frame_totals = frame_counts.groupby("slug")["count"].transform("sum")
            frame_counts["pct"] = 100 * frame_counts["count"] / frame_totals
            fig = px.bar(
                frame_counts, x="slug", y="pct", color="framing", barmode="stack",
                labels={"pct": "% of references", "slug": "Channel"},
                color_discrete_sequence=["#3A0CA3", "#7209B7", "#CDB4DB", "#1B4332", "#2D6A4F", "#95D5B2", "#888888"],
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("**Takeaway:** 'lack_of_faith' = suffering blamed on insufficient faith. 'test' = God testing. 'attack' = spiritual warfare. 'systemic' = social/economic causes.")

            st.subheader("Suffering framings table")
            st.dataframe(
                suffering_df[["slug", "suffering_reference", "framing"]],
                width="stretch", height=400,
            )

with tabs[9]:
    st.subheader("Video Space (UMAP)")
    desc("A map where each dot = one video. Similar language clusters together. Colors = channel.")
    with st.expander("Methodology"):
        st.markdown("""
**Method:** TF-IDF vectors reduced to 2D via UMAP.

**Limitation:** Dimensionality reduction is approximate; distance interpretation is qualitative.
        """)
    try:
        emb, vec = compute_umap_embedding(df)
        keywords = get_top_keywords_per_doc(df, vec, top_k=5)
        plot_df = df[["slug", "title"]].copy()
        plot_df["umap_1"] = emb[:, 0]
        plot_df["umap_2"] = emb[:, 1] if emb.shape[1] > 1 else 0
        plot_df["keywords"] = [", ".join(k) for k in keywords]
        fig = px.scatter(
            plot_df, x="umap_1", y="umap_2",
            color="slug", hover_data=["title", "keywords"],
            labels={"umap_1": "UMAP 1", "umap_2": "UMAP 2"},
            color_discrete_map=CHANNEL_COLORS
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.warning(f"UMAP failed: {e}. Install umap-learn: pip install umap-learn")

    st.subheader("TF-IDF Keywords by Channel")
    desc("Words distinctive to each channel (not just frequent). Higher = more characteristic of that channel.")
    with st.expander("Methodology"):
        st.markdown("""
**Method:** TF-IDF on tokenized text; mean score per channel.

**Limitation:** Single-token only; phrases not captured.
        """)
    kw_df = get_tfidf_keywords(df, top_k=10)
    if not kw_df.empty:
        fig = px.bar(
            kw_df, x="keyword", y="tfidf_mean", color="slug",
            barmode="group", labels={"tfidf_mean": "TF-IDF (mean)", "keyword": "Keyword"},
            color_discrete_map=CHANNEL_COLORS
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, width="stretch")

with tabs[10]:
    st.subheader("Sentiment Over Time")
    desc("Net emotional tone per video, averaged by year. Above zero = more positive.")
    with st.expander("Methodology"):
        st.markdown("""
**Formula:**  `sentiment = (positive_count − negative_count) / total_word_count`

This is **not** per-1,000 words — it's a raw ratio (typically between −0.001 and +0.005). The chart averages this ratio by publish year and channel.

**Positive terms:** `joy, hope, blessed, grateful, peace, love, happy, praise`  
**Negative terms:** `struggle, battle, enemy, attack, fear, worry, anxious, suffer`

**Limitation:** Different scale from all other charts (which use per-1k). A small absolute change can be meaningful. Not comparable to the Emotional Lexicon tab's per-1k values.
        """)
    if "sentiment" in df.columns and "published_at" in df.columns:
        plot_df = df.copy()
        plot_df["published_at"] = pd.to_datetime(plot_df["published_at"], errors="coerce")
        plot_df = plot_df.dropna(subset=["published_at"])
        if not plot_df.empty:
            plot_df["year"] = plot_df["published_at"].dt.year
            agg = plot_df.groupby(["slug", "year"])["sentiment"].mean().reset_index()
            fig = px.line(agg, x="year", y="sentiment", color="slug", markers=True, color_discrete_map=CHANNEL_COLORS)
            st.plotly_chart(fig, width="stretch")

with tabs[11]:
    st.subheader("Raw data")
    desc("Underlying data for each video: channel, title, view count, transcript status, and all computed scores.")
    st.dataframe(df, width="stretch", height=400)
