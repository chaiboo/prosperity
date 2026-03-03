# Prosperity Sermon Corpus — Interactive Dashboard & NLP Spec

## Overview

After corpus scraping completes, build an interactive dashboard for comparative NLP analysis of prosperity vs. non-prosperity evangelical sermons.

---

## Data Fields (from corpus)

**Per video:**
- `channel`, `slug`, `label`, `title`, `video_id`, `url`
- `published_at`, `view_count`, `like_count`, `comment_count`, `duration_seconds`
- `transcript`, `word_count`, `transcript_status`

---

## Core NLP Features (from earlier conversation)

### 1. Market Lexicon Density / Wealth Positivity Index

**Terms (count per 1,000 words):**
`money`, `debt`, `wealth`, `rich`, `poor`, `prosper*`, `abundance`, `seed`, `harvest`, `return`, `invest*`, `business`, `entrepreneur`

**Viz:**
- Channel-level distributions (box/violin)
- Top "market words" bar chart per channel

---

### 2. Individualization vs. Structure Index

**Individual / technique vocabulary:**
`mindset`, `believe`, `declare`, `vision`, `favor`, `breakthrough`, `confidence`, `discipline`, `habit`, `speak`, `words`, `purpose`

**Structural / political economy vocabulary:**
`wages`, `labor`, `inequality`, `class`, `unions`, `corporation*`, `regulation`, `policy`, `housing`, `healthcare`, `system`, `exploitation`

**Score:** `(individual − structural) / word_count` (normalized)

**Viz:** 2D scatter — x = Individualization score, y = Market lexicon density (prosperity channels cluster high/high; controls differ)

---

### 3. "Blessing Causality" Patterns

**Logic:** Co-occurrence + proximity: `faith`/`words`/`seed` → `money`/`blessing`/`abundance`

**Extraction:** Rule-based sentences linking spiritual technique to material outcome ("if you… then God will…" patterns)

**Viz:**
- Sankey / chord: technique terms → outcome terms
- Top 20 causality templates (short excerpts)

---

### 4. Topic Map (lightweight, interpretable)

**Method:**
- TF-IDF keywords by channel
- UMAP clustering for "sermons that sound alike"

**Viz:** "Sermon space" map, colored by channel; hover shows title + top keywords

---

## Additional NLP Suggestions

### 5. Scripture Citation Density
- Count references to books/chapters (e.g. "Genesis 39", "Proverbs 3")
- Compare prosperity vs. control: do prosperity sermons cite fewer passages? Different books?

### 6. Emotional / Affective Lexicon
- Positive emotion: `joy`, `hope`, `blessed`, `grateful`, `peace`
- Negative / struggle: `struggle`, `battle`, `enemy`, `attack`, `fear`
- Ratio or separate scores per channel

### 7. Temporal Framing
- Future-oriented: `will`, `going to`, `one day`, `destiny`, `breakthrough`
- Present: `now`, `today`, `right now`
- Past: `was`, `were`, `testimony`, `story`

### 8. Authority / Certainty Markers
- `God said`, `the Bible says`, `scripture`, `thus saith`
- Hedging: `maybe`, `perhaps`, `might`, `could`

### 9. Call-to-Action Density
- `give`, `sow`, `seed`, `offer`, `tithe`, `donate`
- Imperatives: `declare`, `speak`, `believe`, `receive`

### 10. Named Entity & Person References
- Pastor names, guest speakers, celebrities
- Compare self-promotion vs. external references

### 11. LDA / BERTopic (optional, heavier)
- If you want automated topic discovery beyond TF-IDF
- BERTopic often gives more coherent themes

### 12. Sentiment Over Time
- Track sentiment by `published_at` — does prosperity rhetoric shift over years?

---

## Tech Stack Recommendation

| Layer | Tool |
|-------|------|
| Dashboard | **Streamlit** (fast, Python-native, good for research) |
| NLP | `spaCy` or `nltk`, `scikit-learn` (TF-IDF, clustering) |
| Embeddings / UMAP | `sentence-transformers`, `umap-learn` |
| Viz | `plotly` (interactive), `altair` (declarative) |
| Data | `pandas`, load from `data/processed/corpus.csv` |

**Alternatives:** Dash (more control, steeper), Panel, Gradio (simpler but less flexible).

---

## Project Structure

```
propserity/
├── build_corpus.py          # existing
├── data/
│   ├── raw/
│   └── processed/
│       └── corpus.csv
├── analysis/
│   ├── __init__.py
│   ├── lexicon.py           # market, individual, structural dicts
│   ├── features.py           # compute all derived features
│   ├── causality.py          # blessing causality extraction
│   └── topics.py             # TF-IDF, UMAP
├── dashboard.py              # Streamlit app
├── requirements.txt
└── DASHBOARD_SPEC.md         # this file
```

---

## Implementation Phases

1. **Phase 1 — Feature pipeline**  
   - Load corpus, filter rows with `transcript_status == "ok"`
   - Implement lexicons and compute Market + Individualization scores
   - Save `data/processed/features.csv` for dashboard

2. **Phase 2 — Dashboard shell**  
   - Streamlit app with tabs: Overview, Lexicons, Causality, Topics
   - Basic charts (bar, box, scatter)

3. **Phase 3 — Causality + Topics**  
   - Rule-based causality extraction
   - TF-IDF + UMAP sermon space

4. **Phase 4 — Polish**  
   - Add suggested features (scripture, emotion, temporal)
   - Hover tooltips, filters, export

---

## Quick Start

```bash
# 1. Build corpus (if not done)
python build_corpus.py

# 2. Install dashboard deps
pip install -r requirements.txt

# 3. Run dashboard
streamlit run dashboard.py
```
