# Prosperity YouTube Corpus

Comparative NLP analysis of prosperity theology videos vs. non-prosperity evangelical controls.

## Deploy on Streamlit Community Cloud

1. **Push this repo to GitHub** (create a new repo and push).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**.
4. Configure:
   - **Repository:** `chaiboo/propserity` (or your repo name)
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
5. Click **Deploy**. The app will install dependencies and run the dashboard.

The corpus data (`data/processed/corpus.csv`) is included in the repo, so the dashboard works without running the corpus builder.

## Run locally

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard (corpus must exist)
streamlit run dashboard.py
```

To build or rebuild the corpus (requires YouTube API key):

```bash
export YT_API_KEY='your-key'
python build_corpus.py --fresh
```

## Project structure

- `dashboard.py` — Streamlit app
- `build_corpus.py` — Scrapes transcripts and builds corpus
- `analysis/` — NLP features, lexicons, topics
- `data/processed/corpus.csv` — Combined YouTube corpus
