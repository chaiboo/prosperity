"""Topic map: TF-IDF keywords + UMAP video space."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def get_tfidf_keywords(
    df: pd.DataFrame,
    slug_col: str = "slug",
    text_col: str = "transcript",
    top_k: int = 10,
    max_features: int = 5000,
) -> pd.DataFrame:
    """Top TF-IDF keywords per channel."""
    texts = df[text_col].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for slug in df[slug_col].unique():
        mask = (df[slug_col] == slug).to_numpy()
        sub = X[mask]
        if sub.shape[0] == 0:
            continue
        # Mean TF-IDF per term for this channel
        means = np.asarray(sub.mean(axis=0)).flatten()
        top_idx = np.argsort(-means)[:top_k]
        for i in top_idx:
            if means[i] > 0:
                rows.append({
                    "slug": slug,
                    "keyword": feature_names[i],
                    "tfidf_mean": float(means[i]),
                })
    return pd.DataFrame(rows)


def compute_umap_embedding(
    df: pd.DataFrame,
    text_col: str = "transcript",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """UMAP embedding of videos from TF-IDF vectors."""
    texts = df[text_col].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(texts)
    X_dense = X.toarray()

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(X_dense)
    else:
        # Fallback: PCA if UMAP not installed
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dense)
        pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]))
        embedding = pca.fit_transform(X_scaled)
        if embedding.shape[1] < n_components:
            pad = np.zeros((embedding.shape[0], n_components - embedding.shape[1]))
            embedding = np.hstack([embedding, pad])

    return embedding, vectorizer


def get_top_keywords_per_doc(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    text_col: str = "transcript",
    top_k: int = 5,
) -> List[List[str]]:
    """Top k keywords for each document (for hover tooltips)."""
    X = vectorizer.transform(df[text_col].fillna("").astype(str))
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        data = row.toarray().flatten()
        top_idx = np.argsort(-data)[:top_k]
        keywords = [feature_names[j] for j in top_idx if data[j] > 0]
        rows.append(keywords)
    return rows
