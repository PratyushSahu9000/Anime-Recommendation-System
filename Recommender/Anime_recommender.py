import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="⛩️",
    layout="wide",
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f0f1a; color: #e2e2f0; }
h1 { color: #a78bfa; font-weight: 700; }
h3 { color: #c4b5fd; }

.result-card {
    background: #1c1c2e;
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.result-card:hover { border-color: #7c3aed; }

.rank-badge {
    background: #7c3aed;
    color: white;
    font-weight: 700;
    font-size: 1rem;
    border-radius: 50%;
    min-width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.anime-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e2f0;
    text-transform: capitalize;
    flex: 1;
}

.meta-tag {
    background: #2d2d4e;
    color: #a78bfa;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}

div[data-testid="stTextInput"] input {
    background: #1c1c2e !important;
    color: #e2e2f0 !important;
    border: 1px solid #2d2d4e !important;
    border-radius: 8px !important;
}

div[data-testid="stButton"] > button {
    background: #7c3aed;
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    width: 100%;
}
div[data-testid="stButton"] > button:hover { background: #6d28d9; }

.section-header {
    color: #7c3aed;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Pickle ───────────────────────────────────────────────────────────────
PICKLE_PATH = "../model/anime_model.pkl"   # adjust if your folder layout differs

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["df"], data["tfidf_matrix"], data["embeddings"], data["indices"]


# ─── Recommendation Logic ──────────────────────────────────────────────────────
def remove_same_series(df, sorted_indices, base_title):
    base_words = set(base_title.split())
    filtered = []
    for i in sorted_indices:
        title_words = set(df.iloc[i]["Title"].split())
        if len(base_words & title_words) > 1:
            continue
        filtered.append(i)
    return filtered


def get_index(df, title):
    matches = df[df["Title"].str.contains(re.escape(title), na=False)]
    if matches.empty:
        return None
    return matches.index[0]


def hybrid_recommend(title, df, tfidf_matrix, embeddings,
                     top_n=10, w_sem=0.65, w_tfidf=0.15,
                     w_score=0.10, w_pop=0.10, include_sequels=False):
    title_clean = title.lower().strip()
    idx = get_index(df, title_clean)
    if idx is None:
        return None, "❌ Title not found. Try a different spelling."

    tfidf_sim    = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    semantic_sim = cosine_similarity(embeddings[idx].reshape(1, -1), embeddings).flatten()

    scaler = MinMaxScaler()
    tfidf_sim    = scaler.fit_transform(tfidf_sim.reshape(-1, 1)).flatten()
    semantic_sim = scaler.fit_transform(semantic_sim.reshape(-1, 1)).flatten()

    final_scores = (
        w_sem   * semantic_sim +
        w_tfidf * tfidf_sim +
        w_score * df["score_norm"].values +
        w_pop   * df["pop_norm"].values
    )
    final_scores[idx] = -1

    sorted_idx = final_scores.argsort()[::-1]
    if include_sequels:
        filtered = list(sorted_idx)
    else:
        filtered = remove_same_series(df, sorted_idx, title_clean)
    top        = filtered[:top_n]

    result = df.iloc[top][["Title", "AverageScore", "Popularity", "Genres", "Format"]].copy()
    return result, None


# ─── App UI ────────────────────────────────────────────────────────────────────
st.markdown("# ⛩️ Anime Recommender")
st.markdown("Hybrid semantic + TF-IDF recommendations.")
st.divider()

# Load
try:
    df, tfidf_matrix, embeddings, indices = load_model()
except FileNotFoundError:
    st.error(
        f"**anime_model.pkl not found.**\n\n"
        f"Expected at: `{PICKLE_PATH}`\n\n"
        "Run `build_pickle_kaggle.ipynb` on Kaggle, download the output, "
        "and place it in the `model/` folder.",
        icon="🚨"
    )
    st.stop()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_n   = st.slider("Recommendations", 5, 20, 10)
    st.markdown("**Score weights**")
    w_sem   = st.slider("Semantic similarity",  0.0, 1.0, 0.65, 0.05)
    w_tfidf = st.slider("TF-IDF similarity",    0.0, 1.0, 0.15, 0.05)
    w_score = st.slider("Average score boost",  0.0, 1.0, 0.10, 0.05)
    w_pop   = st.slider("Popularity boost",     0.0, 1.0, 0.10, 0.05)
    total   = w_sem + w_tfidf + w_score + w_pop
    st.caption(f"Weights sum: **{total:.2f}** {'✅' if abs(total-1.0)<0.01 else '⚠️ should equal 1.0'}")

    st.divider()
    include_sequels = st.checkbox("Include sequels & related series", value=False)
    st.divider()
    st.markdown(f"**Dataset:** {len(df):,} anime loaded")

# ─── Search ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("", placeholder="Search anime…  e.g. Bleach, Vinland Saga, Mushishi")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔍 Recommend")

# ─── Suggestions ───────────────────────────────────────────────────────────────
if query and not search_btn:
    matches = df[df["Title"].str.contains(re.escape(query.lower()), na=False)]["Title"].head(6)
    if not matches.empty:
        st.markdown("<div class='section-header'>Suggestions</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(matches), 3))
        for i, title in enumerate(matches):
            with cols[i % 3]:
                if st.button(title.title(), key=f"sug_{i}"):
                    query = title
                    search_btn = True

# ─── Results ───────────────────────────────────────────────────────────────────
if search_btn and query:
    with st.spinner("Finding recommendations…"):
        results, err = hybrid_recommend(
            query, df, tfidf_matrix, embeddings,
            top_n=top_n,
            w_sem=w_sem, w_tfidf=w_tfidf,
            w_score=w_score, w_pop=w_pop,
            include_sequels=include_sequels
        )

    if err:
        st.warning(err)
    else:
        st.markdown(f"### Recommendations for **{query.title()}**")
        st.markdown(f"<div class='section-header'>Top {top_n} results</div>",
                    unsafe_allow_html=True)

        for rank, (_, row) in enumerate(results.iterrows(), 1):
            genres_html = " ".join(
                f"<span class='meta-tag'>{g.strip().title()}</span>"
                for g in str(row["Genres"]).split()[:4] if g.strip()
            )
            score_display = f"{row['AverageScore']:.0f}" if row["AverageScore"] > 0 else "N/A"
            pop_display   = f"{int(row['Popularity']):,}"

            st.markdown(f"""
            <div class="result-card">
                <div class="rank-badge">#{rank}</div>
                <div class="anime-title">{row['Title'].title()}</div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">
                    {genres_html}
                    <span class="meta-tag">⭐ {score_display}</span>
                    <span class="meta-tag">👥 {pop_display}</span>
                    <span class="meta-tag">{str(row['Format']).upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif search_btn and not query:
    st.warning("Please enter an anime title first.")

st.divider()
st.caption("Hybrid recommender · TF-IDF + SentenceTransformers (all-mpnet-base-v2) · Built with Streamlit")
