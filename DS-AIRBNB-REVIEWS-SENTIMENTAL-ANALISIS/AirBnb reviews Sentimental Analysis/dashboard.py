"""
dashboard.py — Streamlit interactive dashboard
Airbnb NYC 2019 – Sentiment Analysis

Run:
    cd "AirBnb reviews Sentimental Analysis"
    ..\..\venv\Scripts\streamlit run dashboard.py
"""
import sys
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

import charts as ch

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb NYC 2019 | Sentiment Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #1e1e3f 100%);
    border-right: 1px solid #2d2d5e;
  }

  /* KPI card */
  .kpi-card {
    background: linear-gradient(135deg, #1e1e3f, #2a2a5e);
    border: 1px solid #3d3d7a;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,229,255,0.07);
    transition: transform 0.2s;
  }
  .kpi-card:hover { transform: translateY(-3px); }
  .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 6px 0; }
  .kpi-label { font-size: 0.85rem; color: #9090c0; letter-spacing: 0.05em; text-transform: uppercase; }

  /* Section header */
  .section-header {
    font-size: 1.3rem; font-weight: 600;
    color: #00e5ff; margin: 28px 0 12px;
    border-bottom: 1px solid #2d2d5e; padding-bottom: 6px;
  }

  /* Predict box */
  .predict-box {
    background: #1e1e3f; border-radius: 12px;
    padding: 20px; border: 1px solid #3d3d7a;
  }
  .badge {
    display: inline-block; padding: 6px 18px; border-radius: 20px;
    font-weight: 600; font-size: 1rem; margin-top: 8px;
  }
  .badge-pos { background: rgba(0,229,255,0.15); color: #00e5ff; border: 1px solid #00e5ff; }
  .badge-neg { background: rgba(255,82,82,0.15);  color: #ff5252; border: 1px solid #ff5252; }
  .badge-neu { background: rgba(255,213,79,0.15); color: #ffd54f; border: 1px solid #ffd54f; }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── NLTK setup ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    for pkg in ["vader_lexicon", "stopwords", "wordnet", "punkt"]:
        nltk.download(pkg, quiet=True)
    return (
        set(stopwords.words("english")),
        WordNetLemmatizer(),
        SentimentIntensityAnalyzer(),
    )

STOP_WORDS, lemmatizer, sia = load_nlp()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

# ── Data loading ──────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent
CLEANED = BASE / "Dataset" / "Cleaned Data"
MODELS  = BASE / "Models"

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    path = CLEANED / "reviews_cleaned.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df

@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    t = MODELS / "tfidf_vectorizer.pkl"
    m = MODELS / "best_model.pkl"
    le = MODELS / "label_encoder.pkl"
    if not (t.exists() and m.exists() and le.exists()):
        return None, None, None
    with open(t, "rb") as f:  tfidf_v = pickle.load(f)
    with open(m, "rb") as f:  model_v = pickle.load(f)
    with open(le, "rb") as f: le_v    = pickle.load(f)
    return tfidf_v, model_v, le_v

df_full     = load_data()
tfidf, mdl, le = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_Bélo.svg",
             width=90)
    st.title("🏠 NYC Sentiment")
    st.markdown("---")

    if df_full is not None:
        boroughs = ["All"] + sorted(df_full["neighbourhood_group"].dropna().unique().tolist())
        sel_borough = st.selectbox("🗺 Borough", boroughs)

        room_types = ["All"] + sorted(df_full["room_type"].dropna().unique().tolist())
        sel_room = st.selectbox("🛏 Room Type", room_types)

        sel_sentiment = st.multiselect(
            "💬 Sentiment",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"],
        )
        st.markdown("---")
        st.caption("Data: Airbnb NYC 2019 · VADER · scikit-learn")
    else:
        st.warning("Run **Notebook 1** first to generate the cleaned dataset.")
        sel_borough  = "All"
        sel_room     = "All"
        sel_sentiment = []

# ── Filter data ───────────────────────────────────────────────────────────────
def apply_filters(df):
    if sel_borough != "All":
        df = df[df["neighbourhood_group"] == sel_borough]
    if sel_room != "All":
        df = df[df["room_type"] == sel_room]
    if sel_sentiment:
        df = df[df["sentiment"].isin(sel_sentiment)]
    return df

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#00e5ff; font-size:2.2rem; margin-bottom:4px;'>
  🏠 Airbnb NYC 2019 — Sentiment Analysis
</h1>
<p style='color:#8080b0; font-size:1rem; margin-top:0;'>
  Explore guest review sentiment across New York City listings
</p>
""", unsafe_allow_html=True)

# ── Guard: no data ────────────────────────────────────────────────────────────
if df_full is None:
    st.error("⚠️ Cleaned dataset not found. Please run **Notebook 1** first.")
    st.stop()

df = apply_filters(df_full)

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar selections.")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
total       = len(df)
pct_pos     = (df["sentiment"] == "Positive").mean() * 100
pct_neg     = (df["sentiment"] == "Negative").mean() * 100
avg_score   = df["compound_score"].mean()
n_listings  = df["listing_id"].nunique()

k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, f"{total:,}",        "#00e5ff", "Total Reviews"),
    (k2, f"{n_listings:,}",   "#a78bfa", "Unique Listings"),
    (k3, f"{pct_pos:.1f}%",   "#00e5ff", "Positive"),
    (k4, f"{pct_neg:.1f}%",   "#ff5252", "Negative"),
    (k5, f"{avg_score:+.3f}", "#ffd54f", "Avg VADER Score"),
]
for col, val, color, label in kpi_data:
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{color};">{val}</div>
      <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "☁️ Word Clouds",
    "🗺 Map",
    "📈 Deep Dive",
    "🤖 Predict",
])

# ══ TAB 1 – Overview ══════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Sentiment Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(ch.sentiment_bar(df), use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">By Borough</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(ch.sentiment_by_borough(df), use_container_width=True)

    st.markdown('<div class="section-header">VADER Compound Score Distribution</div>',
                unsafe_allow_html=True)
    st.plotly_chart(ch.compound_score_hist(df), use_container_width=True)

# ══ TAB 2 – Word Clouds ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Word Clouds by Sentiment</div>',
                unsafe_allow_html=True)
    wc_cols = st.columns(3)
    for col, sentiment in zip(wc_cols, ["Positive", "Neutral", "Negative"]):
        with col:
            fig_wc = ch.wordcloud_figure(df, sentiment)
            st.pyplot(fig_wc, use_container_width=True)

# ══ TAB 3 – Map ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">NYC Listing Locations by Sentiment</div>',
                unsafe_allow_html=True)

    # Merge lat/lon from listings
    listings_path = BASE / "Dataset" / "AB_NYC_2019.csv"
    if listings_path.exists():
        try:
            listings_df = pd.read_csv(listings_path,
                                      usecols=["id","latitude","longitude"])
            df_map = df.merge(listings_df, left_on="listing_id",
                              right_on="id", how="left")
            df_map = df_map.dropna(subset=["latitude","longitude"])
            # Sample for performance
            if len(df_map) > 8000:
                df_map = df_map.sample(8000, random_state=42)
            st.plotly_chart(ch.map_chart(df_map), use_container_width=True)
        except Exception as e:
            st.info(f"Map unavailable: {e}")
    else:
        st.info("AB_NYC_2019.csv not found — run `generate_data.py` first.")

# ══ TAB 4 – Deep Dive ════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Price vs. Sentiment</div>',
                unsafe_allow_html=True)
    if "price" in df.columns:
        st.plotly_chart(ch.price_by_sentiment(df), use_container_width=True)
    else:
        st.info("Price data not available in filtered view.")

    st.markdown('<div class="section-header">Reviews Over Time</div>',
                unsafe_allow_html=True)
    if "date" in df.columns and df["date"].notna().any():
        df_time = df.copy()
        df_time["year_month"] = pd.to_datetime(df_time["date"],
                                               errors="coerce").dt.to_period("M").astype(str)
        time_sent = (
            df_time.groupby(["year_month","sentiment"])
            .size().reset_index(name="count")
        )
        PALETTE = {"Positive":"#00e5ff","Neutral":"#ffd54f","Negative":"#ff5252"}
        fig_time = px.line(
            time_sent, x="year_month", y="count", color="sentiment",
            color_discrete_map=PALETTE,
            title="<b>Review Volume Over Time</b>",
            template="plotly_dark"
        )
        fig_time.update_layout(xaxis_tickangle=-45, title_font_size=16)
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown('<div class="section-header">Sample Reviews</div>',
                unsafe_allow_html=True)
    n_sample = st.slider("Number of sample reviews", 5, 50, 10)
    sample_df = df[["date","reviewer_name","comments","sentiment",
                    "compound_score"]].sample(
        min(n_sample, len(df)), random_state=42
    )
    st.dataframe(
        sample_df.style.applymap(
            lambda v: f"color: {'#00e5ff' if v=='Positive' else '#ff5252' if v=='Negative' else '#ffd54f'}",
            subset=["sentiment"]
        ),
        use_container_width=True, height=300
    )

# ══ TAB 5 – Predict ══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🤖 Real-time Sentiment Predictor</div>',
                unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter a guest review:",
        placeholder="e.g., Amazing location, very clean apartment, the host was super helpful!",
        height=120, key="review_input"
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        use_ml  = st.checkbox("Use ML model", value=True,
                              disabled=(mdl is None))
        use_vader = st.checkbox("Use VADER",  value=True)

    if st.button("🔍 Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text above.")
        else:
            st.markdown('<div class="predict-box">', unsafe_allow_html=True)

            # VADER
            if use_vader:
                score = sia.polarity_scores(user_text)["compound"]
                v_label = "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
                badge_cls = {"Positive":"badge-pos","Negative":"badge-neg","Neutral":"badge-neu"}[v_label]
                st.markdown(
                    f"**VADER:** <span class='badge {badge_cls}'>{v_label}</span>"
                    f"  compound score: `{score:+.4f}`",
                    unsafe_allow_html=True
                )

                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    gauge={"axis":{"range":[-1,1]},
                           "bar":{"color":"#00e5ff"},
                           "steps":[
                               {"range":[-1,-0.05],"color":"#3a1010"},
                               {"range":[-0.05,0.05],"color":"#2a2a10"},
                               {"range":[0.05,1],"color":"#0a2a2a"},
                           ]},
                    title={"text":"VADER Compound Score","font":{"color":"white"}},
                    number={"font":{"color":"white"}},
                    domain={"x":[0,1],"y":[0,1]}
                ))
                fig_gauge.update_layout(
                    template="plotly_dark", height=250,
                    margin=dict(t=40,b=20,l=20,r=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ML model
            if use_ml and mdl is not None:
                cleaned = clean_text(user_text)
                X_vec   = tfidf.transform([cleaned])
                pred    = mdl.predict(X_vec)[0]
                m_label = le.inverse_transform([pred])[0]
                badge_cls = {"Positive":"badge-pos","Negative":"badge-neg","Neutral":"badge-neu"}[m_label]
                st.markdown(
                    f"**ML Model:** <span class='badge {badge_cls}'>{m_label}</span>",
                    unsafe_allow_html=True
                )
            elif use_ml and mdl is None:
                st.info("Run **Notebook 2** to train and save the ML model.")

            st.markdown("</div>", unsafe_allow_html=True)
