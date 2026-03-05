"""
charts.py — Reusable Plotly / Matplotlib chart functions
for the Airbnb NYC 2019 Sentiment Analysis project.
"""
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "Positive": "#00e5ff",
    "Neutral":  "#ffd54f",
    "Negative": "#ff5252",
}
DARK_TEMPLATE = "plotly_dark"


# ── 1. Sentiment bar chart ────────────────────────────────────────────────────
def sentiment_bar(df: pd.DataFrame, title: str = "Sentiment Distribution") -> go.Figure:
    """
    Vertical bar chart showing review counts per sentiment class.
    """
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    fig = px.bar(
        counts, x="sentiment", y="count", color="sentiment",
        color_discrete_map=PALETTE,
        title=f"<b>{title}</b>",
        template=DARK_TEMPLATE,
        text="count",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside",
                      marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        title_font_size=18,
        xaxis_title="Sentiment",
        yaxis_title="Number of Reviews",
        margin=dict(t=60, b=40),
    )
    return fig


# ── 2. Word cloud (returns PIL image as numpy array) ──────────────────────────
def wordcloud_figure(
    df: pd.DataFrame, sentiment: str,
    max_words: int = 80
) -> plt.Figure:
    """
    Returns a Matplotlib figure with a word cloud for the given sentiment.
    """
    WC_COLORS = {"Positive": "Blues", "Neutral": "YlOrBr", "Negative": "Reds"}

    corpus = " ".join(df[df["sentiment"] == sentiment]["cleaned_text"].dropna())
    if not corpus.strip():
        corpus = "no data available"

    wc = WordCloud(
        width=700, height=400, max_words=max_words,
        background_color="#1a1a2e",
        colormap=WC_COLORS.get(sentiment, "viridis"),
    ).generate(corpus)

    fig, ax = plt.subplots(figsize=(8, 4),
                           facecolor="#1a1a2e")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{sentiment} Reviews", color=PALETTE.get(sentiment, "white"),
                 fontsize=14, pad=10)
    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()
    return fig


# ── 3. Price × Sentiment box plot ─────────────────────────────────────────────
def price_by_sentiment(df: pd.DataFrame) -> go.Figure:
    """
    Box plot comparing listing price across sentiment classes.
    Requires 'price' column (from the merged dataset).
    """
    df_plot = df.dropna(subset=["price", "sentiment"]).copy()
    df_plot = df_plot[df_plot["price"] < 1000]   # cap outliers

    fig = px.box(
        df_plot, x="sentiment", y="price", color="sentiment",
        color_discrete_map=PALETTE,
        title="<b>Price Distribution by Sentiment</b>",
        template=DARK_TEMPLATE,
        points="outliers",
    )
    fig.update_layout(
        showlegend=False,
        title_font_size=18,
        xaxis_title="Sentiment",
        yaxis_title="Price per Night (USD)",
        margin=dict(t=60, b=40),
    )
    return fig


# ── 4. Sentiment by borough grouped bar ───────────────────────────────────────
def sentiment_by_borough(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart — sentiment count per NYC borough.
    """
    borough_sent = (
        df.groupby(["neighbourhood_group", "sentiment"])
        .size().reset_index(name="count")
    )
    fig = px.bar(
        borough_sent,
        x="neighbourhood_group", y="count", color="sentiment",
        color_discrete_map=PALETTE, barmode="group",
        title="<b>Sentiment by NYC Borough</b>",
        template=DARK_TEMPLATE,
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title="Borough",
        yaxis_title="Review Count",
        margin=dict(t=60, b=40),
    )
    return fig


# ── 5. VADER compound score histogram ────────────────────────────────────────
def compound_score_hist(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of VADER compound scores coloured by sentiment.
    """
    fig = px.histogram(
        df, x="compound_score", color="sentiment",
        color_discrete_map=PALETTE, nbins=60, marginal="box",
        title="<b>VADER Compound Score Distribution</b>",
        template=DARK_TEMPLATE, opacity=0.8,
    )
    fig.update_layout(title_font_size=18,
                      xaxis_title="Compound Score",
                      yaxis_title="Count")
    return fig


# ── 6. NYC map — listings coloured by dominant sentiment ────────────────────
def map_chart(df: pd.DataFrame) -> go.Figure:
    """
    Scatter Mapbox of NYC listing locations coloured by sentiment.
    Requires 'latitude', 'longitude' columns.
    """
    df_map = df.dropna(subset=["latitude", "longitude", "sentiment"]).copy()

    fig = px.scatter_mapbox(
        df_map,
        lat="latitude", lon="longitude",
        color="sentiment",
        color_discrete_map=PALETTE,
        hover_data={"neighbourhood": True, "room_type": True,
                    "price": True, "latitude": False, "longitude": False},
        zoom=10,
        center={"lat": 40.7128, "lon": -74.0060},
        title="<b>NYC Listings by Sentiment</b>",
        mapbox_style="carto-darkmatter",
        opacity=0.6,
        height=550,
    )
    fig.update_layout(title_font_size=18, margin=dict(t=60, b=0))
    return fig


# ── 7. Model comparison bar chart ─────────────────────────────────────────────
def model_comparison_chart(results_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing model accuracy and F1 scores.
    Expects columns: Model, Accuracy, F1 (weighted).
    """
    melted = results_df.melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    fig = px.bar(
        melted, x="Model", y="Score", color="Metric",
        barmode="group",
        title="<b>ML Model Comparison</b>",
        template=DARK_TEMPLATE,
        color_discrete_sequence=["#00e5ff", "#ffd54f"],
        text="Score",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        title_font_size=18,
        yaxis_range=[0, 1.05],
        margin=dict(t=60, b=60),
        xaxis_tickangle=-20,
    )
    return fig
