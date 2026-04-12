import pandas as pd
import plotly.express as px
import streamlit as st

from database.db import get_news_as_dicts
from dashboard.ui_helpers import (
    init_ui_state,
    get_theme_colors,
    inject_css,
    render_global_sidebar,
    prepare_dataframe,
    apply_filters,
)

st.set_page_config(
    page_title="Sentiment Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data():
    data = get_news_as_dicts()
    return pd.DataFrame(data) if data else pd.DataFrame()


def apply_chart_theme(fig, colors):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=colors["text_primary"],
        title_font_color=colors["text_primary"],
        legend_font_color=colors["text_primary"],
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def plot_sentiment_distribution(df, colors):
    d = df["sentiment_label"].fillna("unknown").value_counts().reset_index()
    d.columns = ["sentiment_label", "count"]

    fig = px.pie(
        d,
        names="sentiment_label",
        values="count",
        hole=0.55,
        title="📊 Sentiment Distribution"
    )
    return apply_chart_theme(fig, colors)


def plot_sentiment_by_stock(df, colors):
    temp = df.dropna(subset=["sentiment_label", "stock_symbol"]).copy()
    grouped = temp.groupby(["stock_symbol", "sentiment_label"]).size().reset_index(name="count")

    fig = px.bar(
        grouped,
        x="stock_symbol",
        y="count",
        color="sentiment_label",
        barmode="group",
        title="📈 Sentiment by Stock"
    )
    return apply_chart_theme(fig, colors)


def plot_avg_confidence(df, colors):
    temp = df.dropna(subset=["sentiment_label", "sentiment_score"]).copy()
    grouped = temp.groupby("sentiment_label", as_index=False)["sentiment_score"].mean()

    fig = px.bar(
        grouped,
        x="sentiment_label",
        y="sentiment_score",
        title="🎯 Average Confidence by Sentiment"
    )
    return apply_chart_theme(fig, colors)


def plot_sentiment_over_time(df, colors):
    temp = df.dropna(subset=["published_local_dt", "sentiment_label"]).copy()
    if temp.empty:
        return None

    temp["published_day"] = temp["published_local_dt"].dt.floor("d")
    grouped = temp.groupby(["published_day", "sentiment_label"]).size().reset_index(name="count")

    fig = px.line(
        grouped,
        x="published_day",
        y="count",
        color="sentiment_label",
        markers=True,
        title="⏳ Sentiment Over Time"
    )
    return apply_chart_theme(fig, colors)


def render_top_articles(df, label, title_text):
    temp = df[df["sentiment_label"] == label].copy()

    if "sentiment_score" in temp.columns:
        temp = temp.sort_values(by="sentiment_score", ascending=False)

    st.subheader(title_text)

    if temp.empty:
        st.info(f"No {label} articles found.")
        return

    for _, row in temp.head(5).iterrows():
        st.markdown(
            f"""
            <div class="headline-card">
                <div class="headline-title">
                    <a href="{row.get('url', '#')}" target="_blank" style="text-decoration:none; color:inherit;">
                        {row.get('title', '')}
                    </a>
                </div>
                <div class="headline-meta">
                    {row.get('stock_symbol', '')} | {row.get('company_name', '')} | {row.get('source', '')}
                    | Score: {row.get('sentiment_score', '')}
                </div>
                <div class="headline-summary">{row.get('summary', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    init_ui_state()

    colors = get_theme_colors()
    inject_css(colors)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-row">
                <div class="hero-icon">📊</div>
                <div>
                    <div class="hero-title">Sentiment Insights</div>
                    <div class="hero-subtitle">
                        Positive and negative analysis across stocks, sources, confidence, and publishing trends.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    if df.empty:
        st.warning("No news records found in the database.")
        return

    timezone_name = render_global_sidebar(df=df, include_show_urls=False)

    df = prepare_dataframe(df, timezone_name)
    filtered_df = apply_filters(df)

    positive_count = int((filtered_df["sentiment_label"] == "positive").sum())
    negative_count = int((filtered_df["sentiment_label"] == "negative").sum())
    neutral_count = int((filtered_df["sentiment_label"] == "neutral").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Positive", positive_count)
    c2.metric("🔴 Negative", negative_count)
    c3.metric("⚪ Neutral", neutral_count)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(plot_sentiment_distribution(filtered_df, colors), width="stretch")
    with g2:
        st.plotly_chart(plot_avg_confidence(filtered_df, colors), width="stretch")

    g3, g4 = st.columns(2)
    with g3:
        st.plotly_chart(plot_sentiment_by_stock(filtered_df, colors), width="stretch")
    with g4:
        fig = plot_sentiment_over_time(filtered_df, colors)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        render_top_articles(filtered_df, "positive", "Top Positive Articles")
    with right:
        render_top_articles(filtered_df, "negative", "Top Negative Articles")


if __name__ == "__main__":
    main()