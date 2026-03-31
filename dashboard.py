import pandas as pd
import streamlit as st
import plotly.express as px

from database.db import get_news_as_dicts
from ui_helpers import (
    init_ui_state,
    get_theme_colors,
    inject_css,
    render_global_sidebar,
    prepare_dataframe,
    apply_filters,
)
from app import main as run_news_collection

st.set_page_config(
    page_title="Stock News Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    data = get_news_as_dicts()
    return pd.DataFrame(data) if data else pd.DataFrame()


def build_kpis(df):
    return {
        "articles": len(df),
        "sources": df["source"].nunique() if not df.empty else 0,
        "stocks": df["stock_symbol"].nunique() if not df.empty else 0,
        "latest": df["collected_local"].iloc[0] if not df.empty else "N/A",
        "avg_per_stock": round(len(df) / df["stock_symbol"].nunique(), 1) if not df.empty and df["stock_symbol"].nunique() else 0,
    }


def build_insights(df):
    if df.empty:
        return {
            "top_source": "N/A",
            "top_stock": "N/A",
            "busiest_hour": "N/A",
            "avg_summary_len": "N/A",
        }

    top_source = df["source"].value_counts().idxmax() if "source" in df else "N/A"
    top_stock = df["stock_symbol"].value_counts().idxmax() if "stock_symbol" in df else "N/A"

    busiest_hour = "N/A"
    if "published_local_dt" in df and df["published_local_dt"].notna().any():
        tmp = df.dropna(subset=["published_local_dt"]).copy()
        tmp["hour"] = tmp["published_local_dt"].dt.strftime("%H:00")
        if not tmp.empty:
            busiest_hour = tmp["hour"].value_counts().idxmax()

    avg_summary_len = int(df["summary"].fillna("").str.len().mean()) if "summary" in df and not df.empty else 0

    return {
        "top_source": top_source,
        "top_stock": top_stock,
        "busiest_hour": busiest_hour,
        "avg_summary_len": f"{avg_summary_len} chars",
    }


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


def plot_sources(df, colors):
    d = df["source"].value_counts().reset_index()
    d.columns = ["source", "count"]

    fig = px.bar(d, x="source", y="count", text="count", title="📰 Articles by Source")
    fig.update_traces(marker_color=colors["accent"])
    fig.update_layout(height=360, xaxis_title="", yaxis_title="Articles")
    return apply_chart_theme(fig, colors)


def plot_stocks(df, colors):
    d = df["stock_symbol"].value_counts().reset_index()
    d.columns = ["stock_symbol", "count"]

    fig = px.pie(d, names="stock_symbol", values="count", hole=0.55, title="📌 Coverage by Stock")
    fig.update_layout(height=360)
    return apply_chart_theme(fig, colors)


def plot_activity(df, colors):
    temp = df.dropna(subset=["published_local_dt"]).copy()
    if temp.empty:
        return None

    temp["published_hour"] = temp["published_local_dt"].dt.floor("h")
    grouped = temp.groupby("published_hour").size().reset_index(name="count")

    fig = px.line(grouped, x="published_hour", y="count", markers=True, title="⏱ Publishing Activity Over Time")
    fig.update_traces(line_color=colors["accent"])
    fig.update_layout(height=360, xaxis_title="Published time", yaxis_title="Articles")
    return apply_chart_theme(fig, colors)


def plot_source_stock_heatmap(df, colors):
    pivot_df = pd.crosstab(df["source"], df["stock_symbol"])
    if pivot_df.empty:
        return None

    fig = px.imshow(
        pivot_df,
        text_auto=True,
        aspect="auto",
        title="🧩 Source × Stock Coverage",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=420, xaxis_title="Stock", yaxis_title="Source")
    return apply_chart_theme(fig, colors)


def plot_top_days(df, colors):
    temp = df.dropna(subset=["published_local_dt"]).copy()
    if temp.empty:
        return None

    temp["weekday"] = temp["published_local_dt"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = temp["weekday"].value_counts().reindex(order, fill_value=0).reset_index()
    grouped.columns = ["weekday", "count"]

    fig = px.bar(grouped, x="weekday", y="count", text="count", title="📅 Publishing Volume by Day")
    fig.update_traces(marker_color=colors["accent_2"])
    fig.update_layout(height=360, xaxis_title="", yaxis_title="Articles")
    return apply_chart_theme(fig, colors)


def render_headlines(df):
    recent_df = df.head(5)

    for _, row in recent_df.iterrows():
        url = row.get("url", "#")
        title = row.get("title", "")
        stock_symbol = row.get("stock_symbol", "")
        company_name = row.get("company_name", "")
        source = row.get("source", "")
        published_local = row.get("published_local", "")
        summary = row.get("summary", "")

        st.markdown(
            f"""
            <div class="headline-card">
                <div class="headline-title">
                    <a href="{url}" target="_blank" style="text-decoration:none; color:inherit;">
                        {title}
                    </a>
                </div>
                <div class="headline-meta">
                    {stock_symbol} | {company_name} | {source} | Published: {published_local}
                </div>
                <div class="headline-summary">{summary}</div>
                <div style="margin-top:8px;">
                    <a href="{url}" target="_blank">Open article</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    init_ui_state()

    colors = get_theme_colors()
    inject_css(colors)

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-row">
                <div class="hero-icon">📈</div>
                <div>
                    <div class="hero-title">Stock News Intelligence</div>
                    <div class="hero-subtitle">
                        Real-time visibility into stock-related news, source coverage, publishing patterns.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("⚙️ News Collection Control")

    c1, c2 = st.columns([1, 1])

    with c1:
        hours_back = st.number_input(
            "Choose how many hours back to collect news",
            min_value=1,
            max_value=168,
            value=24,
            step=1,
        )

    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("▶ Run collection", use_container_width=True)

    if run_clicked:
        with st.spinner(f"Collecting news from the last {hours_back} hours..."):
            try:
                result = run_news_collection(hours_back=int(hours_back))

                if result["success"]:
                    st.success(result["message"])
                    st.info(
                        f"Tracked stocks: {result['tracked_stocks']} | "
                        f"Before filter: {result['before_filter']} | "
                        f"After filter: {result['after_filter']} | "
                        f"After dedup: {result['after_dedup']} | "
                        f"Inserted: {result['inserted']} | "
                        f"Total DB rows: {result['total_rows']}"
                    )
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning(result["message"])

            except Exception as exc:
                st.error(f"Collection failed: {exc}")

    df = load_data()

    if df.empty:
        st.warning("No news records found in the database.")
        return

    timezone_name = render_global_sidebar(df=df, include_show_urls=False)

    df = prepare_dataframe(df, timezone_name)
    filtered_df = apply_filters(df)

    kpis = build_kpis(filtered_df)
    insights = build_insights(filtered_df)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.6, 1])
    c1.metric("📄 Articles", kpis["articles"])
    c2.metric("🌐 Sources", kpis["sources"])
    c3.metric("🏷 Stocks", kpis["stocks"])
    c4.metric("🕒 Latest collected", kpis["latest"])
    c5.metric("📊 Avg / stock", kpis["avg_per_stock"])

    st.markdown("<br>", unsafe_allow_html=True)

    i1, i2, i3, i4 = st.columns(4)
    i1.markdown(f'<div class="insight-box"><div class="insight-label">Top source</div><div class="insight-value">{insights["top_source"]}</div></div>', unsafe_allow_html=True)
    i2.markdown(f'<div class="insight-box"><div class="insight-label">Most covered stock</div><div class="insight-value">{insights["top_stock"]}</div></div>', unsafe_allow_html=True)
    i3.markdown(f'<div class="insight-box"><div class="insight-label">Busiest hour</div><div class="insight-value">{insights["busiest_hour"]}</div></div>', unsafe_allow_html=True)
    i4.markdown(f'<div class="insight-box"><div class="insight-label">Avg summary size</div><div class="insight-value">{insights["avg_summary_len"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    export_df = filtered_df[[
        col for col in [
            "stock_symbol", "company_name", "source", "title", "summary",
            "published_local", "collected_local", "sentiment_label",
            "sentiment_score", "url"
        ] if col in filtered_df.columns
    ]].copy()

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Export filtered data as CSV",
        data=csv_data,
        file_name="stock_news_dashboard_export.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(plot_sources(filtered_df, colors), width="stretch")
    with g2:
        st.plotly_chart(plot_stocks(filtered_df, colors), width="stretch")

    g3, g4 = st.columns(2)
    with g3:
        fig = plot_activity(filtered_df, colors)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    with g4:
        fig = plot_top_days(filtered_df, colors)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

    fig = plot_source_stock_heatmap(filtered_df, colors)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")

    st.subheader("📰 Latest Headlines")
    render_headlines(filtered_df)


if __name__ == "__main__":
    main()