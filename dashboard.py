import pandas as pd
import streamlit as st
import plotly.express as px

from database.db import get_news_as_dicts

st.set_page_config(
    page_title="Stock News Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_theme_colors(mode: str):
    if mode == "Light":
        return {
            "bg": "#f3f6fb",
            "bg_2": "#e8eef7",
            "card": "#ffffff",
            "card_2": "#f8fbff",
            "border": "rgba(100, 116, 139, 0.18)",
            "text_primary": "#0f172a",
            "text_secondary": "#475569",
            "accent": "#2563eb",
            "accent_2": "#0ea5e9",
            "hero_1": "#dbeafe",
            "hero_2": "#eff6ff",
            "sidebar_1": "#e2e8f0",
            "sidebar_2": "#f8fafc",
            "insight_bg": "#eef4ff",
        }

    return {
        "bg": "#0b1220",
        "bg_2": "#0f172a",
        "card": "#121a2b",
        "card_2": "#0f172a",
        "border": "rgba(148, 163, 184, 0.16)",
        "text_primary": "#e5e7eb",
        "text_secondary": "#94a3b8",
        "accent": "#60a5fa",
        "accent_2": "#38bdf8",
        "hero_1": "#0f172a",
        "hero_2": "#172554",
        "sidebar_1": "#0a0f1c",
        "sidebar_2": "#111827",
        "insight_bg": "#111827",
    }


def inject_css(colors):
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, {colors["bg"]} 0%, {colors["bg_2"]} 100%);
                color: {colors["text_primary"]};
            }}

            .block-container {{
                max-width: 96rem;
                padding-top: 4.7rem;
                padding-bottom: 2rem;
            }}

            h1, h2, h3, h4 {{
                color: {colors["text_primary"]} !important;
            }}

            p, span, label, div {{
                color: {colors["text_primary"]};
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {colors["sidebar_1"]} 0%, {colors["sidebar_2"]} 100%);
                border-right: 1px solid {colors["border"]};
            }}

            [data-testid="stSidebar"] * {{
                color: {colors["text_primary"]} !important;
            }}

            div[data-testid="metric-container"] {{
                background: linear-gradient(135deg, {colors["card"]}, {colors["card_2"]});
                border: 1px solid {colors["border"]};
                border-radius: 18px;
                padding: 16px;
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.10);
            }}

            div[data-testid="metric-container"] label {{
                color: {colors["text_secondary"]} !important;
                font-weight: 500;
            }}

            div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
                color: {colors["text_primary"]} !important;
            }}

            .hero {{
                background: linear-gradient(135deg, {colors["hero_1"]} 0%, {colors["hero_2"]} 100%);
                border: 1px solid {colors["border"]};
                border-radius: 24px;
                padding: 24px 26px;
                margin-bottom: 18px;
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.10);
            }}

            .hero-row {{
                display: flex;
                align-items: center;
                gap: 16px;
            }}

            .hero-icon {{
                width: 54px;
                height: 54px;
                border-radius: 16px;
                background: linear-gradient(135deg, {colors["accent"]}, {colors["accent_2"]});
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 28px;
                box-shadow: 0 10px 24px rgba(0,0,0,0.14);
            }}

            .hero-title {{
                font-size: 2.1rem;
                font-weight: 800;
                color: {colors["text_primary"]};
                margin-bottom: 4px;
            }}

            .hero-subtitle {{
                color: {colors["text_secondary"]};
                font-size: 1rem;
                line-height: 1.5;
            }}

            .insight-box {{
                background: {colors["insight_bg"]};
                border: 1px solid {colors["border"]};
                border-radius: 18px;
                padding: 14px 16px;
                box-shadow: 0 8px 22px rgba(0,0,0,0.08);
                min-height: 98px;
            }}

            .insight-label {{
                color: {colors["text_secondary"]};
                font-size: 0.84rem;
                margin-bottom: 8px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}

            .insight-value {{
                color: {colors["text_primary"]};
                font-size: 1.15rem;
                font-weight: 800;
                line-height: 1.35;
            }}

            .headline-card {{
                background: linear-gradient(135deg, {colors["card"]}, {colors["card_2"]});
                border: 1px solid {colors["border"]};
                border-radius: 18px;
                padding: 14px 16px;
                margin-bottom: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            }}

            .headline-title {{
                color: {colors["text_primary"]};
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 6px;
            }}

            .headline-meta {{
                color: {colors["text_secondary"]};
                font-size: 0.84rem;
                margin-bottom: 8px;
            }}

            .headline-summary {{
                color: {colors["text_primary"]};
                font-size: 0.94rem;
                line-height: 1.45;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_data():
    data = get_news_as_dicts()
    return pd.DataFrame(data) if data else pd.DataFrame()


def convert_utc_to_timezone(series, timezone_name):
    if series.empty:
        return series
    dt_series = pd.to_datetime(series, errors="coerce", utc=True)
    return dt_series.dt.tz_convert(timezone_name)


def prepare_dataframe(df, timezone_name):
    df = df.copy()

    if "published_at" in df.columns:
        published_local = convert_utc_to_timezone(df["published_at"], timezone_name)
        df["published_local_dt"] = published_local
        df["published_local"] = published_local.dt.strftime("%Y-%m-%d %H:%M:%S")

    if "collected_at" in df.columns:
        collected_local = convert_utc_to_timezone(df["collected_at"], timezone_name)
        df["collected_local_dt"] = collected_local
        df["collected_local"] = collected_local.dt.strftime("%Y-%m-%d %H:%M:%S")

    df = df.sort_values(by="collected_local_dt", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    return df


def apply_filters(df, selected_stocks, selected_sources, search_text):
    filtered_df = df.copy()

    if selected_stocks:
        filtered_df = filtered_df[filtered_df["stock_symbol"].isin(selected_stocks)]

    if selected_sources:
        filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]

    if search_text:
        q = search_text.lower().strip()
        filtered_df = filtered_df[
            filtered_df["title"].fillna("").str.lower().str.contains(q)
            | filtered_df["summary"].fillna("").str.lower().str.contains(q)
            | filtered_df["company_name"].fillna("").str.lower().str.contains(q)
            | filtered_df["source"].fillna("").str.lower().str.contains(q)
        ]

    return filtered_df


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


def apply_chart_theme(fig, colors, light=False):
    fig.update_layout(
        template="plotly_white" if light else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=colors["text_primary"],
        title_font_color=colors["text_primary"],
        legend_font_color=colors["text_primary"],
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def plot_sources(df, colors, light):
    d = df["source"].value_counts().reset_index()
    d.columns = ["source", "count"]

    fig = px.bar(d, x="source", y="count", text="count", title="📰 Articles by Source")
    fig.update_traces(marker_color=colors["accent"])
    fig.update_layout(height=360, xaxis_title="", yaxis_title="Articles")
    return apply_chart_theme(fig, colors, light)


def plot_stocks(df, colors, light):
    d = df["stock_symbol"].value_counts().reset_index()
    d.columns = ["stock_symbol", "count"]

    fig = px.pie(d, names="stock_symbol", values="count", hole=0.55, title="📌 Coverage by Stock")
    fig.update_layout(height=360)
    return apply_chart_theme(fig, colors, light)


def plot_activity(df, colors, light):
    temp = df.dropna(subset=["published_local_dt"]).copy()
    if temp.empty:
        return None

    temp["published_hour"] = temp["published_local_dt"].dt.floor("h")
    grouped = temp.groupby("published_hour").size().reset_index(name="count")

    fig = px.line(grouped, x="published_hour", y="count", markers=True, title="⏱ Publishing Activity Over Time")
    fig.update_traces(line_color=colors["accent"])
    fig.update_layout(height=360, xaxis_title="Published time", yaxis_title="Articles")
    return apply_chart_theme(fig, colors, light)


def plot_source_stock_heatmap(df, colors, light):
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
    return apply_chart_theme(fig, colors, light)


def plot_top_days(df, colors, light):
    temp = df.dropna(subset=["published_local_dt"]).copy()
    if temp.empty:
        return None

    temp["weekday"] = temp["published_local_dt"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = temp["weekday"].value_counts().reindex(order, fill_value=0).reset_index()
    grouped.columns = ["weekday", "count"]

    fig = px.bar(grouped, x="weekday", y="count", text="count", title="📅 Publishing Volume by Day")
    fig.update_traces(marker_color="#38bdf8")
    fig.update_layout(height=360, xaxis_title="", yaxis_title="Articles")
    return apply_chart_theme(fig, colors, light)


def render_headlines(df):
    recent_df = df.head(5)

    for _, row in recent_df.iterrows():
        st.markdown(
            f"""
            <div class="headline-card">
                <div class="headline-title">{row.get("title", "")}</div>
                <div class="headline-meta">
                    {row.get("stock_symbol", "")} | {row.get("company_name", "")} | {row.get("source", "")}
                    | Published: {row.get("published_local", "")}
                </div>
                <div class="headline-summary">{row.get("summary", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.sidebar.header("Appearance")
    theme_mode = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)

    colors = get_theme_colors(theme_mode)
    light = theme_mode == "Light"
    inject_css(colors)

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-row">
                <div class="hero-icon">📈</div>
                <div>
                    <div class="hero-title">Stock News Intelligence</div>
                    <div class="hero-subtitle">
                        Real-time visibility into stock-related news, source coverage, and publishing patterns.
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

    timezone_options = {
        "Israel": "Asia/Jerusalem",
        "UTC": "UTC",
        "New York": "America/New_York",
        "London": "Europe/London",
    }

    st.sidebar.header("Filters")
    tz_label = st.sidebar.selectbox("Display timezone", list(timezone_options.keys()), index=0)
    timezone_name = timezone_options[tz_label]

    df = prepare_dataframe(df, timezone_name)

    stock_options = sorted(df["stock_symbol"].dropna().unique().tolist())
    source_options = sorted(df["source"].dropna().unique().tolist())

    selected_stocks = st.sidebar.multiselect("Stock symbols", stock_options, default=[])
    selected_sources = st.sidebar.multiselect("Sources", source_options, default=[])
    search_text = st.sidebar.text_input("Search title / summary / source / company")

    filtered_df = apply_filters(df, selected_stocks, selected_sources, search_text)

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
        st.plotly_chart(plot_sources(filtered_df, colors, light), width="stretch")
    with g2:
        st.plotly_chart(plot_stocks(filtered_df, colors, light), width="stretch")

    g3, g4 = st.columns(2)
    with g3:
        fig = plot_activity(filtered_df, colors, light)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    with g4:
        fig = plot_top_days(filtered_df, colors, light)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

    fig = plot_source_stock_heatmap(filtered_df, colors, light)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")

    st.subheader("📰 Latest Headlines")
    render_headlines(filtered_df)


if __name__ == "__main__":
    main()