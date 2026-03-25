import pandas as pd
import streamlit as st


def init_ui_state():
    defaults = {
        "timezone_label": "Israel",
        "selected_stocks": [],
        "selected_sources": [],
        "search_text": "",
        "show_urls": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_theme_colors():
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
        "table_bg": "#121a2b",
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

            h1, h2, h3, h4, h5, h6 {{
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

            div[data-testid="stDataFrame"] {{
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid {colors["border"]};
                box-shadow: 0 10px 28px rgba(0,0,0,0.10);
                background: {colors["table_bg"]};
            }}

            [data-testid="stDownloadButton"] button,
            [data-testid="stButton"] button {{
                border-radius: 12px;
                border: 1px solid {colors["border"]};
                background: {colors["card"]};
                color: {colors["text_primary"]};
            }}

            [data-testid="stTextInput"] input,
            [data-testid="stSelectbox"] div,
            [data-testid="stMultiSelect"] div,
            [data-testid="stDateInput"] input,
            [data-testid="stNumberInput"] input,
            [data-testid="stTextArea"] textarea {{
                background: {colors["card"]} !important;
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

            .control-card {{
                background: linear-gradient(135deg, {colors["card"]}, {colors["card_2"]});
                border: 1px solid {colors["border"]};
                border-radius: 18px;
                padding: 16px;
                margin-bottom: 16px;
                box-shadow: 0 10px 28px rgba(0,0,0,0.10);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_sidebar(df=None, include_show_urls=False):
    st.sidebar.header("Filters")

    timezone_options = {
        "Israel": "Asia/Jerusalem",
        "UTC": "UTC",
        "New York": "America/New_York",
        "London": "Europe/London",
    }

    if st.session_state["timezone_label"] not in timezone_options:
        st.session_state["timezone_label"] = "Israel"

    timezone_index = list(timezone_options.keys()).index(st.session_state["timezone_label"])

    timezone_label = st.sidebar.selectbox(
        "Display timezone",
        list(timezone_options.keys()),
        index=timezone_index,
        key="timezone_label_widget",
    )
    st.session_state["timezone_label"] = timezone_label

    stock_options = []
    source_options = []

    if df is not None and not df.empty:
        if "stock_symbol" in df.columns:
            stock_options = sorted(df["stock_symbol"].dropna().unique().tolist())
        if "source" in df.columns:
            source_options = sorted(df["source"].dropna().unique().tolist())

    st.session_state["selected_stocks"] = [
        item for item in st.session_state["selected_stocks"] if item in stock_options
    ]
    st.session_state["selected_sources"] = [
        item for item in st.session_state["selected_sources"] if item in source_options
    ]

    selected_stocks = st.sidebar.multiselect(
        "Stock symbols",
        stock_options,
        default=st.session_state["selected_stocks"],
        key="selected_stocks_widget",
    )
    st.session_state["selected_stocks"] = selected_stocks

    selected_sources = st.sidebar.multiselect(
        "Sources",
        source_options,
        default=st.session_state["selected_sources"],
        key="selected_sources_widget",
    )
    st.session_state["selected_sources"] = selected_sources

    search_text = st.sidebar.text_input(
        "Search title / summary / source / company",
        value=st.session_state["search_text"],
        key="search_text_widget",
    )
    st.session_state["search_text"] = search_text

    if include_show_urls:
        show_urls = st.sidebar.toggle(
            "Show URLs in table",
            value=st.session_state["show_urls"],
            key="show_urls_widget",
        )
        st.session_state["show_urls"] = show_urls

    return timezone_options[st.session_state["timezone_label"]]


def convert_utc_to_timezone(series, timezone_name):
    if series.empty:
        return series
    dt_series = pd.to_datetime(series, errors="coerce", utc=True)
    return dt_series.dt.tz_convert(timezone_name)


def prepare_dataframe(df, timezone_name, add_display_id=False):
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

    if add_display_id:
        df.insert(0, "display_id", df.index + 1)

    return df


def apply_filters(df):
    filtered_df = df.copy()

    selected_stocks = st.session_state.get("selected_stocks", [])
    selected_sources = st.session_state.get("selected_sources", [])
    search_text = st.session_state.get("search_text", "").lower().strip()

    if selected_stocks:
        filtered_df = filtered_df[filtered_df["stock_symbol"].isin(selected_stocks)]

    if selected_sources:
        filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]

    if search_text:
        filtered_df = filtered_df[
            filtered_df["title"].fillna("").str.lower().str.contains(search_text)
            | filtered_df["summary"].fillna("").str.lower().str.contains(search_text)
            | filtered_df["company_name"].fillna("").str.lower().str.contains(search_text)
            | filtered_df["source"].fillna("").str.lower().str.contains(search_text)
        ]

    return filtered_df