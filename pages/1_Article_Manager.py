import pandas as pd
import streamlit as st

from database.db import (
    get_news_as_dicts,
    delete_news_by_id,
    delete_all_news,
)

st.set_page_config(
    page_title="Articles Manager",
    layout="wide",
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
            "sidebar_1": "#e2e8f0",
            "sidebar_2": "#f8fafc",
            "hero_1": "#dbeafe",
            "hero_2": "#eff6ff",
        }

    return {
        "bg": "#0b1220",
        "bg_2": "#0f172a",
        "card": "#121a2b",
        "card_2": "#0f172a",
        "border": "rgba(148, 163, 184, 0.16)",
        "text_primary": "#e5e7eb",
        "text_secondary": "#94a3b8",
        "sidebar_1": "#0a0f1c",
        "sidebar_2": "#111827",
        "hero_1": "#0f172a",
        "hero_2": "#172554",
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

            h1, h2, h3 {{
                color: {colors["text_primary"]} !important;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {colors["sidebar_1"]} 0%, {colors["sidebar_2"]} 100%);
                border-right: 1px solid {colors["border"]};
            }}

            [data-testid="stSidebar"] * {{
                color: {colors["text_primary"]} !important;
            }}

            .manager-hero {{
                background: linear-gradient(135deg, {colors["hero_1"]} 0%, {colors["hero_2"]} 100%);
                border: 1px solid {colors["border"]};
                border-radius: 24px;
                padding: 22px 24px;
                margin-bottom: 18px;
                box-shadow: 0 16px 40px rgba(0, 0, 0, 0.10);
            }}

            .manager-title {{
                font-size: 1.9rem;
                font-weight: 800;
                color: {colors["text_primary"]};
                margin-bottom: 6px;
            }}

            .manager-subtitle {{
                color: {colors["text_secondary"]};
                font-size: 0.98rem;
            }}

            .control-card {{
                background: linear-gradient(135deg, {colors["card"]}, {colors["card_2"]});
                border: 1px solid {colors["border"]};
                border-radius: 18px;
                padding: 16px;
                margin-bottom: 16px;
                box-shadow: 0 10px 28px rgba(0,0,0,0.10);
            }}

            div[data-testid="stDataFrame"] {{
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid {colors["border"]};
                box-shadow: 0 10px 28px rgba(0,0,0,0.10);
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
    df.insert(0, "display_id", df.index + 1)
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


def main():
    st.sidebar.header("Appearance")
    theme_mode = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
    colors = get_theme_colors(theme_mode)
    inject_css(colors)

    st.markdown(
        """
        <div class="manager-hero">
            <div class="manager-title">Articles Manager</div>
            <div class="manager-subtitle">
                Review the full dataset, filter by stock and source, export rows, and remove records safely when needed.
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
    show_urls = st.sidebar.toggle("Show URLs in table", value=False)

    filtered_df = apply_filters(df, selected_stocks, selected_sources, search_text)

    export_df = filtered_df[[
        col for col in [
            "display_id", "stock_symbol", "company_name", "source", "title", "summary",
            "published_local", "collected_local", "sentiment_label", "sentiment_score", "url"
        ] if col in filtered_df.columns
    ]].copy()

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Export current table as CSV",
        data=csv_data,
        file_name="articles_manager_export.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.subheader("Delete Controls")

    left_col, right_col = st.columns([2.8, 1])

    with left_col:
        delete_id = st.number_input("Display article ID", min_value=1, step=1)

    with right_col:
        if st.button("Delete selected", width="stretch"):
            if delete_id <= len(filtered_df):
                internal_id = int(filtered_df.iloc[int(delete_id) - 1]["id"])
                delete_news_by_id(internal_id)
                st.success(f"Deleted article with display ID {int(delete_id)}")
                st.rerun()

        if st.button("Delete all", width="stretch"):
            delete_all_news()
            st.success("All articles deleted.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🧾 Articles Table")

    display_columns = [
        "display_id",
        "stock_symbol",
        "company_name",
        "source",
        "title",
        "summary",
        "published_local",
        "collected_local",
        "sentiment_label",
        "sentiment_score",
    ]

    if show_urls:
        display_columns.append("url")

    existing_columns = [col for col in display_columns if col in filtered_df.columns]

    st.dataframe(
        filtered_df[existing_columns],
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()