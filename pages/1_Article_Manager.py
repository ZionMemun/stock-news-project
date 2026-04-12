import pandas as pd
import streamlit as st

from database.db import (
    get_news_as_dicts,
    delete_news_by_id,
    delete_all_news,
    get_tracked_stocks,
    add_tracked_stock,
    delete_tracked_stock,
    delete_news_by_stock_symbol,
    delete_news_by_exact_date,
    delete_news_up_to_date,
)
from dashboard.ui_helpers import (
    init_ui_state,
    get_theme_colors,
    inject_css,
    render_global_sidebar,
    prepare_dataframe,
    apply_filters,
)

st.set_page_config(
    page_title="Articles Manager",
    layout="wide",
)


def load_data():
    data = get_news_as_dicts()
    return pd.DataFrame(data) if data else pd.DataFrame()


def main():
    init_ui_state()

    colors = get_theme_colors()
    inject_css(colors)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-row">
                <div class="hero-icon">🗂️</div>
                <div>
                    <div class="hero-title">Articles Manager</div>
                    <div class="hero-subtitle">
                        Manage tracked stocks, inspect article records, export data, and delete records safely when needed.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Tracked Stocks")

    tracked_stocks = get_tracked_stocks()

    stock_card_left, stock_card_right = st.columns([1.4, 1])

    with stock_card_left:
        st.markdown("Add a new stock to the tracking list.")

        new_symbol = st.text_input("Stock symbol").strip().upper()
        new_company = st.text_input("Company name").strip()

        if st.button("Add stock", width="stretch"):
            if new_symbol and new_company:
                add_tracked_stock(new_symbol, new_company)
                st.success(f"Added {new_symbol} - {new_company}")
                st.rerun()
            else:
                st.warning("Please fill in both stock symbol and company name.")

    with stock_card_right:
        st.markdown("Currently tracked stocks:")

        if tracked_stocks:
            for stock in tracked_stocks:
                row_col1, row_col2 = st.columns([3, 1])
                with row_col1:
                    st.write(f"**{stock['stock_symbol']}** — {stock['company_name']}")
                with row_col2:
                    if st.button(
                        "Remove",
                        key=f"remove_{stock['stock_symbol']}",
                        width="stretch"
                    ):
                        delete_tracked_stock(stock["stock_symbol"])
                        st.success(f"Removed {stock['stock_symbol']}")
                        st.rerun()
        else:
            st.info("No tracked stocks yet.")

    st.divider()

    df = load_data()

    if df.empty:
        st.warning("No news records found in the database.")
        return

    timezone_name = render_global_sidebar(df=df, include_show_urls=True)

    df = prepare_dataframe(df, timezone_name, add_display_id=True)
    filtered_df = apply_filters(df)
    show_urls = st.session_state["show_urls"]

    st.subheader("Articles Table")

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

    column_config = {}
    if "url" in existing_columns:
        column_config["url"] = st.column_config.LinkColumn(
            "Article URL",
            display_text="Open article"
        )

    st.dataframe(
        filtered_df[existing_columns],
        column_config=column_config,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Export")

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

    st.divider()

    st.subheader("Delete by Display ID")
    st.markdown('<div class="control-card">', unsafe_allow_html=True)

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

    st.divider()

    st.subheader("Bulk Delete Controls")
    st.markdown('<div class="control-card">', unsafe_allow_html=True)

    bulk_col1, bulk_col2 = st.columns(2)

    with bulk_col1:
        st.markdown("#### Delete by Date")

        date_mode = st.radio(
            "Delete mode",
            ["Only selected day", "Selected day and all previous days"],
            key="bulk_delete_date_mode"
        )

        date_basis = st.selectbox(
            "Use date from",
            ["published_at", "collected_at"],
            key="bulk_delete_date_basis"
        )

        selected_date = st.date_input(
            "Select date",
            key="bulk_delete_date_value"
        )

        selected_date_str = selected_date.strftime("%Y-%m-%d")

        if st.button("Run date-based deletion", width="stretch"):
            if date_mode == "Only selected day":
                delete_news_by_exact_date(selected_date_str, date_column=date_basis)
                st.success(f"Deleted all rows where {date_basis} = {selected_date_str}")
            else:
                delete_news_up_to_date(selected_date_str, date_column=date_basis)
                st.success(f"Deleted all rows where {date_basis} <= {selected_date_str}")

            st.rerun()

    with bulk_col2:
        st.markdown("#### Delete by Stock Symbol")

        bulk_stock_options = sorted(df["stock_symbol"].dropna().unique().tolist())
        selected_stock_for_delete = st.selectbox(
            "Select stock symbol to delete",
            bulk_stock_options,
            key="bulk_delete_stock"
        )

        if st.button("Delete all rows for selected stock", width="stretch"):
            delete_news_by_stock_symbol(selected_stock_for_delete)
            st.success(f"Deleted all rows for stock symbol {selected_stock_for_delete}")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()