import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# Path to the SQLite database file
DB_PATH = Path("data/news.db")


def get_connection():
    """
    Create a connection to the database.
    Ensures that the data directory exists.
    Returns rows as dictionary-like objects.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_table():
    """
    Create the news table if it does not already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT,
            company_name TEXT,
            source TEXT,
            title TEXT,
            url TEXT UNIQUE,
            summary TEXT,
            published_at TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            collected_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def create_tracked_stocks_table():
    """
    Create the tracked_stocks table if it does not already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracked_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT UNIQUE,
            company_name TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_news(news_item):
    """
    Insert a news item into the database.
    Uses INSERT OR IGNORE to avoid duplicate URLs.
    Stores collected_at in UTC.
    """
    conn = get_connection()
    cursor = conn.cursor()

    collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT OR IGNORE INTO news
        (stock_symbol, company_name, source, title, url, summary, published_at, sentiment_label, sentiment_score, collected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        news_item.get("stock_symbol"),
        news_item.get("company_name"),
        news_item.get("source"),
        news_item.get("title"),
        news_item.get("url"),
        news_item.get("summary"),
        news_item.get("published_at"),
        news_item.get("sentiment_label"),
        news_item.get("sentiment_score"),
        collected_at,
    ))

    conn.commit()
    conn.close()


def get_all_news():
    """
    Retrieve all news records from the database.
    Returns a list of sqlite3.Row objects.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM news ORDER BY collected_at DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows


def get_news_as_dicts():
    """
    Retrieve all news records as a list of dictionaries.
    """
    rows = get_all_news()
    return [dict(row) for row in rows]


def delete_news_by_id(news_id):
    """
    Delete a news item from the database by its ID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM news WHERE id = ?", (news_id,))

    conn.commit()
    conn.close()


def delete_all_news():
    """
    Delete all news records from the database.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM news")

    conn.commit()
    conn.close()


def reset_news_table():
    """
    Delete all rows from the news table and reset the AUTOINCREMENT counter.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM news")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'news'")

    conn.commit()
    conn.close()


def get_tracked_stocks():
    """
    Retrieve all tracked stocks.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT stock_symbol, company_name
        FROM tracked_stocks
        ORDER BY stock_symbol
    """)
    rows = cursor.fetchall()

    conn.close()
    return [dict(row) for row in rows]


def add_tracked_stock(stock_symbol, company_name):
    """
    Add a stock to the tracked stocks table.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO tracked_stocks (stock_symbol, company_name)
        VALUES (?, ?)
    """, (stock_symbol, company_name))

    conn.commit()
    conn.close()


def delete_tracked_stock(stock_symbol):
    """
    Remove a stock from the tracked stocks table by symbol.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM tracked_stocks
        WHERE stock_symbol = ?
    """, (stock_symbol,))

    conn.commit()
    conn.close()

def delete_news_by_stock_symbol(stock_symbol):
    """
    Delete all news records for a given stock symbol.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM news
        WHERE stock_symbol = ?
    """, (stock_symbol,))

    conn.commit()
    conn.close()


def delete_news_by_exact_date(target_date, date_column="published_at"):
    """
    Delete all news records where the selected date column matches the target date.
    target_date format: YYYY-MM-DD
    """
    allowed_columns = {"published_at", "collected_at"}
    if date_column not in allowed_columns:
        raise ValueError(f"Unsupported date column: {date_column}")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        DELETE FROM news
        WHERE substr({date_column}, 1, 10) = ?
    """, (target_date,))

    conn.commit()
    conn.close()


def delete_news_up_to_date(target_date, date_column="published_at"):
    """
    Delete all news records where the selected date column
    is on or before the target date.
    target_date format: YYYY-MM-DD
    """
    allowed_columns = {"published_at", "collected_at"}
    if date_column not in allowed_columns:
        raise ValueError(f"Unsupported date column: {date_column}")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        DELETE FROM news
        WHERE substr({date_column}, 1, 10) <= ?
    """, (target_date,))

    conn.commit()
    conn.close()