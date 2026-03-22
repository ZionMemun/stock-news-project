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

def get_news_as_dicts():
    """
    Retrieve all news records as a list of dictionaries.
    """
    rows = get_all_news()
    return [dict(row) for row in rows]


def get_unique_values(column_name):
    """
    Retrieve distinct non-null values from a given column.
    """
    allowed_columns = {"stock_symbol", "company_name", "source", "sentiment_label"}
    if column_name not in allowed_columns:
        raise ValueError(f"Unsupported column: {column_name}")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"SELECT DISTINCT {column_name} FROM news WHERE {column_name} IS NOT NULL ORDER BY {column_name}"
    )
    values = [row[0] for row in cursor.fetchall()]

    conn.close()
    return values