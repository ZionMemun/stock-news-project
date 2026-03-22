import re
import html
import feedparser
from datetime import datetime
from urllib.parse import quote_plus

from ml.preprocess import is_relevant_to_stock


def fetch_google_news(symbol, company_name):
    """
    Fetch news from Google News RSS for a given stock symbol.
    Return only news items that appear relevant to the stock.
    """
    query = quote_plus(f"{symbol} {company_name} stock")
    url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(url)

    results = []

    for entry in feed.entries:
        title = clean_html(entry.get("title"))
        summary = clean_html(entry.get("summary", ""))

        if not is_relevant_to_stock(title, summary, symbol, company_name):
            continue

        results.append({
            "stock_symbol": symbol,
            "company_name": company_name,
            "source": extract_source(entry),
            "title": title,
            "url": entry.get("link"),
            "summary": summary,
            "published_at": parse_date(entry.get("published")),
            "sentiment_label": None,
            "sentiment_score": None,
        })

    return results


def extract_source(entry):
    """
    Extract source name from RSS entry.
    """
    source = entry.get("source")
    if source and isinstance(source, dict):
        return source.get("title", "Google News")

    return "Google News"


def parse_date(date_str):
    """
    Convert RSS date string to a UTC datetime string.
    """
    if not date_str:
        return None

    try:
        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def clean_html(text):
    """
    Remove HTML tags and decode HTML entities.
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text