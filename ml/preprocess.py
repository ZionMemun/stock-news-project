import re
from datetime import datetime, timedelta


def normalize_text(text):
    """
    Normalize text for matching:
    - lowercase
    - strip extra spaces
    """
    if not text:
        return ""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_relevant_to_stock(title, summary, symbol, company_name):
    """
    Check whether a news item is relevant to a stock
    using only the stock symbol and company name.
    """
    combined_text = normalize_text(f"{title or ''} {summary or ''}")
    symbol = normalize_text(symbol)
    company_name = normalize_text(company_name)

    symbol_match = symbol in combined_text if symbol else False
    company_match = company_name in combined_text if company_name else False

    return symbol_match or company_match


def build_model_input(title, summary):
    """
    Build the input text for the sentiment model.
    Use title + summary when summary exists,
    otherwise use title only.
    """
    title = (title or "").strip()
    summary = (summary or "").strip()

    if summary:
        return f"{title} {summary}"

    return title


def filter_recent_news(news_items, hours_back=2):
    """
    Keep only news items published within the last `hours_back` hours.
    """
    cutoff = datetime.now() - timedelta(hours=hours_back)
    filtered_items = []

    for item in news_items:
        published_at = item.get("published_at")
        if not published_at:
            continue

        try:
            published_dt = datetime.strptime(published_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        if published_dt >= cutoff:
            filtered_items.append(item)

    return filtered_items