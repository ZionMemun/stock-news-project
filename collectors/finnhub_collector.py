import requests
from datetime import datetime, timedelta, timezone

from config import FINNHUB_API_KEY
from ml.preprocess import is_relevant_to_stock


def fetch_finnhub_news(symbol, company_name):
    """
    Fetch recent company news for a given stock symbol from Finnhub API.
    Return only news items that appear relevant to the stock.
    """
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=2)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "token": FINNHUB_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    results = []

    for item in data:
        title = item.get("headline")
        summary = item.get("summary")

        if not is_relevant_to_stock(title, summary, symbol, company_name):
            continue

        results.append({
            "stock_symbol": symbol,
            "company_name": company_name,
            "source": item.get("source", "Finnhub"),
            "title": title,
            "url": item.get("url"),
            "summary": summary,
            "published_at": convert_timestamp(item.get("datetime")),
            "sentiment_label": None,
            "sentiment_score": None,
        })

    return results


def convert_timestamp(timestamp_value):
    """
    Convert Unix timestamp to a UTC datetime string.
    """
    if not timestamp_value:
        return None

    return datetime.fromtimestamp(timestamp_value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")