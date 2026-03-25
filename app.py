from database.db import (
    create_table,
    create_tracked_stocks_table,
    insert_news,
    get_all_news,
    get_tracked_stocks,
)
from collectors.finnhub_collector import fetch_finnhub_news
from collectors.google_rss import fetch_google_news
from ml.preprocess import filter_recent_news
from ml.predict import SentimentPredictor


def deduplicate(news_list):
    """
    Remove duplicate news items by URL.
    """
    seen = set()
    unique = []

    for item in news_list:
        url = item.get("url")
        if url and url not in seen:
            seen.add(url)
            unique.append(item)

    return unique


def enrich_with_sentiment(predictor, news_item):
    """
    Add sentiment prediction to one news item.
    """
    result = predictor.predict(
        title=news_item.get("title"),
        summary=news_item.get("summary"),
    )

    news_item["sentiment_label"] = result["sentiment_label"]
    news_item["sentiment_score"] = result["sentiment_score"]

    return news_item


def main():
    create_table()
    create_tracked_stocks_table()

    predictor = SentimentPredictor()

    all_news = []
    tracked_stocks = get_tracked_stocks()

    print("Tracked stocks:", tracked_stocks)

    if not tracked_stocks:
        print("No tracked stocks found. Nothing to collect.")
        return

    for stock in tracked_stocks:
        symbol = stock["stock_symbol"]
        company_name = stock["company_name"]

        finnhub_news = []
        google_news = []

        try:
            finnhub_news = fetch_finnhub_news(symbol, company_name)
            print(f"Finnhub returned {len(finnhub_news)} items for {symbol}")
        except Exception as exc:
            print(f"Finnhub failed for {symbol}: {exc}")

        try:
            google_news = fetch_google_news(symbol, company_name)
            print(f"Google RSS returned {len(google_news)} items for {symbol}")
        except Exception as exc:
            print(f"Google RSS failed for {symbol}: {exc}")

        all_news.extend(finnhub_news)
        all_news.extend(google_news)

    print(f"Total before time filter: {len(all_news)}")

    all_news = filter_recent_news(all_news, hours_back=3)
    print(f"Total after time filter: {len(all_news)}")

    all_news = deduplicate(all_news)
    print(f"Total after deduplication: {len(all_news)}")

    for news_item in all_news:
        news_item = enrich_with_sentiment(predictor, news_item)
        insert_news(news_item)

    rows = get_all_news()

    print(f"Total rows in DB: {len(rows)}")
    print("Sample rows:")

    for row in rows[:10]:
        print(dict(row))


if __name__ == "__main__":
    main()