# from database.db import (
#     create_table,
#     create_tracked_stocks_table,
#     insert_news,
#     get_all_news,
#     get_tracked_stocks,
# )
# from collectors.finnhub_collector import fetch_finnhub_news
# from collectors.google_rss import fetch_google_news
# from ml.preprocess import filter_recent_news
# from ml.predict import SentimentPredictor
#
#
# def deduplicate(news_list):
#     """
#     Remove duplicate news items by URL.
#     """
#     seen = set()
#     unique = []
#
#     for item in news_list:
#         url = item.get("url")
#         if url and url not in seen:
#             seen.add(url)
#             unique.append(item)
#
#     return unique
#
#
# def enrich_with_sentiment(predictor, news_item):
#     """
#     Add sentiment prediction to one news item.
#     """
#     result = predictor.predict(
#         title=news_item.get("title"),
#         summary=news_item.get("summary"),
#     )
#
#     news_item["sentiment_label"] = result["sentiment_label"]
#     news_item["sentiment_score"] = result["sentiment_score"]
#
#     return news_item
#
#
# def main(hours_back=24):
#     """
#     Collect news from all tracked stocks, keep only articles from the last
#     `hours_back` hours, enrich them with sentiment, and insert them into the DB.
#
#     Returns a summary dictionary for the dashboard.
#     """
#     create_table()
#     create_tracked_stocks_table()
#
#     predictor = SentimentPredictor()
#
#     all_news = []
#     tracked_stocks = get_tracked_stocks()
#
#     print("Tracked stocks:", tracked_stocks)
#
#     if not tracked_stocks:
#         print("No tracked stocks found. Nothing to collect.")
#         return {
#             "success": False,
#             "message": "No tracked stocks found. Nothing to collect.",
#             "tracked_stocks": 0,
#             "before_filter": 0,
#             "after_filter": 0,
#             "after_dedup": 0,
#             "inserted": 0,
#             "total_rows": 0,
#         }
#
#     for stock in tracked_stocks:
#         symbol = stock["stock_symbol"]
#         company_name = stock["company_name"]
#
#         finnhub_news = []
#         google_news = []
#
#         try:
#             finnhub_news = fetch_finnhub_news(symbol, company_name)
#             print(f"Finnhub returned {len(finnhub_news)} items for {symbol}")
#         except Exception as exc:
#             print(f"Finnhub failed for {symbol}: {exc}")
#
#         try:
#             google_news = fetch_google_news(symbol, company_name)
#             print(f"Google RSS returned {len(google_news)} items for {symbol}")
#         except Exception as exc:
#             print(f"Google RSS failed for {symbol}: {exc}")
#
#         all_news.extend(finnhub_news)
#         all_news.extend(google_news)
#
#     before_filter = len(all_news)
#     print(f"Total before time filter: {before_filter}")
#
#     all_news = filter_recent_news(all_news, hours_back=hours_back)
#     after_filter = len(all_news)
#     print(f"Total after time filter ({hours_back}h): {after_filter}")
#
#     all_news = deduplicate(all_news)
#     after_dedup = len(all_news)
#     print(f"Total after deduplication: {after_dedup}")
#
#     inserted = 0
#     for news_item in all_news:
#         news_item = enrich_with_sentiment(predictor, news_item)
#         insert_news(news_item)
#         inserted += 1
#
#     rows = get_all_news()
#     total_rows = len(rows)
#
#     print(f"Total rows in DB: {total_rows}")
#     print("Sample rows:")
#
#     for row in rows[:10]:
#         print(dict(row))
#
#     return {
#         "success": True,
#         "message": f"Collection finished successfully for the last {hours_back} hours.",
#         "tracked_stocks": len(tracked_stocks),
#         "before_filter": before_filter,
#         "after_filter": after_filter,
#         "after_dedup": after_dedup,
#         "inserted": inserted,
#         "total_rows": total_rows,
#     }
#
#
# if __name__ == "__main__":
#     result = main(hours_back=24)
#     print(result["message"])

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
    seen = set()   # Stores URLs we have already encountered
    unique = []

    for item in news_list:
        url = item.get("url")  # Safely get the article URL
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

    # Store the predicted sentiment label and confidence score inside the news item
    news_item["sentiment_label"] = result["sentiment_label"]
    news_item["sentiment_score"] = result["sentiment_score"]

    return news_item


def main(hours_back=24):
    """
    Collect news from all tracked stocks, keep only articles from the last
    `hours_back` hours, enrich them with sentiment, and insert them into the DB.

    Returns a summary dictionary for the dashboard.
    """
    create_table()
    create_tracked_stocks_table()

    predictor = SentimentPredictor()

    all_news = []
    tracked_stocks = get_tracked_stocks()

    print("Tracked stocks:", tracked_stocks)

    if not tracked_stocks:
        print("No tracked stocks found. Nothing to collect.")
        return {
            "success": False,
            "message": "No tracked stocks found. Nothing to collect.",
            "tracked_stocks": 0,
            "before_filter": 0,
            "after_filter": 0,
            "after_dedup": 0,
            "inserted": 0,
            "total_rows": 0,
        }

    for stock in tracked_stocks:
        symbol = stock["stock_symbol"]
        company_name = stock["company_name"]

        finnhub_news = []
        google_news = []

        try:
            finnhub_news = fetch_finnhub_news(symbol, company_name)
            print(f"Finnhub returned {len(finnhub_news)} items for {symbol}")
        except Exception as exc:
            # Keep the pipeline running even if one source fails
            print(f"Finnhub failed for {symbol}: {exc}")

        try:
            google_news = fetch_google_news(symbol, company_name)
            print(f"Google RSS returned {len(google_news)} items for {symbol}")
        except Exception as exc:
            # Keep the pipeline running even if one source fails
            print(f"Google RSS failed for {symbol}: {exc}")

        # Merge results from both news sources into one combined list
        all_news.extend(finnhub_news)
        all_news.extend(google_news)

    before_filter = len(all_news)
    print(f"Total before time filter: {before_filter}")

    # Keep only articles published within the selected time window
    all_news = filter_recent_news(all_news, hours_back=hours_back)
    after_filter = len(all_news)
    print(f"Total after time filter ({hours_back}h): {after_filter}")

    # Remove duplicate articles across sources based on URL
    all_news = deduplicate(all_news)
    after_dedup = len(all_news)
    print(f"Total after deduplication: {after_dedup}")

    inserted = 0
    for news_item in all_news:
        news_item = enrich_with_sentiment(predictor, news_item)
        insert_news(news_item)  # Insert into DB (duplicates may still be blocked at DB level)
        inserted += 1

    rows = get_all_news()
    total_rows = len(rows)

    print(f"Total rows in DB: {total_rows}")
    print("Sample rows:")

    # Print only the first 10 rows for quick debugging / inspection
    for row in rows[:10]:
        print(dict(row))  # Convert sqlite row object into a regular dictionary

    return {
        "success": True,
        "message": f"Collection finished successfully for the last {hours_back} hours.",
        "tracked_stocks": len(tracked_stocks),
        "before_filter": before_filter,
        "after_filter": after_filter,
        "after_dedup": after_dedup,
        "inserted": inserted,
        "total_rows": total_rows,
    }


if __name__ == "__main__":
    # Run the pipeline directly when this file is executed as a script
    result = main(hours_back=24)
    print(result["message"])