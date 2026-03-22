# from database.db import create_table, insert_news, get_all_news
# from collectors.finnhub_collector import fetch_finnhub_news
# from collectors.google_rss import fetch_google_news
# from config import STOCKS
# from ml.preprocess import filter_recent_news
#
#
# def deduplicate(news_list):
#     """
#     Remove duplicate news items based on URL.
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
# def main():
#     """
#     Fetch news from all configured sources, keep only recent news,
#     remove duplicates, save results to the database, and print a sample.
#     """
#     create_table()
#
#     all_news = []
#
#     for stock in STOCKS:
#         symbol = stock["symbol"]
#         company_name = stock["company_name"]
#
#         try:
#             all_news.extend(fetch_finnhub_news(symbol, company_name))
#         except Exception as exc:
#             print(f"Finnhub failed for {symbol}: {exc}")
#
#         try:
#             all_news.extend(fetch_google_news(symbol, company_name))
#         except Exception as exc:
#             print(f"Google RSS failed for {symbol}: {exc}")
#
#     all_news = filter_recent_news(all_news, hours_back=2)
#     all_news = deduplicate(all_news)
#
#     for news_item in all_news:
#         insert_news(news_item)
#
#     rows = get_all_news()
#
#     print(f"Total rows in DB: {len(rows)}")
#     print("Sample rows:")
#
#     for row in rows[:10]:
#         print(dict(row))
#
#
# if __name__ == "__main__":
#     main()

from database.db import create_table, insert_news, get_all_news
from collectors.finnhub_collector import fetch_finnhub_news
from collectors.google_rss import fetch_google_news
from config import STOCKS
from ml.preprocess import filter_recent_news


def deduplicate(news_list):
    """
    Remove duplicate news items based on URL.
    """
    seen = set()
    unique = []

    for item in news_list:
        url = item.get("url")
        if url and url not in seen:
            seen.add(url)
            unique.append(item)

    return unique


def main():
    create_table()

    all_news = []

    for stock in STOCKS:
        symbol = stock["symbol"]
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

    all_news = filter_recent_news(all_news, hours_back=48)
    print(f"Total after time filter: {len(all_news)}")

    all_news = deduplicate(all_news)
    print(f"Total after deduplication: {len(all_news)}")

    for news_item in all_news:
        insert_news(news_item)

    rows = get_all_news()

    print(f"Total rows in DB: {len(rows)}")
    print("Sample rows:")

    for row in rows[:10]:
        print(dict(row))


if __name__ == "__main__":
    main()