import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import pandas as pd
import time

BASE_URL = "https://www.dawn.com/business"

START_DATE = date(2025, 1, 1)
END_DATE = date(2026, 8, 12)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0 Safari/537.36"
    )
}


def scrape_day(day):

    url = f"{BASE_URL}/{day.strftime('%Y-%m-%d')}"

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    articles = []

    # Each article is represented by an h2 heading/link
    for heading in soup.select("h2"):

        link = heading.find("a")

        if not link:
            continue

        title = link.get_text(" ", strip=True)
        article_url = link.get("href")

        if not title or not article_url:
            continue

        if article_url.startswith("/"):
            article_url = "https://www.dawn.com" + article_url

        # Find the text/time associated with the article
        parent = heading.parent

        text = parent.get_text(" ", strip=True)

        articles.append({
            "date": day.strftime("%Y-%m-%d"),
            "title": title,
            "url": article_url,
            "raw_text": text
        })

    return articles


all_articles = []

current_date = START_DATE

while current_date <= END_DATE:

    print(f"Scraping {current_date}...")

    try:
        results = scrape_day(current_date)

        print(f"  Found {len(results)} articles")

        all_articles.extend(results)

    except Exception as e:
        print(f"  ERROR: {e}")

    current_date += timedelta(days=1)

    # Don't hammer Dawn
    time.sleep(1)


df = pd.DataFrame(all_articles)

# Remove duplicates
df = df.drop_duplicates(subset=["url"])

# Sort chronologically
df = df.sort_values(
    by=["date", "title"]
)

df.to_csv(
    "dawn_business_2025_to_2026.csv",
    index=False,
    encoding="utf-8-sig"
)

df.to_excel(
    "dawn_business_2025_to_2026.xlsx",
    index=False
)

print("\nDONE")
print("Total articles:", len(df))