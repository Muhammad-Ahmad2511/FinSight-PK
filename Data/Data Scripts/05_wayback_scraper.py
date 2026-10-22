import re
import time
import random
import threading
import unicodedata
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.dawn.com/business"

# === DATE RANGE (change these for each run) ===
START_DATE = date(2014, 1, 1)
END_DATE = date(2014, 12, 31)
OUTPUT_CSV = "dawn_business_2014.csv"

MAX_WORKERS = 1
REQUEST_TIMEOUT = 30
RETRIES = 3
SLEEP_MIN, SLEEP_MAX = 2.5, 5.0   # delay between article requests (avoids 429)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}

_thread_local = threading.local()


def remove_hidden_chars(text):
    if not isinstance(text, str):
        return text
    return ''.join(char for char in text if unicodedata.category(char) not in ['Cf', 'Cc'])


def get_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def get_article_id(url):
    m = re.search(r"/news/(\d+)", url)
    return int(m.group(1)) if m else None


def extract_lead_paragraph(session, article_url):
    """Returns (lead_text, published_date). Either can be empty/None."""
    from datetime import datetime

    for attempt in range(1, RETRIES + 1):
        try:
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
            art_resp = session.get(article_url, timeout=REQUEST_TIMEOUT)
            print(f"    [Attempt {attempt}] {article_url} -> HTTP {art_resp.status_code}")

            if art_resp.status_code == 200:
                art_soup = BeautifulSoup(art_resp.text, "html.parser")

                # publish date from the article page
                pub_date = None
                t = art_soup.find("time", class_="timestamp--date") or art_soup.find("time")
                if t and t.get("datetime"):
                    try:
                        pub_date = datetime.fromisoformat(t["datetime"].replace("Z", "+00:00")).date()
                    except ValueError:
                        pub_date = None

                story_div = art_soup.find("div", class_="story__content")
                if story_div is None:
                    print(f"    [WARNING] story__content NOT FOUND for {article_url}")
                    story_div = art_soup

                for p in story_div.find_all("p"):
                    p_text = p.get_text(" ", strip=True)
                    if p_text.startswith(("•", "-", "Published in Dawn", "Email", "Your Name")):
                        continue
                    if len(p_text) > 60 and not p_text.lower().startswith(
                        ("published in dawn", "email", "your name", "also read:")
                    ):
                        return p_text, pub_date
                return "", pub_date

            elif art_resp.status_code == 429:
                wait = 30 * attempt
                print(f"    [429 rate limited] sleeping {wait}s")
                time.sleep(wait)

            elif art_resp.status_code in (404, 403):
                print(f"    [HARD FAIL {art_resp.status_code}] {article_url}")
                break

        except Exception as e:
            print(f"    [EXCEPTION attempt {attempt}] {article_url}: {e}")
            time.sleep(1.5 * attempt)

    return "", None


def scrape_day(day):
    url = f"{BASE_URL}/{day.strftime('%Y-%m-%d')}"
    session = get_session()

    response = None
    for attempt in range(1, 6):   # 5 attempts
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                break
            wait = 20 * attempt   # 20s, 40s, 60s, 80s, 100s
            print(f"    [LISTING HTTP {response.status_code}] {url} -> waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"    [LISTING EXCEPTION] {url}: {e}")
            time.sleep(10 * attempt)

    if response is None or response.status_code != 200:
        raise RuntimeError(f"Listing page failed for {day}: "
                           f"{response.status_code if response is not None else 'no response'}")

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []
    discarded = 0

    # Step 1: collect all candidate article links from the page
    candidates = []
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
        if "www.dawn.com/news/" not in article_url:
            discarded += 1
            continue
        candidates.append((heading, title, article_url))

    # Smallest article ID on the page = the real articles of that day.
    # Sidebar/trending links are much newer (much bigger IDs).
    ids = [get_article_id(u) for _, _, u in candidates if get_article_id(u)]
    min_id = min(ids) if ids else None

    # Step 2: filter the candidates, then scrape the ones that pass
    seen_urls = set()
    window = 50000 if day.year < 2020 else 3000

    for heading, title, article_url in candidates:
        if article_url in seen_urls:
            discarded += 1
            continue
        seen_urls.add(article_url)

        # pre-filter by ID (skips sidebar items)
        aid = get_article_id(article_url)
        if aid is None or min_id is None or aid > min_id + window:
            discarded += 1
            continue

        lead_text, pub_date = extract_lead_paragraph(session, article_url)

        # article's own publish date (only checked if the page has one)
        if pub_date is not None and pub_date != day:
            discarded += 1
            continue

        raw_text = lead_text if lead_text else title

        articles.append({
            "date": day.strftime("%Y-%m-%d"),
            "title": title,
            "url": article_url,
            "raw_text": raw_text,
            "lead_found": bool(lead_text),
        })

    if discarded:
        print(f"    -> discarded {discarded} mismatched/undated headings")

    return articles


def month_chunks(start, end):
    chunk = []
    current_month = (start.year, start.month)
    d = start
    while d <= end:
        if (d.year, d.month) != current_month:
            yield chunk
            chunk = []
            current_month = (d.year, d.month)
        chunk.append(d)
        d += timedelta(days=1)
    if chunk:
        yield chunk


def load_done_dates():
    try:
        existing = pd.read_csv(OUTPUT_CSV, usecols=["date"])
        return set(existing["date"].unique())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return set()


def _file_has_content(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return bool(f.readline())
    except FileNotFoundError:
        return False


def append_to_csv(df_chunk):
    header = not _file_has_content(OUTPUT_CSV)
    df_chunk.to_csv(OUTPUT_CSV, mode="a", index=False, header=header, encoding="utf-8-sig")


def scrape_month(days_in_month, done_dates):
    days_to_scrape = [d for d in days_in_month if d.strftime("%Y-%m-%d") not in done_dates]

    if not days_to_scrape:
        print(f"  Skipping {days_in_month[0].strftime('%Y-%m')} (already done)")
        return []

    month_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_day = {executor.submit(scrape_day, d): d for d in days_to_scrape}
        for future in as_completed(future_to_day):
            day = future_to_day[future]
            try:
                results = future.result()
                month_results.extend(results)
                print(f"  {day}: {len(results)} verified articles")
            except Exception as e:
                print(f"  {day}: ERROR - {e}")

    return month_results


def main():
    done_dates = load_done_dates()
    if done_dates:
        print(f"Resuming — {len(done_dates)} dates already scraped in {OUTPUT_CSV}")

    for chunk in month_chunks(START_DATE, END_DATE):
        month_label = chunk[0].strftime("%Y-%m")
        print(f"\n=== Month {month_label} ===")

        results = scrape_month(chunk, done_dates)

        if results:
            df_chunk = pd.DataFrame(results)
            df_chunk = df_chunk.drop_duplicates(subset=["url"])
            df_chunk['raw_text'] = df_chunk['raw_text'].apply(remove_hidden_chars)
            df_chunk['title'] = df_chunk['title'].apply(remove_hidden_chars)
            append_to_csv(df_chunk)
            done_dates.update(df_chunk["date"].unique())
            print(f"  Saved {len(df_chunk)} rows -> {OUTPUT_CSV}")

    print("\nALL MONTHS COMPLETED!")
    try:
        final = pd.read_csv(OUTPUT_CSV)
        final = final.drop_duplicates(subset=["url"])
        final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print("Total final articles scraped:", len(final))
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("No articles were successfully scraped.")


if __name__ == "__main__":
    main()