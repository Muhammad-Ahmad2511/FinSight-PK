import requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
import pandas as pd
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.dawn.com/business"

# === 11 YEARS PRODUCTION DATES ===
START_DATE = date(2015, 1, 1)
END_DATE = date(2026, 8, 18)

OUTPUT_CSV = "dawn_business_2015_to_2026.csv"
# MAX_WORKERS = 1 for safe sequential scraping (Cloudflare block-free)
MAX_WORKERS = 1
REQUEST_TIMEOUT = 30
RETRIES = 2

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}

_thread_local = threading.local()

def get_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session

def get_article_date(heading):
    container = heading.find_parent("article") or heading.parent
    if container is None:
        return None
    time_tag = container.find("time")
    if not time_tag or not time_tag.get("datetime"):
        return None
    try:
        return datetime.fromisoformat(time_tag["datetime"]).date()
    except ValueError:
        return None

def scrape_day(day):
    url = f"{BASE_URL}/{day.strftime('%Y-%m-%d')}"
    session = get_session()

    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise last_err

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []
    discarded = 0
    
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

        actual_date = get_article_date(heading)
        if actual_date is None or actual_date != day:
            discarded += 1
            continue

        # --- Anti-Block Delay & Paragraph Extractor ---
        article_text = ""
        try:
            time.sleep(random.uniform(2.5, 5.0))
            art_resp = session.get(article_url, timeout=REQUEST_TIMEOUT)
            
            if art_resp.status_code == 200:
                art_soup = BeautifulSoup(art_resp.text, "html.parser")
                paragraphs = art_soup.find_all("p")
                
                # FIXED: Loop indented inside status_code == 200 block
                for p in paragraphs:
                    p_text = p.get_text(" ", strip=True)
                    
                    # Skip bullet summaries
                    if p_text.startswith(("•", "-")):
                        continue
                        
                    if len(p_text) > 75 and not p_text.lower().startswith(("published in dawn", "email", "your name")):
                        article_text = p_text
                        break
            else:
                print(f"    [Blocked/Error] HTTP {art_resp.status_code} on URL: {article_url}")
                
        except Exception as e:
            print(f"    [Error] Failed to load {article_url}: {e}")

        # Fallback to Title if no lead paragraph found
        if not article_text:
            article_text = title

        articles.append({
            "date": day.strftime("%Y-%m-%d"),
            "title": title,
            "url": article_url,
            "raw_text": article_text
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

def append_to_csv(df_chunk):
    header = not _file_has_content(OUTPUT_CSV)
    df_chunk.to_csv(OUTPUT_CSV, mode="a", index=False, header=header, encoding="utf-8-sig")

def _file_has_content(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return bool(f.readline())
    except FileNotFoundError:
        return False

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
            append_to_csv(df_chunk)
            done_dates.update(df_chunk["date"].unique())
            print(f"  Saved {len(df_chunk)} rows -> {OUTPUT_CSV}")

    print("\nALL MONTHS COMPLETED!")
    try:
        final = pd.read_csv(OUTPUT_CSV)
        final = final.drop_duplicates(subset=["url"])
        final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print("Total final articles scraped:", len(final))
    except pd.errors.EmptyDataError:
        print("No articles were successfully scraped.")

if __name__ == "__main__":
    main()