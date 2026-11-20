import json
import logging
import re
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app.config import DARGLOBAL_RAW_JSON, RAW_DATA_DIR, WASALT_RAW_JSON

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

USER_AGENT = "DarWasaltPropertyAssistant/1.0 (+https://github.com/dar-wasalt-assistant; property research bot)"
SLEEP_INTERVAL = 1.5
MAX_PAGES_PER_SITE = 15

DISALLOWED_KEYWORDS = (
    "login",
    "signin",
    "signup",
    "auth",
    "account",
    "profile",
    "user",
    "my-account",
    "cart",
    "checkout",
    "admin",
    "password",
)


class RobotsChecker:
    def __init__(self, base_url: str, robots_text: str):
        self.base_url = base_url
        self.robots_text = robots_text
        self.parser = urllib.robotparser.RobotFileParser()
        self.parser.parse(robots_text.splitlines())

        # Extract explicit disallow rules for User-agent: *
        self.disallow_rules = []
        is_all_agents = False
        for line in robots_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                is_all_agents = (agent == "*")
            elif is_all_agents and line.lower().startswith("disallow:"):
                path = line.split(":", 1)[1].strip()
                if path:
                    self.disallow_rules.append(path)

    def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path or "/"

        # Check explicit disallow rules
        for disallowed in self.disallow_rules:
            # Handle wildcard or prefix match
            clean_disallow = disallowed.rstrip("*")
            if clean_disallow and clean_disallow in url:
                return False

        # Check standard robotparser
        if not self.parser.can_fetch(USER_AGENT, url) and not self.parser.can_fetch("*", url):
            return False

        # Respect login-protected, personal, or administrative URLs
        url_lower = url.lower()
        if any(keyword in url_lower for keyword in DISALLOWED_KEYWORDS):
            return False

        # Disallow tracking params
        if "utm_source" in url_lower or "sfmc_" in url_lower:
            return False

        return True


def fetch_robots_txt(base_url: str, session: requests.Session) -> RobotsChecker:
    robots_url = urljoin(base_url, "/robots.txt")
    print(f"\n==================================================")
    print(f"Fetching robots.txt from: {robots_url}")
    print(f"==================================================")
    try:
        resp = session.get(robots_url, timeout=10)
        if resp.status_code == 200:
            print(resp.text.strip())
            print(f"==================================================\n")
            return RobotsChecker(base_url, resp.text)
        else:
            print(f"Failed to fetch robots.txt (status {resp.status_code}). Permitting default crawling.")
            return RobotsChecker(base_url, "User-agent: *\nAllow: /\n")
    except Exception as e:
        print(f"Error fetching robots.txt: {e}. Permitting default crawling.")
        return RobotsChecker(base_url, "User-agent: *\nAllow: /\n")


def parse_next_data(html: str) -> str:
    """Extract readable text fields from Next.js __NEXT_DATA__ JSON script tag."""
    soup = BeautifulSoup(html, "lxml")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return ""

    try:
        data = json.loads(script.string)
    except Exception:
        return ""

    collected = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("title", "description", "name", "body", "heading", "subHeading", "content", "summary", "address", "district", "city"):
                    if isinstance(v, str):
                        cleaned = re.sub(r"\s+", " ", v).strip()
                        if len(cleaned) > 20 and not cleaned.endswith((".jpg", ".png", ".webp", ".svg", ".jpeg")):
                            collected.append(f"{k.capitalize()}: {cleaned}")
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in collected:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return "\n\n".join(unique)


def extract_page_content(html: str, url: str) -> tuple[str, str, str | None]:
    """
    Extracts title and main body text.
    Returns (title, text, skip_reason).
    """
    if "Request unsuccessful. Incapsula incident ID" in html or "_Incapsula_Resource" in html:
        return "", "", "Empty JS shell / blocked by Imperva Incapsula challenge (no __NEXT_DATA__ found)"

    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Remove non-content tags
    for tag in soup(["nav", "footer", "header", "script", "style", "noscript", "svg"]):
        tag.decompose()

    # Extract headings, paragraphs, and list items
    content_blocks = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        txt = re.sub(r"\s+", " ", el.get_text(strip=True))
        if len(txt) >= 20 and txt not in content_blocks:
            content_blocks.append(txt)

    body_text = "\n\n".join(content_blocks).strip()

    # If text is too short, check for Next.js __NEXT_DATA__
    if len(body_text) < 200:
        next_data_text = parse_next_data(html)
        if next_data_text:
            body_text = (body_text + "\n\n" + next_data_text).strip() if body_text else next_data_text

    if not body_text or len(body_text) < 100:
        return title, "", "Empty page or JS shell with no readable content"

    return title, body_text, None


def discover_urls(
    source_name: str,
    start_url: str,
    session: requests.Session,
    robots: RobotsChecker,
    max_pages: int = MAX_PAGES_PER_SITE,
) -> list[str]:
    """Find URLs from sitemap.xml if available; otherwise crawl homepage links 1-level deep."""
    urls = []
    sitemap_url = urljoin(start_url, "/sitemap.xml")
    logger.info("[%s] Checking sitemap at %s", source_name, sitemap_url)

    try:
        resp = session.get(sitemap_url, timeout=10)
        if resp.status_code == 200 and ("<urlset" in resp.text or "<sitemapindex" in resp.text):
            root = ET.fromstring(resp.content)
            for child in root:
                loc = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc is not None and loc.text:
                    u = loc.text.strip()
                    if robots.is_allowed(u):
                        urls.append(u)
            logger.info("[%s] Found %d allowed URLs in sitemap.xml", source_name, len(urls))
    except Exception as e:
        logger.info("[%s] Sitemap check skipped or failed: %s", source_name, e)

    if urls:
        # Include start_url if not already present
        if start_url not in urls and robots.is_allowed(start_url):
            urls.insert(0, start_url)
        return urls[:max_pages]

    # Fallback: crawl internal links from start_url one level deep
    logger.info("[%s] Crawling internal links from %s (one level deep)...", source_name, start_url)
    urls = [start_url]
    try:
        resp = session.get(start_url, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "lxml")
            base_domain = urlparse(start_url).netloc
            for a in soup.find_all("a", href=True):
                full_url = urljoin(start_url, a["href"])
                parsed = urlparse(full_url)
                # Keep same domain and English paths if applicable
                if parsed.netloc == base_domain:
                    # Strip fragment & query
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                    if clean_url and clean_url not in urls and robots.is_allowed(clean_url):
                        # For wasalt, prioritize /en pages
                        if "wasalt.sa" in base_domain and not clean_url.startswith("https://wasalt.sa/en"):
                            continue
                        urls.append(clean_url)
                        if len(urls) >= max_pages:
                            break
    except Exception as e:
        logger.error("[%s] Error crawling homepage links: %s", source_name, e)

    return urls[:max_pages]


def scrape_site(source_name: str, start_url: str, output_file: Path) -> dict:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    robots = fetch_robots_txt(start_url, session)
    candidate_urls = discover_urls(source_name, start_url, session, robots, MAX_PAGES_PER_SITE)

    logger.info("[%s] Candidate URLs to scrape (%d): %s", source_name, len(candidate_urls), candidate_urls)

    scraped_pages = []
    skipped_pages = []

    for idx, url in enumerate(candidate_urls, start=1):
        if not robots.is_allowed(url):
            skipped_pages.append({"url": url, "reason": "Disallowed by robots.txt or sensitive path"})
            logger.info("[%s] [%d/%d] SKIPPED (Robots): %s", source_name, idx, len(candidate_urls), url)
            continue

        logger.info("[%s] [%d/%d] Fetching %s...", source_name, idx, len(candidate_urls), url)
        try:
            resp = session.get(url, timeout=12)
            if resp.status_code != 200:
                reason = f"HTTP {resp.status_code}"
                skipped_pages.append({"url": url, "reason": reason})
                logger.warning("[%s] SKIPPED: %s (%s)", source_name, url, reason)
                time.sleep(SLEEP_INTERVAL)
                continue

            title, text, skip_reason = extract_page_content(resp.text, url)
            if skip_reason:
                skipped_pages.append({"url": url, "reason": skip_reason})
                logger.warning("[%s] SKIPPED: %s (%s)", source_name, url, skip_reason)
            else:
                scraped_pages.append({
                    "source": source_name,
                    "url": url,
                    "title": title or url,
                    "text": text,
                })
                logger.info("[%s] SUCCESS: %s (%d chars)", source_name, url, len(text))
        except Exception as e:
            reason = f"Request exception: {str(e)}"
            skipped_pages.append({"url": url, "reason": reason})
            logger.error("[%s] ERROR: %s (%s)", source_name, url, reason)

        time.sleep(SLEEP_INTERVAL)

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraped_pages, f, indent=2, ensure_ascii=False)

    total_chars = sum(len(p["text"]) for p in scraped_pages)

    return {
        "source": source_name,
        "scraped_count": len(scraped_pages),
        "skipped_count": len(skipped_pages),
        "skipped_details": skipped_pages,
        "total_chars": total_chars,
        "pages": scraped_pages,
    }


def main():
    print("\n" + "=" * 60)
    print("STARTING SCRAPER PIPELINE")
    print("=" * 60)

    # 1. Scrape DarGlobal
    darglobal_stats = scrape_site(
        source_name="darglobal",
        start_url="https://www.darglobal.co.uk",
        output_file=DARGLOBAL_RAW_JSON,
    )

    # 2. Scrape Wasalt
    wasalt_stats = scrape_site(
        source_name="wasalt",
        start_url="https://wasalt.sa/en",
        output_file=WASALT_RAW_JSON,
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SCRAPING STATISTICS SUMMARY")
    print("=" * 60)
    for stats in (darglobal_stats, wasalt_stats):
        source = stats["source"]
        print(f"\n--- Source: {source.upper()} ---")
        if source == "darglobal" and stats["scraped_count"] == 0:
            print("Note: blocked by bot protection; collected manually.")
        print(f"Pages successfully scraped: {stats['scraped_count']}")
        print(f"Pages skipped: {stats['skipped_count']}")
        if stats["skipped_details"]:
            print("Skipped reasons breakdown:")
            reasons = {}
            for item in stats["skipped_details"]:
                reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
            for r, count in reasons.items():
                print(f"  - {r}: {count} pages")

        print(f"Total characters collected: {stats['total_chars']:,}")

        # Show 2 sample text snippets
        print(f"\nSample Snippets for {source.upper()}:")
        if stats["pages"]:
            for i, p in enumerate(stats["pages"][:2], 1):
                snippet = p["text"][:300].replace("\n", " ")
                print(f"  [Sample {i}] {p['title']} ({p['url']}):\n    \"{snippet}...\"\n")
        else:
            print("  (No pages collected via automated scraper; manually collected files loaded during ingestion)\n")


if __name__ == "__main__":
    main()
