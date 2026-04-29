import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import re
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Konfiguration ---
DEFAULT_DELAY = 1.0          # Sekunden zwischen Requests (robots.txt Crawl-Delay hat Vorrang)
MAX_PAGES = 50               # Sicherheits-Limit gegen endlose Crawls
REQUEST_TIMEOUT = 10         # Sekunden bis Timeout
USER_AGENT = "RAG-Crawler/1.0"
DEBUG_OUTPUT_DIR = Path("data/debug")


# ---------------------------------------------------------------------------
# Robots.txt
# ---------------------------------------------------------------------------

def load_robots(base_url: str) -> tuple[dict, float]:
    """
    Parst robots.txt manuell, um Allow/Disallow korrekt nach
    Spezifizität zu gewichten (longest match wins).
    Gibt (rules_dict, crawl_delay) zurück.
    """
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    rules = {"allow": [], "disallow": []}
    delay = DEFAULT_DELAY

    try:
        response = requests.get(robots_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        logger.info(f"robots.txt geladen: {robots_url}")

        active = False  # Gilt der aktuelle Block für uns?
        for line in response.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "user-agent":
                active = value == "*" or value.lower() == USER_AGENT.lower()
            elif active:
                if key == "allow":
                    rules["allow"].append(value)
                elif key == "disallow":
                    rules["disallow"].append(value)
                elif key == "crawl-delay":
                    try:
                        delay = float(value)
                    except ValueError:
                        pass

    except Exception as e:
        logger.warning(f"robots.txt nicht erreichbar ({robots_url}): {e} — fahre ohne Einschränkungen fort")

    logger.info(f"Crawl-Delay: {delay}s | Allow-Regeln: {len(rules['allow'])} | Disallow: {len(rules['disallow'])}")
    return rules, delay


def is_allowed(rules: dict, url: str) -> bool:
    """
    Longest-match-wins: Die spezifischste Regel (längster Pfad) gewinnt.
    Bei gleicher Länge hat Allow Vorrang vor Disallow.
    """
    path = urlparse(url).path

    best_length = -1
    best_allowed = True  # Default: erlaubt

    for pattern in rules["allow"]:
        if path.startswith(pattern) and len(pattern) > best_length:
            best_length = len(pattern)
            best_allowed = True

    for pattern in rules["disallow"]:
        if not pattern:  # leeres Disallow = alles erlaubt
            continue
        if path.startswith(pattern) and len(pattern) > best_length:
            best_length = len(pattern)
            best_allowed = False

    return best_allowed

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def fetch(url: str, session: requests.Session) -> requests.Response | None:
    """
    Führt einen GET-Request durch.
    Gibt None zurück bei Fehler (Netzwerk, Timeout, non-200).
    """
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            logger.debug(f"Überspringe (kein HTML, Content-Type: {content_type}): {url}")
            return None

        return response

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout bei: {url}")
    except requests.exceptions.TooManyRedirects:
        logger.warning(f"Zu viele Redirects bei: {url}")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP-Fehler {e.response.status_code} bei: {url}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Netzwerkfehler bei {url}: {e}")

    return None


# ---------------------------------------------------------------------------
# HTML → Text
# ---------------------------------------------------------------------------

def extract_text(html: str) -> tuple[str, BeautifulSoup]:
    """
    Bereinigt HTML und extrahiert sauberen Text.
    Gibt (clean_text, soup) zurück.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    main = (
        soup.find(id="content")
        or soup.find("main")
        or soup.find("article")
        or soup.find(id="main")
        or soup.body
    )

    raw_text = main.get_text(separator="\n", strip=True) if main else ""

    lines = [line for line in raw_text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    return clean_text, soup


def collect_links(soup: BeautifulSoup, base_url: str, origin: str) -> list[str]:
    """
    Sammelt alle internen Links einer Seite (gleiche Domain wie origin).
    Fragmentlinks (#) und mailto: werden ignoriert.
    """
    parsed_origin = urlparse(origin)
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Nur gleiche Domain, nur http(s)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc != parsed_origin.netloc:
            continue

        # Fragment entfernen
        clean = parsed._replace(fragment="").geturl()
        links.append(clean)

    return links


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def scrape_website(
    start_url: str,
    crawl_subpages: bool = True,
    max_pages: int = MAX_PAGES,
    debug_output: bool = False,
) -> list[Document]:
    """
    Crawlt eine Website ab start_url und gibt alle gefundenen Seiten als
    Liste von LangChain Documents zurück.

    Args:
        start_url:      Einstiegs-URL
        crawl_subpages: Wenn True, werden interne Links rekursiv verfolgt
        max_pages:      Maximale Anzahl Seiten (Sicherheitslimit)
        debug_output:   Wenn True, wird HTML/Text in DEBUG_OUTPUT_DIR gespeichert

    Returns:
        Liste von Documents, bereit für run_indexer(sources=...)
    """
    logger.info(f"Starte Scraping: {start_url} | Subpages: {crawl_subpages} | Limit: {max_pages}")

    rp, crawl_delay = load_robots(start_url)

    if not is_allowed(rp, start_url):
        logger.error(f"robots.txt verbietet das Crawlen von: {start_url}")
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    visited: set[str] = set()
    queue: list[str] = [start_url]
    documents: list[Document] = []

    if debug_output:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    while queue and len(visited) < max_pages:
        url = queue.pop(0)

        if url in visited:
            continue

        if not is_allowed(rp, url):
            logger.info(f"robots.txt: überspringe {url}")
            visited.add(url)
            continue

        logger.info(f"[{len(visited)+1}/{max_pages}] Crawle: {url}")

        response = fetch(url, session)
        visited.add(url)

        if response is None:
            continue

        clean_text, soup = extract_text(response.text)

        if not clean_text.strip():
            logger.warning(f"Kein Text extrahiert: {url}")
            continue

        if len(clean_text.strip()) < 50:
            logger.warning(f"Kaum Text extrahiert ({len(clean_text)} Zeichen): {url}")
            continue

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        doc = Document(
            page_content=clean_text,
            metadata={
                "source": url,
                "title": title,
                "source_type": "web",
            }
        )
        documents.append(doc)

        if debug_output:
            slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
            Path(DEBUG_OUTPUT_DIR / f"{slug}.html").write_text(response.text, encoding="utf-8")
            Path(DEBUG_OUTPUT_DIR / f"{slug}.txt").write_text(clean_text, encoding="utf-8")

        if crawl_subpages:
            new_links = collect_links(soup, url, start_url)
            for link in new_links:
                if link not in visited and link not in queue:
                    queue.append(link)

        if len(visited) < max_pages and queue:
            time.sleep(crawl_delay)

    logger.info(f"Scraping abgeschlossen: {len(documents)} Seiten geladen, {len(visited)} besucht")

    if not documents:
        logger.warning("Keine Dokumente erzeugt — bitte URL und robots.txt prüfen")

    return documents


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    url = sys.argv[1] if len(sys.argv) > 1 else None

    if not url:
        print("Usage: python web.py <url> [--no-subpages] [--debug]")
        sys.exit(1)

    crawl_sub = "--no-subpages" not in sys.argv
    debug = "--debug" in sys.argv

    docs = scrape_website(url, crawl_subpages=crawl_sub, debug_output=debug)

    print(f"\n{len(docs)} Dokumente erzeugt:")
    for d in docs:
        print(f"  [{d.metadata['title'][:50]}] {d.metadata['source']}")