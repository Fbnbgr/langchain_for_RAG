import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_DELAY = 1.0
MAX_PAGES = 100
REQUEST_TIMEOUT = 10
USER_AGENT = "RAG-Crawler/1.0"
DEBUG_OUTPUT_DIR = Path("data/debug")
PDF_DOWNLOAD_DIR = Path("data/pdfs")


# ---------------------------------------------------------------------------
# Robots.txt 
# ---------------------------------------------------------------------------

def load_robots(base_url: str) -> tuple[dict, float]:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rules = {"allow": [], "disallow": []}
    delay = DEFAULT_DELAY

    try:
        response = requests.get(robots_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        logger.info(f"robots.txt geladen: {robots_url}")

        active = False
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

    logger.info(f"Crawl-Delay: {delay}s | Allow: {len(rules['allow'])} | Disallow: {len(rules['disallow'])}")
    return rules, delay


def is_allowed(rules: dict, url: str) -> bool:
    path = urlparse(url).path
    best_length = -1
    best_allowed = True

    for pattern in rules["allow"]:
        if path.startswith(pattern) and len(pattern) > best_length:
            best_length = len(pattern)
            best_allowed = True

    for pattern in rules["disallow"]:
        if not pattern:
            continue
        if path.startswith(pattern) and len(pattern) > best_length:
            best_length = len(pattern)
            best_allowed = False

    return best_allowed


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def fetch(url: str, session: requests.Session) -> requests.Response | None:
    """Lädt eine URL. Gibt None zurück bei Fehler."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
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
# PDF 
# ---------------------------------------------------------------------------

def is_pdf_url(url: str, response: requests.Response) -> bool:
    """Erkennt PDFs anhand URL-Endung ODER Content-Type."""
    by_url = urlparse(url).path.lower().endswith(".pdf")
    by_header = "application/pdf" in response.headers.get("Content-Type", "")
    return by_url or by_header


def download_pdf(url: str, response: requests.Response, download_dir: Path) -> Document | None:
    """
    Speichert ein bereits geladenes PDF auf Disk.
    Gibt ein Document zurück, dessen page_content leer ist –
    dein bestehender PDF-Workflow füllt den Inhalt später.
    Das 'source'-Feld zeigt auf den lokalen Pfad.
    """
    download_dir.mkdir(parents=True, exist_ok=True)

    # Dateiname aus URL ableiten, Kollisionen durch Präfix vermeiden
    filename = Path(urlparse(url).path).name or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    # Kollisionen auflösen: falls Datei schon existiert, Zähler anhängen
    target = download_dir / filename
    counter = 1
    while target.exists():
        stem = Path(filename).stem
        target = download_dir / f"{stem}_{counter}.pdf"
        counter += 1

    try:
        target.write_bytes(response.content)
        logger.info(f"PDF gespeichert: {target} (Quelle: {url})")
    except OSError as e:
        logger.error(f"PDF konnte nicht gespeichert werden ({target}): {e}")
        return None

    return Document(
        page_content="",      
        metadata={
            "source": str(target),
            "source_url": url,  
            "source_type": "pdf",
            "title": Path(filename).stem.replace("_", " "),
        },
    )


# ---------------------------------------------------------------------------
# HTML → Text  
# ---------------------------------------------------------------------------

def extract_text(html: str) -> tuple[str, BeautifulSoup]:
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
    return "\n".join(lines), soup


def collect_links(soup: BeautifulSoup, base_url: str, origin: str) -> list[str]:
    parsed_origin = urlparse(origin)
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc != parsed_origin.netloc:
            continue

        links.append(parsed._replace(fragment="").geturl())

    return links


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def scrape_website(
    start_url: str,
    crawl_subpages: bool = True,
    max_pages: int = MAX_PAGES,
    download_pdfs: bool = True,     
    pdf_download_dir: Path = PDF_DOWNLOAD_DIR, 
    debug_output: bool = False,
) -> list[Document]:
    """
    Crawlt eine Website und gibt HTML-Seiten + heruntergeladene PDFs als
    LangChain Documents zurück.

    PDF-Documents haben page_content="" und source_type="pdf".
    Dein bestehender PDF-Workflow kann sie direkt weiterverarbeiten,
    indem er alle docs mit source_type=="pdf" nach source (lokalem Pfad) lädt.
    """
    logger.info(
        f"Starte Scraping: {start_url} | Subpages: {crawl_subpages} "
        f"| PDFs: {download_pdfs} | Limit: {max_pages}"
    )

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

        content_type = response.headers.get("Content-Type", "")

        # ── PDF-Zweig ──────────────────────────────────────────────────────
        if download_pdfs and is_pdf_url(url, response):
            doc = download_pdf(url, response, pdf_download_dir)
            if doc:
                documents.append(doc)
            # PDFs haben keine Links zum Verfolgen → weiter
            if len(visited) < max_pages and queue:
                time.sleep(crawl_delay)
            continue

        # ── HTML-Zweig ─────────────────────────────────────────────────────
        if "text/html" not in content_type:
            logger.debug(f"Überspringe (Content-Type: {content_type}): {url}")
            continue

        clean_text, soup = extract_text(response.text)

        if not clean_text.strip():
            logger.warning(f"Kein Text extrahiert: {url}")
            continue
        if len(clean_text.strip()) < 50:
            logger.warning(f"Kaum Text ({len(clean_text)} Zeichen): {url}")
            continue

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        documents.append(Document(
            page_content=clean_text,
            metadata={"source": url, "title": title, "source_type": "web"},
        ))

        if debug_output:
            slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
            Path(DEBUG_OUTPUT_DIR / f"{slug}.html").write_text(response.text, encoding="utf-8")
            Path(DEBUG_OUTPUT_DIR / f"{slug}.txt").write_text(clean_text, encoding="utf-8")

        if crawl_subpages:
            for link in collect_links(soup, url, start_url):
                if link not in visited and link not in queue:
                    queue.append(link)

        if len(visited) < max_pages and queue:
            time.sleep(crawl_delay)

    logger.info(f"Abgeschlossen: {len(documents)} Dokumente | {len(visited)} besucht")
    if not documents:
        logger.warning("Keine Dokumente erzeugt — URL und robots.txt prüfen")

    return documents


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    url = sys.argv[1] if len(sys.argv) > 1 else None
    if not url:
        print("Usage: python web.py <url> [--no-subpages] [--no-pdfs] [--debug]")
        sys.exit(1)

    docs = scrape_website(
        url,
        crawl_subpages="--no-subpages" not in sys.argv,
        download_pdfs="--no-pdfs" not in sys.argv,
        debug_output="--debug" in sys.argv,
    )

    pdf_docs = [d for d in docs if d.metadata["source_type"] == "pdf"]
    web_docs = [d for d in docs if d.metadata["source_type"] == "web"]

    print(f"\n{len(docs)} Dokumente gesamt ({len(web_docs)} HTML, {len(pdf_docs)} PDF):")
    for d in docs:
        tag = "[PDF]" if d.metadata["source_type"] == "pdf" else "[WEB]"
        title = d.metadata.get("title", "")[:40]
        print(f"  {tag} {title or '—'} → {d.metadata['source']}")