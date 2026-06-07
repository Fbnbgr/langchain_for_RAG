import logging
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from firecrawl import Firecrawl
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
MAX_PAGES = int(os.getenv("MAX_PAGES", 100))
DEBUG_OUTPUT_DIR = Path(os.getenv("DEBUG_OUTPUT_DIR", "data/debug"))


def get_firecrawl_client() -> Firecrawl:
    if not FIRECRAWL_API_KEY:
        raise RuntimeError(
            "FIRECRAWL_API_KEY ist nicht gesetzt. Bitte in der Umgebung definieren."
        )
    return Firecrawl(api_key=FIRECRAWL_API_KEY)


def _to_dict(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    return {k: getattr(metadata, k) for k in getattr(metadata, "__dict__", {})}


def _extract_document_fields(raw_doc: Any) -> tuple[str, str, str, dict[str, Any]]:
    if isinstance(raw_doc, dict):
        metadata = raw_doc.get("metadata", {}) or {}
        markdown = raw_doc.get("markdown") or raw_doc.get("text") or ""
        source_url = (
            metadata.get("source_url")
            or metadata.get("sourceURL")
            or metadata.get("url")
            or raw_doc.get("url", "")
        )
        title = metadata.get("title", "")
    else:
        metadata = _to_dict(getattr(raw_doc, "metadata", {}))
        markdown = getattr(raw_doc, "markdown", None) or getattr(raw_doc, "text", "") or ""
        source_url = (
            metadata.get("source_url")
            or metadata.get("sourceURL")
            or metadata.get("url")
            or getattr(raw_doc, "url", "")
        )
        title = metadata.get("title", "") or getattr(raw_doc, "title", "") or ""

    return markdown or "", source_url or "", title or "", metadata


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            try:
                cleaned[key] = str(value)
            except Exception:
                continue
    return cleaned


def _create_document(raw_doc: Any, download_pdfs: bool) -> Document | None:
    markdown, source_url, title, metadata = _extract_document_fields(raw_doc)
    source_type = "pdf" if source_url.lower().endswith(".pdf") else "web"
    if source_type == "pdf" and not download_pdfs:
        return None

    merged_metadata = {
        "source": source_url,
        "source_url": source_url,
        "title": title,
        "source_type": source_type,
        **metadata,
    }
    merged_metadata = _clean_metadata(merged_metadata)

    return Document(
        page_content=markdown,
        metadata=merged_metadata,
    )


def scrape_website(
    start_url: str,
    crawl_subpages: bool = True,
    max_pages: int = MAX_PAGES,
    download_pdfs: bool = True,
    debug_output: bool = False,
) -> list[Document]:
    """
    Crawlt eine Website mit Firecrawl und gibt die Inhalte als Dokumente zurück.

    Wenn `crawl_subpages` gesetzt ist, wird die Firecrawl-Crawl-API verwendet.
    Andernfalls wird nur die einzelne Start-URL mit der Firecrawl-Scrape-API geladen.
    """
    logger.info(
        f"Starte Firecrawl-Scraping: {start_url} | Subpages: {crawl_subpages} "
        f"| PDFs: {download_pdfs} | Limit: {max_pages}"
    )

    client = get_firecrawl_client()
    documents: list[Document] = []

    if crawl_subpages:
        result = client.crawl(start_url, limit=max_pages, formats=["markdown"])
        raw_docs = getattr(result, "data", None)
        if raw_docs is None and isinstance(result, dict):
            raw_docs = result.get("data", [])
        raw_docs = raw_docs or []
    else:
        result = client.scrape(start_url, formats=["markdown"])
        raw_docs = [result]

    if debug_output:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for raw_doc in raw_docs:
        doc = _create_document(raw_doc, download_pdfs)
        if not doc:
            continue

        documents.append(doc)
        if doc.metadata.get("source_type") == "web":
            logger.info(
                f"Webseite geladen: {doc.metadata.get('source', 'unbekannt')} | "
                f"Titel: {doc.metadata.get('title', '—')}"
            )

        if debug_output:
            source_url = doc.metadata.get("source_url", doc.metadata.get("source", ""))
            slug = urlparse(source_url).path.strip("/").replace("/", "_") or "index"
            Path(DEBUG_OUTPUT_DIR / f"{slug}.md").write_text(doc.page_content, encoding="utf-8")

    logger.info(f"Abgeschlossen: {len(documents)} Dokumente")
    if not documents:
        logger.warning("Keine Dokumente erzeugt — URL oder Firecrawl-Resultat prüfen")

    return documents


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    url = os.getenv("SCRAPE_URL")
    if not url:
        print("Usage: python webscraper.py <url> [--no-subpages] [--no-pdfs] [--debug]")
        sys.exit(1)

    docs = scrape_website(
        url,
        crawl_subpages="--no-subpages" not in sys.argv,
        download_pdfs="--no-pdfs" not in sys.argv,
        debug_output="--debug" in sys.argv,
    )

    pdf_docs = [d for d in docs if d.metadata["source_type"] == "pdf"]
    web_docs = [d for d in docs if d.metadata["source_type"] == "web"]

    print(f"\n{len(docs)} Dokumente gesamt ({len(web_docs)} HTML/MD, {len(pdf_docs)} PDF):")
    for d in docs:
        tag = "[PDF]" if d.metadata["source_type"] == "pdf" else "[WEB]"
        title = d.metadata.get("title", "")[:40]
        print(f"  {tag} {title or '—'} → {d.metadata['source']}")
