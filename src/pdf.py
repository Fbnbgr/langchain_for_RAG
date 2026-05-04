import hashlib
import logging
import os

import pypdf.errors
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PDF_MAGIC_BYTES = b"%PDF-"


def file_hash(path: str) -> str:
    """Berechnet den SHA256-Hash einer Datei, um Änderungen zu erkennen."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


def is_valid_pdf(path: str) -> bool:
    """
    Schneller Vorab-Check: Liest nur die ersten 8 Bytes und prüft den PDF-Magic-Header.
    Fängt den häufigsten Fall ab (HTML/Text-Datei mit .pdf-Endung).
    """
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        if not header.startswith(PDF_MAGIC_BYTES):
            logger.warning(
                f"Ungültiger PDF-Header in '{path}': {header[:8]!r} — Datei wird übersprungen")
            return False
        return True
    except OSError as e:
        logger.warning(f"Datei nicht lesbar '{path}': {e} — wird übersprungen")
        return False


def load_pdfs(pdf_dir: str, existing_hashes: set[str]) -> list[Document]:
    """
    Lädt alle neuen PDFs aus pdf_dir, die noch nicht in existing_hashes sind.
    Ungültige oder korrupte PDFs werden übersprungen und geloggt.
    Gibt eine Liste von Documents mit Metadaten zurück.
    """
    if not os.path.exists(pdf_dir):
        raise RuntimeError(f"PDF-Ordner existiert nicht: {pdf_dir}")

    documents = []
    skipped = []

    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, file)

        # ── 1. Header-Check (billig, vor dem Hashing) ─────────────────────
        if not is_valid_pdf(path):
            skipped.append(file)
            continue

        # ── 2. Duplikat-Check ──────────────────────────────────────────────
        pdf_hash = file_hash(path)
        if pdf_hash in existing_hashes:
            logger.info(f"Überspringe (bereits indexiert): {file}")
            continue

        # ── 3. Laden mit Fehlerbehandlung für korrupte PDFs ───────────────
        logger.info(f"Lade PDF: {file}")
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
        except pypdf.errors.PdfStreamError as e:
            logger.warning(
                f"Korruptes PDF '{file}' (Stream-Fehler): {e} — wird übersprungen")
            skipped.append(file)
            continue
        except pypdf.errors.PdfReadError as e:
            logger.warning(
                f"Korruptes PDF '{file}' (Lesefehler): {e} — wird übersprungen")
            skipped.append(file)
            continue
        except Exception as e:
            logger.warning(
                f"Unerwarteter Fehler beim Laden von '{file}': {type(e).__name__}: {e} — wird übersprungen")
            skipped.append(file)
            continue

        if not docs:
            logger.warning(f"PDF '{file}' ergab keine Seiten — wird übersprungen")
            skipped.append(file)
            continue

        for d in docs:
            d.metadata["source_file"] = file
            d.metadata["file_hash"] = pdf_hash

        logger.info(f"PDF geladen: {file} | Seiten: {len(docs)}")
        documents.extend(docs)

    if skipped:
        logger.warning(f"{len(skipped)} PDF(s) übersprungen (invalid/korrupt): {', '.join(skipped)}")

    return documents