import hashlib
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def file_hash(path: str) -> str:
    """Berechnet den SHA256-Hash einer Datei, um Änderungen zu erkennen."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


def load_pdfs(pdf_dir: str, existing_hashes: set[str]) -> list[Document]:
    """
    Lädt alle neuen PDFs aus pdf_dir, die noch nicht in existing_hashes sind.
    Gibt eine Liste von Documents mit Metadaten zurück.
    """
    if not os.path.exists(pdf_dir):
        raise RuntimeError(f"PDF-Ordner existiert nicht: {pdf_dir}")

    documents = []

    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, file)
        pdf_hash = file_hash(path)

        if pdf_hash in existing_hashes:
            logger.info(f"Überspringe (bereits indexiert): {file}")
            continue

        logger.info(f"Lade PDF: {file}")
        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source_file"] = file
            d.metadata["file_hash"] = pdf_hash

        logger.info(f"PDF: {file} | Seiten: {len(docs)}")
        documents.extend(docs)

    return documents