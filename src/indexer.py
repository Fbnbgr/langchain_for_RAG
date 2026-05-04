import logging
import os
import time
import uuid

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf import load_pdfs
from webscraper import scrape_website

PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "deepset/gbert-base")
SCRAPE_URL = os.getenv("SCRAPE_URL")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 25))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_existing_hashes(vectordb: Chroma) -> set[str]:
    """Liest alle bereits bekannten file_hashes aus der Chroma DB."""
    existing = set()
    try:
        data = vectordb.get(include=["metadatas"])
        for meta in data.get("metadatas", []):
            if meta and "file_hash" in meta:
                existing.add(meta["file_hash"])
    except Exception:
        pass
    return existing


def add_documents(
        vectordb: Chroma,
        embeddings: HuggingFaceEmbeddings,
        chunks: list[Document]
        ) -> None:
    """Fügt Chunks mit stabilen UUIDs in die Chroma DB ein."""
    vectordb._collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings.embed_documents([chunk.page_content for chunk in chunks])
    )


def run_indexer(sources: list[Document] | None = None) -> None:
    """
    Hauptfunktion des Indexers.

    sources: optional eine Liste von Documents (z.B. vom Webscraper).
             Wenn None, werden nur PDFs aus PDF_DIR geladen.
    """
    start = time.time()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    existing_hashes = load_existing_hashes(vectordb)
    logger.info(f"Bekannte Hashes in DB: {len(existing_hashes)}")
    logger.info(f"Chunks gesamt vor Lauf: {vectordb._collection.count()}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Webscraping
    external_docs = scrape_website(SCRAPE_URL, debug_output = True)

    # PDFs laden
    pdf_docs = load_pdfs(PDF_DIR, existing_hashes)

    all_docs = pdf_docs + external_docs

    if not all_docs:
        logger.info("Keine neuen Dokumente gefunden. Nichts zu tun.")
        return

    chunks = splitter.split_documents(all_docs)
    logger.info(f"Chunks erzeugt: {len(chunks)}")

    logger.info("Füge Chunks in Vektordatenbank ein...")
    add_documents(vectordb, embeddings, chunks)

    duration = time.time() - start
    after = vectordb._collection.count()
    logger.info(
        f"Fertig. {len(chunks)} neue Chunks hinzugefügt in {duration:.1f}s. DB-Größe: {after}"
        )


if __name__ == "__main__":
    run_indexer()