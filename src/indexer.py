import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
import time

PDF_DIR = "data/pdfs"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "deepset/gbert-base"

CHUNK_SIZE = 256
CHUNK_OVERLAP = 25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def file_hash(path: str) -> str:
    # Berechnet den SHA256-Hash einer Ingest Datei, um Änderungen zu erkennen
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def load_existing_hashes(vectordb: Chroma) -> set[str]:
    # Set, um doppelte Werte zu vermeiden, da mehrere Dokumente den gleichen Hash haben könnten (z.B. gleiche PDF mit vielen Chunks)
    existing = set()
    try:
        # Nur die Metadaten abrufen, um die Hashes zu extrahieren und nicht die gesamten Vektoren
        data = vectordb.get(include=["metadatas"])
        # Die Metadaten enthalten die "file_hash"-Felder, die zum Vergleich genutzt werden
        for meta in data.get("metadatas", []):
            if meta and "file_hash" in meta:
                existing.add(meta["file_hash"])
    except Exception:
        pass
    return existing

def run_indexer():
    start = time.time()
    if not os.path.exists(PDF_DIR):
        raise RuntimeError(f"PDF-Ordner existiert nicht: {PDF_DIR}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # langchain wrapper um chromadb, um die Dokumente zu speichern und zu verwalten
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    existing_hashes = load_existing_hashes(vectordb)
    logger.info(f"Gefundene bekannte PDFs: {len(existing_hashes)}")

    total_chunks = vectordb._collection.count()
    logger.info(f"Chunks gesamt: {total_chunks}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    new_chunks = []

    for file in os.listdir(PDF_DIR):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(PDF_DIR, file)
        pdf_hash = file_hash(path)

        if pdf_hash in existing_hashes:
            logger.info(f"Überspringe (bereits indexiert): {file}")
            continue

        logger.info(f"Verarbeite neue PDF: {file}")
        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source_file"] = file
            d.metadata["file_hash"] = pdf_hash

        chunks = splitter.split_documents(docs)
        new_chunks.extend(chunks)
        duration = time.time() - start
        logger.info(f"PDF: {file} | Seiten: {len(docs)} | Chunks: {len(chunks)} | Zeit: {duration:.1f}s")

    if not new_chunks:
        logger.info("Keine neuen PDFs gefunden. Nichts zu tun.")
        return

    logger.info(f"Füge die Dokumente der Vektordatenbank hinzu...")

    # vectordb.add_documents(new_chunks)
    vectordb._collection.add(
        ids=[str(i) for i in range(len(new_chunks))],
        documents=[chunk.page_content for chunk in new_chunks],
        metadatas=[chunk.metadata for chunk in new_chunks],
        embeddings=embeddings.embed_documents([chunk.page_content for chunk in new_chunks])
    )
    after = vectordb._collection.count()
    logger.info(f"Fertig. {len(new_chunks)} neue Chunks hinzugefügt. DB: {total_chunks} → {after}")


if __name__ == "__main__":
    run_indexer()
