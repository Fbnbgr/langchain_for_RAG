import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PDF_DIR = "data/pdfs"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


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


def main():
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
    print(f"Gefundene bekannte PDFs: {len(existing_hashes)}")

    total_chunks = vectordb._collection.count()
    print(f"Chunks gesamt: {total_chunks}")

    splitter = RecursiveCharacterTextSplitter(
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
            print(f"Überspringe (bereits indexiert): {file}")
            continue

        print(f"Verarbeite neue PDF: {file}")
        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source_file"] = file
            d.metadata["file_hash"] = pdf_hash

        chunks = splitter.split_documents(docs)
        new_chunks.extend(chunks)

    if not new_chunks:
        print("Keine neuen PDFs gefunden. Nichts zu tun.")
        return

    vectordb.add_documents(new_chunks)
    print(f"Fertig. {len(new_chunks)} neue Chunks hinzugefügt.")


if __name__ == "__main__":
    main()
