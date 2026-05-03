# LLM (LangChain-kompatibel)
from langchain_ollama import ChatOllama

# RAG Chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vectorstore
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# Prompt Template
prompt_template = """[INST]
Du bist ein präziser Dokumentenassistent.

Beantworte die Frage ausschließlich anhand des bereitgestellten Kontexts.

REGELN:
- Antworte nur mit Informationen aus dem Kontext.
- Erfinde keine Informationen.
- Wenn die Antwort nicht eindeutig im Kontext steht, antworte exakt mit:
  "Im Dokument wurde keine Information dazu gefunden."
- Antworte kurz und präzise.
- Antworte ausschließlich auf Deutsch.

KONTEXT:
{context}

FRAGE:
{input}
[/INST]
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "input"]
)

# Config
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "deepset/gbert-base")
TOP_K = int(os.getenv("TOP_K", 5))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.3))
# leichtgewichtiges, multilingual / deutschfähiges Modell für Re-Ranking Crossencoder
cross_encoder = CrossEncoder(os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))

def rerank_candidates(query, docs, cross_encoder, TOP_K=5):
    # Re-Rankt eine Liste von Dokumenten-Chunks nach Relevanz zur Frage.
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(scores, docs))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Filtern mit Schwellenwert
    filtered = [(score, doc) for score, doc in scored_docs if score > SCORE_THRESHOLD]
    if filtered:
        scores_only = [score for score, doc in filtered]
        print(f"Scores nach Sortierung/Filter: min={min(scores_only):.2f}, max={max(scores_only):.2f}, alle={[f'{s:.2f}' for s in scores_only]}")
    else:
        print("Keine Dokumente über Schwellenwert.")

    # Fallback: Wenn kein Dokument über dem Schwellenwert liegt, nimm das bestbewertete Dokument
    return filtered[:TOP_K] if filtered else scored_docs[:1]

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)

# Vector DB
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    # mind. x % Ähnlichkeit, damit es als Treffer gilt
    search_kwargs={"k": TOP_K, "score_threshold": float(os.getenv("SCORE_THRESHOLD", 0.3))}
)

# LLM (lokal, CPU)
llm = ChatOllama(
    model=os.getenv("LLM_MODEL_NAME", "mistral"),
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    max_tokens=512,
    n_ctx=4096,
    n_batch=512,
    n_threads=os.cpu_count(),
    # Mistral-spezifischer Stop-Token, um die Ausgabe zu beenden
    stop=["</s>", "[INST]"],
    verbose=False
)

# QA Chain
qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=PROMPT
)

# BM25 Retriever
_bm25_retriever = None

# lazy loading, damit die Dokumente erst geladen werden, wenn die Evaluation gestartet wird
def get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever

    all_docs = vectordb.get()
    documents = [
        Document(
            page_content=all_docs["documents"][i],
            metadata=all_docs["metadatas"][i]
        )
        for i in range(len(all_docs["documents"]))
    ]

    if not documents:
        raise RuntimeError("Chroma DB ist leer – bitte zuerst den Indexer ausführen.")

    _bm25_retriever = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = TOP_K
    return _bm25_retriever

def hybrid_search(query, k=TOP_K):

    bm25_retriever = get_bm25_retriever()
    
    # Embeddings-basierte Suche (Semantik)
    emb = retriever.invoke(query)
    # print(f"EMB Treffer: {len(emb)}")
    # BM25-basierte Suche (lexikalisch)
    bm25 = bm25_retriever.invoke(query)
    # print(f"BM25 Treffer: {len(bm25)}")

    # Gewichtung
    emb_weight = float(os.getenv("EMB_WEIGHT", 0.7))
    bm25_weight = float(os.getenv("BM25_WEIGHT", 0.3))

    scored = {}

    # reciprocal rank fusion (RRF) für die Kombination der Ergebnisse von Embeddings und BM25
    for rank, doc in enumerate(emb):
        scored[doc.page_content] = scored.get(doc.page_content, 0) + emb_weight * (1 / (rank + 1))
        # print(f"EMB {rank}: {doc.page_content}")

    for rank, doc in enumerate(bm25):
        scored[doc.page_content] = scored.get(doc.page_content, 0) + bm25_weight * (1 / (rank + 1))
        # print(f"BM25 {rank}: {doc.page_content}")

    # sortieren
    sorted_docs = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    # print(f"sorted_docs: {sorted_docs}")

    # zurück zu Documents
    content_to_doc = {doc.page_content: doc for doc in bm25 + emb}

    return [content_to_doc[content] for content, score in sorted_docs[:k]]
