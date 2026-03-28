# LLM (LangChain-kompatibel)
from langchain_community.llms import LlamaCpp

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
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
TOP_K = 8

# leichtgewichtiges, multilingual / deutschfähiges Modell für Re-Ranking Crossencoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

def rerank_candidates(query, docs, cross_encoder, top_k=8):
    # Re-Rankt eine Liste von Dokumenten-Chunks nach Relevanz zur Frage.
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(scores, docs))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return scored_docs[:top_k]

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
    search_kwargs={"k": TOP_K, "score_threshold": 0.2}
)

# LLM (lokal, CPU)
llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=0.1,
    max_tokens=512,
    n_ctx=4096,
    verbose=False
)

# QA Chain
qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=PROMPT
)

# BM25 Retriever
# Alle Dokumente laden aus Chroma
all_docs = vectordb.get()

documents = []
for i in range(len(all_docs["documents"])):
    
    documents.append(
        Document(
            page_content=all_docs["documents"][i],
            metadata=all_docs["metadatas"][i]
        )
    )

# BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents)

bm25_retriever.k = TOP_K

def hybrid_search(query, k=TOP_K):

    # Embeddings-basierte Suche (Semantik)
    emb = retriever.invoke(query)
    print(f"EMB Treffer: {len(emb)}")
    # BM25-basierte Suche (lexikalisch)
    bm25 = bm25_retriever.invoke(query)
    print(f"BM25 Treffer: {len(bm25)}")

    # Gewichtung -> aktuell Semantik höher gewichtet
    emb_weight = 0.7
    bm25_weight = 0.3

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

def cli():
    print("Lokales PDF-RAG System bereit. 'exit' zum Beenden.\n")

    while True:
        query = input("Frage: ")

        if query.lower() in ["exit", "quit"]:
            print("Beendet.")
            break

        # Candidate Retrieval
        candidates = hybrid_search(query)
        
        # Re-Ranking
        reranked = rerank_candidates(query, candidates, cross_encoder, TOP_K)
        for score, doc in reranked:
            print(f"CrossEncoder Score: {score:.3f}")
            print(f"{doc.metadata.get('source_file')} Seite {doc.metadata.get('page')}")
            print(doc.page_content[:500])
            print("\n----------------\n")

        top_docs = [doc for score, doc in reranked]

        # LLM Antwort
        print("LLM-Antwort:")
        result = qa_chain.invoke({
            "context": top_docs,
            "input": query
        })

        print(result)

        # Quellen anzeigen
        print("\nQuellen:\n")
        docs_and_scores = vectordb.similarity_search_with_score(query, k=12)
        for i, (doc, distance) in enumerate(docs_and_scores, 1):

            similarity = 1 - distance

            print(f"Chunk #{i}")
            print(f"Similarity: {similarity:.3f}")
            print(f"Distance:   {distance:.3f}")
            print(f"Quelle:     {doc.metadata.get('source_file')}")
            print(f"Seite:      {doc.metadata.get('page')}")

            print("\nText:")
            print(doc.page_content[:500])

        print("\n-------------------\n")

if __name__ == "__main__":
    cli()


