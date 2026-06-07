import logging
import os

import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# spaCy Modell einmalig laden
try:
    nlp = spacy.load("de_core_news_lg")
    logger.info("spacy Modell geladen")
except OSError:
    raise RuntimeError("spaCy Modell 'de_core_news_lg' nicht gefunden – bitte im Dockerfile einbauen.")

# RAG Chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Vectorstore
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder
from SPARQLWrapper import JSON, SPARQLWrapper

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
FUSEKI_ENDPOINT = os.getenv("FUSEKI_ENDPOINT", "http://fuseki:3030/gnd/sparql")
FUSEKI_USER = os.getenv("FUSEKI_USER", "admin")
FUSEKI_PASSWORD = os.getenv("FUSEKI_PASSWORD", "admin")
GND_ENABLED = os.getenv("GND_ENABLED", "true").lower() == "true"
GND_WEIGHT = float(os.getenv("GND_WEIGHT", 0.3))

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
cross_encoder = CrossEncoder(os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))

# GND SPARQL Retriever

def extract_entities(query: str) -> list[tuple[str, str]]:
    """Extrahiert Entitäten mit spaCy — gibt Liste von (Name, Typ) zurück."""
    doc = nlp(query)
    # PER = Person, LOC = Ort, ORG = Organisation/Körperschaft, MISC = Sonstiges
    relevant_labels = {"PER", "LOC", "ORG", "MISC"}
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in relevant_labels]
    logger.info(f"spaCy Entitäten: {entities}")
    return entities


def query_gnd_person(name: str) -> list[Document]:
    """Sucht eine Person in der GND und gibt strukturierte Infos zurück."""
    # Anfangsbuchstaben kapitalisieren
    name_capitalized = name.title()
    # Name in "Nachname, Vorname" umwandeln
    parts = name_capitalized.strip().split()
    if len(parts) >= 2:
        last = parts[-1]
        first = " ".join(parts[:-1])
        name_capitalized = f"{last}, {first}"

    sparql = SPARQLWrapper(FUSEKI_ENDPOINT)
    sparql.setCredentials(FUSEKI_USER, FUSEKI_PASSWORD)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(f"""
        PREFIX gndo: <https://d-nb.info/standards/elementset/gnd#>
        SELECT ?id ?name ?gender ?birth ?death ?profession ?professionAsLiteral ?relatedPlace ?biography WHERE {{
            ?id gndo:preferredNameForThePerson ?name .
            FILTER(CONTAINS(STR(?name), "{name_capitalized}"))
            OPTIONAL {{ ?id gndo:preferredNameForThePerson ?name }}
            OPTIONAL {{ ?id gndo:gender ?gender }}
            OPTIONAL {{ ?id gndo:geographicAreaCode ?relatedPlace }}
            OPTIONAL {{ ?id gndo:dateOfBirth ?birth }}
            OPTIONAL {{ ?id gndo:dateOfDeath ?death }}
            OPTIONAL {{ ?id gndo:professionOrOccupationAsLiteral ?profession }}
            OPTIONAL {{ ?id gndo:professionOrOccupationAsLiteralAsLiteral ?professionAsLiteral }}
            OPTIONAL {{ ?id gndo:biographicalOrHistoricalInformation ?biography }}
        }} LIMIT 5
    """)
    try:
        results = sparql.query().convert()
        logger.debug(f"GND Rohantwort: {results}")
        bindings = results.get("results", {}).get("bindings", [])
        logger.info(f"GND SPARQL: {len(bindings)} Treffer für '{name}'")

        docs = []
        fields = {
            "name":               "Name",
            "relatedPlace":       "Wirkungsort",
            "gender":             "Geschlecht",
            "birth":              "Geburtsdatum",
            "death":              "Sterbedatum",
            "profession":         "Beruf/Funktion",
            "professionAsLiteral":"Beruf/Funktion (Literal)",
            "biography":          "Biografie",
        }

        for r in bindings:
            gnd_id = r.get("id", {}).get("value", "")
            text = f"GND-Person: {name}\n"
            text += f"GND-ID: {gnd_id}\n"
            for key, label in fields.items():
                if key in r:
                    text += f"{label}: {r[key]['value']}\n"

            docs.append(Document(
                page_content=text,
                metadata={"source": "GND", "gnd_id": gnd_id, "entity": name}
            ))
            logger.debug(f"GND Dokument (id={gnd_id}):\n{text}")

        logger.info(f"GND: {len(docs)} Dokumente erstellt für '{name}'")
        return docs
    except Exception:
        logger.exception("GND SPARQL Fehler für '%s'", name, exc_info=True)
        return []


def query_gnd_general(name: str) -> list[Document]:
    """Fallback: sucht name in allen GND-Entitätstypen (Ort, Körperschaft etc.)."""
    name_capitalized = name.title()
    sparql = SPARQLWrapper(FUSEKI_ENDPOINT)
    sparql.setCredentials(FUSEKI_USER, FUSEKI_PASSWORD)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(f"""
        PREFIX gndo: <https://d-nb.info/standards/elementset/gnd#>
        SELECT ?id ?type ?label WHERE {{
            ?id gndo:preferredName "{name_capitalized}"@de .
            OPTIONAL {{ ?id a ?type }}
            BIND("{name_capitalized}" AS ?label)
        }} LIMIT 5
    """)
    try:
        results = sparql.query().convert()
        docs = []
        for r in results["results"]["bindings"]:
            gnd_id = r.get("id", {}).get("value", "")
            ent_type = r.get("type", {}).get("value", "unbekannt").split("#")[-1]
            text = f"GND-Eintrag: {name}\nTyp: {ent_type}\nGND-ID: {gnd_id}\n"
            docs.append(Document(
                page_content=text,
                metadata={"source": "GND", "gnd_id": gnd_id, "entity": name}
            ))
        return docs
    except Exception:
        logger.exception("GND SPARQL Fallback Fehler für '%s'", name)
        return []


def gnd_search(query: str) -> list[Document]:
    """Hauptfunktion: extrahiert Entitäten mit spaCy und sucht in GND."""
    if not GND_ENABLED:
        return []

    entities = extract_entities(query)
    # Wenn spaCy keine Entitäten findet, versuchen wir einen allgemeinen GND-Suchlauf
    if not entities:
        logger.debug("Keine Entitäten erkannt für '%s' — mache allgemeinen GND-Fallback", query)
        return query_gnd_general(query)

    docs = []
    for name, ent_type in entities:
        if ent_type == "PER":
            results = query_gnd_person(name)
        else:
            # LOC, ORG, MISC → allgemeiner Fallback
            results = query_gnd_general(name)
        # Falls nichts gefunden, nochmal mit Fallback versuchen
        if not results:
            results = query_gnd_general(name)
        logger.debug("GND-Ergebnisse für '%s': %d", name, len(results))
        docs.extend(results)

    return docs

# Reranking mit Cross-Encoder
def rerank_candidates(query, docs, cross_encoder, TOP_K=5):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    filtered = [(score, doc) for score, doc in scored_docs if score > SCORE_THRESHOLD]
    if filtered:
        scores_only = [score for score, doc in filtered]
        print(
            f"Scores nach Sortierung/Filter: min={min(scores_only):.2f},"
            f"max={max(scores_only):.2f},"
            f"alle={[f'{s:.2f}' for s in scores_only]}")
    else:
        print("Keine Dokumente über Schwellenwert.")
    return filtered[:TOP_K] if filtered else scored_docs[:1]


# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Vector DB
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": TOP_K, "score_threshold": float(os.getenv("SCORE_THRESHOLD", 0.3))}
)

# LLM
llm = ChatOllama(
    model=os.getenv("LLM_MODEL_NAME", "mistral"),
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    max_tokens=512,
    n_ctx=4096,
    n_batch=512,
    n_threads=os.cpu_count(),
    stop=["</s>", "[INST]"],
    verbose=False
)

# QA Chain
qa_chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)

# BM25
_bm25_retriever = None

def get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever
    all_docs = vectordb.get()
    documents = [
        Document(page_content=all_docs["documents"][i], metadata=all_docs["metadatas"][i])
        for i in range(len(all_docs["documents"]))
    ]
    if not documents:
        raise RuntimeError("Chroma DB ist leer – bitte zuerst den Indexer ausführen.")
    _bm25_retriever = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = TOP_K
    return _bm25_retriever


def hybrid_search(query, k=TOP_K):
    bm25_retriever = get_bm25_retriever()

    emb = retriever.invoke(query)
    bm25 = bm25_retriever.invoke(query)
    # gnd = gnd_search(query)   

    emb_weight = float(os.getenv("EMB_WEIGHT", 0.7))
    bm25_weight = float(os.getenv("BM25_WEIGHT", 0.3))

    scored = {}

    # RRF für Embeddings
    for rank, doc in enumerate(emb):
        scored[doc.page_content] = scored.get(doc.page_content, 0) + emb_weight * (1 / (rank + 1))

    # RRF für BM25
    for rank, doc in enumerate(bm25):
        scored[doc.page_content] = scored.get(doc.page_content, 0) + bm25_weight * (1 / (rank + 1))

    # GND direkt hinzufügen (kein Ranking, immer mit festem Gewicht)
    gnd_docs_map = {}
    for rank, doc in enumerate(gnd_docs_map):
        scored[doc.page_content] = scored.get(doc.page_content, 0) + GND_WEIGHT * (1 / (rank + 1))
        gnd_docs_map[doc.page_content] = doc

    sorted_docs = sorted(scored.items(), key=lambda x: x[1], reverse=True)

    content_to_doc = {doc.page_content: doc for doc in bm25 + emb}
    content_to_doc.update(gnd_docs_map) 

    return [content_to_doc[content] for content, score in sorted_docs[:k]]