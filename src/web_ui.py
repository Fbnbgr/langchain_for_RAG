import logging
import os

from flask import Flask, render_template_string, request

from retrieval import TOP_K, cross_encoder, gnd_search, hybrid_search, qa_chain, rerank_candidates

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>RAG Web UI</title></head>
<body>
    <h1>Frage an das RAG-System</h1>
    <form method="POST">
        <input type="text" name="query" placeholder="Gib deine Frage ein" required>
        <button type="submit">Fragen</button>
    </form>
    {% if answer %}
        <h2>Antwort:</h2>
        <p>{{ answer }}</p>
        <h2>Quellen:</h2>
        <ul>
        {% for source in sources %}
            <li>{{ source }}</li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    sources = []
    if request.method == 'POST':
        query = request.form['query']
        # Verwende bestehende Logik aus retrieval.py
        candidates = hybrid_search(query)
        reranked = rerank_candidates(query, candidates, cross_encoder, TOP_K)
        gnd = gnd_search(query) or []  
        top_docs = [doc for score, doc in reranked] + gnd
        logger.debug(f"Top-Dokumente: {[doc.metadata.get('source_file') or doc.metadata.get('source', 'unbekannt') for doc in top_docs]}")
        
        # Antwort generieren
        answer = qa_chain.invoke({"context": top_docs, "input": query})
        
        # Quellen sammeln
        sources = [
            f"{doc.metadata.get('source_file') or doc.metadata.get('source', 'unbekannt')} "
            f"Seite {doc.metadata.get('page', '—')}: {doc.page_content[:200]}..."
            for doc in top_docs
]
    
    return render_template_string(HTML_TEMPLATE, answer=answer, sources=sources)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)