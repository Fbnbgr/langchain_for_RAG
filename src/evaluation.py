import logging
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.cost import TokenUsage
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
    NoiseSensitivity,
)

from data.evaluation.examples import examples
from retrieval import TOP_K, cross_encoder, hybrid_search, qa_chain, rerank_candidates

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GRADER_MODEL = os.getenv("GRADER_MODEL", "mistral")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "deepset/gbert-base")

grader_llm = LangchainLLMWrapper(ChatOllama(
    model=GRADER_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    max_tokens=512,
    n_ctx=2048,
))

grader_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
)

def get_token_usage_for_ollama(response) -> TokenUsage:
    input_tokens = 0
    output_tokens = 0

    for gen_list in response.generations:
        for gen in gen_list:
            info = getattr(gen, "generation_info", {}) or {}
            input_tokens += info.get("prompt_eval_count", 0)
            output_tokens += info.get("eval_count", 0)

    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

def run_pipeline(question: str) -> dict:
    """RAG-Pipeline ausführen und Antwort + Chunks zurückgeben."""
    candidates = hybrid_search(question)
    reranked = rerank_candidates(question, candidates, cross_encoder, TOP_K)
    top_docs = [doc for score, doc in reranked]

    answer = qa_chain.invoke({
        "context": top_docs,
        "input": question
    })

    return {
        "answer": answer,
        "contexts": [doc.page_content for doc in top_docs],
    }


def build_dataset() -> EvaluationDataset:
    """Beispielfragen durch die Pipeline laufen lassen und Dataset bauen."""
    samples = []

    for example in examples:
        question = example["inputs"]["question"]
        ground_truth = example["outputs"]["answer"]

        print(f"  → {question}")
        result = run_pipeline(question)

        samples.append(SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=ground_truth,
        ))

    return EvaluationDataset(samples=samples)


def evaluation():

    dataset = build_dataset()

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),        # Halluziniert das LLM?
            AnswerRelevancy(),    # Beantwortet die Antwort die Frage?
            ContextPrecision(),   # Sind die Chunks relevant? (braucht ground_truth)
            ContextRecall(),      # Wurden alle relevanten Infos gefunden?
            FactualCorrectness(),  # faktische Korrektheit der Antwort
            NoiseSensitivity(),    # Robustheit gegen schlechte Chunks
            ContextEntityRecall(), # Wurden wichtige Entitäten aus den Chunks in der Antwort verwendet?
        ],
        llm=grader_llm,
        embeddings=grader_embeddings,
        token_usage_parser=get_token_usage_for_ollama,
    )

    print("\n=== Ergebnisse ===")
    print(results)
    print(results.total_tokens())

    # Als CSV speichern für spätere Analyse
    df = results.to_pandas()
    df.to_csv("data/evaluation/results.csv", index=False)
    print("Ergebnisse gespeichert: data/evaluation/results.csv")

    return results


if __name__ == "__main__":
    evaluation()