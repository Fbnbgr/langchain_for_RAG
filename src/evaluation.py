from typing_extensions import Annotated, TypedDict
from langchain_ollama import ChatOllama
from pathlib import Path
from langsmith import Client
from langsmith.evaluation import evaluate
from retrieval import hybrid_search, rerank_candidates, qa_chain, cross_encoder, TOP_K
import os
from langsmith import traceable
from dotenv import load_dotenv
import json
import logging

# import for test cases
from data.evaluation.examples import examples
env_path = dotenv_path=Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path = env_path)

logger = logging.getLogger(__name__)

client = Client()

dataset_name = "test-dataset"

try:
    dataset = client.read_dataset(dataset_name=dataset_name)
    # Alte Examples löschen und neu anlegen
    existing = list(client.list_examples(dataset_id=dataset.id))
    for ex in existing:
        client.delete_example(ex.id)
    print(f"Dataset gefunden, {len(existing)} alte Examples gelöscht")
except Exception:
    dataset = client.create_dataset(dataset_name=dataset_name)
    print("Dataset neu erstellt")

client.create_examples(
    dataset_id=dataset.id,
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
)

@traceable
def target(inputs: dict) -> dict:
    query = inputs["question"]
    
    # Candidate Retrieval
    candidates = hybrid_search(query)
    
    # Re-Ranking
    reranked = rerank_candidates(query, candidates, cross_encoder, TOP_K)
    top_docs = [doc for score, doc in reranked]
    
    # LLM Antwort
    answer = qa_chain.invoke({
        "context": top_docs,
        "input": query
    })
    
    return {"answer": answer}

# Grade output schema
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Grade prompt
correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
grader_llm = ChatOllama(
    model="mistral",
    base_url="http://host.docker.internal:11434",
    temperature=0.1,
    # max anzahl an auszugebenden Tokens (soll nur 1 Token ausgeben)
    max_tokens=64,
    # Kontextgröße
    n_ctx=2048,
    # parallele Verarbeitung von x Tokens
    n_batch=512,
    n_threads=os.cpu_count(),
    # stoppt nach dem JSON-Output
    stop=["}", "\n"],
    # logs
    verbose=True
)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    # Run evaluator (Anpassung auf llamacpp)
    prompt = f"{correctness_instructions}\n\n{answers}\n\nAntworte nur mit JSON: {{\"correct\": true}} oder {{\"correct\": false}}"
    
    grade = grader_llm.invoke(prompt)
    
    try:
        raw = grade.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)
        return bool(result["correct"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Grader-Output konnte nicht geparst werden: {grade.content!r} — Fehler: {e}")
        return False

@traceable
def evaluation():
    print("Starte Evaluation")
    evaluation_results = evaluate(
            target,
            data=dataset_name,
            evaluators=[correctness],
            experiment_prefix="rag-doc-relevance",
            metadata={"version": "LCEL context, mistral-7b"},
        )
    return evaluation_results

if __name__ == "__main__":
    results = evaluation()
    print("Evaluation finished")
    