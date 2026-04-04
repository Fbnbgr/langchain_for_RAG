from typing_extensions import Annotated, TypedDict
from langchain_community.llms import LlamaCpp
from pathlib import Path
from langsmith import Client
from langsmith.evaluation import evaluate
from retrieval import hybrid_search, rerank_candidates, qa_chain, cross_encoder, TOP_K
import os
from langsmith import traceable
from dotenv import load_dotenv
import time

# import for test cases
from data.evaluation.examples import examples
env_path = dotenv_path=Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path = env_path)

LLM_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

client = Client()

dataset_name = "examples"

try:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"Dataset bereits vorhanden: {dataset.id}")
except Exception:
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
    )
    print(f"Dataset erstellt und {len(examples)} Examples hochgeladen")

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
grader_llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
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
    
    return "true" in grade.lower()

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
    # print(results)
    print("Evaluation finished")
    