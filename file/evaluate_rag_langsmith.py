"""Upload and evaluate the RAG benchmark in LangSmith.

Set these environment variables before running:
    LANGSMITH_API_KEY=your_langsmith_key
    GOOGLE_API_KEY=your_google_key
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.evaluation import load_evaluator
from langsmith import Client

from llm import GeminiRAG
from retriever import ChromaRetriever
from dotenv import load_dotenv

load_dotenv()

DATASET_NAME = "nabl-112a-rag-evaluation-30"
DATASET_FILE = Path(__file__).with_name("nabl_112a_rag_evaluation_30.json")
JUDGE_MODEL = "gemini-3" \
".6-flash"

CONTEXT_PRECISION_CRITERIA = """
Score Context Precision from 0 to 10. The prediction is the retrieved context.
Assess whether its passages directly address the user's question. Penalize
irrelevant, distracting, or only weakly related passages. Do not judge answer
quality. 0 means entirely irrelevant; 10 means every passage is directly useful.
"""

CONTEXT_RECALL_CRITERIA = """
Score Context Recall from 0 to 10. The prediction is the retrieved context and
the reference is the ground-truth answer. Assess whether the context contains
all material facts needed to produce the complete ground-truth answer. Do not
penalize wording differences. 0 means none of the needed facts are present;
10 means all material facts are present.
"""

FAITHFULNESS_CRITERIA = """
Score Faithfulness from 0 to 10. The input contains the question and retrieved
context; the prediction is the generated answer. Assess whether every factual
claim in the answer is supported by the retrieved context. Ignore whether the
answer is complete. 0 means unsupported or fabricated; 10 means fully grounded.
"""

ANSWER_RELEVANCY_CRITERIA = """
Score Answer Relevancy from 0 to 10. The input is the user's question and the
prediction is the generated answer. Assess whether the answer directly answers
the question, including all requested parts, without unnecessary tangents.
Do not use the reference to judge factual correctness. 0 means irrelevant;
10 means direct, focused, and complete for the question.
"""


def require_environment() -> None:
    # required = ("LANGSMITH_API_KEY", "GOOGLE_API_KEY")
    # missing = [key for key in required if not os.getenv(key)]
    # if missing:
    #     raise EnvironmentError(f"Missing environment variable(s): {', '.join(missing)}")
    os.environ["LANGSMITH_API_KEY"]
    os.environ["GOOGLE_API_KEY"]
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "nabl-rag-evaluation")


def as_context_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def context_text(value: Any) -> str:
    contexts = as_context_list(value)
    return "\n\n--- CONTEXT CHUNK ---\n\n".join(contexts) or "[NO CONTEXT RETRIEVED]"


def load_rows() -> list[dict[str, Any]]:
    with DATASET_FILE.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError("The evaluation file must contain a JSON list.")
    required = {"question", "contexts", "answer", "ground_truth"}
    for index, row in enumerate(rows, start=1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
    return rows


def upload_dataset(client: Client, rows: list[dict[str, Any]]) -> None:
    """Create the benchmark once; safely reuse it on later executions."""
    try:
        client.read_dataset(dataset_name=DATASET_NAME)
        return
    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="NABL 112A RAG benchmark with ground-truth answers.",
        )

    for row in rows:
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "question": row["question"],
                "contexts": as_context_list(row["contexts"]),
            },
            outputs={
                "answer": row["answer"],
                "ground_truth": row["ground_truth"],
            },
        )


RETRIEVER = None
RAG = None


def get_pipeline() -> tuple[ChromaRetriever, GeminiRAG]:
    global RETRIEVER, RAG
    if RETRIEVER is None:
        RETRIEVER = ChromaRetriever(db_path="./chroma_db", collection_name="documents")
    if RAG is None:
        RAG = GeminiRAG(os.environ["GOOGLE_API_KEY"])
    return RETRIEVER, RAG


def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Replace this body only if your production pipeline uses another entry point."""
    import query_gen

    question = str(inputs["question"])
    retriever, rag = get_pipeline()
    _, max_tokens = query_gen.filter_tokens(question)
    queries = query_gen.generate_queries(question=question, max_l=max_tokens)
    documents = retriever.retrieve_multiple(queries, top_k=5)
    contexts = [document["content"] for document in documents]
    answer = rag.answer(question, documents)
    return {"answer": answer, "contexts": contexts}


JUDGE_LLM = None
JUDGES = None


def get_judges() -> dict[str, Any]:
    global JUDGE_LLM, JUDGES
    if JUDGES is None:
        JUDGE_LLM = ChatGoogleGenerativeAI(
            model=JUDGE_MODEL,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0,
        )
        JUDGES = {
            "context_precision": load_evaluator(
                "score_string", llm=JUDGE_LLM, criteria=CONTEXT_PRECISION_CRITERIA
            ),
            "context_recall": load_evaluator(
                "score_string", llm=JUDGE_LLM, criteria=CONTEXT_RECALL_CRITERIA
            ),
            "faithfulness": load_evaluator(
                "score_string", llm=JUDGE_LLM, criteria=FAITHFULNESS_CRITERIA
            ),
            "answer_relevancy": load_evaluator(
                "score_string", llm=JUDGE_LLM, criteria=ANSWER_RELEVANCY_CRITERIA
            ),
        }
    return JUDGES


def score_out_of_ten(judge: Any, *, input_text: str, prediction: str, reference: str) -> float:
    result = judge.evaluate_strings(
        input=input_text,
        prediction=prediction,
        reference=reference,
    )
    raw_score = float(result["score"])
    return max(0.0, min(1.0, raw_score / 10.0))


def rag_metrics_evaluator(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> list[dict[str, float | str]]:
    """LangSmith custom evaluator: emits four normalized feedback scores."""
    question = str(inputs["question"])
    retrieved_context = context_text(outputs.get("contexts"))
    generated_answer = str(outputs.get("answer", ""))
    ground_truth = str(reference_outputs["ground_truth"])
    judges = get_judges()

    scores = {
        "context_precision": score_out_of_ten(
            judges["context_precision"],
            input_text=question,
            prediction=retrieved_context,
            reference="No reference answer is needed for context precision.",
        ),
        "context_recall": score_out_of_ten(
            judges["context_recall"],
            input_text=question,
            prediction=retrieved_context,
            reference=ground_truth,
        ),
        "faithfulness": score_out_of_ten(
            judges["faithfulness"],
            input_text=f"Question:\n{question}\n\nRetrieved context:\n{retrieved_context}",
            prediction=generated_answer,
            reference="Judge only whether the answer is supported by the retrieved context.",
        ),
        "answer_relevancy": score_out_of_ten(
            judges["answer_relevancy"],
            input_text=question,
            prediction=generated_answer,
            reference="No reference answer is needed for answer relevancy.",
        ),
    }
    return [{"key": metric, "score": score} for metric, score in scores.items()]


def main() -> None:
    require_environment()
    rows = load_rows()
    client = Client()
    upload_dataset(client, rows)

    results = client.evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[rag_metrics_evaluator],
        experiment_prefix="nabl-rag-llm-judge-evaluation",
        max_concurrency=2,
    )



if __name__ == "__main__":
    main()
