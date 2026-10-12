"""Upload and evaluate the RAG benchmark in LangSmith.

Required environment variables:
    LANGSMITH_API_KEY=your_langsmith_key
    GOOGLE_API_KEY=your_google_key
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_classic.evaluation import load_evaluator
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import query_gen
from llm import OpenaiRAG
from hybridRag import ChromaHybridRag

load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]
langsmith_api_key = os.environ["LANGSMITH_API_KEY"]

faithfulness_prompt = PromptTemplate(
    input_variables=["input", "prediction", "criteria"],
    template="""
You are an expert RAG evaluation judge.

Evaluation criterion:
{criteria}

Input:
{input}

Prediction:
{prediction}

Evaluate the prediction according to the criterion.
Choose an integer score from 1 to 10.

Explain your reasoning briefly, then finish with the score
in exactly this format: [[9]]

The final output MUST contain a double-bracketed integer
from [[1]] to [[10]]. Do not use "Score: 9" instead.
"""
)

DATASET_NAME = "nabl-112a-rag-evaluation-30"
DATASET_FILE = Path(__file__).with_name(
    "nabl_112a_rag_evaluation_30.json"
)

CONTEXT_PRECISION_CRITERIA = """
Score from 0 to 10. The prediction is the retrieved context.
Evaluate whether the retrieved passages directly address the question.
Penalize irrelevant or weakly related passages. Do not judge answer quality.
0 means entirely irrelevant; 10 means all passages are useful.
"""

CONTEXT_RECALL_CRITERIA = """
Score from 0 to 10. The prediction is the retrieved context, and the
reference is the ground-truth answer. Evaluate whether the context contains
all material facts needed to produce the ground-truth answer.
0 means no necessary facts are present; 10 means all are present.
"""

FAITHFULNESS_CRITERIA = """
Score from 0 to 10. The input contains the question and retrieved context.
The prediction is the generated answer. Evaluate whether every factual
claim is supported by the retrieved context. Ignore completeness.
0 means unsupported or fabricated; 10 means fully grounded.
"""

ANSWER_RELEVANCY_CRITERIA = """
Score from 0 to 10. The input is the user's question, and the prediction
is the generated answer. Evaluate whether the answer directly addresses
all requested parts without unnecessary tangents. Do not use the reference
to judge factual correctness.
0 means irrelevant; 10 means direct, focused, and complete.
"""

RETRIEVER = None
RAG = None
JUDGES = None

def require_environment() -> None:
    required = ("LANGSMITH_API_KEY", "OPENAI_API_KEY")
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise EnvironmentError(
            f"Missing environment variable(s): {', '.join(missing)}"
        )

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
    return "\n\n--- CONTEXT CHUNK ---\n\n".join(
        contexts
    ) or "[NO CONTEXT RETRIEVED]"

def load_rows() -> list[dict[str, Any]]:
    with DATASET_FILE.open(encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "The evaluation file must contain a non-empty JSON list."
        )

    required = {"question", "contexts", "answer", "ground_truth"}

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be a JSON object.")
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"Row {index} is missing fields: {sorted(missing)}"
            )

    return rows

# Uploads the selected dataset to langsmith evaluator in the cloud
def upload_dataset(client: Client, rows: list[dict[str, Any]],) -> None:
    """Create the dataset if it does not already exist."""

    try:
        client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Reusing existing dataset: {DATASET_NAME}")
        return

    except LangSmithNotFoundError:
        pass

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="NABL 112A RAG benchmark with ground-truth answers.",
    )

    examples = [
        {
            "inputs": {
                "question": row["question"],
                "contexts": as_context_list(row["contexts"]),
            },
            "outputs": {
                "answer": row["answer"],
                "ground_truth": row["ground_truth"],
            },
        } for row in rows
    ]

    client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )

    print(f"Uploaded {len(examples)} examples.")

# Returns preset RAG retriever and llm model objects
def get_pipeline() -> tuple[ChromaHybridRag, OpenaiRAG]:
    global RETRIEVER, RAG

    if RETRIEVER is None:
        RETRIEVER = ChromaHybridRag(
            db_path="./chroma_db",
            collection_name="documents",
        )

    if RAG is None:
        #RAG = GeminiRAG(os.environ["GOOGLE_API_KEY"])
        RAG = OpenaiRAG(openai_api_key)

    return RETRIEVER, RAG

# Retrieves relevant chunks and generate answers with llm
def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the production retrieval and generation pipeline."""

    question = str(inputs["question"])
    rag_retriever, llm_model = get_pipeline()
    f_t, max_tokens = query_gen.filter_tokens(question)
    queries = query_gen.generate_queries(
        question=f_t,
        max_l=max_tokens,
    )
    documents = rag_retriever.retrieve_multiple(
        queries,
        top_k=5,
    )
    contexts = [
        str(document["content"])
        for document in documents
        if document.get("content") is not None
    ]
    answer = llm_model.answer(question, documents)
    return {
        "answer": str(answer),
        "contexts": contexts,
    }

# Returns judge llm models
def get_judges() -> dict[str, Any]:
    global JUDGES
    if JUDGES is None:
        criteria = {
            "context_precision": CONTEXT_PRECISION_CRITERIA,
            "context_recall": CONTEXT_RECALL_CRITERIA,
            "faithfulness": FAITHFULNESS_CRITERIA,
            "answer_relevancy": ANSWER_RELEVANCY_CRITERIA,
        }

        judge_llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=openai_api_key,
            temperature=0,
        )

        JUDGES = {}

        for name, description in criteria.items():
            evaluator_type = (
                "labeled_score_string"
                if name == "context_recall"
                else "score_string"
            )

            JUDGES[name] = load_evaluator(
                evaluator_type,
                llm=judge_llm,
                criteria={name: description},
            )

    return JUDGES


def score_out_of_ten(judge, input_text: str, prediction: str, reference: str = "") -> float:
    kwargs = {
        "input": input_text,
        "prediction": prediction,
    }

    if reference:
        kwargs["reference"] = reference

    result = judge.evaluate_strings(**kwargs)

    score = result.get("score")
    if score is None:
        raise ValueError(f"Evaluator returned no score: {result}")

    return float(score) / 10.0

def rag_metrics_evaluator(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any],) -> list[dict[str, Any]]:
    """Return normalized scores from 0.0 to 1.0."""

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
            reference="",
        ),
        "context_recall": score_out_of_ten(
            judges["context_recall"],
            input_text=question,
            prediction=retrieved_context,
            reference=ground_truth,
        ),
        "faithfulness": score_out_of_ten(
            judges["faithfulness"],
            input_text=(
                f"Question:\n{question}\n\n"
                f"Retrieved context:\n{retrieved_context}"
            ),
            prediction=generated_answer,
            reference="",
        ),
        "answer_relevancy": score_out_of_ten(
            judges["answer_relevancy"],
            input_text=question,
            prediction=generated_answer,
            reference="",
        ),
    }

    return [
        {"key": metric, "score": score} for metric, score in scores.items()
    ]

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
    print("Evaluation submitted successfully.")
    print(results)

if __name__ == "__main__":
    main()
