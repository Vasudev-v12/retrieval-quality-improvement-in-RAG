"""Score the RAG pipeline on the domain benchmarks with Ragas.

For each domain (built by benchmarks/prepare_benchmarks.py):
  1. Chunk the domain corpus into its own Chroma collection ("bench_<domain>"),
     so the benchmarks never touch the main "documents" collection.
  2. Run the production pipeline on every question
     (query_gen keyword queries -> ChromaHybridRag dense+BM25 ->
     context_expansion centroid / neighbor-window second pass -> OpenaiRAG answer).
  3. Score with Ragas: context precision, context recall, faithfulness,
     answer relevancy and answer correctness, plus a non-LLM retrieval check
     (did any retrieved chunk come from the question's gold source document?).

With --baseline, step 2 is replaced by naive RAG: one dense similarity search
with the raw question -> top_k chunks -> OpenaiRAG answer, so the pipeline's
gains can be measured against it.

Pipeline outputs are cached in results/ragas/<domain>_outputs.json (or
results/ragas_<mode>/ for an --expansion other than "none", and
results/ragas_baseline/ for --baseline) and scores in <domain>_scores.csv;
a re-run only fills in missing or failed (NaN) score cells. Use
--rerun-pipeline after changing the pipeline, --rescore to re-judge.

Required environment variables:
    OPENAI_API_KEY=your_openai_key

Usage (from the `file` directory):
    python ragas_eval.py                                # all three domains, 50 questions each
    python ragas_eval.py --domains medical --limit 5    # quick smoke test
    python ragas_eval.py --expansion none               # full pipeline without the second pass
    python ragas_eval.py --baseline                     # naive RAG: question -> dense search -> LLM
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AnswerCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402

import rag_optimizer.query_gen as query_gen  # noqa: E402
from rag_optimizer.context_expansion import EXPANSION_MODES, ContextExpander  # noqa: E402
from rag_optimizer.hybridRag import ChromaHybridRag  # noqa: E402
from rag_optimizer.llm import OpenaiRAG  # noqa: E402

load_dotenv()

DOMAINS = ("legal", "medical", "laboratory")
BENCH_DIR = Path(__file__).with_name("benchmarks") / "data"
RESULTS_DIR = Path(__file__).with_name("results") / "ragas"
DB_PATH = "./chroma_db"

TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_domain(domain: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folder = BENCH_DIR / domain
    if not (folder / "questions.json").exists():
        raise FileNotFoundError(
            f"{folder} not found - run `python benchmarks/prepare_benchmarks.py` first."
        )
    questions = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    with (folder / "corpus.jsonl").open(encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]
    return questions, corpus


def build_index(domain: str, corpus: list[dict[str, Any]], rebuild: bool) -> ChromaHybridRag:
    """Chunk the corpus into a per-domain collection (only if it is empty)."""
    name = f"bench_{domain}"
    retriever = ChromaHybridRag(db_path=DB_PATH, collection_name=name)
    if rebuild and retriever.collection.count():
        retriever.client.delete_collection(name)
        retriever = ChromaHybridRag(db_path=DB_PATH, collection_name=name)

    if retriever.collection.count() == 0:
        print(f"[{domain}] indexing {len(corpus)} documents into '{name}' ...")
        for doc in corpus:
            # document_id = gold doc id, so chunk ids are "<doc_id>_chunk_<i>"
            # and retrieval hits can be traced back to their source document.
            retriever.insert_string(
                doc["text"],
                source_name=doc["title"][:200],
                document_id=doc["doc_id"],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
    print(f"[{domain}] collection '{name}' has {retriever.collection.count()} chunks")
    return retriever


def run_pipeline(question: str, retriever: ChromaHybridRag, rag: OpenaiRAG,
                 expansion: str) -> dict[str, Any]:
    """Keyword queries -> hybrid retrieval -> second-pass expansion -> answer."""
    queries = [entry["query"] for entry in query_gen.build_queries(question)["queries"]]
    documents = retriever.retrieve_multiple(queries, top_k=TOP_K)
    documents = ContextExpander(retriever).expand(documents, mode=expansion)
    return {
        "queries": queries,
        "retrieved_ids": [doc["id"] for doc in documents],
        "expansion_sources": [doc["expansion"] for doc in documents],
        "contexts": [str(doc["content"]) for doc in documents if doc.get("content") is not None],
        "answer": str(rag.answer(question, documents)),
    }


def run_baseline(question: str, retriever: ChromaHybridRag, rag: OpenaiRAG) -> dict[str, Any]:
    """Naive RAG: one dense similarity search with the raw question -> answer."""
    results = retriever.collection.query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "distances"],
    )
    documents = [
        {"id": doc_id, "content": content, "distance": dist}
        for doc_id, content, dist in zip(results["ids"][0], results["documents"][0], results["distances"][0])
    ]
    return {
        "queries": [question],
        "retrieved_ids": [doc["id"] for doc in documents],
        "contexts": [str(doc["content"]) for doc in documents if doc.get("content") is not None],
        "answer": str(rag.answer(question, documents)),
    }


def collect_outputs(domain: str, questions: list[dict[str, Any]], retriever: ChromaHybridRag,
                    rag: OpenaiRAG, rerun: bool, expansion: str, baseline: bool) -> list[dict[str, Any]]:
    cache_file = RESULTS_DIR / f"{domain}_outputs.json"
    cache = {}
    if cache_file.exists() and not rerun:
        cache = {row["id"]: row for row in json.loads(cache_file.read_text(encoding="utf-8"))}

    rows = []
    for i, q in enumerate(questions, start=1):
        if q["id"] not in cache:
            print(f"[{domain}] {i}/{len(questions)} {q['question'][:80]}")
            if baseline:
                out = run_baseline(q["question"], retriever, rag)
            else:
                out = run_pipeline(q["question"], retriever, rag, expansion)
            prefix = f"{q['source_doc_id']}_chunk_"
            gold_chunks = sum(1 for cid in out["retrieved_ids"] if cid.startswith(prefix))
            cache[q["id"]] = {
                "id": q["id"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "source_doc_id": q["source_doc_id"],
                **out,
                "gold_doc_hit": gold_chunks > 0,
                "gold_chunk_share": gold_chunks / len(out["retrieved_ids"]) if out["retrieved_ids"] else 0.0,
            }
            # Save after every question so an interrupted run can resume.
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(list(cache.values()), indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        rows.append(cache[q["id"]])
    return rows


def make_metrics() -> list:
    # AnswerCorrectness rather than FactualCorrectness: it sees the question
    # when splitting the reference into claims, so short references such as
    # "30%" or "Saponaria vaccaria" (LitQA2) can still be matched.
    return [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
        AnswerCorrectness(),
    ]


def ragas_score(domain: str, rows: list[dict[str, Any]], judge_model: str, workers: int,
                rescore: bool) -> pd.DataFrame:
    """Score every (row, metric) cell that is missing or failed in the saved CSV.

    Judge calls sometimes time out and Ragas records NaN; re-running the
    script retries just those cells instead of re-scoring everything.
    """
    scores_file = RESULTS_DIR / f"{domain}_scores.csv"
    metrics = make_metrics()
    names = [m.name for m in metrics]

    df = pd.DataFrame({
        "id": [row["id"] for row in rows],
        "user_input": [row["question"] for row in rows],
        "retrieved_contexts": [row["contexts"] for row in rows],
        "response": [row["answer"] for row in rows],
        "reference": [row["ground_truth"] for row in rows],
    })
    if scores_file.exists() and not rescore:
        saved = pd.read_csv(scores_file).set_index("id")
        for name in names:
            if name in saved:
                df[name] = df["id"].map(saved[name])

    llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    for metric in metrics:
        if metric.name not in df:
            df[metric.name] = float("nan")
        todo = [i for i in range(len(rows)) if pd.isna(df.at[i, metric.name])]
        if not todo:
            continue
        print(f"[{domain}] {metric.name}: scoring {len(todo)} rows")
        dataset = EvaluationDataset.from_list([
            {
                "user_input": rows[i]["question"],
                "retrieved_contexts": rows[i]["contexts"] or ["[NO CONTEXT RETRIEVED]"],
                "response": rows[i]["answer"],
                "reference": rows[i]["ground_truth"],
            }
            for i in todo
        ])
        result = evaluate(
            dataset=dataset,
            metrics=[metric],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=workers, timeout=300),
        )
        df.loc[todo, metric.name] = result.to_pandas()[metric.name].to_numpy()
        df.to_csv(scores_file, index=False, encoding="utf-8")

    df["gold_doc_hit"] = [row["gold_doc_hit"] for row in rows]
    df["gold_chunk_share"] = [row["gold_chunk_share"] for row in rows]
    df["n_contexts"] = [len(row["contexts"]) for row in rows]
    df.to_csv(scores_file, index=False, encoding="utf-8")
    return df


METRIC_COLUMNS = {
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "answer_correctness": "answer_correctness",
    "gold_doc_hit": "gold_doc_hit_rate",
    "gold_chunk_share": "gold_chunk_share",
}


def summarize(domain: str, df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"domain": domain, "questions": len(df)}
    for column, label in METRIC_COLUMNS.items():
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce")
            summary[label] = round(float(values.mean()), 4)
            if values.isna().any():
                summary[f"{label}_failed"] = int(values.isna().sum())
    return summary


def write_report(summaries: list[dict[str, Any]], judge_model: str, pipeline: str) -> None:
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    labels = list(METRIC_COLUMNS.values())
    lines = [
        f"# Ragas benchmark results (judge: {judge_model}, top_k={TOP_K}, chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}, "
        f"{pipeline})",
        "",
        "| domain | n | " + " | ".join(labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for s in summaries:
        lines.append(f"| {s['domain']} | {s['questions']} | "
                     + " | ".join(f"{s.get(label, float('nan')):.3f}" for label in labels) + " |")
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary.md").write_text(report, encoding="utf-8")
    print("\n" + report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions per domain")
    parser.add_argument("--judge-model", default="gpt-4.1-mini", help="LLM used by the Ragas metrics")
    parser.add_argument("--workers", type=int, default=8, help="parallel Ragas judge calls")
    parser.add_argument("--rebuild-index", action="store_true", help="re-chunk the benchmark corpora")
    parser.add_argument("--rerun-pipeline", action="store_true", help="ignore cached pipeline outputs (implies --rescore)")
    parser.add_argument("--rescore", action="store_true", help="ignore saved Ragas scores")
    parser.add_argument("--expansion", choices=EXPANSION_MODES, default="both",
                        help="second-pass context expansion (centroid search / neighbor window)")
    parser.add_argument("--baseline", action="store_true",
                        help="naive RAG: dense search with the raw question, no query generation, "
                             "BM25 or expansion (ignores --expansion)")
    args = parser.parse_args()

    # Each mode gets its own results folder so runs can be compared
    # and cached outputs from one mode are never reused for another.
    global RESULTS_DIR
    if args.baseline:
        RESULTS_DIR = RESULTS_DIR.with_name("ragas_baseline")
        pipeline = "baseline: question -> dense search -> LLM"
    else:
        if args.expansion != "none":
            RESULTS_DIR = RESULTS_DIR.with_name(f"ragas_{args.expansion}")
        pipeline = f"expansion={args.expansion}"

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rag = OpenaiRAG(openai_api_key)
    summaries = []
    for domain in args.domains:
        questions, corpus = load_domain(domain)
        questions = questions[: args.limit] if args.limit else questions
        retriever = build_index(domain, corpus, args.rebuild_index)
        rows = collect_outputs(domain, questions, retriever, rag, args.rerun_pipeline,
                               args.expansion, args.baseline)
        print(f"[{domain}] scoring {len(rows)} answers with Ragas ...")
        df = ragas_score(domain, rows, args.judge_model, args.workers,
                         rescore=args.rerun_pipeline or args.rescore)
        summaries.append(summarize(domain, df))

    write_report(summaries, args.judge_model, pipeline)


if __name__ == "__main__":
    main()
