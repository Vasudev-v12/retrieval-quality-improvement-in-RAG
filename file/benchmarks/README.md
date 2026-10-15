# Domain benchmarks

50 question / reference-answer pairs per domain, each tied to a source document
that is chunked into the RAG store. Built by `prepare_benchmarks.py`, scored by
`../ragas_eval.py`.

| Domain | Questions from | Source documents (corpus) | Corpus size |
|---|---|---|---|
| **legal** (India) | [IndicLegalQA](https://data.mendeley.com/datasets/gf8n8cnmvc) – QA pairs over Supreme Court of India judgments (CC BY 4.0) | Full judgment PDFs from the [Indian Supreme Court Judgments](https://registry.opendata.aws/indian-supreme-court-judgments/) open dataset (CC BY 4.0) | 50 judgments |
| **medical** | [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) `pqa_labeled` – expert-labelled (MIT) | PubMed abstracts; gold abstract for each question + the other 950 labelled abstracts as distractors | 1000 abstracts |
| **laboratory** | [LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench) ProtocolQA (19) + LitQA2 (31) (CC BY-SA 4.0) | ProtocolQA protocol text; LitQA2 open-access full text from [Europe PMC](https://europepmc.org/) + 19 distractor papers | 19 protocols + 50 papers |

## Why these sources

- **Indian datasets first.** IndicLegalQA is the only one of the three domains
  with a public Indian benchmark. It has no judgment text, so each case is
  matched to its judgment PDF by decision date and party names (965/1253
  cases match). Questions are kept only if they name the case, are not
  trivial ("Who is the respondent?") and most of the answer's words appear in
  the judgment.
- **Medical:** Indian medical QA sets (e.g. MedMCQA from AIIMS/NEET-PG) have
  no source documents, so they can't test retrieval. PubMedQA is the standard
  medical QA set that ships its source passage.
- **Laboratory:** NABL publishes accreditation documents but no QA
  benchmark, so LAB-Bench is used. ProtocolQA's 108 protocols are only ~24
  base protocols, each with a different injected error; indexing several
  variants would put contradictory near-duplicates in the store, so only one
  question (and its variant) per base protocol is used. LitQA2 fills the rest,
  using papers whose full text Europe PMC has and that contain the question's
  key passage. The existing NABL 112A set (`../nabl_112a_rag_evaluation_30.json`)
  remains the India-specific lab check via `../eval.py`.

All QA is open-ended: LAB-Bench multiple-choice distractors are kept in
`metadata` but not shown to the pipeline.

## Files

`data/<domain>/questions.json`
```json
{"id", "question", "ground_truth", "source_doc_id", "reference_contexts", "metadata"}
```
`data/<domain>/corpus.jsonl` – one document per line: `{"doc_id", "title", "text", "source"}`.

## Rebuild

```bash
cd file
python benchmarks/prepare_benchmarks.py            # all domains, seed 42
python benchmarks/prepare_benchmarks.py --domains legal --n 50
```
Downloads are cached in `benchmarks/.cache/` (git-ignored).

LAB-Bench asks that its data not be used for model training; its canary
string is kept in each laboratory question's `metadata`.
