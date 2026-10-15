# Ragas benchmark results (judge: gpt-4.1-mini, top_k=5, chunk=1000/200, baseline: question -> dense search -> LLM)

| domain | n | context_precision | context_recall | faithfulness | answer_relevancy | answer_correctness | gold_doc_hit_rate | gold_chunk_share |
|---|---|---|---|---|---|---|---|---|
| legal | 50 | 0.490 | 0.560 | 0.891 | 0.838 | 0.353 | 0.840 | 0.552 |
| medical | 50 | 0.952 | 0.703 | 0.897 | 0.707 | 0.422 | 1.000 | 0.328 |
| laboratory | 50 | 0.535 | 0.500 | 0.762 | 0.706 | 0.341 | 0.920 | 0.724 |
