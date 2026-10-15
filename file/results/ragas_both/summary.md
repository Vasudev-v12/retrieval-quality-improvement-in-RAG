# Ragas benchmark results (judge: gpt-4.1-mini, top_k=5, chunk=1000/200, expansion=both)

| domain | n | context_precision | context_recall | faithfulness | answer_relevancy | answer_correctness | gold_doc_hit_rate | gold_chunk_share |
|---|---|---|---|---|---|---|---|---|
| legal | 50 | 0.541 | 0.840 | 0.992 | 0.846 | 0.441 | 0.960 | 0.525 |
| medical | 50 | 0.859 | 0.649 | 0.971 | 0.786 | 0.427 | 1.000 | 0.163 |
| laboratory | 50 | 0.419 | 0.580 | 0.772 | 0.800 | 0.374 | 0.960 | 0.618 |
