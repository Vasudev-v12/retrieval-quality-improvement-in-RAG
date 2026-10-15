# Ragas benchmark results (judge: gpt-4.1-mini, top_k=5, chunk=1000/200)

| domain | n | context_precision | context_recall | faithfulness | answer_relevancy | answer_correctness | gold_doc_hit_rate | gold_chunk_share |
|---|---|---|---|---|---|---|---|---|
| legal | 50 | 0.574 | 0.710 | 0.939 | 0.879 | 0.411 | 0.960 | 0.514 |
| medical | 50 | 0.880 | 0.681 | 0.941 | 0.741 | 0.398 | 1.000 | 0.220 |
| laboratory | 50 | 0.432 | 0.540 | 0.768 | 0.818 | 0.396 | 0.960 | 0.627 |


# Ragas benchmark results (judge: gpt-4.1-mini, top_k=5, chunk=1000/200, expansion=both)

| domain | n | context_precision | context_recall | faithfulness | answer_relevancy | answer_correctness | gold_doc_hit_rate | gold_chunk_share |
|---|---|---|---|---|---|---|---|---|
| legal | 50 | 0.541 | 0.840 | 0.992 | 0.846 | 0.441 | 0.960 | 0.525 |
| medical | 50 | 0.859 | 0.649 | 0.971 | 0.786 | 0.427 | 1.000 | 0.163 |
| laboratory | 50 | 0.419 | 0.580 | 0.772 | 0.800 | 0.374 | 0.960 | 0.618 |

# Ragas benchmark results (judge: gpt-4.1-mini, top_k=5, chunk=1000/200, baseline: question -> dense search -> LLM)

| domain | n | context_precision | context_recall | faithfulness | answer_relevancy | answer_correctness | gold_doc_hit_rate | gold_chunk_share |
|---|---|---|---|---|---|---|---|---|
| legal | 50 | 0.490 | 0.560 | 0.891 | 0.838 | 0.353 | 0.840 | 0.552 |
| medical | 50 | 0.952 | 0.703 | 0.897 | 0.707 | 0.422 | 1.000 | 0.328 |
| laboratory | 50 | 0.535 | 0.500 | 0.762 | 0.706 | 0.341 | 0.920 | 0.724 |