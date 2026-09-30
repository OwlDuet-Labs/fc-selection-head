### BFCL across-category selector comparison — 0.6B-cats (base `mlx-community/Qwen3-0.6B-bf16`)

| category | n | vanilla acc | trained acc | Δ |
|---|---:|---:|---:|---:|
| live_multiple | 134 | 80.6 | 91.0 | +10.4 |
| live_irrelevance | 127 | 55.9 | 96.1 | +40.2 |

_Scoring: live_multiple/live_simple = exact-match (pred == gold index); live_irrelevance = abstain-correct (pred == NONE index); live_relevance = call-correct (pred != NONE index). See data/build_bfcl_router_split.py for the abstention convention._
