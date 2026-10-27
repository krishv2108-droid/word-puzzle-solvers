## Fusion weight sweep on DEV (n=100 puzzles with LLM candidates, out-of-fold scorer)

| LLM bonus (log-odds per unit confidence) | Avg groups (of 4) | 95% CI | Fully solved | Gain vs learned alone |
|---|---|---|---|---|
| 0 | 0.68 | 0.50 to 0.88 | 4.0% | +0.00 (+0.00 to +0.00) |
| 0.5 | 0.74 | 0.55 to 0.95 | 4.0% | +0.06 (+0.01 to +0.12) |
| 1 | 0.76 | 0.57 to 0.96 | 4.0% | +0.08 (+0.00 to +0.17) |
| 2 | 0.81 | 0.61 to 1.03 | 6.0% | +0.13 (+0.02 to +0.26) |
| 4 | 0.82 | 0.62 to 1.03 | 5.0% | +0.14 (+0.01 to +0.28) |
| 8 | 0.81 | 0.61 to 1.03 | 6.0% | +0.13 (-0.07 to +0.33) |
| 16 | 0.74 | 0.55 to 0.95 | 5.0% | +0.06 (-0.17 to +0.28) |
| 32 | 0.66 | 0.48 to 0.86 | 4.0% | -0.02 (-0.24 to +0.19) |

Selected bonus (best dev mean; ties to the smaller): **4**
