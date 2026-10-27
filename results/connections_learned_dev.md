## Learned pair scorer: ablation on DEV (n=478, 5-fold CV by puzzle)

| Features | Avg groups (of 4) | 95% CI | Fully solved | Paired gain vs baseline (95% CI) |
|---|---|---|---|---|
| mpnet cosine, untrained (current baseline) | 0.72 | 0.63 to 0.80 | 3.6% | n/a |
| mpnet cosine only (trained) | 0.74 | 0.66 to 0.83 | 4.2% | +0.03 (-0.01 to +0.07) |
| mpnet embedding features | 0.75 | 0.67 to 0.84 | 4.0% | +0.04 (-0.01 to +0.08) |
| mpnet + MiniLM embedding features | 0.79 | 0.71 to 0.89 | 4.6% | +0.08 (+0.02 to +0.15) |
| bge-large embedding features | 0.71 | 0.63 to 0.80 | 3.3% | -0.00 (-0.09 to +0.08) |
| mpnet + MiniLM + bge-large embedding features | 0.89 | 0.79 to 0.99 | 6.3% | +0.17 (+0.10 to +0.25) |
| string features only | 0.06 | 0.04 to 0.09 | 0.0% | -0.65 (-0.74 to -0.56) |
| all features (3 embedding models + string) | 0.90 | 0.80 to 1.00 | 6.3% | +0.18 (+0.11 to +0.26) |
