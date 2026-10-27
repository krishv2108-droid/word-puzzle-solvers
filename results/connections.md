> **Upper bound.** Every puzzle here predates the models' training cutoffs, so LLM scores are an upper bound on performance on genuinely new puzzles. No clean post-cutoff holdout exists in this project. The memorization probe below only detects puzzles a model can recite from a date; finding none does not prove a puzzle is unseen.

### Memorization probe
- **qwen3-8b local** recalls 0.00 of 4 category names from a puzzle's number and date alone (chance control: 0.01); 0% of puzzles flagged likely-seen (>=2 of 4 matched).

## All 150 puzzles

| Approach | N | Fully solved | Avg groups (of 4) | Yellow | Green | Blue | Purple | ms/puzzle (mean / p95) | $/puzzle | In tok | Out tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 150 | 0.00% | 0.01 | 0.0% | 0.7% | 0.0% | 0.0% | 0.01 / 0.01 | $0.0000 | 0 | 0 |
| embedding (all-MiniLM-L6-v2) | 150 | 1.33% | 0.39 | 14.7% | 12.0% | 7.3% | 5.3% | 69.42 / 29.78 | $0.0000 | 0 | 0 |
| embedding (all-mpnet-base-v2) | 150 | 2.67% | 0.59 | 20.7% | 18.0% | 13.3% | 6.7% | 50.25 / 50.14 | $0.0000 | 0 | 0 |
| learned pair scorer | 150 | 3.33% | 0.70 | 25.3% | 20.0% | 15.3% | 9.3% | 173.26 / 163.65 | $0.0000 | 0 | 0 |
| llm (qwen3-8b local) | 150 | 2.00% | 0.43 | 19.3% | 11.3% | 6.7% | 5.3% | 9716.10 / 11201.15 | $0.0000 | 363 | 195 |
| hybrid, no embeddings (qwen3-8b local) | 150 | 1.33% | 0.36 | 16.0% | 8.7% | 8.7% | 2.7% | 25897.49 / 53552.44 | $0.0000 | 378 | 561 |
| hybrid (qwen3-8b local) | 150 | 3.33% | 0.55 | 19.3% | 16.0% | 14.0% | 6.0% | 25879.35 / 53550.42 | $0.0000 | 378 | 561 |
| learned + LLM (qwen3-8b local) | 150 | 6.00% | 0.83 | 29.3% | 24.0% | 20.0% | 9.3% | 25895.94 / 53563.41 | $0.0000 | 378 | 561 |

### Statistical uncertainty
95% bootstrap confidence intervals, n=150 puzzles.

| Approach | Avg groups (of 4) | 95% CI | Fully solved | 95% CI |
|---|---|---|---|---|
| random | 0.01 | 0.00 to 0.02 | 0.0% | 0.0% to 0.0% |
| embedding (all-MiniLM-L6-v2) | 0.39 | 0.29 to 0.51 | 1.3% | 0.0% to 3.3% |
| embedding (all-mpnet-base-v2) | 0.59 | 0.45 to 0.73 | 2.7% | 0.7% to 5.3% |
| learned pair scorer | 0.70 | 0.55 to 0.85 | 3.3% | 0.7% to 6.7% |
| llm (qwen3-8b local) | 0.43 | 0.30 to 0.57 | 2.0% | 0.0% to 4.7% |
| hybrid, no embeddings (qwen3-8b local) | 0.36 | 0.25 to 0.48 | 1.3% | 0.0% to 3.3% |
| hybrid (qwen3-8b local) | 0.55 | 0.41 to 0.71 | 3.3% | 0.7% to 6.7% |
| learned + LLM (qwen3-8b local) | 0.83 | 0.66 to 1.01 | 6.0% | 2.7% to 10.0% |

Paired differences in average groups (A minus B, same puzzles):

| A | B | A - B | 95% CI | Significant at 95%? |
|---|---|---|---|---|
| learned + LLM (qwen3-8b local) | embedding (all-mpnet-base-v2) | +0.24 | +0.11 to +0.37 | yes |
| learned + LLM (qwen3-8b local) | learned pair scorer | +0.13 | +0.03 to +0.24 | yes |
| learned pair scorer | embedding (all-mpnet-base-v2) | +0.11 | +0.01 to +0.22 | yes |
| learned + LLM (qwen3-8b local) | hybrid (qwen3-8b local) | +0.27 | +0.15 to +0.41 | yes |
| hybrid (qwen3-8b local) | embedding (all-mpnet-base-v2) | -0.03 | -0.17 to +0.11 | no |
| hybrid (qwen3-8b local) | llm (qwen3-8b local) | +0.13 | +0.00 to +0.26 | no |
| hybrid (qwen3-8b local) | hybrid, no embeddings (qwen3-8b local) | +0.19 | +0.11 to +0.29 | yes |
| embedding (all-mpnet-base-v2) | llm (qwen3-8b local) | +0.16 | +0.01 to +0.32 | yes |
| embedding (all-mpnet-base-v2) | embedding (all-MiniLM-L6-v2) | +0.19 | +0.09 to +0.30 | yes |

## Most recent third (2024-08-11 to 2025-02-28, n=50)

| Approach | N | Fully solved | Avg groups (of 4) | Yellow | Green | Blue | Purple | ms/puzzle (mean / p95) | $/puzzle | In tok | Out tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 50 | 0.00% | 0.00 | 0.0% | 0.0% | 0.0% | 0.0% | 0.01 / 0.01 | $0.0000 | 0 | 0 |
| embedding (all-MiniLM-L6-v2) | 50 | 4.00% | 0.50 | 18.0% | 12.0% | 14.0% | 6.0% | 26.63 / 28.18 | $0.0000 | 0 | 0 |
| embedding (all-mpnet-base-v2) | 50 | 2.00% | 0.58 | 20.0% | 16.0% | 16.0% | 6.0% | 43.38 / 51.70 | $0.0000 | 0 | 0 |
| learned pair scorer | 50 | 0.00% | 0.54 | 26.0% | 14.0% | 12.0% | 2.0% | 133.80 / 183.38 | $0.0000 | 0 | 0 |
| llm (qwen3-8b local) | 50 | 0.00% | 0.22 | 10.0% | 10.0% | 2.0% | 0.0% | 9232.00 / 10832.25 | $0.0000 | 364 | 199 |
| hybrid, no embeddings (qwen3-8b local) | 50 | 0.00% | 0.18 | 8.0% | 4.0% | 6.0% | 0.0% | 22054.35 / 31381.23 | $0.0000 | 379 | 556 |
| hybrid (qwen3-8b local) | 50 | 2.00% | 0.40 | 14.0% | 8.0% | 14.0% | 4.0% | 22028.72 / 31352.01 | $0.0000 | 379 | 556 |
| learned + LLM (qwen3-8b local) | 50 | 0.00% | 0.60 | 26.0% | 20.0% | 14.0% | 0.0% | 22049.01 / 31371.76 | $0.0000 | 379 | 556 |

## Split by memorization flag (qwen3-8b local probe)

| Approach | N | Fully solved | Avg groups (of 4) | Yellow | Green | Blue | Purple | ms/puzzle (mean / p95) |
|---|---|---|---|---|---|---|---|---|
| llm (qwen3-8b local) - not flagged | 150 | 2.00% | 0.43 | 19.3% | 11.3% | 6.7% | 5.3% | 9716.10 / 11201.15 |
| hybrid, no embeddings (qwen3-8b local) - not flagged | 150 | 1.33% | 0.36 | 16.0% | 8.7% | 8.7% | 2.7% | 25897.49 / 53552.44 |
| hybrid (qwen3-8b local) - not flagged | 150 | 3.33% | 0.55 | 19.3% | 16.0% | 14.0% | 6.0% | 25879.35 / 53550.42 |
