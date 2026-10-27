## Wordle — 2315 original answers, 12972 allowed guesses

| Approach | N | Solve rate | Win ≤6 | Avg steps | Worst | ms/puzzle (mean / p95) |
|---|---|---|---|---|---|---|
| frequency | 2315 | 100.00% | 99.35% | 3.622 | 9 | 0.66 / 0.74 |
| entropy | 2315 | 100.00% | 100.00% | 3.464 | 6 | 18.19 / 29.39 |
| minimax | 2315 | 100.00% | 100.00% | 3.514 | 5 | 18.98 / 29.27 |
| frequency (hard) | 2315 | 100.00% | 99.35% | 3.622 | 9 | 0.67 / 0.75 |
| entropy (hard) | 2315 | 100.00% | 99.70% | 3.602 | 8 | 3.86 / 18.14 |
| minimax (hard) | 2315 | 100.00% | 99.48% | 3.651 | 8 | 3.95 / 17.71 |

**frequency** — guess-count distribution
```
 1 | █ 1 (0.0%)
 2 | ███ 132 (5.7%)
 3 | ████████████████████████ 955 (41.3%)
 4 | ████████████████████████ 957 (41.3%)
 5 | █████ 214 (9.2%)
 6 | █ 41 (1.8%)
 7 | █ 10 (0.4%)
 8 | █ 3 (0.1%)
 9 | █ 2 (0.1%)
```

**entropy** — guess-count distribution
```
 2 | █ 44 (1.9%)
 3 | ████████████████████████ 1217 (52.6%)
 4 | ████████████████████ 990 (42.8%)
 5 | █ 63 (2.7%)
 6 | █ 1 (0.0%)
```

**minimax** — guess-count distribution
```
 1 | █ 1 (0.0%)
 2 | █ 59 (2.5%)
 3 | ███████████████████████ 1065 (46.0%)
 4 | ████████████████████████ 1129 (48.8%)
 5 | █ 61 (2.6%)
```

**frequency (hard)** — guess-count distribution
```
 1 | █ 1 (0.0%)
 2 | ███ 132 (5.7%)
 3 | ████████████████████████ 955 (41.3%)
 4 | ████████████████████████ 957 (41.3%)
 5 | █████ 214 (9.2%)
 6 | █ 41 (1.8%)
 7 | █ 10 (0.4%)
 8 | █ 3 (0.1%)
 9 | █ 2 (0.1%)
```

**entropy (hard)** — guess-count distribution
```
 2 | ██ 79 (3.4%)
 3 | ████████████████████████ 1041 (45.0%)
 4 | ██████████████████████ 969 (41.9%)
 5 | ████ 183 (7.9%)
 6 | █ 36 (1.6%)
 7 | █ 5 (0.2%)
 8 | █ 2 (0.1%)
```

**minimax (hard)** — guess-count distribution
```
 1 | █ 1 (0.0%)
 2 | ██ 97 (4.2%)
 3 | ██████████████████████ 931 (40.2%)
 4 | ████████████████████████ 1031 (44.5%)
 5 | █████ 202 (8.7%)
 6 | █ 41 (1.8%)
 7 | █ 11 (0.5%)
 8 | █ 1 (0.0%)
```

_Games are capped at 12 guesses; 'Win ≤6' is the official win rate. ms/puzzle excludes one-time setup (word lists + pattern matrix loaded from the on-disk cache in 0.01s; a cold build is ~1.8s; then per solver, incl. opening-move search: frequency: 0.00s, entropy: 0.11s, minimax: 0.10s, frequency (hard): 0.00s, entropy (hard): 0.10s, minimax (hard): 0.10s)._
