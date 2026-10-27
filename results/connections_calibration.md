## Group confidence calibration (fit on DEV out-of-fold, checked on TEST, 600 groups)

Brier score: **0.116** (always predicting the base rate: 0.144); expected calibration error: **0.026**

| Stated confidence | Groups | Mean stated | Actually correct |
|---|---|---|---|
| 0% to 10% | 212 | 6.7% | 4.7% |
| 10% to 20% | 197 | 13.9% | 12.7% |
| 20% to 35% | 103 | 26.4% | 23.3% |
| 35% to 50% | 37 | 41.0% | 35.1% |
| 50% to 100% | 51 | 72.2% | 64.7% |
