---
title: Word Puzzle Solvers
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Wordle and Connections solvers with an interactive UI
---

# Word Puzzle Solvers

Interactive solvers for Wordle and the Connections word-grouping puzzle.

- **Wordle assistant**: enter the word you played and the colours; get the best next guesses
  ranked by expected information gain.
- **Connections**: enter 16 words; get a proposed grouping with a calibrated confidence per
  group, or ask for one group at a time.

On 150 held-out past Connections puzzles the solver gets about 0.7 of the 4 groups right on
average, so treat its suggestions as hints.

Source code, evaluation harness and full results: see the project repository on GitHub.
Not affiliated with The New York Times. The Connections model was trained on a public dataset of
past puzzles; no puzzle text is stored in this Space.
