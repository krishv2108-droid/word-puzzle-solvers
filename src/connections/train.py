"""Train the Connections model that the web app serves, on ALL available puzzles.

    python -m src.connections.train          # writes data/cache/connections_model.joblib

The saved bundle holds the pair scorer and the confidence calibrator (fitted on out-of-fold
predictions over every puzzle). It is derived from NYT puzzle data, so it is gitignored and
built locally; it is not the model evaluated in results/ (that one is trained on dev puzzles
only so the test sample stays unseen). Reported accuracy comes from results/, never from this model.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from .calibration import Calibrator, collect
from .data import load_puzzles
from .experiments import BGE, MINILM, MPNET, embedders, oof_logits
from .learned import feature_matrix, fit_pair_model

BUNDLE = Path(__file__).resolve().parents[2] / "data" / "cache" / "connections_model.joblib"


def main() -> None:
    puzzles = load_puzzles()
    X, y, names = feature_matrix(puzzles, embedders())
    cols = list(range(len(names)))
    calibrator = Calibrator().fit(*collect(puzzles, oof_logits(puzzles, X, y, cols)))
    model = fit_pair_model(X[:, :, cols], y)
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "names": names, "cols": cols, "calibrator": calibrator,
                 "embedders": {"mpnet": MPNET, "minilm": MINILM, "bge": BGE}}, BUNDLE)
    print(f"trained on {len(puzzles)} puzzles; saved {BUNDLE}")


if __name__ == "__main__":
    main()
