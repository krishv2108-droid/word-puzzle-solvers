"""Regression tests for bugs found in a full debugging pass. Each test fails on the old code."""
import glob
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.connections_service import ConnectionsService
from src.api.main import app
from src.api.wordle_service import WordleService
from src.common.cachefile import atomic_open, load_npy, save_npy
from src.common.evaluate import EpisodeResult, summarize
from src.common.llm import LLMReply, ReplyCache
from src.common.solver import Usage
from src.connections import partition as P
from src.connections.solvers import baselines
from src.connections.solvers.baselines import EmbeddingModel
from src.wordle.engine import feedback_to_str, score_guess
from src.wordle.patterns import build_pattern_matrix, load_pattern_matrix

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- NaN is not valid JSON ---------------------------------------------------------------

def test_summary_of_an_approach_that_never_solves_is_strict_json():
    results = [EpisodeResult("1", False, 1, 0.01, Usage(), {"groups_correct": 0.0})] * 3
    d = summarize("never solves", results).to_dict()
    assert d["mean_steps_solved"] is None
    json.dumps(d, allow_nan=False)  # raises on NaN/Infinity


@pytest.mark.parametrize("path", sorted(glob.glob(str(ROOT / "results" / "**" / "*.json"), recursive=True)))
def test_committed_results_are_strict_json(path):
    def reject(constant):
        raise AssertionError(f"{path} contains non-standard JSON constant {constant}")

    json.load(open(path), parse_constant=reject)


# --- caches must survive interrupted runs and read-only filesystems ------------------------

def test_atomic_write_never_leaves_a_partial_file(tmp_path):
    target = tmp_path / "cache.bin"
    target.write_bytes(b"old complete contents")
    with pytest.raises(RuntimeError), atomic_open(target) as f:
        f.write(b"half of the new conten")
        raise RuntimeError("process killed mid-write")
    assert target.read_bytes() == b"old complete contents"
    assert [p.name for p in tmp_path.iterdir()] == ["cache.bin"]  # no temp file left behind


def test_load_npy_treats_truncated_missing_and_wrong_shape_as_misses(tmp_path):
    good = tmp_path / "a.npy"
    assert save_npy(good, np.arange(6).reshape(2, 3))
    assert load_npy(good, shape=(2, 3)) is not None
    assert load_npy(good, shape=(3, 2)) is None
    (tmp_path / "b.npy").write_bytes(good.read_bytes()[:20])  # truncated
    assert load_npy(tmp_path / "b.npy") is None
    assert load_npy(tmp_path / "missing.npy") is None


def test_save_npy_reports_failure_instead_of_raising_on_a_read_only_directory(tmp_path):
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert save_npy(tmp_path / "x.npy", np.zeros(3)) is False
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)


def test_llm_reply_cache_ignores_a_corrupt_entry_and_survives_read_only_dirs(tmp_path):
    cache, req = ReplyCache(tmp_path), {"q": 1}
    cache._path(req).write_text('{"data": {"x"')  # run was killed while writing
    assert cache.get(req) is None
    reply = LLMReply({"ok": True}, None, Usage(input_tokens=5), 1.5)
    cache.put(req, reply)  # overwrites the corrupt entry
    hit = cache.get(req)
    assert hit.data == {"ok": True} and hit.from_cache and hit.usage.input_tokens == 5

    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        cache.put({"q": 2}, reply)  # must not raise
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)


def test_pattern_matrix_cache_rebuilds_when_the_file_is_corrupt(tmp_path):
    words = ["crane", "slate", "irate", "trace"]
    first = load_pattern_matrix(words, words, cache_dir=tmp_path)
    (cache_file,) = tmp_path.glob("*.npy")
    cache_file.write_bytes(cache_file.read_bytes()[:30])
    again = load_pattern_matrix(words, words, cache_dir=tmp_path)
    assert (again == first).all() and (again == build_pattern_matrix(words, words)).all()


def test_partition_table_cache_rebuilds_when_the_file_is_corrupt(tmp_path, monkeypatch):
    bad_cache = tmp_path / "partitions16.npy"
    bad_cache.write_bytes(b"\x93NUMPY garbage")
    monkeypatch.setattr(P, "CACHE", bad_cache)
    P.all_partitions.cache_clear()
    try:
        parts = P.all_partitions()
        assert parts.shape == (2_627_625, 4) and load_npy(bad_cache, shape=(2_627_625, 4)) is not None
    finally:
        P.all_partitions.cache_clear()


# --- embedding cache: a web server must neither crash nor grow without bound ---------------

class FakeEncoder:
    def __init__(self):
        self.kwargs = None

    def encode(self, texts, **kw):
        return np.tile(np.array([[0.6, 0.8]]), (len(texts), 1))


def test_embedding_model_keeps_working_when_the_cache_directory_is_read_only(tmp_path):
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        model = EmbeddingModel("fake/readonly", cache_dir=tmp_path)
        model._model = FakeEncoder()
        out = model.vectors(["A", "B"])  # used to raise PermissionError
        assert out.shape == (2, 2)
        assert model._path is None  # persistence switched off after the first failure
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)


def test_embedding_model_memory_cache_is_bounded_and_results_are_still_correct():
    model = EmbeddingModel("fake/bounded", cache_dir=None, max_entries=10)
    model._model = FakeEncoder()
    for i in range(20):
        out = model.vectors([f"w{i}-{j}" for j in range(16)])  # 16 new words > the bound itself
        assert out.shape == (16, 2)
    assert len(model._cache) <= 10


def test_embedding_model_ignores_a_corrupt_disk_cache(tmp_path):
    (tmp_path / "emb_fake_corrupt.npz").write_bytes(b"PK\x03\x04 truncated")
    model = EmbeddingModel("fake/corrupt", cache_dir=tmp_path)
    model._model = FakeEncoder()
    assert model.vectors(["A"]).shape == (1, 2)
    assert EmbeddingModel("fake/corrupt", cache_dir=tmp_path)._cache  # rewritten atomically, loads fine


def test_known_models_are_loaded_at_a_pinned_revision(monkeypatch):
    seen = {}

    class Spy(FakeEncoder):
        def __init__(self, name, **kw):
            seen[name] = kw

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Spy)
    for name in baselines.MODEL_REVISIONS:
        EmbeddingModel(name, cache_dir=None).vectors(["x"])
    assert {n: kw["revision"] for n, kw in seen.items()} == baselines.MODEL_REVISIONS
    assert all(len(r) == 40 for r in baselines.MODEL_REVISIONS.values())


def test_connections_service_refuses_to_load_without_its_libraries(monkeypatch, tmp_path):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "sentence_transformers" else real(name, *a, **k))
    with pytest.raises(ImportError, match="sentence_transformers"):
        ConnectionsService.load(tmp_path / "model.joblib")  # fails now, not as a 500 on the first request


# --- HTTP behaviour --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ["/", "/api/health", "/static/app.js"])
def test_head_requests_are_allowed(client, path):
    # uptime monitors and link-preview bots send HEAD; it used to be 405 on / and /api/health
    assert client.head(path).status_code == 200
    assert client.get(path).status_code == 200


def test_static_files_must_be_revalidated_so_deploys_are_not_shadowed(client):
    assert client.get("/static/style.css").headers["cache-control"] == "no-cache"


def test_no_match_message_explains_that_the_answer_may_be_a_newer_word():
    svc = WordleService()
    r = svc.suggest([("crane", "ggggg"), ("slate", "ggggg")])  # contradictory
    assert r["remaining"] == 0 and "newer word" in r["message"] and "2,315" in r["message"]
    # a real game whose answer is outside the original list ('aahed' is allowed but never an answer)
    r = svc.suggest([("soare", feedback_to_str(score_guess("soare", "aahed")))])
    assert r["remaining"] > 0 or "newer word" in r["message"]


# --- reproducibility pins ----------------------------------------------------------------------

def test_wordle_word_lists_match_the_pinned_checksums():
    sums = load_script("download_wordle").SHA256
    for name, expected in sums.items():
        assert hashlib.sha256((ROOT / "data" / "wordle" / name).read_bytes()).hexdigest() == expected


def test_connections_dataset_matches_its_pinned_checksum_when_present():
    mod = load_script("download_connections")
    assert len(mod.REVISION) == 40 and mod.REVISION in mod.URL
    path = ROOT / "data" / "connections" / "Connections_Data.csv"
    if not path.exists():
        pytest.skip("dataset not downloaded")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == mod.SHA256
