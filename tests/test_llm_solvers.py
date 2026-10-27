import json
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest

from src.common.evaluate import evaluate
from src.common.llm import AnthropicJSONClient, LLMReply, ReplyCache, cost_usd
from src.common.solver import Usage
from src.connections.env import ConnectionsEnv, ConnectionsState, validate_partition
from src.connections.probe import labels_matched, probe_summary, run_probe
from src.connections.solvers.baselines import EmbeddingModel
from src.connections.solvers.hybrid import HybridSolver
from src.connections.solvers.llm import LLMSolver, build_partition
from tests.test_connections import make_puzzle


class FakeClient:
    """Returns canned JSON; counts calls; reports fixed usage."""

    def __init__(self, data=None, failure=None):
        self.data, self.failure, self.calls = data, failure, []

    def complete_json(self, **kw):
        self.calls.append(kw)
        usage = Usage(input_tokens=100, output_tokens=50, cost_usd=0.01, api_calls=1)
        return LLMReply(self.data if not self.failure else None, self.failure, usage, 0.5)


def true_groups(p, conf=0.6, key="words"):
    return [{"category": g.label, key: sorted(g.words), "confidence": conf} for g in p.groups]


# --- cost / client ----------------------------------------------------------------------

def test_cost_formula():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)
    assert cost_usd("claude-sonnet-5-5", 0, 1_000_000) == pytest.approx(10.0)
    assert cost_usd("claude-sonnet-5-5", 0, 0, cache_read=1_000_000) == pytest.approx(0.20)


def make_client(tmp_path, create):
    c = object.__new__(AnthropicJSONClient)
    c._anthropic = anthropic
    c._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    c.cache = ReplyCache(tmp_path)
    return c


def fake_response(text, stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop, content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=2000,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


def call(client):
    return client.complete_json(model="claude-haiku-5-5", system="s", user="u", schema={"type": "object"})


def test_client_parses_json_prices_usage_and_sends_structured_output_config(tmp_path):
    seen = {}

    def create(**kw):
        seen.update(kw)
        return fake_response('{"a": 1}')

    reply = call(make_client(tmp_path, create))
    assert reply.data == {"a": 1} and reply.failure is None
    assert reply.usage.cost_usd == pytest.approx(cost_usd("claude-haiku-5-5", 1000, 2000))
    assert seen["output_config"]["format"]["type"] == "json_schema"
    assert seen["output_config"]["effort"] == "medium"
    assert "temperature" not in seen and "tool_choice" not in seen  # both 400 on these models


def test_client_cache_hit_skips_api_and_keeps_original_usage(tmp_path):
    n = []

    def create(**kw):
        n.append(1)
        return fake_response('{"a": 1}')

    client = make_client(tmp_path, create)
    first, second = call(client), call(client)
    assert len(n) == 1 and second.from_cache and second.data == {"a": 1}
    assert second.usage.cost_usd == pytest.approx(first.usage.cost_usd)


@pytest.mark.parametrize("stop,text,failure", [
    ("refusal", "", "refusal"), ("max_tokens", '{"a":', "max_tokens"), ("end_turn", "not json", "invalid_json"),
])
def test_client_reports_failures_and_does_not_cache_them(tmp_path, stop, text, failure):
    client = make_client(tmp_path, lambda **kw: fake_response(text, stop))
    reply = call(client)
    assert reply.data is None and reply.failure == failure and reply.usage.output_tokens == 2000
    assert not list(tmp_path.glob("*.json"))


def test_client_configuration_errors_abort_instead_of_counting_as_model_failures(tmp_path):
    response = SimpleNamespace(status_code=400, headers={}, request=SimpleNamespace())

    def create(**kw):
        raise anthropic.BadRequestError("credit balance is too low", response=response, body=None)

    with pytest.raises(anthropic.BadRequestError):
        call(make_client(tmp_path, create))


def test_client_api_error_becomes_failure(tmp_path):
    def create(**kw):
        raise anthropic.APIConnectionError(request=SimpleNamespace())

    reply = call(make_client(tmp_path, create))
    assert reply.data is None and reply.failure == "api_error"


# --- partition building / LLM solver -----------------------------------------------------

def test_build_partition_maps_case_and_whitespace():
    p = make_puzzle()
    groups = [{"category": "x", "words": [f" {w.lower()} " for w in sorted(g.words)], "confidence": 0.5}
              for g in p.groups]
    proposal, repaired = build_partition(p.words, groups)
    validate_partition(proposal.groups, p.words)
    assert repaired == 0 and {frozenset(g) for g in proposal.groups} == {g.words for g in p.groups}


def test_build_partition_repairs_bad_output_into_a_valid_partition():
    p = make_puzzle()
    g = true_groups(p)
    g[1]["words"][0] = "NOT_ON_BOARD"  # invented word -> group rejected
    g[2]["words"] = g[2]["words"][:3]  # group of three -> rejected
    proposal, repaired = build_partition(p.words, g)
    validate_partition(proposal.groups, p.words)
    assert repaired == 8


def test_greedy_partition_falls_for_a_high_confidence_decoy():
    p = make_puzzle()
    w = sorted(p.groups[0].words) + sorted(p.groups[1].words)
    decoy = {"category": "decoy", "words": [w[0], w[1], w[2], w[4]], "confidence": 0.95}
    proposal, _ = build_partition(p.words, true_groups(p) + [decoy])
    assert frozenset(decoy["words"]) in {frozenset(g) for g in proposal.groups}  # greedy took it


def test_last_group_by_elimination_is_not_counted_as_a_repair():
    p = make_puzzle()
    g = true_groups(p)
    g[3]["words"] = g[3]["words"][:3]  # model gave only three words for one group
    proposal, repaired = build_partition(p.words, g)
    assert repaired == 0
    assert {frozenset(x) for x in proposal.groups} == {x.words for x in p.groups}  # deduced correctly


def test_llm_solver_solves_tracks_usage_and_metrics():
    p = make_puzzle()
    solver = LLMSolver(FakeClient({"groups": true_groups(p)}), "claude-haiku-5-5")
    (r,) = evaluate(solver, ConnectionsEnv(), [p])
    assert r.solved and r.usage.api_calls == 1 and r.usage.cost_usd == pytest.approx(0.01)
    assert r.metrics["repaired_words"] == 0 and r.metrics["llm_failure"] == 0


def test_llm_failure_still_returns_valid_partition_and_is_counted():
    p = make_puzzle()
    solver = LLMSolver(FakeClient(failure="refusal"), "claude-sonnet-5-5")
    (r,) = evaluate(solver, ConnectionsEnv(), [p])
    assert not r.solved and r.metrics["llm_failure"] == 1 and r.metrics["repaired_words"] == 16


# --- hybrid ------------------------------------------------------------------------------

def test_hybrid_resolves_decoy_globally_where_greedy_fails():
    p = make_puzzle()
    w = sorted(p.groups[0].words) + sorted(p.groups[1].words)
    decoy = {"category": "decoy", "words": [w[0], w[1], w[2], w[4]], "confidence": 0.95}
    cands = true_groups(p) + [decoy]

    greedy = LLMSolver(FakeClient({"groups": cands}), "claude-haiku-5-5")
    (g,) = evaluate(greedy, ConnectionsEnv(), [p])
    assert not g.solved

    hybrid = HybridSolver(FakeClient({"candidates": cands}), "claude-haiku-5-5", alpha=0.0)
    (h,) = evaluate(hybrid, ConnectionsEnv(), [p])
    assert h.solved and h.metrics["candidates_valid"] == 5 and h.metrics["groups_from_llm"] == 4


def test_hybrid_uses_embeddings_to_fill_groups_the_llm_missed():
    p = make_puzzle()
    group_of = {w: g.level for g in p.groups for w in g.words}

    class Fake(EmbeddingModel):
        def __init__(self):
            super().__init__("fake")

        def vectors(self, words):
            centers = np.random.default_rng(0).normal(size=(4, 8))
            rng = np.random.default_rng(1)
            vs = [centers[group_of[x]] + 0.01 * rng.normal(size=8) for x in words]
            return np.stack([v / np.linalg.norm(v) for v in vs])

    only_two = true_groups(p)[:2]
    h = HybridSolver(FakeClient({"candidates": only_two}), "claude-haiku-5-5", embedder=Fake(), alpha=0.25)
    (r,) = evaluate(h, ConnectionsEnv(), [p])
    assert r.solved and r.metrics["groups_from_llm"] == 2


def test_hybrid_survives_llm_failure():
    p = make_puzzle()
    h = HybridSolver(FakeClient(failure="max_tokens"), "claude-haiku-5-5", alpha=0.0)
    (r,) = evaluate(h, ConnectionsEnv(), [p])
    validate_partition(h.next_action(ConnectionsState(p.words)).groups, p.words)
    assert r.metrics["llm_failure"] == 1


# --- memorization probe --------------------------------------------------------------------

def test_labels_matched_fuzzy_and_one_to_one():
    truth = ["RAINY DAYS", "MIRRORED WORDS", "TYPING KEYS", "___ BOX"]
    assert labels_matched(["rainy days", "Mirrored Words!", "typing keys", "box"], truth) == 3
    assert labels_matched(["rainy days", "rainy days", "rainy days", "rainy days"], truth) == 1
    assert labels_matched(["COLORS", "ANIMALS", "FOODS", "TOOLS"], truth) == 0
    # one differing word is not a recollection, even though the characters are close
    assert labels_matched(["board games"], ["card games"]) == 0
    assert labels_matched(["mirrored word"], ["MIRRORED WORDS"]) == 1


def test_probe_flags_recited_puzzles_and_not_guesses():
    base = make_puzzle()
    from src.connections.data import Group, Puzzle

    names = ["rivers of europe", "things with keys", "famous bills", "words before fish", "pasta shapes",
             "moons of saturn", "hidden body parts", "silent letters", "types of knot", "cheeses",
             "chess openings", "kinds of bread", "dances", "board games", "mountain ranges", "cloud types",
             "card games", "gemstones", "poker terms", "jazz greats", "tea varieties", "bridge types",
             "fabric weaves", "coffee drinks"]

    def variant(i):
        groups = tuple(Group(names[4 * i + k], k, g.words) for k, g in enumerate(base.groups))
        return Puzzle(i + 10, f"2024-01-{i + 1:02d}", base.words, groups)

    ps = [variant(i) for i in range(6)]

    class Recall:
        def __init__(self, truth_by_id):
            self.truth_by_id = truth_by_id

        def complete_json(self, **kw):
            pid = int(kw["user"].split("#")[1].split(",")[0])
            cats = self.truth_by_id.get(pid, ["colors", "animals", "foods", "tools"])
            return LLMReply({"categories": cats}, None, Usage(), 0.0)

    recited = Recall({p.id: [g.label for g in p.groups] for p in ps})
    res = run_probe(recited, "claude-haiku-5-5", ps)
    assert all(r.flagged and r.matched == 4 for r in res) and probe_summary(res)["flagged_rate"] == 1.0
    assert all(r.control_matched == 0 for r in res)  # recalled labels don't match OTHER puzzles
    guessing = run_probe(Recall({}), "claude-haiku-5-5", ps)
    assert not any(r.flagged for r in guessing)


# --- local model client ---------------------------------------------------------------------

def test_extract_json_handles_think_blocks_prose_fences_and_junk():
    from src.common.local_llm import extract_json

    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('<think>maybe {"x": 0}</think>\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('Sure! Here you go: {"a": {"b": 2}} Hope that helps') == {"a": {"b": 2}}
    assert extract_json("no json here") is None
    assert extract_json('{"a": 1') is None  # truncated
    assert extract_json("[1, 2, 3]") is None  # arrays are not objects


def make_local(tmp_path, outputs):
    from src.common.local_llm import LocalJSONClient

    c = LocalJSONClient(cache_dir=tmp_path)
    queue = list(outputs)
    c.generated = []

    def fake_generate(name, messages):
        c.generated.append(messages)
        return queue.pop(0), 10, 5

    c._generate = fake_generate
    return c


def test_local_client_parses_costs_nothing_and_caches(tmp_path):
    c = make_local(tmp_path, ['{"a": 1}'])
    first, second = call(c), call(c)
    assert first.data == {"a": 1} and first.usage.cost_usd == 0.0 and first.usage.api_calls == 1
    assert second.from_cache and len(c.generated) == 1
    assert "JSON schema" in c.generated[0][0]["content"]  # schema is put in the prompt


def test_local_client_retries_once_on_invalid_json_then_succeeds(tmp_path):
    c = make_local(tmp_path, ["I think the answer is blue", '{"a": 2}'])
    reply = call(c)
    assert reply.data == {"a": 2} and len(c.generated) == 2
    assert reply.usage.input_tokens == 20 and reply.usage.output_tokens == 10  # both attempts counted
    assert c.generated[1][-1]["role"] == "user" and "not valid JSON" in c.generated[1][-1]["content"]


def test_local_client_reports_failure_after_retry_and_does_not_cache(tmp_path):
    c = make_local(tmp_path, ["nope", "still nope"])
    reply = call(c)
    assert reply.data is None and reply.failure == "invalid_json"
    assert not list(tmp_path.glob("*.json"))


def test_local_client_plugs_into_the_llm_solver(tmp_path):
    p = make_puzzle()
    good = json.dumps({"groups": true_groups(p)})
    solver = LLMSolver(make_local(tmp_path, [good]), "mlx-community/Qwen3-8B-4bit")
    (r,) = evaluate(solver, ConnectionsEnv(), [p])
    assert r.solved and r.usage.cost_usd == 0.0


def test_cached_replies_still_report_their_original_latency():
    p = make_puzzle()

    class Cached(FakeClient):
        def complete_json(self, **kw):
            r = super().complete_json(**kw)
            return LLMReply(r.data, None, r.usage, 12.5, from_cache=True)

    for solver in (LLMSolver(Cached({"groups": true_groups(p)}), "claude-haiku-5-5"),
                   HybridSolver(Cached({"candidates": true_groups(p)}), "claude-haiku-5-5", alpha=0.0)):
        (r,) = evaluate(solver, ConnectionsEnv(), [p])
        assert r.seconds >= 12.5 and r.usage.cached_latency_s == 12.5
