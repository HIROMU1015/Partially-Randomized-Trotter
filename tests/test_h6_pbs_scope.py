"""Exercise the full requested scope with artificial preparation and Q3 data."""

from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/h6_fixed_run"
sys.path.insert(0, str(RUNNER))
from h6_scope import calibration_design, case_rows, digest, read, rounds
from run_h6_scope import features, optimize, optimizer
from h6_scope_worker import write


def artificial_preparation(job):
    import time

    kind, cell, index, output, owner = job
    time.sleep(0.04)
    if cell.get("fail"):
        raise ValueError("artificial error " * 2000)
    write(output, dict(kind=kind, index=index, owner=owner))


def test_preparation_pools_separate_cpu_slots_and_memory(tmp_path):
    import h6_preparation_pool as pool

    signal_jobs = [("signal", {}, i, tmp_path / f"signal-{i}.json", "fixture") for i in range(5)]
    price_jobs = [("price", {"profile": (0, 1 << 14)}, 0, tmp_path / "heavy.json", "fixture")]
    price_jobs += [("price", {"profile": (1024, 0), "fail": i == 2}, i, tmp_path / f"light-{i}.json", "fixture")
                   for i in range(4)]
    metrics = []
    assert not pool.signals(signal_jobs, artificial_preparation, 5, metrics.append)
    assert metrics[-1]["phase"] == "signal" and metrics[-1]["done"] == 5
    assert metrics[-1]["worker_limit"] == 5
    memory = 3 * pool.GIB
    errors = pool.prices(price_jobs, artificial_preparation, 5, memory, lambda c: c["profile"], metrics.append)
    assert set(errors) == {"light-2.json"}
    assert (tmp_path / "heavy.json").is_file()
    assert set(p.name for p in tmp_path.glob("light-*.json")) == {"light-0.json", "light-1.json", "light-3.json"}
    assert metrics[-1]["done"] == 4 and metrics[-1]["peak_workers"] == 2
    assert metrics[-1]["peak_reserved_bytes"] <= memory
    assert metrics[-1]["peak_worker_rss_bytes"] > 0


def test_price_pool_starts_more_than_one_unmeasured_job_when_memory_fits(tmp_path):
    import h6_preparation_pool as pool

    jobs = [("price", {}, i, tmp_path / f"price-{i}.json", "fixture") for i in range(4)]
    metrics = []
    memory = 8 * pool.GIB
    assert not pool.prices(jobs, artificial_preparation, 4, memory, lambda _: (1024, 0), metrics.append)
    assert metrics[-1]["done"] == metrics[-1]["peak_workers"] == 4
    assert metrics[-1]["peak_reserved_bytes"] <= memory


def test_price_pool_prioritizes_larger_circuits_in_the_same_memory_group(tmp_path):
    import h6_preparation_pool as pool

    jobs = [("price", {"size": (n, 0)}, n, tmp_path / f"price-{n}.json", "fixture") for n in (600, 1000, 800)]
    assert not pool.prices(jobs, artificial_preparation, 1, 4 * pool.GIB, lambda c: c["size"], lambda _: None)
    completed = sorted((task[3] for task in jobs), key=lambda path: path.stat().st_mtime_ns)
    assert [read(path)["index"] for path in completed] == [1000, 800, 600]


def test_starting_long_price_job_leaves_room_for_short_jobs(tmp_path):
    import time
    import h6_preparation_pool as pool

    heavy = ("price", {"size": (0, 1 << 15), "seconds": 0.18}, 0, tmp_path / "long.json", "fixture")
    medium = ("price", {"size": (0, 1 << 14), "seconds": 0.13}, 0, tmp_path / "medium.json", "fixture")
    short = [("price", {"size": (1024, 0), "seconds": 0.025}, i, tmp_path / f"short-{i}.json", "fixture")
             for i in range(5)]
    memory = 8 * pool.GIB + (16 << 20)
    metrics = []

    def execute(job):
        with pool.price_phase("artificial"):
            time.sleep(job[1]["seconds"])
            write(job[3], {"index": job[2]})

    assert not pool.prices([heavy, medium, *short], execute, 8, memory, lambda c: c["size"], metrics.append)
    assert all(job[3].stat().st_mtime_ns < heavy[3].stat().st_mtime_ns for job in short)
    assert metrics[-1]["peak_workers"] >= 3
    assert metrics[-1]["peak_reserved_bytes"] <= memory
    phases = metrics[-1]["phase_metrics"]["artificial"]
    assert phases["tasks"] == 7 and phases["peak_rss_bytes"] > 0


def test_price_memory_learning_requires_two_successful_measurements():
    import h6_preparation_pool as pool

    forecast, low, high = 10 * pool.GIB, 2 * pool.GIB, 8 * pool.GIB
    assert pool.reserved_bytes(forecast, low, 0, 1) == 2 * forecast
    assert pool.reserved_bytes(forecast, low, 1, 1) == forecast
    assert pool.reserved_bytes(forecast, low, 2, 1) == 3 * pool.GIB
    assert pool.reserved_bytes(forecast, high, 2, 1) == 12 * pool.GIB


def test_scoped_price_archive_preserves_events_and_price_reuse(tmp_path, monkeypatch):
    import hashlib
    import json
    import lzma
    from types import SimpleNamespace

    import h6_scope_worker as worker
    from run_h6_scope import observation

    price_dir = tmp_path / "prices"
    price_dir.mkdir()
    cell = {"id": "path", "method": "PR"}
    path = price_dir / "path-0.json"
    event_a = {"application_sequence": ["rotación"] * 500, "phase": {"real": 1.0, "imag": 0.0}}
    event_b = {"application_sequence": ["identity"] * 150, "phase": {"real": -1.0, "imag": 0.0}}
    a = SimpleNamespace(to_dict=lambda: event_a)
    b = SimpleNamespace(to_dict=lambda: event_b)
    data = dict(candidate=cell, index=0, compiler="artificial", axes={"X": {"rz": 42}, "Y": {"rz": 43}},
                events=[(a, b, a), (), (b, a)])
    calls = []

    def artificial_price(*args):
        calls.append(args)
        return data

    monkeypatch.setattr(worker, "price", artificial_price)
    job = ("price", cell, 0, path, "fixture")
    worker.execute(job)
    summary = read(path)
    archive = price_dir / summary["events"]["file"]
    events = [[event_a, event_b, event_a], [], [event_b, event_a]]
    assert lzma.decompress(archive.read_bytes()) == (json.dumps(events) + "\n").encode()
    assert summary["events"]["size_bytes"] == archive.stat().st_size
    assert summary["events"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    plan = dict(scope={"price_paths": 1}, owner="fixture", compiler="artificial")
    assert observation(plan, cell, tmp_path, "X")[0] == 42
    assert observation(plan, cell, tmp_path, "Y")[0] == 43
    worker.execute(job)
    archive.unlink()
    with pytest.raises(FileNotFoundError):
        worker.execute(job)
    with pytest.raises(FileNotFoundError):
        observation(plan, cell, tmp_path, "X")
    write(path, dict(summary, events=events))
    assert observation(plan, cell, tmp_path, "X")[0] == 42
    worker.execute(job)
    assert len(calls) == 1


def test_scope_continuation_keeps_old_events_and_seeds_without_rerunning_completed_jobs(tmp_path, monkeypatch):
    import fcntl
    from types import SimpleNamespace

    import h6_scope_worker as worker

    old, new = tmp_path / "old", tmp_path / "new"
    (old / "signals").mkdir(parents=True)
    (old / "prices").mkdir()
    old_lock = old / ".h6_scope.lock"
    cell = {"id": "pr", "method": "PR", "prefix": 1}
    cases = [dict(rounds=[cell])]
    scope = {"price_paths": 2}
    design = {"anchors": [cell], "holdout": None}
    state = SimpleNamespace(cases=cases, binding={"system": "H6"}, target=-3.0, compiler="test",
                            preparations={1: SimpleNamespace(preparation_hash="prep")})
    write(old / "PLAN.json", dict(owner="old-owner", cases=cases, binding=state.binding, target=state.target,
                                 compiler=state.compiler, scope=scope, designs=[design]))
    write(old / "signals/pr.json", dict(owner="old-owner", candidate=cell, signal={"raw": {"real": 0.5}}))
    original = dict(owner="old-owner", candidate=cell, index=0, cohort="H6_SCOPE_old-owner", compiler="test",
                    preparation_hash="prep", axes={"X": {"rz": 20}, "Y": {"rz": 22}}, seeds=[123],
                    events=[[{"component": "previous"}]])
    write(old / "prices/pr-0.json", original)
    (old / "prices/pr-1.started").touch()
    monkeypatch.setattr(worker, "startup", lambda *_: state)
    monkeypatch.setattr(worker, "calibration_design", lambda *_: {"group": design})
    seen = []

    def artificial_price(candidate, index, cohort):
        seen.append((candidate, index, cohort))
        return dict(original, index=index, cohort=cohort, events=[])

    def execute_all(jobs, execute, *args):
        for job in jobs:
            execute(job)
        return {}

    monkeypatch.setattr(worker, "price", artificial_price)
    monkeypatch.setattr(worker.pool, "signals", execute_all)
    monkeypatch.setattr(worker.pool, "prices", execute_all)
    with old_lock.open("w") as previous:
        fcntl.flock(previous, fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            worker.collect(scope, tmp_path, new, 2, 2, 2 * worker.pool.GIB, previous=old)
    assert worker.collect(scope, tmp_path, new, 2, 2, 2 * worker.pool.GIB, previous=old) == 0
    assert seen == [(cell, 1, "H6_SCOPE_old-owner")]
    continued = read(new / "CONTINUED_PREPARATION.json")
    assert (continued["imported_signals"], continued["imported_prices"]) == (1, 1)
    imported = worker.read_price(new / "prices/pr-0.json")
    assert imported["owner"] == read(new / "PLAN.json")["owner"] != original["owner"]
    assert imported["seeds"] == original["seeds"]
    assert (old / "prices/pr-1.started").is_file() and read(old / "prices/pr-0.json") == original
    assert worker.collect(scope, tmp_path, new, 2, 2, 2 * worker.pool.GIB, previous=old) == 0
    assert len(seen) == 1


def artificial_plan(run, scope=None):
    scope = scope or read(RUNNER / "h6_scope_example.json")
    scope["scaled_shots"]["grid"].update(min_k=0, max_k=0)
    cases = []
    for case in case_rows(scope, 19):
        case["rounds"] = list(rounds(scope, case, Fraction(case["prefix"] + 10, 100)))
        case["id"] = digest(case)
        cases.append(case)
    designs = calibration_design(cases, scope)
    owner = "artificial"
    price_dir, signal_dir = run / "prices", run / "signals"
    price_dir.mkdir()
    signal_dir.mkdir()
    for case in cases:
        for cell in case["rounds"]:
            write(signal_dir / f"{cell['id']}.json", dict(candidate=cell, owner=owner,
                  signal={"raw": {"real": 0.5, "imag": 0.25}}))
    for design in designs.values():
        for cell in [*design["anchors"], *([design["holdout"]] if design["holdout"] else [])]:
            assert cell["m"] is None or cell["m"] <= scope["calibration_max_m"]
            assert cell["T"] <= float(scope["h"]) * 2**scope["calibration_max_m"]
            f = features(cell)
            count = 10 + 7 * f["q"] + 5 * f["qR"]
            for idx in range(scope["price_paths"] if cell["method"] == "PR" else 1):
                write(price_dir / f"{cell['id']}-{idx}.json", dict(candidate=cell, owner=owner, index=idx,
                    compiler="artificial", axes={"X": {"rz": count}, "Y": {"rz": count + 1}}))
    return dict(scope=scope, owner=owner, binding={"system": "H6", "rank": 19}, compiler="artificial",
                target=-3.2360662799087647, cases=cases, designs=list(designs.values()))


def test_all_requested_combinations_reach_common_shot_optimizer(tmp_path):
    plan = artificial_plan(tmp_path)
    assert len(plan["cases"]) == 130
    assert any(c.get("calibration_only") for d in plan["designs"] for c in d["anchors"])
    seen = set()

    def artificial_q3(cid, vector):
        assert len(vector) == 26
        seen.add(cid)
        return 0, 0

    test_backend = optimizer.FixtureBackend(artificial_q3, Fraction("0.00159360010199040") ** 2)
    budgets = dict.fromkeys(plan["scope"]["methods"], None)
    _, held = optimize(plan, tmp_path, tmp_path, 1, 1, budgets, execute=False, artificial_backend=test_backend)
    report = read(tmp_path / "ROUND_RESULT.json")
    assert report["counts"] == {"FULL_PF": 12, "TRUNCATED_PF": 48, "PR_DF": 70}
    assert report["ready"] == len(seen) == 130
    assert not held
    prices = next((tmp_path / "cases").glob("*/REGISTERED_PRICES.json"))
    before = prices.read_bytes()
    optimize(plan, tmp_path, tmp_path, 1, 1, budgets, execute=False, artificial_backend=test_backend)
    assert prices.read_bytes() == before

    results, _ = optimize(plan, tmp_path, tmp_path, 2, 8, dict.fromkeys(budgets, 0), execute=False)
    assert all(method["pending_request"] and method["new_calls"] == 0 for method in results.values())


def test_entire_prefix_domain_and_arbitrary_widths_reach_optimizer(tmp_path):
    scope = read(RUNNER / "h6_scope_example.json")
    scope.update(M=13, price_paths=2, methods={
        "FULL_PF": {"formulas": ["4th(m5_best)"], "width_over_h": [10]},
        "TRUNCATED_PF": {"formulas": ["4th"], "prefixes": "all", "width_over_h": [3, 10]},
        "PR_DF": {"formulas": ["2nd"], "prefixes": "all", "width_over_h": [3, 10],
                  "kappas": ["3/2"], "K": [2, 4]},
    })
    plan = artificial_plan(tmp_path, scope)
    assert all({c["prefix"] for c in plan["cases"] if c["method"] == method} == set(range(19))
               for method in ("TRUNCATED_PF", "PR_DF"))
    assert all(d["target_step"] == d["calibration_step"] for d in plan["designs"])
    assert any(c["m"] is None and d["target_step"] == d["calibration_step"]
               for d in plan["designs"] for c in d["anchors"])
    seen = set()

    def artificial_q3(cid, vector):
        assert len(vector) == 28
        seen.add(cid)
        return 0, 0

    backend = optimizer.FixtureBackend(artificial_q3, Fraction("0.00159360010199040") ** 2)
    _, held = optimize(plan, tmp_path, tmp_path, 1, 1, dict.fromkeys(scope["methods"], None),
                       execute=False, artificial_backend=backend)
    assert not held
    assert len(seen) == len(plan["cases"]) == 115
    full_id = next(c["id"] for c in plan["cases"] if c["method"] == "FULL_PF")
    axes = read(tmp_path / "cases" / full_id / "REGISTERED_PRICES.json")["methods"]["Full PF"]["axes"]
    assert any(axis["kind"] == "CONDITIONAL_COMPILED_PRICE_MODEL" for axis in axes)
    assert all(axis["qualified_E_C"] is False for axis in axes)


def test_missing_same_step_calibration_holds_only_affected_candidate(tmp_path):
    scope = read(RUNNER / "h6_scope_example.json")
    scope["methods"] = {"FULL_PF": {"formulas": ["2nd"], "width_over_h": [2, 64]}}
    plan = artificial_plan(tmp_path, scope)
    assert all(d["target_step"] == d["calibration_step"] for d in plan["designs"])
    assert any(not d["anchors"] and d.get("reason") == "INSUFFICIENT_SAME_STEP_CALIBRATION"
               for d in plan["designs"])
    backend = optimizer.FixtureBackend(lambda *_: (0, 0), Fraction("0.00159360010199040") ** 2)
    _, held = optimize(plan, tmp_path, tmp_path, 1, 1, {"FULL_PF": None},
                       execute=False, artificial_backend=backend)
    assert set(held) == {plan["cases"][1]["id"]}
    assert "insufficient same-step calibration" in next(iter(held.values()))
    assert read(tmp_path / "ROUND_RESULT.json")["ready"] == 1


def test_prefix_bounds_and_single_round():
    scope = read(RUNNER / "h6_scope_example.json")
    scope.update(M=0, calibration_max_m=0, methods={
        "TRUNCATED_PF": {"formulas": ["2nd"], "prefixes": [1], "width_over_h": [3]},
    })
    case = next(case_rows(scope, 19))
    assert case["prefix"] == 1
    assert [r["m"] for r in rounds(scope, case)] == [0]
    scope["methods"]["TRUNCATED_PF"]["prefixes"] = [19]
    with pytest.raises(ValueError, match="Invalid H6 range"):
        list(case_rows(scope, 19))


def test_preview_reports_no_scientific_calls():
    r = subprocess.run([sys.executable, "-B", str(RUNNER / "run_h6_scope.py"), "--plan-only",
                        "--scope", str(RUNNER / "h6_scope_example.json")], check=True, capture_output=True, text=True)
    assert '"scientific_calls": 0' in r.stdout


def test_scoped_q3_adapter_accepts_materialized_candidate(tmp_path):
    from run_h6_scope import candidate_inputs, price_models, retain, sha
    from h6_scope import LABELS

    sys.path.insert(0, str(RUNNER / "q3-pilot"))
    import h6_q3_adapter
    import run_q3

    scope = read(RUNNER / "h6_scope_example.json")
    scope.update(M=13, methods={
        "PR_DF": {"formulas": ["2nd"], "prefixes": [1], "width_over_h": [3], "kappas": [3], "K": 4},
    })
    plan = artificial_plan(tmp_path, scope)
    case = plan["cases"][0]
    models, errors = price_models(plan, tmp_path)
    assert not errors
    models_dir = tmp_path / "price_models"
    models_dir.mkdir()
    for key, item in models.items():
        retain(models_dir / f"{digest(str(key))}.json", item)
    axes, entry, registry = candidate_inputs(tmp_path, plan, case, models)
    shots = [1] * len(case["rounds"])
    cost = sum(Fraction(row["unit_price_RZ"]["numerator"], row["unit_price_RZ"]["denominator"]) for row in axes)
    request = dict(request_id="artificial-request", method=case["method"], configuration_id=case["id"],
        N_X=shots, N_Y=shots, cost={"numerator": cost.numerator, "denominator": cost.denominator},
        origin={"M": 13}, expected={"configuration_id": case["id"]},
        route=dict(kind="h6_df19_scoped_q3", scope_owner=plan["owner"], view_commit=run_q3.Q3_SOURCE,
                   configurations={case["id"]: entry}, registered_prices=str(registry),
                   registered_prices_sha256=sha(registry)))
    label, _, _, _, _, identity = h6_q3_adapter.request_inputs(request)
    assert label == LABELS["PR_DF"]
    assert identity["M"] == 13
    request["route"]["scope_owner"] = "different"
    with pytest.raises(ValueError, match="horizon or owner"):
        h6_q3_adapter.request_inputs(request)
    request["route"]["scope_owner"] = plan["owner"]
    escaped = read(registry)
    model_axis = next(row for row in escaped["methods"][label]["axes"]
                      if row["kind"] == "CONDITIONAL_COMPILED_PRICE_MODEL")
    model_axis["price_evidence"]["calibration_step"] = {"numerator": 19, "denominator": 10}
    write(registry, escaped)
    request["route"]["registered_prices_sha256"] = sha(registry)
    with pytest.raises(ValueError, match="calibration step differs"):
        h6_q3_adapter.request_inputs(request)


@pytest.mark.parametrize("method,formula", [("FULL", "4th"), ("B0I", "4th(m5_best)"), ("PR", "2nd")])
def test_native_signal_period_matches_repeated_application(method, formula, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from scipy.linalg import expm
    from trotterlib.rte import finite_rte_distribution, finite_taylor_operator
    from trotterlib.h46_m5_plan import weights
    import h6_scope_worker as worker

    state = np.array([1, 1j]) / np.sqrt(2)
    matrices = [np.array([[1.0, 0.2], [0.2, 0.4]]), np.array([[0.3, 0.1j], [-0.1j, 0.8]])]
    tail = np.diag([0.15, -0.10])
    T, q = 1.25, 5
    cell = dict(method=method, formula=formula, prefix=1, T=T, q=q, K=2,
                R_outer=3 if method == "PR" else None)
    ctx = SimpleNamespace(state=state, matrices=matrices,
        helpers={"core": SimpleNamespace(weights=weights, allocation=lambda _, r: [r])},
        tail={1: tail}, lambdas={1: 0.2}, preparations={1: SimpleNamespace(extracted_identity_coefficient=0.03)},
        constant=0.21, identity={1: 0.04})
    monkeypatch.setattr(worker, "STATE", ctx)
    worker.head_blocks.cache_clear()
    result = worker.signal(cell)["signal"]["raw"]
    z = state.copy()
    for _ in range(q):
        for piece in weights(formula):
            t = T * piece / q
            for matrix in matrices:
                z = expm(-0.5j * t * matrix) @ z
            if method == "PR":
                tau = 0.2 * (t / 3)
                step = finite_taylor_operator(tail / 0.2, tau, 2)
                normalizer = finite_rte_distribution(tau, 2).exact_finite_distribution
                for _ in range(3):
                    z = step @ z / normalizer
            for matrix in reversed(matrices):
                z = expm(-0.5j * t * matrix) @ z
    scalar = {"FULL": 0.21, "B0I": 0.25, "PR": 0.24}[method]
    assert complex(result["real"], result["imag"]) == pytest.approx(
        np.vdot(state, z) * np.exp(-1j * T * scalar), abs=2e-14)
    worker.head_blocks.cache_clear()
