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
        "FULL_PF": {"formulas": ["4th(m5_best)"], "width_over_h": [64]},
        "TRUNCATED_PF": {"formulas": ["4th"], "prefixes": "all", "width_over_h": [3, 64]},
        "PR_DF": {"formulas": ["2nd"], "prefixes": "all", "width_over_h": [3, 64],
                  "kappas": ["3/2"], "K": [2, 4]},
    })
    plan = artificial_plan(tmp_path, scope)
    assert all({c["prefix"] for c in plan["cases"] if c["method"] == method} == set(range(19))
               for method in ("TRUNCATED_PF", "PR_DF"))
    assert any(d["target_step"] != d["calibration_step"] for d in plan["designs"])
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
    assert any(axis["kind"] == "CONDITIONAL_STEP_TRANSFER_PRICE" for axis in axes)
    assert all(axis["qualified_E_C"] is False for axis in axes)


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
