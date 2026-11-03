"""Fresh trajectory -> conditional +/-1 measurement, with no default H4 run.

Only pure legacy RTE definitions are loaded. The molecular context belongs to
the separately invoked runner. Synthetic tests use a separate public namespace.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CARRIER = "4591b8d75a2a6a55c74a51225f0c61648023c8ee"
V2 = "78c49d0b5f1d71e63b5959e78813237923bbc648"
B_RESULT = "e22d7718c47358ff650dbc04e8e72f6ab1c8bf41"
SEED_NAMESPACE = "rpe_fresh_iid_measurement_a_20261103_v1_public"
DIAGNOSTIC_FAMILY_ALPHA = .05  # Separate from every saved RPE alpha/beta.
DIAGNOSTIC_COMPARISONS = 12   # 4 axes x (trajectory, shot, conditional gap).
MEAN_ROUNDOFF_ATOL = 1e-12
MATRIX_ATOL = 1e-10


def fixed_bytes(commit, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{path}"])


@lru_cache(maxsize=1)
def legacy_functions():
    """Unchanged FunctionDefs only; no old runner/module import."""
    path = "src/trotterlib/rpe_hadamard_failure_validation.py"
    source = fixed_bytes(CARRIER, path).decode()
    names = {"_seed_for_shot", "_apply_event_to_state", "_trajectory_signal"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Legacy pure function inventory mismatch")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"np": np, "math": math, "hashlib": hashlib}
    exec(compile(module, "<unchanged-legacy-rpe-functions>", "exec"), ns)
    return {n: ns[n] for n in names}


@lru_cache(maxsize=1)
def legacy_rte():
    """rte.py has NumPy/stdlib imports only; avoid trotterlib.__init__."""
    path = "src/trotterlib/rte.py"
    if (ROOT/path).read_bytes() != fixed_bytes(CARRIER, path):
        raise ValueError("Legacy sampler SOURCE mismatch")
    sys.path.insert(0, str(ROOT/"src/trotterlib"))
    module = importlib.import_module("rte")
    if Path(module.__file__).resolve() != (ROOT/path).resolve():
        raise ValueError("Wrong rte module")
    return module


@dataclass(frozen=True)
class AxisInput:
    m: int
    q: int
    r: int
    k: int
    axis: str
    shots: int
    target_mean: float
    saved_alpha: str
    saved_betas: tuple[str, str, str]
    saved_epsilon: str
    state_label: str = "physical_df_ground_state"
    signal_kind: str = "attenuated_event_mean_signal"

    def __post_init__(self):
        if self.axis not in ("cosine", "sine") or self.shots < 1:
            raise ValueError("Invalid axis/N")
        if self.state_label != "physical_df_ground_state" or self.signal_kind != "attenuated_event_mean_signal":
            raise ValueError("Control/corrected signals are not physical measurement marginals")
        if not math.isfinite(self.target_mean) or not -1 <= self.target_mean <= 1:
            raise ValueError("Invalid saved marginal")


def fixed_axis_inputs():
    """Metadata/literal transfer only; no saved-value probability evaluation."""
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(fixed_bytes(V2,
        "docs/research/pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv").decode())))
    inputs = []
    for m, q, n in ((0, 1, 510), (3, 8, 454)):
        row, = [r for r in rows if r["ld"] == "3" and r["round_index"] == str(m)]
        if (row["q_m"],row["r_m"],row["K_m"],row["delta_time"]) != (str(q),"1","2","0.02"):
            raise ValueError("Fixed tuple mismatch")
        signal = json.loads(row["saved_complex_json"])
        axes = json.loads(row["axes_results_json"])
        for axis, coordinate in (("cosine","real"),("sine","imag")):
            saved = axes[axis]
            if int(saved["saved_N"]) != n or row["saved_signal_kind"] != "attenuated_event_mean_signal":
                raise ValueError("Fixed N/measurement-law mismatch")
            inputs.append(AxisInput(m,q,1,2,axis,n,float(signal[coordinate]),saved["saved_alpha"],
                (row["saved_beta_PF"],row["saved_beta_RTE"],row["saved_beta_stat"]),
                saved["saved_epsilon_coordinate"]))
    return tuple(inputs)


def stream_seed(q, axis, shot, role, *, namespace=SEED_NAMESPACE):
    if role not in ("trajectory", "measurement") or axis not in ("cosine", "sine") or shot < 0:
        raise ValueError("Invalid RNG stream key")
    # New public reproducibility recipe, not a reused private seed.
    raw = hashlib.sha256(f"{namespace}:{role}".encode()).digest()
    master = int.from_bytes(raw[:8], "big") & ((1 << 63)-1)
    return legacy_functions()["_seed_for_shot"](master, q, axis, shot)


def check_stream_keys(inputs, *, namespace=SEED_NAMESPACE):
    keys, seeds = set(), set()
    for spec in inputs:
        for shot in range(spec.shots):
            for role in ("trajectory", "measurement"):
                key = (spec.q,spec.axis,shot,role)
                seed = stream_seed(*key[:3],role,namespace=namespace)
                if key in keys or seed in seeds:
                    raise ValueError("Duplicate request/stream seed; STOP without reseeding")
                keys.add(key);seeds.add(seed)
    return len(keys)


def conditional_probability(signal, axis):
    signal = complex(signal)
    if not math.isfinite(signal.real) or not math.isfinite(signal.imag) or abs(signal) > 1+MATRIX_ATOL:
        raise ValueError("Invalid conditional unitary signal")
    if axis not in ("cosine","sine"):
        raise ValueError("Invalid axis")
    raw = float(signal.real if axis == "cosine" else signal.imag)
    if not -1-MEAN_ROUNDOFF_ATOL <= raw <= 1+MEAN_ROUNDOFF_ATOL:
        raise ValueError("Conditional +/-1 mean is outside numerical allowance")
    used = min(1.,max(-1.,raw))  # Only explicitly allowed roundoff, recorded.
    return raw,used,(1+used)/2,raw != used


def one_shot(spec, shot, *, components, distribution, signal_fn,
             sampler=None, namespace=SEED_NAMESPACE):
    if not 0 <= shot < spec.shots:
        raise ValueError("Shot index outside fixed N")
    sampler = legacy_rte().sample_rte_events if sampler is None else sampler
    events = sampler(components,distribution,sample_count=spec.q*spec.r,
                     seed=stream_seed(spec.q,spec.axis,shot,"trajectory",namespace=namespace))
    if len(events) != spec.q*spec.r:
        raise ValueError("Missing/invalid trajectory request: event count mismatch")
    z = complex(signal_fn(events,spec.q,spec.r))
    raw,used,p,rounded = conditional_probability(z,spec.axis)
    rng = np.random.Generator(np.random.PCG64(stream_seed(spec.q,spec.axis,shot,"measurement",namespace=namespace)))
    outcome = 1 if float(rng.random()) < p else -1
    event_records = [{"order":e.taylor_order,"rotation":e.rotation_component_id,
                      "products":list(e.product_component_ids),"rotation_angle":e.rotation_angle,
                      "phase":{"real":e.phase.real,"imag":e.phase.imag},
                      "probability":e.event_probability,"normalization":e.event_normalization} for e in events]
    return {"q":spec.q,"axis":spec.axis,"shot":shot,"signal_real":z.real,"signal_imag":z.imag,
            "raw_conditional_mean":raw,"conditional_mean":used,"p_plus":p,
            "roundoff_adjusted":rounded,"outcome":outcome,"events":event_records}


def summarize_axis(spec, rows):
    """Twelve prespecified bounded-mean diagnostics, not an RPE success test."""
    rows = sorted(rows,key=lambda r:r["shot"])
    if len(rows) != spec.shots or [r["shot"] for r in rows] != list(range(spec.shots)):
        raise ValueError("Missing/duplicate shots; no partial inference")
    if any(r["q"]!=spec.q or r["axis"]!=spec.axis or r["outcome"] not in (-1,1)
           or not math.isfinite(r["conditional_mean"]) or not -1<=r["conditional_mean"]<=1 for r in rows):
        raise ValueError("Invalid axis records")
    trajectory = math.fsum(r["conditional_mean"] for r in rows)/spec.shots
    measured = math.fsum(r["outcome"] for r in rows)/spec.shots
    h = math.sqrt(2*math.log(2*DIAGNOSTIC_COMPARISONS/DIAGNOSTIC_FAMILY_ALPHA)/spec.shots)
    # All three bounded-mean comparisons share the same prespecified width.
    checks = {"trajectory_to_saved":abs(trajectory-spec.target_mean)<=h+MEAN_ROUNDOFF_ATOL,
              "measurement_to_saved":abs(measured-spec.target_mean)<=h+MEAN_ROUNDOFF_ATOL,
              "measurement_to_conditional_average":abs(measured-trajectory)<=h}
    return {"q":spec.q,"axis":spec.axis,"shots":spec.shots,"saved_target_mean":spec.target_mean,
            "trajectory_conditional_mean_average":trajectory,"measurement_sample_mean":measured,
            "conditional_measurement_gap":measured-trajectory,"simultaneous_diagnostic_half_width":h,
            "sufficient_separation_for_high_probability_detection_gt":2*h+2*MEAN_ROUNDOFF_ATOL,
            "roundoff_adjustment_count":sum(r["roundoff_adjusted"] for r in rows),
            "checks":checks,"diagnostic_observation":"NO_DISCREPANCY_DETECTED" if all(checks.values()) else "DISCREPANCY_DETECTED",
            "scientific_status":"UNDETERMINED" if all(checks.values()) else "INCONSISTENT",
            "equivalence_certified":False,"RPE_success_probability_evaluated":False}
