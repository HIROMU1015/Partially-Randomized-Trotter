"""Measurement/decoder plumbing only; no scientific backend or default sampler.

The callbacks below must be supplied by the caller. They are exercised only with
artificial fixtures in this preparation. Statistical IID is an external backend
condition, never established by call counts or distinct request keys.
"""
from __future__ import annotations

import ast
import cmath
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parents[3]
CARRIER = "4591b8d75a2a6a55c74a51225f0c61648023c8ee"
V2 = "78c49d0b5f1d71e63b5959e78813237923bbc648"
MODEL_PATH = "artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json"
V2_ROWS_PATH = "docs/research/pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv"
DECODER_PATH = "src/trotterlib/rpe_four_round_phase_validation.py"
PHYSICAL = "physical_df_ground_state"
AXES = ("cosine", "sine")


def fixed_bytes(commit: str, path: str) -> bytes:
    """Read a local immutable object; no fetch, runner, or environment changes."""
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{path}"])


@dataclass(frozen=True)
class AxisSettings:
    shots: int
    alpha: str  # Preserve the saved decimal; no allocation/optimization here.


@dataclass(frozen=True)
class RoundSettings:
    ld: int
    m: int
    q: int
    r: int
    k: int
    delta: float
    cosine: AxisSettings
    sine: AxisSettings
    beta_pf: str
    beta_rte: str
    beta_stat: str
    epsilon_coordinate: str
    state_label: str
    signal_kind: str
    physical_fingerprint: str
    identity_status: str

    def __post_init__(self):
        counts = (self.ld, self.m, self.q, self.r, self.k,
                  self.cosine.shots, self.sine.shots)
        if any(type(v) is not int for v in counts):
            raise ValueError("Counts must be integers, excluding bool.")
        if self.m < 0 or self.q != 2**self.m:
            raise ValueError("Round/q must be dyadic.")
        if not math.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("delta must be finite and positive.")
        if self.ld not in (3, 12) or self.state_label != PHYSICAL:
            raise ValueError("This adapter connects the two physical-input laws only.")
        expected_kind = "attenuated_event_mean_signal" if self.ld == 3 else "physical_PF_signal"
        if self.signal_kind != expected_kind:
            raise ValueError("Corrected/control signals are not measurement means.")
        if (self.ld == 3 and (self.r < 1 or self.k < 0 or self.k % 2)) or (
            self.ld == 12 and (self.r, self.k) != (0, 0)
        ):
            raise ValueError("Partial needs finite RTE; Full is tail-free r0/K0.")
        for axis in (self.cosine, self.sine):
            if axis.shots < 1 or not 0 < float(axis.alpha) < 1:
                raise ValueError("Invalid fixed N/alpha.")
        for value in (self.beta_pf, self.beta_rte, self.beta_stat, self.epsilon_coordinate):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("Invalid saved budget.")


def load_fixed_rounds(ld: int) -> tuple[RoundSettings, ...]:
    """Copy settings only; do not decode signals, evaluate probabilities or E0."""
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(fixed_bytes(V2, V2_ROWS_PATH).decode())))
    model = json.loads(fixed_bytes(CARRIER, MODEL_PATH))
    selected = model["coherent_reoptimization"]["best_by_ld"][str(ld)]
    best = selected["best"]
    result = []
    for m, saved in enumerate(best["rounds"]):
        row, = [v for v in rows if int(v["ld"]) == ld and int(v["round_index"]) == m]
        if any(int(row[c]) != int(saved[c]) for c in ("q_m", "r_m", "K_m")):
            raise ValueError("V2/model tuple mismatch.")
        axes = json.loads(row["axes_results_json"])
        for name in AXES:
            if (axes[name]["saved_N"] != str(saved["axes"][name]["shots"])
                    or axes[name]["saved_alpha"] != str(saved["axes"][name]["alpha"])):
                raise ValueError("V2/model N/alpha mismatch.")
        result.append(RoundSettings(
            ld, m, int(saved["q_m"]), int(saved["r_m"]), int(saved["K_m"]),
            float(selected["delta_time"]),
            *(AxisSettings(int(saved["axes"][a]["shots"]), str(saved["axes"][a]["alpha"])) for a in AXES),
            str(best["beta_pf_budget"]), str(best["beta_rte_budget"]), str(best["beta_stat_budget"]),
            str(saved["epsilon_coordinate"]), PHYSICAL, row["saved_signal_kind"],
            row["saved_physical_state_fingerprint"], row["physical_identity_status"],
        ))
    if len(result) != 18 or result[-1].q != 131072 or selected["delta_time"] != .02:
        raise ValueError("Fixed 18-round scope mismatch.")
    return tuple(result)


@dataclass(frozen=True)
class MeasurementRequest:
    settings: RoundSettings
    axis: str
    shot: int
    trajectory_key: tuple[int, int, str, int] | None

    @property
    def expected_event_count(self) -> int:
        return self.settings.q * self.settings.r


@dataclass(frozen=True)
class MeasuredMeans:
    settings: RoundSettings
    cosine: float
    sine: float
    trajectory_policy: str = "external_results"

    def __post_init__(self):
        if any(not math.isfinite(v) or not -1 <= v <= 1 for v in (self.cosine, self.sine)):
            raise ValueError("Each measured +/-1 sample mean must lie in [-1,1].")

    @property
    def signal(self) -> complex:
        # A finite-sample estimate can have modulus >1; do not project to a disk.
        return complex(self.cosine, self.sine)


def marginal_coordinate(signal: complex, settings: RoundSettings, axis: str) -> float:
    """Validate a declared physical marginal, without normalization correction."""
    if axis not in AXES:
        raise ValueError("Unknown Hadamard axis.")
    if not math.isfinite(abs(signal)) or abs(signal) > 1:
        raise ValueError("A true physical marginal must lie in the unit disk.")
    return signal.real if axis == "cosine" else signal.imag


def collect_virtual_round(
    settings: RoundSettings,
    trajectory_factory: Callable[[MeasurementRequest], object],
    measure_one: Callable[[MeasurementRequest, object], int],
    *,
    trajectory_policy: str = "fresh_per_shot",
) -> MeasuredMeans:
    """Request one +/-1 result per shot; reuse is an explicit non-V2 diagnostic.

    The factory maps q,r,K,delta to a trajectory request in an authorized future
    backend. It must not cache fresh keys. Full never calls the RTE factory.
    Neither callback is supplied by this module.
    """
    if trajectory_policy not in ("fresh_per_shot", "reuse_within_axis"):
        raise ValueError("Unknown trajectory policy.")
    means = []
    for axis in AXES:
        n = getattr(settings, axis).shots
        cached = None
        total = 0
        for shot in range(n):
            key = None if settings.ld == 12 else (
                settings.ld, settings.m, axis,
                shot if trajectory_policy == "fresh_per_shot" else 0,
            )
            request = MeasurementRequest(settings, axis, shot, key)
            if settings.ld == 3 and (trajectory_policy == "fresh_per_shot" or shot == 0):
                cached = trajectory_factory(request)
            outcome = measure_one(request, cached)
            if isinstance(outcome, bool) or outcome not in (-1, 1):
                raise ValueError("Measurement callback must return +/-1, not a bit.")
            total += outcome
        means.append(total / n)
    policy = "deterministic_full" if settings.ld == 12 else trajectory_policy
    return MeasuredMeans(settings, *means, trajectory_policy=policy)


@lru_cache(maxsize=1)
def original_decoder():
    """Reuse the three unchanged FunctionDefs; omit scientific module imports."""
    import numpy as np
    tree = ast.parse(fixed_bytes(CARRIER, DECODER_PATH).decode())
    names = {"_wrap_phase", "_phase_distance", "reconstruct_rpe_phase"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Missing decoder definition.")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "math": math}
    exec(compile(module, "<unchanged-rpe-decoder>", "exec"), namespace)
    return namespace["reconstruct_rpe_phase"]


def decode_measurements(
    rounds: Sequence[RoundSettings], means: Sequence[MeasuredMeans],
) -> dict:
    """No E0/oracle parameter. Output is conditional on an external alias window."""
    if not rounds or len(rounds) != len(means):
        raise ValueError("One measured record is required per round.")
    if any(s.m != m or s.q != 2**m or s.ld != rounds[0].ld or s.delta != rounds[0].delta
           or z.settings != s for m, (s, z) in enumerate(zip(rounds, means))):
        raise ValueError("Round order/settings/state/measurement law mismatch.")
    metadata = dict(
        alias_window="phase [-pi,pi); energy (-pi/delta,pi/delta]",
        alias_assumption_verified=False, state_preparation_verified=False,
        radius_lower_bound_verified=False, statistical_iid_verified=False,
        v2_fresh_call_policy_requested=all(
            z.settings.ld == 12 or z.trajectory_policy == "fresh_per_shot" for z in means),
        success_probability_evaluated=False,
    )
    if any(z.signal == 0 for z in means):
        return dict(status="ZERO_SIGNAL", energy=None, **metadata)
    phases = [cmath.phase(z.signal) for z in means]
    result = original_decoder()(phases, [s.q for s in rounds])
    ambiguous = any(math.isclose(r["distance_from_previous"], r["next_smallest_branch_distance"],
                                 rel_tol=0, abs_tol=1e-12) for r in result["rounds"][1:])
    final = result["final_phase_estimate"]
    boundary = any(math.isclose(abs(p), math.pi, rel_tol=0, abs_tol=1e-12)
                   for p in (phases[0], final))
    status = "AMBIGUOUS_BRANCH" if ambiguous else "ALIAS_BOUNDARY" if boundary else "DECODED_CONDITIONAL"
    return dict(status=status, energy=-final / rounds[0].delta if status == "DECODED_CONDITIONAL" else None,
                reconstruction=result, **metadata)
