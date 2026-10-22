"""Stdlib-only AX-2A preparation records, never scientific execution."""

import hashlib
import json
import math
from pathlib import Path

BASE_COMMIT = "b2e1bf65e21893b6c617223b42313623d3186f12"
STATUS = "AX2A_PREPARED_AX2B_NOT_AUTHORIZED"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prefix_candidates(rank):
    if type(rank) is not int or rank <= 0:
        raise ValueError("Actual DF rank must be a positive integer.")
    # AX-0 fractions and rounding exactly; no model/bias/cost input.
    numerators = (0, 1, 2, 4, 6, 8)
    return tuple(sorted({min(rank, max(0, (n * rank + 4) // 8 + d))
                         for n in numerators for d in (-1, 0, 1)}))


def axis_headroom(epsilon_axis, bias, uncertainty):
    """Three-state numerical eligibility; no operational bias prediction."""
    if bias is None or uncertainty is None:
        return {"status": "UNDETERMINED", "eligible": None, "headroom": None}
    values = (epsilon_axis, bias, uncertainty)
    if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
        raise ValueError("Headroom inputs must be finite.")
    if epsilon_axis <= 0 or bias < 0 or uncertainty < 0:
        raise ValueError("Invalid epsilon, bias or uncertainty.")
    headroom = epsilon_axis - bias - uncertainty
    if headroom > 0:
        status, eligible = "ELIGIBLE", True
    elif max(0.0, bias - uncertainty) >= epsilon_axis:
        status, eligible = "INELIGIBLE", False
    else:
        status, eligible = "UNDETERMINED", None
    return {"status": status, "eligible": eligible, "headroom": headroom}


def pilot_draft():
    """Model-independent structural tasks. All caps are proposals, not grants."""
    h4 = [
        {"id": "H4_LEGACY_STATE_ACTION", "scope": "legacy rank12 saved snapshot",
         "checks": ["dense versus action", "Qiskit/OpenFermion ordering", "scalar phase", "primitive sector"]},
        {"id": "H4_FINITE_AND_B0", "scope": "same legacy target",
         "checks": ["K2/K4/K6", "corrected/raw", "signed discard/PF decomposition", "empty tail"]},
        {"id": "H4_STRONG_BASELINES", "scope": "same legacy target",
         "checks": ["global S4", "negative time", "ordinary control versus directional IR", "B2 backbone fairness"]},
    ]
    h6 = [
        {"id": f"H6_B1_{order}_q{q}", "method": "B1", "order": order,
         "prefix_fraction": 1.0, "q": q, "R": None, "K": None}
        for order in ("2nd", "4th") for q in (1, 8)
    ] + [
        {"id": "H6_B2_short", "method": "B2", "order": "2nd", "prefix_fraction": 0.5,
         "q": 1, "R": 1, "K": 2},
        {"id": "H6_B3_long", "method": "B3", "order": "2nd", "prefix_fraction": 0.0,
         "q": 8, "R": 64, "K": 6},
    ]
    return {
        "schema_version": "track_a_ax2b_technical_pilot_draft_v1",
        "status": "DRAFT_NOT_AUTHORIZATION", "science_authorized": False,
        "ax2b_authorized": False, "next_stage_authorized": False, "mandatory_stop": True,
        "task_selection": "structural/model-independent before reference results",
        "H4_tasks": h4, "H6_tasks": h6, "H8_tasks": [],
        "proposed_target": {"model": "linear H-chain", "basis": "STO-3G", "geometry_angstrom": 1.0,
                            "T": 0.8, "epsilon_min": 0.001, "H6_df_rank": None,
                            "H6_df_tol": None, "state_identity": None},
        "proposed_caps": {"cpu_cores": 1, "blas_threads": 1, "processes": 1,
                          "gpu": False, "address_space_bytes": 8589934592,
                          "wall_seconds_per_H4_task": 900, "wall_seconds_per_H6_task": 1800,
                          "total_wall_seconds": 14400, "output_bytes": 536870912,
                          "total_wrappers": 64, "cost_trajectories_per_random_cell": 2,
                          "tail_matvecs_per_signal_path": 448,
                          "tail_matvecs_corrected_and_raw_total": 896},
        "assigned_resources": None,
        "unresolved_before_launch": ["snapshot hash and creation allowance", "DF tolerance semantics and actual rank",
                                     "primitive sector certification", "H4 callback/wrapper lowering",
                                     "reference numerical allowance", "state and seed identities",
                                     "actual CPU/RAM allocation", "per-circuit gate and matvec caps",
                                     "source/plan freeze", "separate explicit launch authorization"],
        "information_scope": "all exposed H6 science values are development; no H8 truth access",
        "cap_scope": "technical profiling only; not an adequate main-campaign search bound",
    }


def build_preparation(root: Path, review_path: Path):
    """Hash source and freeze existing FEW by reference; no fit or signal read."""
    root, review_path = Path(root), Path(review_path)
    saved_path = Path("artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/model_fits.json")
    files = [
        "src/trotterlib/df_hamiltonian.py", "src/trotterlib/pf_c_system_size_validation.py",
        "src/trotterlib/df_gpu_statevector.py", "src/trotterlib/rte.py",
        "src/trotterlib/product_formula.py", "src/trotterlib/pf_decomposition.py",
        "src/trotterlib/df_partial_s2.py", "src/trotterlib/df_partial_s2_repeated.py",
        "src/trotterlib/df_rpe_hadamard_compiled_cost.py",
        "src/trotterlib/pr2_matched_accuracy_m1_execution.py",
        "src/trotterlib/research_direction_energy_tail_pareto.py",
        "src/trotterlib/df_partial_randomized_pf.py",
        "src/trotterlib/rpe_hadamard_compiled_cost_proxy.py", str(saved_path),
        "docs/research/track_a_ax0_research_contract.md",
        "docs/research/track_a_ax0_benchmark_protocol.md",
        "docs/research/track_a_ax0_compute_budget.md",
        "docs/research/track_a_ax1a_preanalysis_contract.md",
    ]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    return {
        "schema_version": "track_a_ax2a_preparation_v1", "status": STATUS,
        "base_commit": BASE_COMMIT,
        "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "existing_source_and_contract_hashes": hashes,
        "frozen_H4_model": {"path": str(saved_path), "sha256": hashes[str(saved_path)],
                            "model": "PRED_BASE_FEW_PARAM", "fold": "full210", "refit": False},
        "primary_RQ": "RQ-R", "secondary_RQ": "RQ-P1", "operational_required": False,
        "science_authorized": False, "ax2b_authorized": False,
        "next_stage_authorized": False, "mandatory_stop": True,
        "new_scientific_calculation_count": 0,
        "validation_scope": "synthetic implementation only; native DF full wrapper remains unvalidated",
        "pilot_draft": pilot_draft(),
    }
