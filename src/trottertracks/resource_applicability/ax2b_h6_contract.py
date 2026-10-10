"""Stdlib-only H6 technical proposal. No input access or execution grant."""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path
import runpy

from .ax2a_preparation import digest

REVIEW_REF = "d6510db9326e9335bedd03d0d07c490561e23112"
PHASES = ("input_reference", "correctness", "wrapper_cost")


@lru_cache(maxsize=1)
def _pf_iterator():
    # This existing module is pure stdlib; importing its package would run
    # trotterlib.__init__ and eagerly import numerical libraries.
    path = Path(__file__).resolve().parents[2] / "trotterlib/pf_decomposition.py"
    return runpy.run_path(str(path))["iter_pf_steps"]


def h6_cells(actual_rank=None):
    if actual_rank is not None and (type(actual_rank) is not int or actual_rank < 2):
        raise ValueError("ACTUAL_RANK_AT_LEAST_TWO_REQUIRED")
    p = (actual_rank + 1) // 2 if actual_rank is not None else "(L+1)//2"
    full = actual_rank if actual_rank is not None else "L"
    specs = [("B0_S2_q2", "B0", "2nd", p, 2, None, None, 1)]
    specs += [(f"B1_S{order}_q{q}", "B1", formula, full, q, None, None, 1)
              for order, formula in ((2, "2nd"), (4, "4th")) for q in (1, 2)]
    specs += [("B2_K2_q2_R4", "B2", "2nd", p, 2, 4, 2, 2),
              ("B3_K6_q2_R4", "B3", "2nd", 0, 2, 4, 6, 2)]
    return [{"id": "H6_" + name, "method": method, "formula": order,
             "prefix": prefix, "q": q, "R": R, "r": R // q if R else None,
             "K": K, "replicas": replicas}
            for name, method, order, prefix, q, R, K, replicas in specs]


def cost_seed(cell_id, replica):
    if type(replica) is not int or replica < 0:
        raise ValueError("REPLICA_INDEX")
    value = f"track_a_h6_pilot_draft_v1_cost_seed:{cell_id}:{replica}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def wrapper_tasks(cells):
    tasks, seeds = [], set()
    for cell in cells:
        for replica in range(cell["replicas"]):
            seed = cost_seed(cell["id"], replica) if cell["R"] else None
            if seed is not None:
                if seed in seeds:
                    raise ValueError("COST_SEED_COLLISION")
                seeds.add(seed)
            for control in ("ordinary", "symmetric_directional"):
                for axis in ("cosine", "sine"):
                    tasks.append({"cell_id": cell["id"], "replica": replica,
                                  "control": control, "axis": axis, "seed": seed})
    return tasks


def primitive_time_schedule(cell, *, T=0.8):
    """Actual merged global PF and unmerged directional/partial schedules.

    Returned times include negative S4 pieces, undo branch signs, and partial
    micro-step times. No numerical action is performed, no coverage is pruned.
    Scalar phase is recorded separately from primitive evolution times.
    """
    if type(cell["prefix"]) is not int or cell["prefix"] < 0:
        raise ValueError("UNRESOLVED_ACTUAL_RANK")
    if isinstance(T, bool) or not math.isfinite(T):
        raise ValueError("TIME")
    count, delta = cell["prefix"] + 1, T / cell["q"]
    if cell["method"] in ("B2", "B3"):
        ordinary = [(i, delta / 2) for i in range(count)]
        ordinary += [(i, delta / 2) for i in reversed(range(count))]
        directional = [(i, t, "UNCONTROLLED") for i, t in ordinary[:count]]
        directional += [(i, t, "DIRECTIONAL") for i, t in ordinary[count:]]
        tail_time = delta / cell["r"]
    else:
        w = 1 / (2 - 2**(1/3))
        weights = [1.0] if cell["formula"] == "2nd" else [1 - 2*w, w]
        ordinary = [(i, delta * fraction) for i, fraction in _pf_iterator()(count, weights)]
        pieces = [1.0] if cell["formula"] == "2nd" else [w, weights[0], w]
        directional = [(i, delta * fraction / 2, mode) for fraction in pieces
                       for mode, indices in (("UNCONTROLLED", range(count)),
                                             ("DIRECTIONAL", reversed(range(count))))
                       for i in indices]
        tail_time = None
    times = {(i, t) for i, t in ordinary}
    for i, t, mode in directional:
        times.add((i, t))
        if mode == "DIRECTIONAL":
            times.add((i, -t))
    return {"ordinary_one_outer_step": ordinary, "directional_one_outer_step": directional,
            "unique_primitive_times": sorted(times), "tail_micro_time": tail_time,
            "q": cell["q"], "scalar_total_time": T,
            "probe_policy": "structural sector proof + independent small oracle + registered numerical probes",
            "molecular_probe_plan_sealed": False}


def preparation_plan(actual_rank=None):
    cells = h6_cells(actual_rank)
    tasks = wrapper_tasks(cells)
    assert len(cells) == 7 and len(tasks) == 36
    return {"schema": "track_a_ax2b_post_review_preparation_v1",
            "status": "H6_NOT_AUTHORIZED", "contract_status": "DRAFT_NOT_AUTHORIZATION",
            "science_authorized": False, "input_generation_authorized": False,
            "launch_allowed": False, "mandatory_stop": True, "next_stage_authorized": False,
            "review_repository_ref": REVIEW_REF, "actual_rank": actual_rank,
            "target": {"model": "linear_H6", "geometry_angstrom": 1.0, "basis": "sto-3g",
                       "n_qubits": 12, "sector_dimension": 400, "nelec_alpha": 3, "nelec_beta": 3,
                       "T": 0.8, "epsilon_signal_diagnostic": 0.001,
                       "definition": "serialized binary64 H_DF and normalized saved designated state"},
            "df_policy": {"name": "TOL_ONLY_NO_CONFIG_FALLBACK", "df_tol": 1e-8,
                          "final_rank_supplied": False, "cutoff": 0.0},
            "cells": cells, "wrapper_tasks": tasks,
            "primitive_schedules": ([{"cell_id": c["id"], **primitive_time_schedule(c)} for c in cells]
                                     if actual_rank is not None else None),
            "caps_proposed": {"phase_wall_seconds": dict(zip(PHASES, (1800,1800,3600))),
                              "total_wall_seconds": 7200, "address_space_bytes": 8*2**30,
                              "output_bytes": 512*2**20, "log_bytes": 65536, "diagnostics": 1024,
                              "compile": 36, "trajectory": 4, "occurrence": 8,
                              "primitive": 2000, "control_probe": 256,
                              "solver_matvec": 10000, "reference_matvec": 20000,
                              "reference_matvec_per_action": 20000,
                              "deterministic_actions_per_cell": 100000,
                              "tail_matvecs_corrected_and_raw": {"B2": 24, "B3": 56},
                              "untranspiled_instructions": 1000000, "transpiled_instructions": 5000000},
            "compiler_proposed": {"basis_gates": ["rz","sx","x","cx"], "optimization_level": 1,
                                  "seed_transpiler": 17, "backend": None, "coupling_map": None},
            "cost_primary": "symmetric_directional", "ordinary_cost": "paired sensitivity",
            "cost_samples_per_random_cell": 2, "quantum_shots": None, "G": None,
            "numerical_allowance_certified": False, "accuracy_eligibility": "UNDETERMINED",
            "assigned_resources": None, "source_commit": None, "input_binding": None,
            "input_generation_budget": None, "authorization": None,
            "unresolved": ["separate input-generation scope/budget/authorization", "input/state/sector hashes",
                           "molecular backend integration", "actual primitive probe coverage and instruction bounds",
                           "source-bound plan and environment", "host CPU/RAM assignment", "separate pilot launch grant"],
            "retry": False, "resume": False, "H8_tasks": [], "gpu": False}


def reject_scientific_launch(*args, **kwargs):
    """No executable molecular port or launch authorization exists in this stage."""
    raise RuntimeError("H6_NOT_AUTHORIZED:DRAFT_NOT_AUTHORIZATION")
