"""H6 orchestration components, tested only with injected synthetic ports.

No concrete molecular backend or executable scientific launcher is supplied.
Frozen H4 modules remain unchanged. Every terminal requires a human next step.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .ax2a_preparation import digest
from .ax2b_h6_contract import PHASES, preparation_plan
from .ax2b_limits import CallBudget, output_size


class BoundedWriter:
    def __init__(self, output, *, byte_cap, diagnostics_cap=1024, reserve=65536):
        self.output = Path(output)
        if any(type(v) is not int or v < 0 for v in (byte_cap, diagnostics_cap, reserve)) or byte_cap <= reserve:
            raise ValueError("OUTPUT_BUDGET")
        self.byte_cap, self.reserve = byte_cap, reserve
        self.diagnostics = CallBudget(records=diagnostics_cap)

    def write(self, name, value, *, diagnostic=False, terminal=False):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("OUTPUT_NAME")
        payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
        limit = self.byte_cap if terminal else self.byte_cap - self.reserve
        if output_size(self.output) + len(payload) > limit:
            raise RuntimeError("OUTPUT_WRITE_CAP")
        if diagnostic:
            self.diagnostics.take("records")
        with (self.output / name).open("xb") as stream:
            stream.write(payload)


def bounded_matvec(action, total_budget, *, name, per_action_cap):
    """Both counters increment *before* matvec/rmatvec work; no after-call gate."""
    local = CallBudget(calls=per_action_cap)
    def apply(vector):
        if total_budget.used[name] >= total_budget.limits[name]:
            raise RuntimeError("CALL_BUDGET:" + name)
        local.take("calls")
        total_budget.take(name)
        return action(vector)
    return apply, local


def bounded_sector_matrix(action, dimension, total_budget, *, per_action_cap=20000):
    if type(dimension) is not int or not 1 <= dimension <= 400:
        raise ValueError("SECTOR_REFERENCE_DIMENSION")
    import numpy as np
    apply, local = bounded_matvec(action, total_budget, name="reference_matvec", per_action_cap=per_action_cap)
    matrix = np.empty((dimension, dimension), dtype=np.complex128)
    for i in range(dimension):
        vector = np.zeros(dimension, dtype=np.complex128)
        vector[i] = 1
        column = np.asarray(apply(vector), dtype=np.complex128)
        if column.shape != (dimension,) or not np.isfinite(column).all():
            raise ValueError("REFERENCE_ACTION_RESULT")
        matrix[:, i] = column
    return matrix, local.used["calls"]


def bounded_solver(action, dimension, initial, total_budget, *, solver):
    """Caller-supplied solver; no molecule/state generation entry point.

    SciPy-compatible options are a proposal. Both directions share the same
    counter. Post-solve diagnostics/residual calls must use that same operator.
    A molecular use needs separately authorized input generation.
    """
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    if type(dimension) is not int or not 3 <= dimension <= 400:
        raise ValueError("SOLVER_DIMENSION")
    v0 = np.asarray(initial, dtype=np.complex128)
    if v0.shape != (dimension,) or not np.isfinite(v0).all() or abs(np.linalg.norm(v0)-1) > 1e-12:
        raise ValueError("SOLVER_INITIAL_STATE")
    apply, counter = bounded_matvec(action, total_budget, name="solver_matvec", per_action_cap=10000)
    operator = LinearOperator((dimension, dimension), matvec=apply, rmatvec=apply, dtype=np.complex128)
    result = solver(operator, k=1, which="SA", tol=1e-12, maxiter=1000,
                    ncv=min(40, dimension), v0=v0.copy())
    return result, operator, counter


def finite_scale_guard(*, log_B, intermediate_norm, absolute_discrepancy,
                       absolute_gate=1e-9, relative_gate=1e-10):
    """Fixed mixed-scale engineering agreement; never u_bound or eligibility.

    Overflow or subnormal raw attenuation stops this initial technical policy.
    No tolerance, R, seed or rank is changed to rescue a failing cell.
    """
    import sys
    values = (log_B, intermediate_norm, absolute_discrepancy, absolute_gate, relative_gate)
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("NONFINITE_OR_NEGATIVE_SCALE")
    if log_B > math.log(sys.float_info.max):
        raise ValueError("NORMALIZATION_OVERFLOW")
    if log_B > -math.log(sys.float_info.min):
        raise ValueError("RAW_ATTENUATION_SUBNORMAL_OR_UNDERFLOW")
    threshold = absolute_gate + relative_gate * max(1.0, intermediate_norm)
    if not math.isfinite(threshold) or absolute_discrepancy > threshold:
        raise ValueError("SCALE_AGREEMENT")
    return {"threshold": threshold, "evidence_kind": "TECHNICAL_AGREEMENT", "certified": False}


def exercise_synthetic_controller(port, writer, *, actual_rank):
    """Exercise the complete schedule with an explicitly synthetic injected port.

    This is NOT a molecular launch API. The port's Boolean correctness is a
    fixture witness; the future molecular port needs substantive checks.
    """
    if getattr(port, "synthetic_only", False) is not True:
        raise RuntimeError("MOLECULAR_PORT_NOT_IMPLEMENTED_OR_AUTHORIZED")
    plan = preparation_plan(actual_rank)
    caps = plan["caps_proposed"]
    budget = CallBudget(**{name: caps[name] for name in ("compile","trajectory","occurrence")})
    completed, compiled = 0, 0
    try:
        for phase in PHASES:
            writer.write("phase_" + phase + ".json", {"phase": phase})
            if phase == "input_reference":
                port.setup(plan)
            elif phase == "correctness":
                for cell in plan["cells"]:
                    value = port.correctness(cell)
                    if value.get("technical_pass") is not True:
                        raise ValueError("CORRECTNESS:" + cell["id"])
                    writer.write(cell["id"] + "_correctness.json", value)
                    completed += 1
            else:
                for cell in plan["cells"]:
                    for replica in range(cell["replicas"]):
                        group = [t for t in plan["wrapper_tasks"] if t["cell_id"] == cell["id"] and t["replica"] == replica]
                        trajectory = None
                        if cell["R"] is not None:
                            budget.take("trajectory")
                            trajectory = []
                            for occurrence in range(cell["q"]):
                                budget.take("occurrence")
                                trajectory.append(port.occurrence(cell, group[0]["seed"], occurrence))
                        before = digest(trajectory)
                        try:
                            for task in group:
                                budget.take("compile")
                                value = port.wrapper(task, trajectory)
                                if digest(trajectory) != before:
                                    raise ValueError("PREPARED_EVENTS_MUTATED")
                                writer.write("wrapper_%02d.json" % compiled, value)
                                compiled += 1
                        finally:
                            port.release_group()
        status, reason = "SYNTHETIC_CONTROLLER_COMPLETE", None
    except Exception as error:
        status, reason = "SYNTHETIC_CONTROLLER_STOP", str(error)[:512]
    terminal = {"status": status, "reason": reason, "completed_correctness_cells": completed,
                "compiled_wrappers": compiled, "calls": budget.used,
                "synthetic_only": True, "H6_status": "H6_NOT_AUTHORIZED",
                "numerical_allowance_certified": False, "accuracy_eligibility": "UNDETERMINED",
                "N": None, "G": None, "mandatory_stop": True, "next_stage_authorized": False}
    writer.write("worker_terminal.json", terminal, terminal=True)
    return terminal
