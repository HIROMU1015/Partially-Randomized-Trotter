"""Unchanged AX-2A schedule definitions; metadata-only import bindings.

Source commit 31cdff47f3282d2898c153a1af22f90e500be4c6
src/trottertracks/resource_applicability/ax2a_control_plan.py
Only imports are adapted: existing baseline coefficients, Python-int scope.
"""
from dataclasses import dataclass
import math
from ._baselines import trotter_2nd_list, yoshida_4th_list


def require_integer_count(value, *, name, minimum=0):
    if type(value) is not int:
        raise TypeError(name+" must be a Python integer")
    if value < minimum:
        raise ValueError(name+" is below its minimum")
    return value


def _get_w_list(formula):
    if formula == "2nd": return trotter_2nd_list()
    if formula == "4th": return yoshida_4th_list()
    raise ValueError("Only S2/Yoshida4 are connected")


@dataclass(frozen=True)
class ControlStage:
    term: int
    time: float
    mode: str

    def branch_time(self, control: int) -> float:
        control = require_integer_count(control, name="control")
        if control not in (0, 1):
            raise ValueError("control must be integer 0 or 1.")
        if self.mode == "UNCONTROLLED":
            return self.time
        if self.mode == "DIRECTIONAL":
            return self.time if control else -self.time
        if self.mode == "ORDINARY":
            return self.time if control else 0.0
        raise ValueError("Unknown control mode.")


def partial_s2_control_plan(num_deterministic: int, delta: float, *, has_tail: bool):
    """Forward uncontrolled, central ordinary, reversed directional.

    The tail stage denotes a unitary event/trajectory, never the nonunitary
    corrected finite mean. Its internal event phase must also be controlled.
    A separate ordinary-controlled scalar phase is required.
    """
    count = require_integer_count(num_deterministic, name="num_deterministic")
    if isinstance(delta, bool) or not math.isfinite(delta) or type(has_tail) is not bool:
        raise ValueError("Invalid delta or tail flag.")
    forward = [ControlStage(i, delta / 2, "UNCONTROLLED") for i in range(count)]
    middle = [ControlStage(count, delta, "ORDINARY")] if has_tail else []
    reverse = [ControlStage(i, delta / 2, "DIRECTIONAL") for i in reversed(range(count))]
    return tuple(forward + middle + reverse)


def deterministic_control_plan(num_terms: int, T: float, q: int, formula: str):
    """Same second/fourth approximation as deterministic_pf_state, unmerged.

    Fourth-order composes three symmetric second-order pieces, including
    the negative middle time. The IR intentionally retains undo pairs.
    """
    count = require_integer_count(num_terms, name="num_terms")
    q = require_integer_count(q, name="q", minimum=1)
    if isinstance(T, bool) or not math.isfinite(T) or formula not in ("2nd", "4th"):
        raise ValueError("Invalid time or formula.")
    weights = _get_w_list(formula)
    pieces = weights if formula == "2nd" else (weights[1], weights[0], weights[1])
    result = []
    for _ in range(q):
        for fraction in pieces:
            result.extend(partial_s2_control_plan(count, T * fraction / q, has_tail=False))
    return tuple(result)
