"""Opt-in Q1 readout and count decoder. No scientific input is loaded on import.

All confidence statements are conditional on the preserved prefix success
event and fixed independent-bit/ideal-gate model. This module does not compile,
sample, qualify a physical feedback provider, or invoke the Q0 campaign.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
Q0_COMMIT = '34327b904540556bce7509f2a3168cbff36daab7'
LEGACY_PATH = 'src/trotterlib/rpe_four_round_phase_validation.py'
CAP = F(2, 5)
FEEDBACK_CAP = F('0.000748774257')  # Common conservative saved Q0 condition.
REFERENCE_TIME = F.from_float(.02)  # Preserve the legacy energy conversion literal.


class UnqualifiedPrefix(ValueError):
    pass


@lru_cache(None)
def _q0_module():
    # Reuse the existing definitions; no arm_run(), worst(), margin(), etc.
    path = ROOT/'scripts/diagnose_pr_reference_q0.py'
    spec = importlib.util.spec_from_file_location('_q1_q0_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.I = module.setup()
    return module


def _q0():
    module = _q0_module()
    if module.mp.mp.prec != 192 or module.mp.iv.prec != 192:
        raise RuntimeError('Q1 requires the preserved 192-bit interval context')
    return module


@lru_cache(None)
def _legacy_decoder():
    # The full validation module imports scientific builders. Reuse only its
    # original three pure functions, following the Q0 pinned-AST reuse pattern.
    import numpy as np
    text = subprocess.check_output(['git', 'show', Q0_COMMIT+':'+LEGACY_PATH], cwd=ROOT)
    wanted = {'_wrap_phase', '_phase_distance', 'reconstruct_rpe_phase'}
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise RuntimeError('Pinned decoder functions unavailable')
    ns = {'np': np, 'math': math, 'Sequence': Sequence, 'Any': Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), LEGACY_PATH, 'exec'), ns)
    return ns['reconstruct_rpe_phase']


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name+' must be an integer in range')
    return value


@dataclass(frozen=True)
class AxisCounts:
    n: int
    plus: int

    def __post_init__(self):
        _integer(self.n, 'n', 1)
        _integer(self.plus, 'plus')
        if self.plus > self.n:
            raise ValueError('plus exceeds n')

    @property
    def mean(self):
        return F(2*self.plus, self.n)-1


@dataclass(frozen=True)
class RoundCounts:
    m: int
    X: AxisCounts
    Y: AxisCounts

    def __post_init__(self):
        _integer(self.m, 'm')
        if self.m > 16 or not isinstance(self.X, AxisCounts) or not isinstance(self.Y, AxisCounts):
            raise ValueError('Expected m0..16 X/Y count records')


@dataclass(frozen=True)
class Prefix:
    history: tuple
    legacy_result: dict
    base_phase_center: Any
    last_principal_phase: Any
    energy_center: Any
    energy_interval: Any
    h: F
    q_last: int
    branch_certificates: tuple
    saved_arm: str | None = None
    status: str = 'CONDITIONAL_ON_SAVED_PREFIX_SUCCESS_EVENT'


@dataclass(frozen=True)
class FeedbackAngle:
    parameter: float
    reduced_ideal_interval: Any
    arithmetic_error_upper: F
    period_shift: int
    bound_prefix: Prefix | None = None
    definition: str = '2 * exact_arg_of_m16_counts + pi/2 modulo 2pi'


@dataclass(frozen=True)
class SignalModel:
    rho_lower: F
    bias_upper: F
    feedback_limit: F = FEEDBACK_CAP
    source: str = 'ARTIFICIAL_OR_CALLER_SUPPLIED_MODEL_NOT_A_QUALIFICATION'
    saved_arm: str | None = None

    def __post_init__(self):
        if not (0 < self.rho_lower <= 1 and 0 <= self.bias_upper < CAP
                and self.feedback_limit >= 0):
            raise ValueError('Invalid signal bounds')


def saved_q0_model(arm):
    """Read the saved model on explicit scientific use; not called by Q1 tests."""
    if arm not in ('Bprime', 'Reference_B1'):
        raise ValueError('Unknown Q0 arm')
    path = 'artifacts/pr_reference_q0/2026-11-09/run01/INPUTS.json'
    data = json.loads(subprocess.check_output(['git', 'show', Q0_COMMIT+':'+path], cwd=ROOT))
    row = data['arms'][arm]
    return SignalModel(F.from_float(row['raw_radius_lower']),
                       CAP-F.from_float(row['statistical_phase_lower']),
                       source=Q0_COMMIT+':'+path+'#'+arm, saved_arm=arm)


def _count_phase(row):
    q = _q0()
    x, y = row.X.mean, row.Y.mean
    if not x and not y:
        raise UnqualifiedPrefix('Zero X/Y sample vector has no phase')
    if not y and x < 0:
        return -q.mp.iv.pi  # Match the old [-pi,pi) convention at the cut.
    return q.mp.iv.atan2(q.iv(y), q.iv(x))


def _period_near(value, reference):
    """Choose an integer representative, then retain its FULL interval.

    Correctness does not require a correctly rounded integer at a period cut:
    changing the integer changes only a full 2pi representative.
    """
    q = _q0()
    a, b = q.endpoints((q.iv(reference)-value)/(2*q.mp.iv.pi))
    shift = int(q.mp.nint((a+b)/2))
    return value+2*shift*q.mp.iv.pi, shift


def _circular_distance(value):
    q = _q0()
    return q.mp.iv.atan2(abs(q.mp.iv.sin(value)), q.mp.iv.cos(value))


def _saved_prefix_shots(arm):
    base = 'f2638a7845685a119619382b850441f5d8d33713'
    if arm == 'Bprime':
        path = 'artifacts/pr_reference_bprime/2026-11-07/ERROR_RPE.json'
        data = json.loads(subprocess.check_output(['git', 'show', base+':'+path], cwd=ROOT))
        return tuple(row['shots_per_axis'] for row in data['rows'][:17])
    if arm == 'Reference_B1':
        path = 'artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json'
        data = json.loads(subprocess.check_output(['git', 'show', base+':'+path], cwd=ROOT))
        return tuple(row['arms']['reference']['shots_per_axis'] for row in data['rows'][:17])
    raise ValueError('Unknown saved prefix arm')


def prefix_from_counts(history, *, h=REFERENCE_TIME, saved_arm=None):
    """Bind all m0..16 count records to the preserved branch decoder.

    Interval checks establish the SAME nearest branches for exact count args.
    They do not observe whether the stochastic prefix success event occurred.
    The saved per-round .4 phase premise and alias margin remain assumptions.
    """
    q = _q0()
    history = tuple(history)
    if len(history) != 17 or tuple(r.m for r in history) != tuple(range(17)):
        raise ValueError('Exactly one ordered m0..16 history is required')
    if type(h) is not F or h <= 0:
        raise ValueError('h must be an explicit positive rational reference time')
    if saved_arm is not None:
        thresholds = _saved_prefix_shots(saved_arm)
        if h != REFERENCE_TIME or len(thresholds) != 17 or any(
                (row.X.n, row.Y.n) != (n, n) for row, n in zip(history, thresholds)):
            raise UnqualifiedPrefix('Counts do not match the preserved prefix shot schedule')
    phases = [_count_phase(r) for r in history]
    # Past data only: no target energy, state, covariance, or future count input.
    approximate = [math.atan2(float(r.Y.mean), float(r.X.mean)) for r in history]
    old = _legacy_decoder()(approximate, [1 << r.m for r in history])
    previous = None
    certificates = []
    for m, (phase, row) in enumerate(zip(phases, old['rounds'])):
        power = 1 << m
        aligned, phase_shift = _period_near(phase, F.from_float(row['measured_round_phase']))
        unwrapped = (aligned+2*row['selected_branch']*q.mp.iv.pi)/power
        center, base_shift = _period_near(unwrapped, F.from_float(row['base_phase_estimate']))
        if previous is not None:
            chosen = _circular_distance(center-previous)
            neighbors = [_circular_distance(center+sign*2*q.mp.iv.pi/power-previous)
                         for sign in (-1, 1)]
            if any(q.endpoints(chosen)[1] >= q.endpoints(v)[0] for v in neighbors):
                raise UnqualifiedPrefix('Nearest branch is ambiguous or differs from the old decoder')
        certificates.append(dict(m=m, branch=row['selected_branch'], phase_period=phase_shift,
                                 base_period=base_shift, nearest_branch_verified=True))
        previous = center
    # The saved alias promise places the true base phase at least CAP away
    # from +/-pi. On the good prefix event this representative is unambiguous.
    qlast = 1 << 16
    guard = q.mp.iv.pi-q.iv(CAP)+q.iv(CAP)/qlast
    if q.endpoints(abs(previous))[1] >= q.endpoints(guard)[0]:
        raise UnqualifiedPrefix('Alias-edge history cannot support the saved linear energy interval')
    ec = -previous/q.iv(h)
    w = q.iv(CAP)/(qlast*q.iv(h))
    energy = q.I([q.endpoints(ec-w)[0], q.endpoints(ec+w)[1]])
    return Prefix(history, old, previous, phases[-1], ec, energy, h, qlast, tuple(certificates), saved_arm)


def _feedback_phase(phase, prefix=None):
    q = _q0()
    # 2*q16*center differs from 2*principal by an integer times 2pi.
    # Work with the small principal phase; never multiply a rounded energy.
    ideal = 2*phase+q.mp.iv.pi/2
    reduced, shift = _period_near(ideal, F(0))
    lo, hi = q.endpoints(reduced)
    value = float((lo+hi)/2)
    if not math.isfinite(value):
        raise ArithmeticError('Nonfinite feedback parameter')
    error = abs(q.I(value)-reduced)
    upper = q.exact(error._mpi_[1])
    return FeedbackAngle(value, reduced, upper, shift, prefix)


def feedback_from_prefix(prefix):
    if not isinstance(prefix, Prefix) or prefix.q_last != 1 << 16:
        raise ValueError('A bound m16 prefix is required')
    return _feedback_phase(prefix.last_principal_phase, prefix)


def feedback_readout(core, feedback, *, family):
    """Raw opt-in wrapper for an EXISTING core. No compile/reset/sampling.

    Match old wrapper order: H, core, P(-phi), H, measurement. Caller keeps
    the old Reference reset-after-compile/before-execution convention.
    """
    from qiskit import QuantumCircuit
    if family not in ('Bprime', 'Reference_B1') or not isinstance(feedback, FeedbackAngle):
        raise ValueError('Explicit family and feedback are required')
    if type(feedback.parameter) is not float or not math.isfinite(feedback.parameter):
        raise ValueError('Finite binary64 readout parameter required')
    if feedback.bound_prefix is not None and feedback.bound_prefix.saved_arm not in (None, family):
        raise ValueError('Readout family differs from the bound prefix arm')
    n = core.num_qubits
    if core.num_clbits or n < 2 or (family == 'Reference_B1' and n % 2 != 1):
        raise ValueError('Expected an unmeasured core with ancilla last')
    if any(i.operation.name == 'measure' for i in core.data):
        raise ValueError('Core already contains a readout')
    circuit = QuantumCircuit(n, 1)
    circuit.h(n-1)
    circuit.compose(core, inplace=True)
    circuit.p(-feedback.parameter, n-1)
    circuit.h(n-1)
    circuit.measure(n-1, 0)
    return circuit


def decode_counts(prefix, feedback, plus, model, *, additional_feedback_error=None, n=340):
    """Q0 inversion for actual counts, conditional on declared scientific bounds.

    Missing P/lowering/provider error is UNKNOWN, never silently zero.
    The prefix's statistical width is u; it is not charged again as e_phi.
    Additional error must be an angular-axis bound. A general operator/raw
    disk requires separate model qualification; it is not an angle in radians.
    """
    q = _q0()
    if not isinstance(prefix, Prefix) or not isinstance(feedback, FeedbackAngle) or not isinstance(model, SignalModel):
        raise TypeError('Bound prefix, feedback and signal model required')
    if feedback.bound_prefix is not prefix:
        raise ValueError('Feedback was generated for a different prefix')
    if n != 340 or type(n) is not int:
        raise ValueError('Q1 uses fixed n=340; no adaptive shot count')
    count = AxisCounts(n, plus)
    output = dict(status='UNQUALIFIED_FEEDBACK_ERROR', n=n, plus=plus,
                  prior_energy_interval=q.pack(prefix.energy_interval),
                  feedback_parameter=feedback.parameter, feedback_arithmetic_error=str(feedback.arithmetic_error_upper),
                  model_source=model.source, prefix_status=prefix.status,
                  saved_prefix_schedule=prefix.saved_arm or 'UNBOUND_ARTIFICIAL_OR_EXTERNAL_PREFIX',
                  conditional_failure_budget='34/720 + 1/360 = 1/20',
                  probability_premises='Saved prefix sufficiency, fresh IID trials, and numerical acceptance coverage remain conditional',
                  formal_RPE_or_resource_qualification=False)
    if model.saved_arm is not None and prefix.saved_arm != model.saved_arm:
        output.update(status='UNQUALIFIED_PREFIX_SCHEDULE', reason='Saved model requires the matching preserved prefix shot schedule')
        return output
    if additional_feedback_error is None:
        return output
    if type(additional_feedback_error) is not F or additional_feedback_error < 0:
        raise ValueError('Explicit nonnegative rational additional error required')
    total_error = feedback.arithmetic_error_upper+additional_feedback_error
    output['total_feedback_error_upper'] = str(total_error)
    if total_error > model.feedback_limit:
        output['reason'] = 'Feedback error exceeds the configured sufficient limit'
        return output
    U, D = q.iv(2*CAP), q.iv(model.bias_upper+total_error)
    if q.verdict(U+D, q.mp.iv.pi/2) != 'PASS':
        output.update(status='UNQUALIFIED_MULTIPLE_SINE_BRANCHES', reason='All branches must be retained; Q0 monotone inverse is inapplicable')
        return output
    eps = q.epsilon(n)
    output['confidence_mean_interval'] = q.pack(q.iv(count.mean)+q.I([-q.endpoints(eps)[1],q.endpoints(eps)[1]]))
    result = q.hull(n, plus, q.iv(model.rho_lower), D, U, eps)
    if result is None:
        output.update(status='INCONSISTENT_COUNTS', estimate=None)
        return output
    lo, hi, diameter = result
    time = 2*prefix.q_last*q.iv(prefix.h)
    lower, upper = prefix.energy_center-hi/time, prefix.energy_center-lo/time
    energy = q.I([q.endpoints(lower)[0],q.endpoints(upper)[1]])
    # The explicit hull includes the ideal-center enclosure once.
    energy_diameter = energy.delta
    output.update(phase_hull=[q.pack(lo),q.pack(hi)], phase_diameter=q.pack(diameter),
                  energy_hull=q.pack(energy), energy_diameter=q.pack(energy_diameter),
                  energy_midpoint_enclosure=q.pack((lower+upper)/2),
                  energy_estimate_rational=str((q.exact(energy._mpi_[0])+q.exact(energy._mpi_[1]))/2),
                  status='CONDITIONAL_WIDTH_SUFFICIENT' if q.verdict(energy_diameter,2*q.iv(CAP)/time)=='PASS' else 'INSUFFICIENT_OR_UNRESOLVED_WIDTH')
    return output
