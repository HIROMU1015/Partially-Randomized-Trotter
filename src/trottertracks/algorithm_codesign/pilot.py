"""Bounded BF-1 objectives/search. Importing this module performs no run."""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from fractions import Fraction
import math
import time
import resource
import mpmath as mp
import numpy as np

from .adapter import allocate, stationary_template, tail_statistics
from .domain import DPS, Point, fixed_references
from .numerics import EPS, Spectrum, gamma, phase_values

Q_VALUES = (1, 2, 4, 8)
R_VALUES = (5, 10, 20, 40, 80)
PRIMARY_EPSILON = .01
BRIDGE_EPSILON = .05
ALPHA = .05


def parameter_score(bias, u, log_b, deterministic, random, epsilon):
    allowance = epsilon/math.sqrt(2)-np.asarray(bias)-u
    if not np.all(np.isfinite(allowance)) or np.min(allowance) <= 0:
        return dict(feasible=False, reason="nonpositive_guarded_allowance", value=None, lower=None, shots=None)
    if u > .01*epsilon/math.sqrt(2):
        return dict(feasible=False, reason="UNRESOLVED_NUMERICAL_MARGIN", value=None, lower=None, shots=None)
    log_prefactor = math.log(2*math.log(2/(ALPHA/2)))+2*log_b
    logs = log_prefactor-2*np.log(allowance)
    if float(np.max(logs)) > 600:
        return dict(feasible=False, reason="unrepresentable_shot_budget", value=None, lower=None, shots=None)
    shots = [math.ceil(math.exp(float(v))*(1+gamma(64))) for v in logs]
    optimistic = epsilon/math.sqrt(2)-np.maximum(0, np.asarray(bias)-u)
    lower_shots = [max(1, math.floor(math.exp(log_prefactor-2*math.log(float(v)))*(1-gamma(64)))) for v in optimistic]
    cost = deterministic+random
    cost_guard = gamma(4096)*max(1, cost)
    return dict(feasible=True, reason=None, shots=shots,
                value=sum(shots)*(cost+cost_guard), lower=sum(lower_shots)*max(0, cost-cost_guard))


class Limits:
    def __init__(self):
        self.wall = time.monotonic()
        self.cpu = time.process_time()

    def check(self):
        if time.monotonic()-self.wall >= 4*3600 or time.process_time()-self.cpu >= 8*3600:
            raise RuntimeError("STOP_RESOURCE_CAP")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 4*1024**3:
            raise RuntimeError("STOP_MEMORY_CAP")


@dataclass
class Task:
    matrices: dict
    scalar: float
    lambda_r: float
    state: np.ndarray
    matrix_assembly_bound: float = 0.


class Evaluator:
    def __init__(self, task: Task, limits=None, on_cell=None):
        self.task = task
        self.limits = limits or Limits()
        self.on_cell = on_cell or (lambda row: None)
        self.spectra = {key: Spectrum(matrix) for key, matrix in task.matrices.items()}
        self.deterministic_count = len(task.matrices)-1
        if self.deterministic_count < 1:
            raise ValueError("BF-1 requires a nonempty deterministic backbone")
        if self.spectra['R'].norm > task.lambda_r*(1+1e-10):
            raise ValueError("RTE lambda does not dominate the residual generator")
        self.state = np.asarray(task.state, dtype=complex)
        norm = float(np.linalg.norm(self.state))
        if abs(norm-1) > 1e-12:
            raise ValueError("Frozen state normalization gate failed")
        self.state = self.state/norm
        h = sum(task.matrices.values(), task.scalar*np.eye(len(self.state), dtype=complex))
        full = Spectrum(h)
        action, error, _ = full.action(self.state, .8)
        self.target = complex(np.vdot(self.state, action))
        self.target_error = error+.8*task.matrix_assembly_bound+gamma(8*len(self.state))*float(np.linalg.norm(action))
        self.ideal = {}
        self.finite = {}
        self.coefficients = {}
        self._cache_only = False

    @contextmanager
    def cached_diagnostics(self):
        """Post-search diagnostics cannot compute a new signal/cell."""
        if self._cache_only:
            raise RuntimeError('Nested diagnostic phase is not allowed')
        self._cache_only = True
        try:
            yield
        finally:
            self._cache_only = False

    def _signal(self, point, q, allocation=None, K=2):
        self.limits.check()
        one, factors = stationary_template(point, self.deterministic_count, q)
        vector = self.state.copy()
        error = 0.
        tail_index = 0
        for factor in factors:
            with mp.workdps(DPS):
                exact = factor.time.value(point.basis)*mp.mpf(4)/(5*q)
                t = float(exact)
                lowering_error = float(abs(exact-mp.mpf(t)))+1e-65
            r = None
            if factor.generator == "R" and allocation is not None:
                r = allocation[tail_index % len(allocation)]
                tail_index += 1
            vector, action_error, op_norm = self.spectra[factor.generator].action(vector, t, lowering_error, r, K)
            error = op_norm*error+action_error
            if not math.isfinite(error):
                raise ValueError("UNRESOLVED_NUMERICAL_MARGIN")
            self.limits.check()
        phase, phase_error = phase_values(np.array([self.task.scalar*.8]))
        value = complex(phase[0]*np.vdot(self.state, vector))
        error += phase_error*float(np.linalg.norm(vector))+gamma(8*len(vector)+16)*float(np.linalg.norm(vector))
        error += gamma(2)*abs(self.task.scalar*.8)*float(np.linalg.norm(vector))
        error += .8*self.task.matrix_assembly_bound*max(1, float(np.linalg.norm(vector)))
        return dict(signal=[value.real, value.imag], u_signal=error+self.target_error,
                    bias=[abs(value.real-self.target.real), abs(value.imag-self.target.imag)],
                    deterministic_actions=sum(f.generator != 'R' for f in factors),
                    ideal_actions=len(factors), tail_count=sum(f.generator == 'R' for f in one))

    def ideal_cell(self, point, q):
        key = point.identity, q
        if key not in self.ideal:
            if self._cache_only:
                raise RuntimeError('MISSING_POSTSEARCH_IDEAL_CELL')
            if len(self.ideal) >= 400:
                raise RuntimeError("STOP_IDEAL_CELL_CAP")
            self.ideal[key] = self._signal(point, q)
            self.on_cell(dict(kind="ideal", coefficient=point.identity, q=q, **self.ideal[key]))
        return self.ideal[key]

    def finite_cell(self, point, q, budget, K):
        key = point.identity, q, budget, K
        if key not in self.finite:
            if self._cache_only:
                raise RuntimeError('MISSING_POSTSEARCH_FINITE_CELL')
            if len(self.finite) >= 4000:
                raise RuntimeError("STOP_FINITE_CELL_CAP")
            one, _ = stationary_template(point, self.deterministic_count, q)
            tails = [f.time for f in one if f.generator == 'R']
            allocation = allocate(point, tails, budget)
            if allocation is None:
                return None
            with mp.workdps(DPS):
                times = [float(t.value(point.basis)*mp.mpf(4)/(5*q)) for t in tails]
            try:
                stats = tail_statistics(times, allocation, self.task.lambda_r, q, K)
                signal = self._signal(point, q, allocation, K)
            except (OverflowError, FloatingPointError, ValueError) as exc:
                row = dict(valid=False, reason=str(exc), coefficient=point.identity, q=q, R_bud=budget, K=K)
            else:
                row = dict(valid=True, coefficient=point.identity, q=q, R_bud=budget, K=K,
                           allocation=list(allocation), tail_times_per_step=times,
                           Gamma=q/.8*math.fsum(abs(t) for t in times), **stats, **signal)
            self.finite[key] = row
            self.on_cell(dict(kind="finite", **row))
        return self.finite[key]

    def cells(self, point, epsilon, *, finite=True):
        self.coefficients[point.identity] = point
        for q in Q_VALUES:
            ideal = self.ideal_cell(point, q)
            one, _ = stationary_template(point, self.deterministic_count, q)
            tails = [f.time for f in one if f.generator == 'R']
            for budget in R_VALUES:
                allocation = allocate(point, tails, budget)
                if allocation is None:
                    continue
                with mp.workdps(DPS):
                    times = [float(t.value(point.basis)*mp.mpf(4)/(5*q)) for t in tails]
                try:
                    stats2 = tail_statistics(times, allocation, self.task.lambda_r, q, 2)
                except OverflowError:
                    stats2 = dict(tail_bound=math.inf)
                eligible4 = (stats2['tail_bound'] > epsilon/math.sqrt(2)/4 and
                             max(ideal['bias'])+ideal['u_signal'] <= epsilon/math.sqrt(2)/2)
                for K in ((2, 4) if eligible4 else (2,)):
                    try:
                        stats = tail_statistics(times, allocation, self.task.lambda_r, q, K)
                    except OverflowError:
                        continue
                    actual = self.finite_cell(point, q, budget, K) if finite else None
                    yield q, budget, K, ideal, stats, actual

    def objective(self, point, arm):
        record = self.objective_record(point, arm)
        return record['value'] if record['feasible'] else math.inf

    def objective_record(self, point, arm):
        """The original design objective, with its minimizing cell preserved."""
        if arm not in ('O', 'L', 'F'):
            raise ValueError('Unknown design objective')
        rows, rejected = [], []
        if arm == 'O':
            for q in Q_VALUES:
                ideal = self.ideal_cell(point, q)
                score = parameter_score(ideal['bias'], ideal['u_signal'], 0., ideal['ideal_actions'], 0., PRIMARY_EPSILON)
                if score['feasible']:
                    rows.append(dict(value=score['value'], lower=score['lower'], q=q, R_bud=None, K=None,
                                     bias=ideal['bias'], u_signal=ideal['u_signal'], log_b=0.,
                                     deterministic_actions=ideal['ideal_actions'], random_actions=0.,
                                     work_definition='ideal formula length including exact tail factors', shots=score['shots']))
                else:
                    rejected.append(score['reason'])
        else:
            for q, budget, K, ideal, stats, actual in self.cells(point, PRIMARY_EPSILON, finite=arm == 'F'):
                if arm == 'L':
                    bias = [b+stats['tail_bound'] for b in ideal['bias']]
                    log_b = stats['log_b_leading']
                    u_signal = ideal['u_signal']
                    deterministic = ideal['deterministic_actions']
                    random = q*budget
                elif actual and actual['valid']:
                    bias, u_signal, log_b = actual['bias'], actual['u_signal'], actual['log_b']
                    deterministic, random = actual['deterministic_actions'], actual['random_actions']
                else:
                    rejected.append(actual['reason'] if actual else 'missing_finite_cell')
                    continue
                score = parameter_score(bias, u_signal, log_b, deterministic, random, PRIMARY_EPSILON)
                if score['feasible']:
                    rows.append(dict(value=score['value'], lower=score['lower'], q=q, R_bud=budget, K=K,
                                     bias=bias, u_signal=u_signal, log_b=log_b, shots=score['shots'],
                                     ideal_bias=ideal['bias'], tail_bound=stats['tail_bound'],
                                     deterministic_actions=deterministic, random_actions=random,
                                     work_definition='leading work' if arm == 'L' else 'finite expected event work'))
                else:
                    rejected.append(score['reason'])
        winner = min(rows, key=lambda r: (r['value'], r['q'], r['R_bud'] or 0, r['K'] or 0)) if rows else None
        return dict(arm=arm, feasible=winner is not None, value=winner['value'] if winner else None,
                    best_cell=winner, feasible_cell_count=len(rows), rejected_cell_count=len(rejected),
                    rejected_reasons=sorted(set(rejected)))

    def rescore(self, points, epsilon):
        rows = []
        for point in points:
            for q, budget, K, ideal, stats, actual in self.cells(point, epsilon):
                if not actual or not actual['valid']:
                    continue
                score = parameter_score(actual['bias'], actual['u_signal'], actual['log_b'],
                                        actual['deterministic_actions'], actual['random_actions'], epsilon)
                if score['feasible']:
                    rows.append(dict(**actual, **score, epsilon=epsilon, label=point.label,
                                     coefficient_max_abs=max(abs(w) for w in point.values())))
        return rows


def search(domain, initial, evaluator, arm):
    """Exactly 16 shared starts + 16 refinements per arm, same policy."""
    points, scores = list(initial), {}
    for point in points:
        scores[point.identity] = evaluator.objective(point, arm)
    for _ in range(16):
        intervals = []
        for component in range(3):
            interior = [p for p in points if p.component == component]
            endpoints = [domain.point(component, 0., 'virtual_endpoint'),
                         domain.point(component, domain.lengths[component], 'virtual_endpoint')]
            ordered = sorted(interior+endpoints, key=lambda p: p.arc)
            for left, right in zip(ordered, ordered[1:]):
                if left.arc == right.arc:
                    continue
                endpoint_score = min(scores.get(left.identity, math.inf), scores.get(right.identity, math.inf))
                values = left.values()
                s, d = values[0]+values[1], values[0]-values[1]
                intervals.append((endpoint_score, component, s, d, right.arc-left.arc, left, right))
        if not intervals:
            raise ValueError("Frozen initial points cannot support refinement")
        if all(math.isinf(x[0]) for x in intervals):
            interval = min(intervals, key=lambda x: (-x[4], x[1], x[2], x[3]))
        else:
            interval = min(intervals, key=lambda x: x[:4])
        left, right = interval[-2:]
        point = domain.point(left.component, (left.arc+right.arc)/2, f"{arm}_refinement_{len(points)-16}")
        if point.identity in scores:
            raise ValueError("Refinement generated duplicate coefficient; STOP without replacement")
        points.append(point)
        scores[point.identity] = evaluator.objective(point, arm)
    return points, [dict(point=p.record(), objective=None if math.isinf(scores[p.identity]) else scores[p.identity]) for p in points]


def best(rows):
    return min(rows, key=lambda r: (r['value'], r['q'], r['R_bud'], r['K'], r['coefficient'])) if rows else None


def classify(f_rows, ref_rows, l_rows):
    """One primary route only. All cases return mandatory STOP."""
    f, reference, leading = best(f_rows), best(ref_rows), best(l_rows)
    common = dict(mandatory_stop=True, automatic_next_stage=None, BF2_authorized=False,
                  science_retry_authorized=False, primary_route="finite action ratio <= 0.95")
    if not f or not reference or not leading:
        return dict(**common, outcome="INCONCLUSIVE", reason="missing_feasible_F_or_reference_or_L")
    ratio = f['value']/reference['value']
    ratio_upper = f['value']/min(r['lower'] for r in ref_rows)
    ratio_lower = min(r['lower'] for r in f_rows)/reference['value']
    uncertainty = ratio_upper-ratio_lower
    # A primary win automatically requires a finite-design point absent from
    # the complete L search set, not a different objective score at one point.
    unique = f['coefficient'] not in {r['coefficient'] for r in l_rows}
    corners = []
    for wd in (.98, 1.02):
        for wr in (.98, 1.02):
            def weighted(row):
                return sum(row['shots'])*(wd*row['deterministic_actions']+wr*row['random_actions'])
            corners.append(min(map(weighted, f_rows))/min(map(weighted, ref_rows)))
    boundary = any(r['q'] == max(Q_VALUES) or r['R_bud'] == max(R_VALUES)
                   or r.get('coefficient_max_abs', 0.) >= 2-1e-12 for r in (f, reference, leading))
    if boundary:
        outcome, reason = 'INCONCLUSIVE', 'best_at_coefficient_cap_or_upper_q_or_R_boundary'
    elif uncertainty > .01:
        outcome, reason = 'INCONCLUSIVE', 'UNRESOLVED_NUMERICAL_MARGIN'
    elif ratio <= .95 and ratio_upper <= .95 and unique and max(corners) < 1:
        outcome, reason = 'BF-C', 'primary_materiality_pass_in_frozen_family'
    elif ratio_lower <= .95 < ratio_upper:
        outcome, reason = 'INCONCLUSIVE', 'primary_threshold_interval_crossing'
    elif ratio >= .99:
        outcome, reason = 'BF-A', 'less_than_1_percent_finite_decision_gain'
    else:
        outcome, reason = 'BF-B', 'submaterial_finite_decision_gain'
    return dict(**common, outcome=outcome, reason=reason, primary_ratio=ratio,
                ratio_interval=[ratio_lower, ratio_upper], ratio_uncertainty=uncertainty,
                F_vs_L_ratio=f['value']/leading['value'], F_unique_from_L=unique,
                corner_ratios=corners, best_F=f, best_reference=reference, best_L=leading,
                secondary_q_change=f['q'] != reference['q'])


def secondary_frontier(rows):
    """Frozen five-axis descriptive frontier; it cannot change BF-C."""
    rows = list({(r['coefficient'], r['q'], r['R_bud'], r['K']): r for r in rows}.values())
    if not rows:
        return []
    values = np.array([[r['bias'][0]+r['u_signal'], r['bias'][1]+r['u_signal'],
                        r['log_b'], r['deterministic_actions'], r['random_actions']] for r in rows])
    frontier = []
    for i, row in enumerate(rows):
        dominated = np.all(values <= values[i], axis=1)&np.any(values < values[i], axis=1)
        if not np.any(dominated):
            frontier.append({k: row[k] for k in ('coefficient', 'q', 'R_bud', 'K')})
    return frontier
