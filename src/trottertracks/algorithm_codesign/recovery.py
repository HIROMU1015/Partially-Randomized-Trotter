"""Post-hoc replay of frozen BF-1 rules against saved cells only."""
from __future__ import annotations

from contextlib import contextmanager
import builtins
import copy
import json
import math
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

from . import adapter, pilot
from .cross_objectives import attribute_F_winner, cross_score
from .domain import Domain, Point, fixed_references
from .freeze import canonical


class RecoveryFailure(RuntimeError):
    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.detail = dict(reason=reason, **detail)


def parse_cells(text):
    """Reject ambiguous/unsupported records; never choose a duplicate winner."""
    ideal, finite = {}, {}
    for number, line in enumerate(text.splitlines(), 1):
        row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        kind = row.get('kind')
        if kind not in ('ideal', 'finite') or row.get('q') not in pilot.Q_VALUES:
            raise RecoveryFailure('RESCUE_INVALID_ORIGINAL_RECORD', line=number)
        coefficient = row.get('coefficient')
        if not isinstance(coefficient, str) or len(coefficient) != 64:
            raise RecoveryFailure('RESCUE_INVALID_COEFFICIENT_IDENTITY', line=number)
        if kind == 'ideal':
            key, destination = (coefficient, row['q']), ideal
            value = {k: v for k, v in row.items() if k not in ('kind', 'coefficient', 'q')}
        else:
            if row.get('R_bud') not in pilot.R_VALUES or row.get('K') not in (2, 4):
                raise RecoveryFailure('RESCUE_INVALID_ORIGINAL_RECORD', line=number)
            key, destination = (coefficient, row['q'], row['R_bud'], row['K']), finite
            value = {k: v for k, v in row.items() if k != 'kind'}
        if key in destination:
            raise RecoveryFailure('RESCUE_DUPLICATE_ORIGINAL_CELL', line=number, key=list(key))
        destination[key] = value
    if len(ideal) > 400 or len(finite) > 4000:
        raise RecoveryFailure('RESCUE_ORIGINAL_CELL_CAP_EXCEEDED')
    return ideal, finite


class ReadOnlyEvaluator(pilot.Evaluator):
    """Reuse source objective/rescore methods, bypass all physical initialization."""
    def __init__(self, ideal, finite, lambda_r, deterministic_count, limits):
        if not math.isfinite(lambda_r) or lambda_r <= 0 or deterministic_count != 4:
            raise RecoveryFailure('RESCUE_INVALID_SAVED_TASK_METADATA')
        self.ideal = MappingProxyType({k: MappingProxyType(copy.deepcopy(v)) for k, v in ideal.items()})
        self.finite = MappingProxyType({k: MappingProxyType(copy.deepcopy(v)) for k, v in finite.items()})
        self.task = SimpleNamespace(lambda_r=lambda_r)
        self.deterministic_count = deterministic_count
        self.limits = limits
        self.coefficients = {}
        self._cache_only = False
        self.phase = 'UNSTARTED'
        self.lookups = dict(ideal=0, finite=0)

    def _lookup(self, kind, key):
        self.limits.check()
        cache = self.ideal if kind == 'ideal' else self.finite
        if key not in cache:
            raise RecoveryFailure('RESCUE_INCOMPLETE_MISSING_ORIGINAL_CELL',
                                  phase=self.phase, kind=kind, key=list(key))
        self.lookups[kind] += 1
        return cache[key]

    def ideal_cell(self, point, q):
        return self._lookup('ideal', (point.identity, q))

    def finite_cell(self, point, q, budget, K):
        return self._lookup('finite', (point.identity, q, budget, K))

    def _signal(self, *args, **kwargs):
        raise RecoveryFailure('RESCUE_FORBIDDEN_SIGNAL_ACQUISITION')

    def snapshot(self):
        # Include nested values, not only key/count equality.
        return canonical(dict(ideal=[[list(k), dict(v)] for k, v in sorted(self.ideal.items())],
                              finite=[[list(k), dict(v)] for k, v in sorted(self.finite.items())]))


@contextmanager
def forbid_physical_evaluation():
    """Deny physical providers even if a future replay edit accidentally calls one."""
    def denied(*args, **kwargs):
        raise RecoveryFailure('RESCUE_FORBIDDEN_PHYSICAL_EVALUATION')
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if ('science_input' in name or name.split('.')[0] in
                ('trotterlib', 'cupy', 'qiskit', 'openfermion', 'openfermionpyscf')):
            raise RecoveryFailure('RESCUE_FORBIDDEN_IMPORT', module=name)
        return original_import(name, *args, **kwargs)
    with patch.object(pilot.Evaluator, '__init__', denied), \
            patch.object(pilot.Evaluator, '_signal', denied), \
            patch.object(pilot.Spectrum, '__init__', denied), \
            patch.object(pilot.Spectrum, 'action', denied), \
            patch.object(pilot.np, 'load', denied), \
            patch.object(adapter, 'dense_operator', denied), \
            patch.object(adapter, 'polynomial', denied), \
            patch.object(builtins, '__import__', guarded_import):
        yield


def replay(original, frozen, ideal, finite, limits):
    """Source-bound mathematical replay; a missing original cell stops immediately."""
    report = dict(schema='bf1_read_only_recovery_result_v1',
                  status='BF1_READ_ONLY_RECOVERY_INCOMPLETE',
                  evidence_scope='post-hoc mechanical replay of source-bound one-shot data',
                  original_status=original['status'], original_outcome=original['outcome'],
                  primary_recovered=False, cross_objectives_recovered=False,
                  science_execution_authorized=False, science_retry_authorized=False,
                  mandatory_stop=True, automatic_next_stage=None, BF2_authorized=False,
                  added_ideal_cells=0, added_finite_cells=0, added_science_signals=0,
                  added_science_candidates=0, bridge=dict(status='NOT_REACHED'), searches={})
    evaluator = ReadOnlyEvaluator(ideal, finite, original['input_audit']['lambda_r'],
                                 original['input_audit']['native_generator_count'], limits)
    before = evaluator.snapshot()
    report['original_cell_counts'] = dict(ideal=len(ideal), finite=len(finite))
    try:
        with forbid_physical_evaluation():
            evaluator.phase = 'FORMULA_DOMAIN_CHECK'
            initial = [Point.from_record(p) for p in frozen['initial_points']]
            fixed = [Point.from_record(p) for p in frozen['fixed_references']]
            domain = Domain()
            _, regenerated = domain.initial_points()
            if (len(initial) != 16 or len(fixed) != 4 or
                    tuple(domain.lengths) != tuple(frozen['arclength']['lengths']) or
                    [p.identity for p in regenerated] != [p.identity for p in initial] or
                    [p.identity for p in fixed_references(domain)] != [p.identity for p in fixed]):
                raise RecoveryFailure('RESCUE_FROZEN_DOMAIN_IDENTITY_MISMATCH')
            points = {}
            for arm in ('O', 'L', 'F'):
                evaluator.phase = 'SEARCH_REPLAY_'+arm
                points[arm], report['searches'][arm] = pilot.search(domain, initial, evaluator, arm)
                if len(points[arm]) != 32 or len({p.identity for p in points[arm]}) != 32:
                    raise RecoveryFailure('RESCUE_SEARCH_BUDGET_OR_IDENTITY_MISMATCH', arm=arm)
            # Preserve original runner insertion order; do not infer origins
            # from the order of saved physical records.
            reference_points = {p.identity: p for p in points['O']+points['L']+fixed}
            F_points = {p.identity: p for p in points['F']}
            evaluator.phase = 'PRIMARY_REFERENCE_RESCORE'
            reference = evaluator.rescore(list(reference_points.values()), .01)
            evaluator.phase = 'PRIMARY_F_RESCORE'
            finite_rows = evaluator.rescore(list(F_points.values()), .01)
            L_ids = {p.identity for p in points['L']}
            leading = [r for r in reference if r['coefficient'] in L_ids]
            evaluator.phase = 'PRIMARY_CLASSIFY_AND_FRONTIER'
            report.update(primary_recovered=True, primary_decision=pilot.classify(finite_rows, reference, leading),
                          primary_F=finite_rows, primary_reference=reference,
                          secondary_frontier=pilot.secondary_frontier(finite_rows+reference))
            all_points = {p.identity: p for p in list(reference_points.values())+list(F_points.values())}
            if not set(all_points) <= {k[0] for k in ideal}:
                raise RecoveryFailure('RESCUE_COEFFICIENT_NOT_IN_ORIGINAL_CACHE')
            report['coefficients'] = [p.record() for p in all_points.values()]
            report['search_identity_validation'] = dict(points_per_arm={a:len(p) for a,p in points.items()},
                union_coefficients=len(all_points), all_identities_in_original_cache=True,
                origins_inferred_from_record_order=False, alternate_search_performed=False)
            evaluator.phase = 'POSTHOC_CROSS_OBJECTIVES'
            cross = cross_score(evaluator, points, fixed)
            report.update(cross_objectives_recovered=True, cross_objectives=cross,
                          objective_attribution=attribute_F_winner(cross, report['primary_decision']),
                          attribution_scope='POSTHOC_RECOVERED_OBJECTIVE_ATTRIBUTION')
            # Bridge is optional and cannot erase a completely recovered primary.
            evaluator.phase = 'OPTIONAL_BRIDGE_CACHE_ONLY'
            try:
                bridge = evaluator.rescore(list(all_points.values()), .05)
            except RecoveryFailure as exc:
                if exc.detail['reason'] != 'RESCUE_INCOMPLETE_MISSING_ORIGINAL_CELL':
                    raise
                report['bridge'] = dict(status='MISSING_FROM_ORIGINAL_RUN', missing=exc.detail)
            else:
                report['bridge'] = dict(status='POSTHOC_RECOVERED_CACHE_ONLY', epsilon=.05, rows=bridge)
            report['status'] = 'BF1_READ_ONLY_RECOVERY_COMPLETE'
    except RecoveryFailure as exc:
        report['failure'] = exc.detail
    except Exception as exc:
        report['failure'] = dict(reason='RESCUE_UNEXPECTED_REPLAY_FAILURE', phase=evaluator.phase,
                                 exception_type=type(exc).__name__, message=str(exc))
    finally:
        unchanged = before == evaluator.snapshot()
        report['cache_values_unchanged'] = unchanged
        report['cache_lookups'] = evaluator.lookups
        report['reconstructed_coefficient_count'] = len(evaluator.coefficients)
        report['final_cell_counts'] = dict(ideal=len(evaluator.ideal), finite=len(evaluator.finite))
        if not unchanged or report['final_cell_counts'] != report['original_cell_counts']:
            report.update(status='BF1_READ_ONLY_RECOVERY_INCOMPLETE',
                          failure=dict(reason='RESCUE_ORIGINAL_CACHE_CHANGED'))
    return report
