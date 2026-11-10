"""Track D method selection and existing PR/RPE accounting connection.

SavedCDF8Study reads frozen outputs; it never predicts prices, solves shots or
reaggregates H4. aggregate_supplied delegates explicit caller-supplied evidence
to the existing scalar accounting function. No builder/provider is imported.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math

from . import track_d_b6b_bc as bc
from . import c1_stage3_accuracy as accounting

TOTAL = '8828b298975c88a2c0d1fd57ff6ed905233840f5'
B6B = '7ffe8b281b76cef3a6fd06f43413904d6766d0ec'
DIRECTORY = 'artifacts/track_d_bc_total/2026-11-08/run01/'
SCOPE = bc.SCOPE
METHODS = bc.METHODS
SCENARIOS = ('A', 'F', 'S')


@dataclass(frozen=True)
class ScientificInputs:
    binding: dict
    target: dict
    rounds: tuple  # (m, q); no price or control-method information
    law_model: str
    decoder: str


@dataclass(frozen=True)
class ControlProfile:
    compiler: dict
    cut_policy: str
    head_tail_C1: bool = False
    topology: str = 'topology_free'
    phase_policy: str = 'declared_global_phase_preserved'


@dataclass(frozen=True)
class MethodPrices:
    scientific_inputs: ScientificInputs
    profile: ControlProfile
    method: str
    rows: tuple
    horizon_qualification: str = 'UNQUALIFIED'


def _same(a, b):
    # Preserve the fixed metadata's numeric types; bool is not compiler level 1.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def _method(method):
    if method not in METHODS:
        raise ValueError('Unknown control method; use symmetric_directional or reference')


def _validate_prices(table):
    _method(table.method)
    if table.horizon_qualification not in ('qualified_horizon', 'explicit_conditional_horizon'):
        raise ValueError('Explicit full-horizon price qualification required')
    expected = {(m, axis) for m, q in table.scientific_inputs.rounds for axis in ('X', 'Y')}
    keys = [(r['m'], r['axis']) for r in table.rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Complete unique round/axis price coverage required')
    by_m = dict(table.scientific_inputs.rounds)
    for row in table.rows:
        if (row['method'] != table.method or row['q'] != by_m[row['m']]
                or not _same(row['binding'], table.scientific_inputs.binding)
                or not _same(row['compiler'], table.profile.compiler)
                or row['scope'] != SCOPE):
            raise ValueError('Price science/method/compiler/scope mismatch')
        if (not row.get('source') or row.get('estimate_unit') != 'whole_path_one_interrogation'
                or row.get('attenuation_price_multiplier') is not False):
            raise ValueError('Explicit whole-wrapper provenance and normalization required')
        if row.get('long_round_coverage', True) is not True:
            raise ValueError('Row contradicts declared horizon qualification')
        basis = row.get('expectation_basis')
        if basis not in ('qualified_expectation', 'explicit_conditional_model'):
            raise ValueError('Finite sample alone is not an expected-price qualification')
        if basis == 'explicit_conditional_model' and not row.get('model_assumptions'):
            raise ValueError('Conditional-price assumptions required')
        if set(row['mean']) != set(bc.METRICS):
            raise ValueError('Exactly six additive resources; qubits are separate')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v < 0 for v in row['mean'].values()):
            raise ValueError('Finite nonnegative prices required')
        if type(row.get('peak_qubits')) is not int or row['peak_qubits'] <= 0:
            raise ValueError('Explicit positive logical width required')


def aggregate_supplied(inputs, accuracy, plans, prices):
    """Aggregate already qualified caller evidence; never solve N or build gates.

    Qualification remains the upstream caller's responsibility. This helper
    checks identity/coverage, not the truth of a supplied scientific theorem.
    Real numerical use requires its own execution scope; tests use toy scalars.
    """
    if (not _same(asdict(inputs), asdict(prices.scientific_inputs))
            or not _same(accuracy.get('binding'), inputs.binding)
            or not _same(accuracy.get('target'), inputs.target)):
        return dict(status='UNQUALIFIED_SCIENTIFIC_INPUTS', totals=None, peak_qubits=None)
    plans = tuple(plans)
    if tuple((p['m'], 2**p['m']) for p in plans) != inputs.rounds:
        raise ValueError('Shot plans must match the explicit dyadic horizon')
    if any(type(p.get('shots')) is not int or p['shots'] <= 0 for p in plans):
        raise ValueError('Supplied equal-X/Y positive integer shots required')
    _validate_prices(prices)
    target_ids = {p['target_id'] for p in plans}
    if len(target_ids) != 1:
        raise ValueError('One shared scientific target required')
    target_id = next(iter(target_ids))
    if any(r.get('target_id', target_id) != target_id for r in prices.rows):
        raise ValueError('Price target differs from shot target')
    costs = [dict(r, target_id=target_id, long_round_coverage=True) for r in prices.rows]
    # No second implementation of N_X*C_X + N_Y*C_Y or interval accounting.
    result = accounting.aggregate_totals(accuracy, plans, costs,
        cost_scope=SCOPE, expected_scope=SCOPE)
    return dict(result, method=prices.method,
        peak_qubits=max(r['peak_qubits'] for r in prices.rows) if result.get('totals') is not None else None,
        actual_success_guaranteed=False, qualification='CALLER_EVIDENCE_CONDITIONAL',
        uncertainty_note='Missing intervals/covariance/bias are not zero',
        attenuation_price_multiplier=False, depth_is_hardware_runtime=False)


class SavedCDF8Study:
    """Read-only selector for the fixed B4/B6b/8828b298 evidence domain."""
    def __init__(self, root):
        self.sources = {}
        def read(commit, path):
            return json.loads(bc.blob(root, commit, path, self.sources))
        self._handoff = read(bc.B4, bc.BDIR+'/Stage_C_handoff.json')
        self._preflight = read(TOTAL, DIRECTORY+'preflight.json')
        self._result = read(TOTAL, DIRECTORY+'result.json')
        self._source = read(TOTAL, DIRECTORY+'SOURCE.json')
        self._stops = {
            'total': read(TOTAL, DIRECTORY+'STOP.json'),
            'B4': read(bc.B4, bc.BDIR+'/run01/STOP.json'),
            'B6b': read(B6B, 'artifacts/track_d_b6b_bc/2026-11-08/run01/STOP.json')}
        self._conditions = read(B6B, 'artifacts/track_d_b6b_bc/2026-11-08/run01/common_N_conditions.json')
        self._prices = {s: read(TOTAL, DIRECTORY+s+'/prices52.json') for s in SCENARIOS}
        self._totals = {s: read(TOTAL, DIRECTORY+s+'/result.json') for s in SCENARIOS}
        meta = self._preflight['b4_metadata']
        if (self._handoff['target'] != meta['target'] or self._handoff['rows'] != meta['rows']
                or self._handoff['total_shots'] != meta['stored_total_shots']
                or self._result['stored_shots'] != meta['stored_total_shots']
                or meta['stored_total_shots'] != 14738
                or meta['binding'] != bc.binding()
                or self._result['status'] != 'CONDITIONAL_IDEAL_RPE_TOTAL_ESTIMATE'
                or self._result['actual_success_guaranteed'] is not False
                or self._conditions['ideal']['common_N'] is not True):
            raise ValueError('Fixed saved B4/B6b/total metadata mismatch')
        if (self._handoff['binding']['original_h'] != bc.PACKET
                or self._handoff['binding']['cdf'] != bc.CDF
                or self._handoff['energy']['decoder'] != meta['decoder']
                or self._source['execution_source'] != self._result['source_commit']
                or any(s.get('mandatory_stop') is not True for s in self._stops.values())):
            raise ValueError('Fixed identity/SOURCE/decoder/STOP mismatch')
        self._inputs = ScientificInputs(deepcopy(meta['binding']), deepcopy(meta['target']),
            tuple((r['m'], r['q']) for r in meta['rows']),
            'B4_ideal_finite_RTE_IID_exact_native', meta['decoder'])
        if self._inputs.rounds != tuple((m, 2**m) for m in range(13)):
            raise ValueError('Fixed saved horizon mismatch')
        self._profile = ControlProfile(deepcopy(bc.P0), 'q1:event16;q>=2:complete_step_q/2')
        for scenario in SCENARIOS:
            if len(self._prices[scenario]) != 52:
                raise ValueError('Fixed saved 52 price records required')
            if set(self._totals[scenario]['totals']) != set(bc.METRICS):
                raise ValueError('Fixed saved resource definitions differ')
            for method in METHODS:
                rows = tuple(r for r in self._prices[scenario] if r['method'] == method)
                _validate_prices(MethodPrices(self._inputs, self._profile, method, rows,
                                               'explicit_conditional_horizon'))
                for r in rows:
                    q = r['q']
                    cut = (dict(kind='complete_event_within_step', after=16) if q == 1
                           else dict(kind='complete_step', after=q//2))
                    label = 'DIRECT_FINITE_SAMPLE_PRICE_INPUT' if q <= 32 else 'EXPLORATORY_LONG_Q_PROXY'
                    if (r['cut'] != cut or r['scenario'] != scenario
                            or r['price_classification'] != label
                            or r['price_uncertainty_qualified'] is not False):
                        raise ValueError('Fixed saved cut/scenario/qualification mismatch')

    @property
    def inputs(self):
        return deepcopy(self._inputs)

    @property
    def profile(self):
        return deepcopy(self._profile)

    def common_shots(self, inputs, *, lane='ideal'):
        if not _same(asdict(inputs), asdict(self._inputs)):
            return dict(status='ADDITIONAL_SCIENTIFIC_QUALIFICATION_REQUIRED', common_N=False,
                rows=None, stored_total_shots=None, actual_success_guaranteed=False)
        assessment = bc.common_N(self._conditions['premises'], lane=lane)
        return dict(assessment, rows=deepcopy(self._handoff['rows']) if assessment['common_N'] else None,
            stored_total_shots=self._handoff['total_shots'] if assessment['common_N'] else None,
            shot_origin='SAVED_B4_OUTPUT_NOT_RECOMPUTED',
            source=dict(commit=bc.B4, path=bc.BDIR+'/Stage_C_handoff.json'),
            equivalence_source=dict(commit=B6B, path='docs/research/track_d_b6b_bc_integration_20261108/MEASUREMENT_EQUIVALENCE.md'))

    def select(self, inputs, method, *, scenario='A', profile=None, lane='ideal'):
        """Select saved totals and prices. No model prediction or aggregation."""
        _method(method)
        if scenario not in SCENARIOS:
            raise ValueError('Unknown saved price scenario; no fallback')
        shots = self.common_shots(inputs, lane=lane)
        profile = self.profile if profile is None else profile
        if not _same(asdict(profile), asdict(self._profile)):
            shots = dict(status='UNQUALIFIED_CONTROL_PROFILE', common_N=False,
                         rows=None, stored_total_shots=None, actual_success_guaranteed=False)
        if not shots['common_N']:
            return dict(status='UNQUALIFIED_OR_UNSUPPORTED_CONTEXT', shot_evidence=shots,
                prices=None, total_resources=None, peak_qubits=None,
                actual_success_guaranteed=False, fallback_used=False)
        rows = tuple(deepcopy(r) for r in self._prices[scenario] if r['method'] == method)
        column = 'symmetric' if method == 'symmetric_directional' else 'reference'
        # These are saved scalar selections, not sums over real H4 records.
        totals = {g: self._totals[scenario]['totals'][g][column] for g in bc.METRICS}
        return dict(status='SAVED_CONDITIONAL_EVALUATION_REFERENCED', method=method, scenario=scenario,
            shot_evidence=shots, prices=MethodPrices(deepcopy(inputs), deepcopy(profile), method, rows,
                                                    'explicit_conditional_horizon'),
            total_resources=deepcopy(totals), peak_qubits=self._result['peak_qubits'][method],
            uncertainty=deepcopy(self._totals[scenario]['uncertainty']),
            source=dict(commit=TOTAL, path=DIRECTORY+scenario+'/result.json',
                execution_source=self._source['execution_source']),
            price_source=dict(commit=TOTAL, path=DIRECTORY+scenario+'/prices52.json'),
            total_origin='SAVED_RESULT_NOT_REAGGREGATED', new_shot_solver_calls=0,
            actual_success_guaranteed=False, confirmatory_resource_advantage=False,
            width_is_additive=False, attenuation_price_multiplier=False,
            depth_is_hardware_runtime=False, method_optimizations_already_included=True)
