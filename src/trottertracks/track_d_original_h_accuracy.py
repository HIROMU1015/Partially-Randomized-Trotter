"""Original-H RPE adapter. No saved inputs, scientific CLI or work on import.

Explicit callers supply original-H transported operator errors (DF included
once), targets and qualified premises. This module neither proves those
premises nor generates them. Real-data evaluation needs separate permission.
"""
from dataclasses import asdict, dataclass
import json
import math
from numbers import Real

from . import c1_stage3_accuracy as accuracy

STATE = 'exact_original_H_ground_state_oracle'
ERROR_KIND = 'original_H_corrected_operator_norm_DF_included_once'
IDEAL = 'ideal_finite_RTE_law'
ACTUAL = 'actual_implemented_law'
DECODER = 'consecutive_dyadic_nearest_branch_principal_energy_v1'


def _number(value, name, *, positive=False, signed=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (not signed and value < 0)
            or (positive and value <= 0)):
        raise ValueError(name + ' must be a finite real in its declared range')
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' identity/provenance required')


@dataclass(frozen=True)
class SignalBinding:
    original_h: str
    cdf: str
    sector: str
    law: str
    source: str
    implementation: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            _text(value, name)

    @property
    def target_id(self):
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class OriginalHTarget:
    energy_ha: float
    base_time: float
    beta_total: float
    beta_stat: float
    alpha_axis: float
    alpha_total: float
    maximum_round_index: int
    energy_window_ha: tuple[float, float]

    def __post_init__(self):
        for name in ('energy_ha', 'base_time', 'beta_total', 'beta_stat',
                     'alpha_axis', 'alpha_total'):
            _number(getattr(self, name), name, positive=True)
        if not 0 < self.beta_stat < self.beta_total < math.pi / 3:
            raise ValueError('Strict dyadic branch/phase budget required')
        if self.alpha_axis >= 1 or self.alpha_total >= 1:
            raise ValueError('Failure probabilities must be below one')
        if type(self.maximum_round_index) is not int or self.maximum_round_index < 0:
            raise ValueError('Explicit nonnegative maximum round index required')
        if not isinstance(self.energy_window_ha, tuple) or len(self.energy_window_ha) != 2:
            raise ValueError('Explicit immutable energy prior interval required')
        lo, hi = (_number(x, 'energy window', signed=True) for x in self.energy_window_ha)
        if lo >= hi or max(abs(lo), abs(hi))*self.base_time + self.beta_total >= math.pi:
            raise ValueError('Energy window/phase margin crosses principal alias branch')

    def horizon(self):
        # No own-DF reserve: transported DF error is already in the signal.
        required = accuracy._legacy('rpe_target_round_horizon_validation').required_rpe_round_horizon(
            target_energy_precision=self.energy_ha, beta_rpe=self.beta_total,
            delta_time=self.base_time)
        if self.maximum_round_index < required['maximum_round_index_M']:
            raise ValueError('Declared horizon cannot meet energy target')
        q = 1 << self.maximum_round_index
        achieved = self.beta_total/(q*self.base_time)
        previous = None if q == 1 else self.beta_total/((q >> 1)*self.base_time)
        return dict(required, minimum_required_round_index=required['maximum_round_index_M'],
                    maximum_round_index_M=self.maximum_round_index, q_max=q,
                    round_count_M_plus_one=self.maximum_round_index+1,
                    achieved_energy_resolution_ha=achieved,
                    previous_q_energy_resolution_ha=previous,
                    previous_power_fails=previous is None or previous > self.energy_ha,
                    minimal_power_of_two_horizon=self.maximum_round_index == required['maximum_round_index_M'])


@dataclass(frozen=True)
class OriginalHRoundEvidence:
    m: int
    attenuation: float
    operator_error: float  # corrected operator norm to exact original-H evolution
    implementation_signal_error: float | None  # raw complex-signal units
    finite_law_signal_error: float | None       # raw complex-signal units
    binding: SignalBinding
    law_model: str
    alias_branch_verified: bool
    iid_per_shot: bool
    operator_provenance: str
    alias_provenance: str | None
    iid_provenance: str | None
    implementation_provenance: str | None
    law_provenance: str | None
    state_model: str = STATE
    operator_error_kind: str = ERROR_KIND
    evidence_kind: str = 'numerically_qualified_pilot'

    @classmethod
    def from_b1_row(cls, row, *, target, **context):
        """Select saved B1 operator fields, never its historical N/phase target.

        Caller binds the packet to H/CDF/sector/law/SOURCE; this reads no file.
        B1 numerical norms/engineering guard cannot become rigorous evidence.
        """
        m = row['m']
        if type(m) is not int or m < 0 or row['q'] != 1 << m:
            raise ValueError('B1 round/q mismatch')
        if row['time'] != target.base_time * (1 << m):
            raise ValueError('B1 time differs from target')
        return cls(m=m, attenuation=row['attenuation'],
                   operator_error=row['original_H_corrected_operator_error'],
                   evidence_kind='numerically_qualified_pilot', **context)


def assess_round(target, binding, evidence):
    """Conditional sufficient N, with equal X/Y coordinate/failure allowances."""
    if not isinstance(target, OriginalHTarget) or not isinstance(binding, SignalBinding):
        raise ValueError('Explicit original-H target and binding required')
    if not isinstance(evidence, OriginalHRoundEvidence) or evidence.binding != binding:
        raise ValueError('Original-H/CDF/sector/law/SOURCE/implementation binding mismatch')
    if evidence.state_model != STATE or evidence.operator_error_kind != ERROR_KIND:
        raise ValueError('Original-H state and DF-in-signal operator units required')
    if type(evidence.m) is not int or not 0 <= evidence.m <= target.maximum_round_index:
        raise ValueError('Round outside declared dyadic horizon')
    if evidence.law_model not in (IDEAL, ACTUAL):
        raise ValueError('Unsupported law model')
    if evidence.evidence_kind not in ('numerically_qualified_pilot', 'rigorous_bound'):
        raise ValueError('Explicit deterministic-error evidence qualification required')
    _text(evidence.operator_provenance, 'operator')
    A = _number(evidence.attenuation, 'attenuation')
    op = _number(evidence.operator_error, 'original-H operator error')
    if A > 1:
        raise ValueError('Raw measured attenuation cannot exceed one')
    for name in ('implementation_signal_error', 'finite_law_signal_error'):
        value = getattr(evidence, name)
        if value is not None:
            _number(value, name)
    if type(evidence.alias_branch_verified) is not bool or type(evidence.iid_per_shot) is not bool:
        raise ValueError('Alias and IID conditions must be explicit booleans')
    if evidence.law_model == IDEAL and any(
            x is not None and x != 0 for x in
            (evidence.implementation_signal_error, evidence.finite_law_signal_error)):
        raise ValueError('Ideal IID/exact-native model requires explicit zero model errors')
    missing = [name for name in ('alias_provenance', 'iid_provenance',
                                'implementation_provenance', 'law_provenance')
               if not isinstance(getattr(evidence, name), str) or not getattr(evidence, name).strip()]
    unresolved = (missing or not evidence.alias_branch_verified or not evidence.iid_per_shot
                  or evidence.implementation_signal_error is None
                  or evidence.finite_law_signal_error is None)
    metadata = dict(target_id=binding.target_id, binding=asdict(binding), target=asdict(target),
                    state_model=STATE, law_model=evidence.law_model,
                    evidence_kind=evidence.evidence_kind, operator_error_kind=ERROR_KIND,
                    input_errors=dict(attenuation=A, corrected_operator_error=op,
                                      implementation_raw=evidence.implementation_signal_error,
                                      law_raw=evidence.finite_law_signal_error),
                    provenance={k: getattr(evidence, k) for k in
                                ('operator_provenance', 'alias_provenance', 'iid_provenance',
                                 'implementation_provenance', 'law_provenance')},
                    qualification='CONDITIONAL_IDEAL_MODEL' if evidence.law_model == IDEAL
                                  else 'CONDITIONAL_IMPLEMENTED_ERROR_PREMISES',
                    unconditional_success_claim=False, success_probability_evaluated=False)
    if unresolved:
        result = dict(status='UNRESOLVED_SIGNAL_OR_SAMPLING_PREMISES', m=evidence.m, shots=None,
                      missing_provenance=missing)
    else:
        raw = math.fsum((A*op, evidence.implementation_signal_error, evidence.finite_law_signal_error))
        # Common arithmetic uses attributes also present on own-DF evidence.
        from types import SimpleNamespace
        shared = SimpleNamespace(m=evidence.m, target_id=binding.target_id,
                                 evidence_kind=evidence.evidence_kind, law_model=evidence.law_model)
        try:
            result = accuracy._assess_raw_signal(target, shared, attenuation=A, raw_error=raw)
        except (OverflowError, ZeroDivisionError):
            result = dict(status='UNRESOLVED_NUMERICAL_SHOT_RANGE', m=evidence.m, shots=None)
    return dict(result, **metadata)


def assess_energy(target, binding, round_plans):
    """Full dyadic horizon/failure union; DF already included, no Weyl addition."""
    if not isinstance(target, OriginalHTarget) or not isinstance(binding, SignalBinding):
        raise ValueError('Explicit original-H target and binding required')
    plans = tuple(round_plans)
    for p in plans:
        if (p['binding'] != asdict(binding) or p['target'] != asdict(target)
                or p['state_model'] != STATE or p['operator_error_kind'] != ERROR_KIND
                or p['target_id'] != binding.target_id):
            raise ValueError('Round target/state/SOURCE binding differs')
        if type(p['m']) is not int or p['m'] < 0:
            raise ValueError('Integer round required')
        if p['status'] == 'CONDITIONAL_ROUND_PASS':
            if (p['alpha_axis'] != target.alpha_axis or type(p['shots']) is not int
                    or p['shots'] <= 0 or not 0 < p['beta_total'] <= target.beta_total
                    or p['evidence_kind'] not in ('numerically_qualified_pilot', 'rigorous_bound')):
                raise ValueError('Round result differs from declared phase/failure model')
    if len({p['law_model'] for p in plans}) > 1:
        raise ValueError('Ideal and actual round lanes cannot be mixed')
    result = accuracy._assess_energy_interval(target, plans, energy_shift_ha=0.)
    return dict(result, state_model=STATE, df_counting='already_in_signal_once',
                decoder=DECODER, energy_phase_convention='phi=-h*E; E=-phi/h',
                unconditional_success_claim=False, success_probability_evaluated=False)


def decode_energy(target, binding, round_plans, coordinates):
    """Decode supplied means, without sampling or evaluating success probability.

    Principal alias window is required by OriginalHTarget. Shifted branches,
    weights/postselection and nonconsecutive dyadic rounds are unsupported.
    """
    plans = tuple(round_plans)
    assessment = assess_energy(target, binding, plans)
    if not assessment['same_accuracy']:
        return dict(status='UNQUALIFIED_DECODER_INPUT', energy_estimate_ha=None)
    xy = tuple(coordinates)
    if len(xy) != len(plans):
        raise ValueError('One X/Y mean pair per round required')
    phases = []
    for x, y in xy:
        x, y = _number(x, 'X mean', signed=True), _number(y, 'Y mean', signed=True)
        if abs(x) > 1 or abs(y) > 1:
            raise ValueError('Unweighted binary-outcome means required')
        if x == y == 0:
            return dict(status='UNRESOLVED_ZERO_COORDINATES', energy_estimate_ha=None)
        phases.append(math.atan2(y, x))
    decoded = accuracy._legacy('rpe_four_round_phase_validation').reconstruct_rpe_phase(
        phases, [1 << p['m'] for p in plans])
    if any(math.isclose(p['distance_from_previous'], p['next_smallest_branch_distance'],
                        abs_tol=1e-12, rel_tol=0.) for p in decoded['rounds'][1:]):
        return dict(status='UNRESOLVED_BRANCH_TIE', energy_estimate_ha=None)
    energy = -decoded['final_phase_estimate']/target.base_time
    width = assessment['energy_error_bound_ha']
    lo, hi = target.energy_window_ha
    interval = (max(lo, energy-width), min(hi, energy+width))
    if interval[0] > interval[1]:
        return dict(status='ENERGY_BRANCH_OUTSIDE_PRIOR', energy_estimate_ha=None)
    return dict(status='CONDITIONAL_DECODED_ENERGY', energy_estimate_ha=energy,
                energy_interval_ha=interval, decoded=decoded,
                interval_condition='all declared deterministic and coordinate error events hold',
                success_probability_evaluated=False, unconditional_success_claim=False)
