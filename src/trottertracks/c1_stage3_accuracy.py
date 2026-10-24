"""CA energy/signal/shot/cost connection. Import is stdlib-only.

No H4 values are embedded or loaded. Molecular calls require separate user
permission. Artificial tests use scalar evidence and mock cost providers.
"""
from dataclasses import dataclass
import importlib
import math


def _legacy(name):
    # Reuse the existing package loader, preserving actual relative imports
    # while avoiding trotterlib's eager plotting/circuit __init__ imports.
    from .c1_stage1 import legacy_df_module
    package = legacy_df_module().__package__
    return importlib.import_module(package+'.'+name)


def _finite(value, name, *, positive=False):
    if value is None or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(name+' must be finite and '+('positive' if positive else 'nonnegative'))
    return float(value)


@dataclass(frozen=True)
class AccuracyTarget:
    energy_ha: float
    df_reserve_fraction: float
    base_time: float
    beta_total: float
    beta_stat: float
    alpha_axis: float
    alpha_total: float = .05

    def __post_init__(self):
        for name in ['energy_ha', 'base_time', 'beta_total', 'beta_stat', 'alpha_axis']:
            _finite(getattr(self, name), name, positive=True)
        if not 0 < self.df_reserve_fraction < 1:
            raise ValueError('DF reserve must be an explicit proper fraction')
        if not 0 < self.beta_stat < self.beta_total < math.pi/3:
            raise ValueError('Strict RPE branch margin and positive deterministic phase room required')
        if not self.alpha_axis < 1 or not 0 < self.alpha_total < 1:
            raise ValueError('Axis failure probability must be below one')

    @property
    def df_cap(self):
        return self.energy_ha*self.df_reserve_fraction

    def horizon(self):
        return _legacy('rpe_target_round_horizon_validation').required_rpe_round_horizon(
            target_energy_precision=self.energy_ha-self.df_cap,
            beta_rpe=self.beta_total, delta_time=self.base_time)


@dataclass(frozen=True)
class RoundEvidence:
    m: int
    attenuation: float
    pf_operator_error: float
    rte_corrected_operator_error: float
    synthesis_signal_error: float | None
    finite_law_signal_error: float | None
    target_id: str
    state_model: str
    alias_branch_verified: bool
    iid_per_shot: bool
    evidence_kind: str = 'numerically_qualified_pilot'
    law_model: str = 'actual_implemented_law'


def assess_round(target, evidence):
    """Convert errors to the SAME raw complex-signal norm before combining.

    Ideal observed signal is A exp(-i E_x t) in an own-DF ground state.
    δ=A(εPF+εRTE)+εsynth+εlaw, ρ>=A-δ,
    βdet<=asin(δ/A). Statistical coordinate tolerance uses this observed ρ.
    No DF energy error (Ha) or Stage2 max-entry tolerance enters δ.
    """
    if type(evidence.m) is not int or evidence.m < 0 or not evidence.target_id:
        raise ValueError('Round index and target identity required')
    if evidence.state_model != 'own_DF_ground_state_oracle':
        raise ValueError('HF survival signal is not an energy eigenstate guarantee')
    if not evidence.alias_branch_verified or not evidence.iid_per_shot:
        raise ValueError('Alias branch and independent per-shot outcomes are required')
    A = _finite(evidence.attenuation, 'attenuation')
    if A > 1:
        raise ValueError('Raw measured attenuation cannot exceed one')
    if A == 0:
        return dict(status='INFEASIBLE_RADIUS', m=evidence.m, shots=None,
                    reason='Observed attenuation is zero at the declared precision')
    if evidence.evidence_kind not in ('numerically_qualified_pilot', 'rigorous_bound'):
        raise ValueError('Evidence qualification must be declared')
    if evidence.law_model not in ('actual_implemented_law', 'ideal_finite_RTE_law', 'deterministic'):
        raise ValueError('Unknown law model')
    if evidence.law_model in ('ideal_finite_RTE_law', 'deterministic') and evidence.finite_law_signal_error != 0:
        raise ValueError('A reference-defined law must explicitly set its own law error to zero')
    if evidence.synthesis_signal_error is None or evidence.finite_law_signal_error is None:
        return dict(status='UNRESOLVED_SIGNAL_ERROR', m=evidence.m, shots=None,
                    reason='Unknown synthesis/actual-law error cannot be assigned zero')
    eps_pf = _finite(evidence.pf_operator_error, 'PF operator error')
    eps_rte = _finite(evidence.rte_corrected_operator_error, 'corrected RTE operator error')
    eps_syn = _finite(evidence.synthesis_signal_error, 'synthesis signal error')
    eps_law = _finite(evidence.finite_law_signal_error, 'finite-law signal error')
    delta = math.fsum((A*eps_pf, A*eps_rte, eps_syn, eps_law))
    rho = A-delta
    if rho <= 0:
        return dict(status='INFEASIBLE_RADIUS', m=evidence.m, shots=None)
    beta_det = math.asin(delta/A)
    if beta_det+target.beta_stat > target.beta_total:
        return dict(status='INFEASIBLE_PHASE', m=evidence.m, shots=None,
                    beta_deterministic=beta_det)
    coordinate = rho*math.sin(target.beta_stat)/math.sqrt(2.)
    shots = _legacy('rpe_resource_accounting')._hoeffding_shots(coordinate, target.alpha_axis)
    return dict(status='CONDITIONAL_ROUND_PASS', m=evidence.m, shots=shots,
                target_id=evidence.target_id, raw_signal_error=delta,
                radius_lower=rho, beta_deterministic=beta_det,
                beta_total=beta_det+target.beta_stat, epsilon_coordinate=coordinate,
                alpha_axis=target.alpha_axis, evidence_kind=evidence.evidence_kind,
                law_model=evidence.law_model,
                quantum_shots_drawn=0)


def assess_energy(target, df_error_bound_ha, round_plans, *, state_model,
                  alias_branch_verified):
    """Weyl energy shift + final RPE energy interval, after phase conversion.

    DF is counted exactly once. Own-DF state is an explicit conditional input;
    state preparation cost/real-device error is not silently assigned zero.
    """
    df = _finite(df_error_bound_ha, 'DF energy bound')
    if state_model != 'own_DF_ground_state_oracle' or not alias_branch_verified:
        raise ValueError('Ground-state/alias assumptions must be justified')
    plans = tuple(round_plans)
    horizon = target.horizon()
    expected = list(range(horizon['maximum_round_index_M']+1))
    if [row['m'] for row in plans] != expected:
        return dict(status='INCOMPLETE_ROUNDS', same_accuracy=False)
    if any(row['status'] != 'CONDITIONAL_ROUND_PASS' for row in plans):
        return dict(status='UNQUALIFIED_ROUNDS', same_accuracy=False)
    if len({row['target_id'] for row in plans}) != 1:
        raise ValueError('All rounds must target the same arm and candidate')
    if df > target.df_cap:
        return dict(status='DF_RESERVE_EXCEEDED', same_accuracy=False)
    alpha = math.fsum(2*row['alpha_axis'] for row in plans)
    if alpha > target.alpha_total:
        return dict(status='FAILURE_PROBABILITY_EXCEEDED', same_accuracy=False)
    epsilon = df+target.beta_total/(target.base_time*(1<<plans[-1]['m']))
    return dict(status='CONDITIONAL_CA_PASS' if epsilon <= target.energy_ha else 'ENERGY_TARGET_FAIL',
                same_accuracy=epsilon <= target.energy_ha, energy_error_bound_ha=epsilon,
                statistical_failure_upper=alpha, state_preparation='ORACLE_COST_EXCLUDED',
                guarantee='conditional_on_declared_deterministic_error_evidence',
                numerical_pilot=any(p['evidence_kind'] != 'rigorous_bound' for p in plans))


METRICS = ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')


def bounded_cost_intervals(samples, *, support_bounds=None, alpha_per_metric=None,
                           support_provenance=None):
    """IID Hoeffding cost intervals, conditional on declared valid supports.

    Observed sample minima/maxima are not support bounds. Unknown bounds give
    point estimates only. The caller allocates alpha simultaneously over all
    candidate/metric selections; reused calibration is never resampled here.
    """
    rows=tuple(samples)
    if not rows:raise ValueError('At least one classical cost sample required')
    mean={k:math.fsum(_finite(row[k],'cost') for row in rows)/len(rows) for k in METRICS}
    result=dict(mean=mean,classical_cost_samples=len(rows),lower=None,upper=None,
                simultaneous_interval_qualified=False)
    if support_bounds is None or not support_provenance:return result
    if alpha_per_metric is None or not 0<alpha_per_metric<1:
        raise ValueError('An explicit simultaneous failure allocation is required')
    lower,upper={},{}
    for k in METRICS:
        lo,hi=(_finite(v,'cost support') for v in support_bounds[k])
        if any(not lo<=row[k]<=hi for row in rows):raise ValueError('Sample outside declared cost support')
        radius=(hi-lo)*math.sqrt(math.log(2/alpha_per_metric)/(2*len(rows)))
        lower[k]=max(lo,mean[k]-radius);upper[k]=min(hi,mean[k]+radius)
    return dict(**{**result,'lower':lower,'upper':upper,'simultaneous_interval_qualified':True},
                support_provenance=support_provenance,alpha_per_metric=alpha_per_metric)


def aggregate_totals(accuracy, round_plans, costs, *, cost_scope,
                     expected_scope='serial_native_kernel_composition_without_state_preparation'):
    """Sum N_X C_X+N_Y C_Y, including execution/measurement counts.

    Cost records cover the FULL one-interrogation scope at each round. Long
    serial-kernel composition is an explicitly implemented policy, never an
    extrapolation of whole-circuit-optimized Stage2 depths. Classical cost
    sample counts are not additional quantum-shot multipliers.
    """
    if not accuracy.get('same_accuracy'):
        return dict(status='NO_TOTAL_UNQUALIFIED_ACCURACY', totals=None)
    if cost_scope != expected_scope:
        raise ValueError('Subcircuit, optimized-full and serial-scheduled costs cannot be mixed')
    plans = tuple(round_plans)
    lookup = {(row['m'], row['axis']): row for row in costs}
    if len(lookup) != len(costs) or set(lookup) != {(p['m'], a) for p in plans for a in ['X', 'Y']}:
        return dict(status='INCOMPLETE_COST_COVERAGE', totals=None)
    totals = {k: [] for k in METRICS}
    lower = {k: [] for k in METRICS}; upper = {k: [] for k in METRICS}
    intervals_qualified = True
    for p in plans:
        for axis in ['X', 'Y']:
            row = lookup[p['m'], axis]
            if row['target_id'] != p['target_id'] or row['scope'] != cost_scope:
                raise ValueError('Cost identity/scope mismatch')
            if not row.get('long_round_coverage', False):
                return dict(status='UNQUALIFIED_LONG_ROUND_COST', totals=None)
            intervals_qualified &= row.get('simultaneous_interval_qualified', False)
            for key in METRICS:
                mean = _finite(row['mean'][key], 'expected cost')
                totals[key].append(p['shots']*mean)
                if row.get('lower') is None or row.get('upper') is None:
                    intervals_qualified = False
                else:
                    lo, hi = row['lower'][key], row['upper'][key]
                    if not 0 <= lo <= mean <= hi or not math.isfinite(hi):
                        raise ValueError('Cost interval must enclose its point estimate')
                    lower[key].append(p['shots']*lo);upper[key].append(p['shots']*hi)
    return dict(status='CONDITIONAL_TOTAL_AVAILABLE', totals={k: math.fsum(v) for k, v in totals.items()},
                intervals={k: [math.fsum(lower[k]), math.fsum(upper[k])] for k in METRICS}
                    if intervals_qualified else None,
                shots=sum(2*p['shots'] for p in plans),
                executions=sum(2*p['shots'] for p in plans),
                cost_scope=cost_scope, state_preparation_cost=None,
                classical_cost_samples_are_shot_multipliers=False,
                unconditional_total_resource_claim=False)


def compare_totals(cdf, xdf):
    if cdf.get('totals') is None or xdf.get('totals') is None:
        return dict(verdict='INDETERMINATE_MISSING_QUALIFICATION')
    if cdf['cost_scope'] != xdf['cost_scope']:
        raise ValueError('Comparison scopes differ')
    if cdf['intervals'] is None or xdf['intervals'] is None:
        return dict(verdict='POINT_ESTIMATES_ONLY', superiority=False)
    a, b = cdf['intervals']['rz_count'], xdf['intervals']['rz_count']
    verdict = 'CDF_LOWER_CONDITIONAL_INTERROGATION_COST' if a[1] < b[0] else (
        'XDF_LOWER_CONDITIONAL_INTERROGATION_COST' if b[1] < a[0] else 'INDETERMINATE_OVERLAP')
    return dict(verdict=verdict, unconditional_total_resource_claim=False)


def evaluate_evidence_packet(packet):
    """Numerical assessment entry for a future permitted evidence evaluation.

    Evidence generation uses the existing CAR/native/RTE and reference bridge;
    no new simulator, fitter, or state preparation is hidden in this function.
    """
    target=AccuracyTarget(**packet['target'])
    plans=[assess_round(target,RoundEvidence(**row)) for row in packet['rounds']]
    accuracy=assess_energy(target,packet['df_error_bound_ha'],plans,
        state_model=packet['state_model'],alias_branch_verified=packet['alias_branch_verified'])
    resources=aggregate_totals(accuracy,plans,packet.get('costs',[]),cost_scope=packet['cost_scope'])
    return dict(accuracy=accuracy,rounds=plans,resources=resources,
                scientific_target='common_physical_sector_ground_energy',mandatory_stop=True)
