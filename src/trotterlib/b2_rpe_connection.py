"""Explicit generation-prefix B2 -> conditional RPE accounting and no-prep cost.

No input loading, state solve, schedule selection, or execution on import.
Every numerical value and scientific identity comes from the caller. Molecular
use requires its separately approved execution scope; synthetic tests qualify
software connections only, never energy accuracy or RPE success.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from types import SimpleNamespace

from . import o1_m1_resource_bridge as accounting
from .df_partial_s2_repeated import (
    DFPartialS2RepeatedRequest, make_df_partial_s2_repeated_request,
)
from .df_rte_circuit import DFRTEEventSequenceCircuitRequest
from .pauli_signal_adapter import digest
from .rpe_hadamard_interrogation import RPE_HADAMARD_INTERROGATION_SCOPE
from .rpe_resource_accounting import RPE_COST_METRICS, RPERoundSpecification
from .rte import finite_rte_attenuation, finite_rte_distribution, require_integer_count
from .rte_compiled_cost import compiler_settings_hash
from trottertracks.resource_applicability.ax2a_native_df import (
    partial_native_from_step_requests, build_native_hadamard_wrapper,
)


@dataclass(frozen=True)
class RPEConnectionContext:
    """Common physical/comparison contract, supplied independently of Agent A.

    State identity is the caller's original-H state identity, never a DF fallback.
    None leaves a scientific condition unevaluated and prevents shot accounting.
    Identity/provenance declarations do not prove the ground-state assumption.
    """
    original_hamiltonian: str
    df_hamiltonian: str
    original_state: str
    sector: str
    spin_orbital_order: str
    source: str
    compiler: object
    backend_hash: str
    cost_metric: str
    model_error_bound: float | None
    energy_precision: float | None
    success_probability: float | None
    alias_branch_conditions: str | None
    state_role: str = 'original_H_ground_state'
    cost_scope: str = RPE_HADAMARD_INTERROGATION_SCOPE
    state_preparation_included: bool = False

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (
            self.original_hamiltonian, self.df_hamiltonian, self.original_state,
            self.sector, self.spin_orbital_order, self.source, self.backend_hash)):
            raise ValueError('Explicit Hamiltonian/state/sector/order/SOURCE/backend identities required')
        if self.state_role != 'original_H_ground_state':
            raise ValueError('Original-H state required; DF state substitution is forbidden')
        if (self.cost_metric not in RPE_COST_METRICS
                or self.cost_scope != RPE_HADAMARD_INTERROGATION_SCOPE
                or self.state_preparation_included is not False):
            raise ValueError('Common full no-prep cost scope and supported metric required')
        compiler_settings_hash(self.compiler)
        for name in ('model_error_bound', 'energy_precision', 'success_probability'):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v < 0):
                raise ValueError('Invalid '+name)
        if self.energy_precision is not None and self.energy_precision <= 0:
            raise ValueError('Positive energy precision required')
        if self.success_probability is not None and not 0 < self.success_probability < 1:
            raise ValueError('Success probability must lie in (0,1)')

    @property
    def context_hash(self):
        fields = asdict(self)
        fields['compiler'] = compiler_settings_hash(self.compiler)
        return digest(fields)

    def require_evaluated_conditions(self):
        if any(getattr(self, k) is None for k in ('model_error_bound', 'energy_precision',
                                                  'success_probability', 'alias_branch_conditions')):
            raise ValueError('Scientific conditions remain unevaluated')
        if not self.alias_branch_conditions:
            raise ValueError('Explicit alias/branch conditions required')


def prepare_b2(hamiltonian, *, L_D):
    """Exactly the existing B2 preparation; no rank search, DF fit or state solve."""
    from .pr2_matched_accuracy_m1_execution import _prepare
    return _prepare(hamiltonian, require_integer_count(L_D, name='L_D'))


@dataclass(frozen=True)
class B2RPERound:
    context: RPEConnectionContext
    preparation: object
    specification: RPERoundSpecification
    L_D: int
    r: int
    K: int
    delta: float
    q: int
    config: object
    distribution: object

    def __post_init__(self):
        p, c, d = self.preparation, self.config, self.distribution
        for name, minimum in (('L_D', 0), ('r', 1), ('K', 0), ('q', 1)):
            require_integer_count(getattr(self, name), name=name, minimum=minimum)
        if (self.L_D != p.ld or p.deterministic_fragment_indices != tuple(range(self.L_D))
                or p.randomized_block_indices != tuple(range(self.L_D,
                    self.L_D+len(p.randomized_block_indices)))
                or p.identity_policy != 'extract_identity_phase' or p.coefficient_atol != 0
                or p.product_formula != '2nd'):
            raise ValueError('B2 requires generation-prefix, exact identity extraction and partial S2')
        if self.context.df_hamiltonian != p.hamiltonian_hash:
            raise ValueError('DF Hamiltonian identity mismatch')
        if (self.q != self.specification.q_m or self.delta != self.specification.delta_time
                or self.delta <= 0 or c.evolution_time != self.delta
                or c.rte_steps != self.r or c.finite_taylor_order != self.K):
            raise ValueError('Explicit round/q/delta/r/K mismatch')
        accounting._validate_law(self, c, d)
        canonical = finite_rte_distribution(c.dimensionless_step_time, self.K)
        if d != canonical or c.distribution_normalization != canonical.exact_finite_distribution:
            raise ValueError('Noncanonical finite-RTE law')

    @property
    def tail(self):
        return self.preparation.rte_preparation.symbolic_tail

    @property
    def context_hash(self):
        return self.context.context_hash

    @property
    def preparation_hash(self):
        # r/K/delta/q belong to per-round law metadata, not the multi-round arm.
        return digest(dict(context=self.context_hash, B2=self.preparation.preparation_hash,
                           control='ordinary', event_order='step_major_circuit_append_order'))

    @property
    def binding(self):
        return digest(dict(context=self.context_hash, arm=self.preparation_hash,
                           round=self.specification.round_index, q=self.q, delta=self.delta,
                           law=accounting._law_hash(self.config)))

    def sampled_request(self, *, seed):
        return make_df_partial_s2_repeated_request(self.preparation, step_time=self.delta,
            repetition_count=self.q, rte_config=self.config, rte_distribution=self.distribution,
            seed=seed, controlled=True, ancilla_qubit=self.preparation.num_system_qubits)

    def explicit_request(self, events):
        """Replay validated events without RNG; chronological q outer / r inner order."""
        events = tuple(events)
        accounting._canonical_events(self, self.config, self.distribution, events, self.q)
        prep = self.preparation.rte_preparation
        n = self.preparation.num_system_qubits
        specs = tuple(prep.component_specs)
        occurrences = tuple(DFRTEEventSequenceCircuitRequest(
            events=events[i:i+self.r], component_specs=specs, controlled=True, ancilla_qubit=n,
            tail_id=self.tail.tail_id, tail_hash=self.tail.tail_hash, occurrence_rte_steps=self.r)
            for i in range(0, len(events), self.r))
        return DFPartialS2RepeatedRequest(self.preparation, self.delta, self.q, self.config,
            self.distribution, rte_occurrences=occurrences, controlled=True, ancilla_qubit=n,
            step_seeds=(0,)*self.q, sampling_policy='explicit_canonical_events_no_RNG')

    def native(self, request, *, max_instructions):
        if (request.preparation != self.preparation or request.step_time != self.delta
                or request.repetition_count != self.q or request.rte_config != self.config
                or request.rte_distribution != self.distribution):
            raise ValueError('Request differs from this round/law/preparation')
        events = tuple(e for occurrence in request.rte_occurrences for e in occurrence.events)
        accounting._canonical_events(self, self.config, self.distribution, events, self.q)
        return partial_native_from_step_requests(request.iter_step_requests(),
            control_policy='ordinary', max_instructions=max_instructions)

    def price(self, provider, *, sample_count, seed, max_instructions):
        _check_provider(self.context, provider)
        return provider.monte_carlo_with_builder(self, self.config, self.distribution,
            q=self.q, sample_count=sample_count, seed=seed,
            build_trajectory=lambda events: self.native(self.explicit_request(events),
                max_instructions=max_instructions),
            wrapper_builder=lambda evolution, axis: build_native_hadamard_wrapper(evolution,
                axis=axis, include_measurement=True, max_instructions=max_instructions))


def _check_provider(context, provider):
    if provider.compiler != context.compiler or provider.backend_hash != context.backend_hash:
        raise ValueError('Common compiler/backend mismatch')


@dataclass(frozen=True)
class SignalValue:
    value: complex
    state_identity: str
    hamiltonian_identity: str
    state_role: str = 'original_H_ground_state'


@dataclass(frozen=True)
class B2SignalRecord:
    """Five distinct same-round signals, all on the supplied original-H state."""
    round_binding: str
    original: SignalValue
    df: SignalValue
    ideal_tail_pf: SignalValue
    finite_corrected: SignalValue
    raw_measured: SignalValue
    normalization_product: float
    implementation_phase_bound: float

    def bounds(self, round, *, radius_allowance, complex_allowance, phase_allowance,
               normalization_tolerance):
        for v in (radius_allowance, complex_allowance, phase_allowance, normalization_tolerance,
                  self.implementation_phase_bound):
            if not math.isfinite(v) or v < 0:
                raise ValueError('Explicit finite nonnegative numerical allowances required')
        if self.round_binding != round.binding:
            raise ValueError('Signal round/law/SOURCE/compiler binding mismatch')
        ctx = round.context
        for name in ('original', 'df', 'ideal_tail_pf', 'finite_corrected', 'raw_measured'):
            signal = getattr(self, name)
            if signal.state_identity != ctx.original_state or signal.state_role != ctx.state_role:
                raise ValueError('Original-H state mismatch; DF state substitution is forbidden')
            expected_h = ctx.original_hamiltonian if name == 'original' else ctx.df_hamiltonian
            if signal.hamiltonian_identity != expected_h:
                raise ValueError('Signal Hamiltonian identity mismatch')
            if not math.isfinite(signal.value.real) or not math.isfinite(signal.value.imag):
                raise ValueError('Nonfinite signal')
        A = (1. if round.config is None else
             finite_rte_attenuation(round.config, tail_evolutions=round.q))
        if (not math.isfinite(self.normalization_product) or self.normalization_product < 1
                or not math.isclose(self.normalization_product*A, 1., rel_tol=0,
                                    abs_tol=normalization_tolerance)
                or abs(self.raw_measured.value-A*self.finite_corrected.value) > normalization_tolerance
                or abs(self.raw_measured.value) > 1+normalization_tolerance):
            raise ValueError('Raw/corrected normalization mismatch or double application')
        phase = lambda a, b: abs(math.atan2((a*b.conjugate()).imag, (a*b.conjugate()).real))
        if any(abs(getattr(self, k).value) == 0 for k in ('original', 'df', 'ideal_tail_pf')):
            raise ValueError('Undefined signal phase')
        radius = abs(self.ideal_tail_pf.value)-radius_allowance
        return dict(reference_radius_lower_bound=radius,
            corrected_signal_error_bound=abs(self.finite_corrected.value-self.ideal_tail_pf.value)+complex_allowance,
            pf_phase_bound=phase(self.ideal_tail_pf.value, self.df.value)+phase_allowance,
            model_phase_bound=phase(self.df.value, self.original.value)+phase_allowance,
            implementation_phase_bound=self.implementation_phase_bound+phase_allowance,
            radius_provenance='same-round original-H-state ideal-tail PF radius minus allowance; empirical not certified')


def account_signal_round(round, signals, *, allocation, beta_rpe, cost,
                     radius_allowance, complex_allowance, phase_allowance, normalization_tolerance):
    round.context.require_evaluated_conditions()
    _check_cost(round.context, cost)
    bounds = signals.bounds(round, radius_allowance=radius_allowance,
        complex_allowance=complex_allowance, phase_allowance=phase_allowance,
        normalization_tolerance=normalization_tolerance)
    result = accounting.account_radius_round(context_hash=round.context_hash,
        arm_hash=round.preparation_hash, specification=round.specification, allocation=allocation,
        beta_rpe=beta_rpe, config=round.config, cost=cost, cost_metric=round.context.cost_metric,
        backend_hash=round.context.backend_hash, **bounds)
    result.update(raw_signal_radius=abs(signals.raw_measured.value),
        corrected_signal_radius=abs(signals.finite_corrected.value),
        ideal_tail_pf_radius=abs(signals.ideal_tail_pf.value),
        phase_budgets=dict(PF=allocation.beta_pf_budget, RTE=allocation.beta_rte_budget,
                          statistical=allocation.beta_stat_budget), round_binding=round.binding)
    return result


def _check_cost(context, cost):
    if cost is not None and (cost.cosine_expected_cost.compiler != context.compiler
                            or cost.sine_expected_cost.compiler != context.compiler):
        raise ValueError('Common X/Y compiler mismatch')


def aggregate_b2_rounds(rounds, *, required_max_round, total_alpha_budget, context):
    context.require_evaluated_conditions()
    rounds = tuple(rounds)
    if any(r['context_hash'] != context.context_hash for r in rounds):
        raise ValueError('Mixed physical/SOURCE/compiler comparison context')
    if total_alpha_budget > 1-context.success_probability+1e-15:
        raise ValueError('Failure budget exceeds declared success condition')
    return accounting.aggregate_registered_rounds(rounds, required_max_round=required_max_round,
                                                 total_alpha_budget=total_alpha_budget)


def build_strong_baseline(hamiltonian, *, context, specification, substeps, formula,
                          basis_policy, relative_fusion):
    """Full S2 or standard three-stage Yoshida4 on the same complete saved factors.

    Exact interleaved spin lift is opt-in; StrongBuilder rejects ineligible spin
    structure without projecting it. Relative Gaussian fusion retains its analytic
    vacuum-phase correction. No phase alignment or state preparation is applied.
    """
    from .df_partial_randomized_pf import df_hamiltonian_hash
    from .product_formula import _get_w_list
    from trottertracks import c1_h4_strong_baseline as strong
    if df_hamiltonian_hash(hamiltonian) != context.df_hamiltonian:
        raise ValueError('Baseline DF input mismatch')
    substeps = require_integer_count(substeps, name='substeps', minimum=1)
    if formula not in ('Full_S2', 'Yoshida4'):
        raise ValueError('Explicit Full_S2 or standard Yoshida4 required')
    model, reports = strong.prepare_xdf_basis(hamiltonian.one_body, hamiltonian.lambdas,
        hamiltonian.g_matrices, hamiltonian.constant, source_identity=context.source,
        input_identity=context.df_hamiltonian, basis_policy=basis_policy)
    weights = (1.,) if formula == 'Full_S2' else _get_w_list('4th')
    pieces = weights if formula == 'Full_S2' else (weights[1], weights[0], weights[1])
    dt = Fraction(specification.delta_time)/substeps
    schedule = tuple(item for w in pieces for item in strong.native.ordered_schedule(
        len(model.blocks), dt*Fraction(w)))*(specification.q_m*substeps)
    circuit = strong.StrongBuilder(relative_fusion=relative_fusion, basis_policy=basis_policy).sequence(
        model, schedule, optimized=True, scalar_time=Fraction(specification.delta_time)*specification.q_m)
    arm_hash = digest(dict(context=context.context_hash, formula=formula, substeps=substeps,
                           basis_policy=basis_policy, relative_fusion=relative_fusion))
    return SimpleNamespace(circuit=circuit, arm_hash=arm_hash, context=context,
        specification=specification, spin_reports=reports, preparation_hash=arm_hash,
        context_hash=context.context_hash, q=specification.q_m, config=None,
        binding=digest(dict(context=context.context_hash, arm=arm_hash,
            round=specification.round_index, q=specification.q_m, delta=specification.delta_time,
            law='deterministic_no_RTE')))


def price_strong_baseline(baseline, provider):
    _check_provider(baseline.context, provider)
    return provider.deterministic(baseline.circuit, context_hash=baseline.context.context_hash,
        arm_hash=baseline.arm_hash, q=baseline.specification.q_m,
        delta_time=baseline.specification.delta_time)
