"""Explicit matched-dictionary resource glue; no files, defaults or science run.

Molecular preparation/build/pricing/shot arithmetic require a separately fixed
execution scope. Only caller-supplied complete event sets are priced exactly;
representative circuit checks are never treated as an expectation or MC sample.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
from types import SimpleNamespace

from qiskit import QuantumCircuit

from .o1_m1_matched_split import extract_matched_df_tail, pauli_tail_coefficients
from .o1_m1_circuit_signal_validation import one_body_circuit
from .df_trotter.ops import build_one_body_gaussian_block
from .df_rte_tail import prepare_df_rte_event_inputs
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .pauli_signal_adapter import PauliHamiltonian, append_pauli_event, array_hash, digest
from .rte import RTEComponent, _make_event, finite_rte_attenuation, make_rte_config, require_integer_count
from .finite_rte_signal_validation import _hoeffding_shots, _explicit_cutoff_tolerance
from .rpe_resource_accounting import (RPE_COST_METRICS, RPE_NUMERICAL_PHASE_MODEL_LIMIT,
    RPERoundCompiledCost, circuit_cost_metric)
from .rpe_hadamard_interrogation import RPE_HADAMARD_INTERROGATION_SCOPE, wrap_controlled_no_prep
from .rte_compiled_cost import (CompiledMetricAccumulator, canonical_qiskit_circuit_fingerprint,
    canonical_backend_fingerprint_or_none, compiler_settings_hash,
    circuit_cost_from_metric_statistics, transpile_and_measure_cost)


def _law_hash(config):
    return ("deterministic_no_RTE" if config is None else digest(dict(tail=config.tail_hash,
        r=config.rte_steps, K=config.finite_taylor_order, delta=config.evolution_time)))


@dataclass(frozen=True)
class MatchedResourcePreparation:
    split: object
    dictionary: str
    context_hash: str
    preparation_hash: str
    tail: object
    retained: object
    labels: object
    df_preparation: object


def prepare_matched_resources(split, *, dictionary):
    """Keep the same complete T_eff, signed square tail and original state.

    Explicit scientific algebra/Gaussian preparation; never called on import.
    """
    if dictionary not in ("P", "DF"):
        raise ValueError("dictionary must be P or DF")
    inputs = split.inputs
    context = digest(dict(physical_split="T_eff/V_square", identity="faithful_identity_in_tail",
        scalar=inputs.nuclear_repulsion, state_field="state", state=array_hash(inputs.state),
        bare=array_hash(inputs.spin_one), two=array_hash(inputs.spin_two),
        retained=array_hash(inputs.df_one_body), weights=array_hash(inputs.df_lambdas),
        factors=array_hash(inputs.df_g_matrices), sector_projection=False))
    labels, df_preparation = {}, None
    if dictionary == "P":
        coefficients, transfer = pauli_tail_coefficients(split, identity_policy="faithful_identity_in_tail")
        if transfer != 0:
            raise ValueError("Matched resource bridge fixes faithful tail identity")
        model = PauliHamiltonian.from_terms(coefficients, num_qubits=inputs.num_qubits)
        lam = math.fsum(abs(c) for _, c in coefficients)
        if lam <= 0:
            raise ValueError("PR bridge requires a nonzero tail")
        components = tuple(RTEComponent(str(i), abs(c)/lam, abs(c), 1 if c>0 else -1,
                                      is_identity=set(label)=={"I"})
                           for i,(label,c) in enumerate(coefficients))
        labels = {str(i): label for i,(label,c) in enumerate(coefficients)}
        tail = SimpleNamespace(tail_id="M1_matched_P", lambda_r=lam, components=components,
            tail_hash=digest(dict(context=context, terms=coefficients, hamiltonian=model.hamiltonian_hash)))
    else:
        extraction = extract_matched_df_tail(split, identity_policy="faithful_identity_in_tail")
        df_preparation = prepare_df_rte_event_inputs(extraction)
        tail = df_preparation.symbolic_tail
    return MatchedResourcePreparation(split, dictionary, context,
        digest(dict(context=context, dictionary=dictionary, tail=tail.tail_hash)), tail,
        build_one_body_gaussian_block(inputs.df_one_body), labels, df_preparation)


def configure_law(preparation, *, delta_time, r, K):
    r=require_integer_count(r,name="r",minimum=1)
    if not math.isfinite(delta_time) or delta_time<=0:
        raise ValueError("Positive finite delta_time required")
    return make_rte_config(preparation.tail, evolution_time=delta_time, rte_steps=r,
        finite_taylor_order=K,truncation_tolerance=_explicit_cutoff_tolerance(
            preparation.tail.lambda_r*delta_time/r,K))


def _canonical_events(preparation, config, distribution, events, q):
    q = require_integer_count(q, name="q", minimum=1)
    if (config.tail_hash != preparation.tail.tail_hash or config.lambda_r != preparation.tail.lambda_r
            or distribution.exact_finite_distribution != config.distribution_normalization
            or distribution.dimensionless_step_time != config.dimensionless_step_time
            or distribution.finite_taylor_order != config.finite_taylor_order):
        raise ValueError("Event law differs from the prepared dictionary/config")
    if len(events) != q*config.rte_steps:
        raise ValueError("One independently specified event per q*r short step required")
    index = {c.component_id:i for i,c in enumerate(preparation.tail.components)}
    for event in events:
        try:
            selected = [index[event.rotation_component_id], *[index[c] for c in event.product_component_ids]]
            canonical = _make_event(selected, preparation.tail.components, distribution,
                                    distribution.orders.index(event.taylor_order))
        except (KeyError, ValueError) as exc:
            raise ValueError("Unknown component/order in event") from exc
        if event != canonical:
            raise ValueError("Event coefficients, phase, probability or basis differ from law")


def build_controlled_trajectory(preparation, config, distribution, *, q, events):
    """No RNG: compose exactly the supplied fresh-event sequence, scalar once/step."""
    q=require_integer_count(q,name="q",minimum=1)
    events = tuple(events)
    _canonical_events(preparation, config, distribution, events, q)
    n = preparation.split.inputs.num_qubits
    half = one_body_circuit(preparation.retained, config.evolution_time/2, controlled=True)
    circuit = QuantumCircuit(n+1)
    builder = (QiskitDFRTEEventCircuitBuilder(basis_registry=preparation.df_preparation.basis_registry)
               if preparation.dictionary == "DF" else None)
    cursor = 0
    for _ in range(q):
        circuit.compose(half, inplace=True)
        for _ in range(config.rte_steps):
            event = events[cursor]; cursor += 1
            if builder is not None:
                central = builder.build_event(preparation.df_preparation.request_for_event(
                    event, controlled=True, ancilla_qubit=n)).circuit
                circuit.compose(central, inplace=True)
            else:
                central = QuantumCircuit(n)
                append_pauli_event(central, event, preparation.labels)
                circuit.append(central.to_gate().control(1), [n, *range(n)])
        circuit.compose(half, inplace=True)
        circuit.p(-config.evolution_time*preparation.split.inputs.nuclear_repulsion, n)
    return circuit


def build_strong_df_baseline(split, *, delta_time, q, substeps, basis_policy):
    """Existing StrongBuilder with all saved factors, exact-zero precision/fusion.

    The spin route is opt-in and rejects inputs lacking its exact spin structure.
    No legacy input loader, scientific run contract or saved price is reused.
    """
    from trottertracks import c1_h4_strong_baseline as strong
    q = require_integer_count(q, name="q", minimum=1)
    substeps = require_integer_count(substeps, name="substeps", minimum=1)
    if basis_policy not in (strong.LEGACY, strong.SPIN):
        raise ValueError("Explicit supported strong-baseline basis policy required")
    dt = float(delta_time)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite delta_time required")
    inputs = split.inputs
    model, _ = strong.prepare_xdf_basis(inputs.df_one_body, inputs.df_lambdas,
        inputs.df_g_matrices, inputs.nuclear_repulsion, source_identity="M1_supplied_saved_factors",
        input_identity=digest(dict(retained=array_hash(inputs.df_one_body), factors=array_hash(inputs.df_g_matrices),
                                   weights=array_hash(inputs.df_lambdas), scalar=inputs.nuclear_repulsion)),
        basis_policy=basis_policy)
    schedule = strong.native.ordered_schedule(len(model.blocks), Fraction(dt)/substeps)*(q*substeps)
    return strong.StrongBuilder(relative_fusion=True,basis_policy=basis_policy).sequence(
        model, schedule, optimized=True, scalar_time=q*Fraction(dt))


def _compile_pair(circuit, compiler, *, compile_function, backend):
    costs = {}
    for axis in ("cosine", "sine"):
        wrapper = wrap_controlled_no_prep(circuit, ancilla_qubit=circuit.num_qubits-1,
                                          axis=axis, include_measurement=True)
        fp = canonical_qiskit_circuit_fingerprint(wrapper)
        cost = compile_function(wrapper, compiler, backend=backend, actual_circuit_fingerprint=fp)
        if cost.compiler != compiler or any(not math.isfinite(float(getattr(cost,k))) or getattr(cost,k)<0 for k in RPE_COST_METRICS):
            raise ValueError("Compile context or metric differs")
        costs[axis] = cost
    return costs


class MatchedNoPrepCostProvider:
    """Exact complete-trajectory expectation using existing full-wrapper primitives.

    A representative set cannot be relabelled exact: count, uniqueness,
    canonical finite-law events and complete probability mass are checked before
    any build. No sampling implementation or molecular defaults are introduced.
    """
    def __init__(self, compiler, *, compile_function=transpile_and_measure_cost, backend=None):
        self.compiler, self.compile_function, self.backend = compiler, compile_function, backend
        self.backend_hash = canonical_backend_fingerprint_or_none(backend)
        if self.backend_hash is None:
            raise ValueError("Exact pricing requires a canonical backend context")

    def _result(self, accumulators, *, context_hash, arm_hash, q, delta_time, law_hash, method):
        expected = {axis:circuit_cost_from_metric_statistics(acc.finalize(),compiler=self.compiler,
            estimate_kind=method, fidelity_level=5) for axis,acc in accumulators.items()}
        fingerprint = digest(dict(context=context_hash, arm=arm_hash, q=q, delta=delta_time,
            law=law_hash, compiler=compiler_settings_hash(self.compiler), backend=self.backend_hash,
            scope=RPE_HADAMARD_INTERROGATION_SCOPE))
        return RPERoundCompiledCost(expected["cosine"], expected["sine"], None, None, method, None,
            circuit_cost_scope=RPE_HADAMARD_INTERROGATION_SCOPE, cost_model_fingerprint=fingerprint,
            metadata=(("context_hash",context_hash),("arm_hash",arm_hash),("q",q),("delta_time",delta_time),
                      ("law_hash",law_hash),("backend_hash",self.backend_hash),
                      ("state_preparation_included",False)))

    def exact(self, preparation, config, distribution, *, q, trajectories):
        q=require_integer_count(q,name="q",minimum=1)
        trajectories = tuple(tuple(events) for events in trajectories)
        for events in trajectories:
            _canonical_events(preparation,config,distribution,events,q)
        # Saturate the size calculation at supplied count, avoiding enormous
        # integers when a clearly incomplete long-round set is supplied.
        per_event = sum(len(preparation.tail.components)**(order+1) for order,p in
                        zip(distribution.orders,distribution.order_probabilities) if p>0)
        expected_count = 1
        for _ in range(q*config.rte_steps):
            expected_count *= per_event
            if expected_count > len(trajectories):
                raise ValueError("Exact pricing requires the complete trajectory space")
        identities = [tuple((e.taylor_order,e.selected_component_ids) for e in events) for events in trajectories]
        weights = [math.prod(e.event_probability for e in events) for events in trajectories]
        if (len(trajectories)!=expected_count or len(set(identities))!=expected_count
                or not math.isclose(math.fsum(weights),1.,rel_tol=0,abs_tol=1e-12)):
            raise ValueError("Incomplete, duplicate or non-normalized exact trajectory set")
        accumulators = {axis:CompiledMetricAccumulator(weighted=True) for axis in ("cosine","sine")}
        for events,weight in zip(trajectories,weights,strict=True):
            pair = _compile_pair(build_controlled_trajectory(preparation,config,distribution,q=q,events=events),
                self.compiler,compile_function=self.compile_function,backend=self.backend)
            for axis,cost in pair.items(): accumulators[axis].update(cost,weight=weight)
        return self._result(accumulators, context_hash=preparation.context_hash,
            arm_hash=preparation.preparation_hash,q=q,delta_time=config.evolution_time,law_hash=_law_hash(config),
            method="exact_complete_matched_trajectory_expectation")

    def deterministic(self, circuit, *, context_hash, arm_hash, q, delta_time):
        """One qualified deterministic strong-DF circuit, with the same scope."""
        q=require_integer_count(q,name="q",minimum=1)
        if not context_hash or not arm_hash or not math.isfinite(delta_time) or delta_time<=0:
            raise ValueError("Explicit deterministic context and positive time required")
        pair = _compile_pair(circuit,self.compiler,compile_function=self.compile_function,backend=self.backend)
        accumulators = {axis:CompiledMetricAccumulator(weighted=True) for axis in pair}
        for axis,cost in pair.items(): accumulators[axis].update(cost,weight=1.)
        return self._result(accumulators,context_hash=context_hash,arm_hash=arm_hash,q=q,
            delta_time=delta_time,law_hash="deterministic_no_RTE",method="deterministic_full_no_prep")


def account_radius_round(*, context_hash, arm_hash, specification, allocation,
    beta_rpe, reference_radius_lower_bound, corrected_signal_error_bound,
    pf_phase_bound, model_phase_bound, implementation_phase_bound, radius_provenance,
    cost_metric, config=None, cost=None, backend_hash="no_backend"):
    """Conditional arithmetic, never automatic energy/branch certification.

    Caller supplies a justified PF/reference radius, not a raw radius that has
    already been attenuated. No unit-radius default and no copied total B field.
    """
    values = (reference_radius_lower_bound,corrected_signal_error_bound,pf_phase_bound,
              model_phase_bound,implementation_phase_bound,beta_rpe)
    if any(not math.isfinite(float(x)) or x<0 for x in values) or not 0<reference_radius_lower_bound<=1:
        raise ValueError("Finite nonnegative bounds and explicit reference radius in (0,1] required")
    if not context_hash or not arm_hash or not radius_provenance or cost_metric not in RPE_COST_METRICS:
        raise ValueError("Explicit context, arm, radius provenance and metric required")
    if not 0<beta_rpe<=RPE_NUMERICAL_PHASE_MODEL_LIMIT or not 0<allocation.beta_stat_budget<=RPE_NUMERICAL_PHASE_MODEL_LIMIT:
        raise ValueError("beta_rpe and statistical phase budget must lie in (0, pi/2]")
    if not backend_hash:
        raise ValueError("Explicit canonical backend context required")
    attenuation = 1. if config is None else finite_rte_attenuation(config,tail_evolutions=specification.q_m)
    if config is not None and config.evolution_time != specification.delta_time:
        raise ValueError("Config time differs from round delta")
    rho = attenuation*(reference_radius_lower_bound-corrected_signal_error_bound)
    beta_rte = (math.asin(corrected_signal_error_bound/reference_radius_lower_bound)
                if corrected_signal_error_bound<reference_radius_lower_bound else None)
    reasons=[]
    if rho<=0 or attenuation<=0: reasons.append("nonpositive_radius")
    if beta_rte is None or beta_rte>allocation.beta_rte_budget: reasons.append("RTE_phase_budget")
    if pf_phase_bound>allocation.beta_pf_budget: reasons.append("PF_phase_budget")
    if allocation.beta_total+model_phase_bound+implementation_phase_bound>beta_rpe: reasons.append("total_phase_budget")
    if allocation.beta_stat_budget<=0: reasons.append("nonpositive_statistical_budget")
    shots = {}
    if not reasons:
        for axis,alpha in (("cosine",allocation.alpha_cosine),("sine",allocation.alpha_sine)):
            shots[axis]=_hoeffding_shots(attenuation=attenuation,reference_radius=reference_radius_lower_bound,
                signal_error_bound=corrected_signal_error_bound,beta_stat_budget=allocation.beta_stat_budget,
                alpha_axis=alpha)
    total=None; compiler_hash=None
    if cost is not None:
        md=dict(cost.metadata)
        if (cost.circuit_cost_scope!=RPE_HADAMARD_INTERROGATION_SCOPE or not cost.cost_model_fingerprint
            or md.get("context_hash")!=context_hash or md.get("arm_hash")!=arm_hash
            or md.get("q")!=specification.q_m or md.get("delta_time")!=specification.delta_time
            or md.get("law_hash")!=_law_hash(config) or md.get("backend_hash")!=backend_hash
            or md.get("state_preparation_included") is not False):
            raise ValueError("Cost is not this arm/round's full no-prep estimate")
        compiler_hash=compiler_settings_hash(cost.cosine_expected_cost.compiler)
        if shots:
            total=math.fsum((shots['cosine']*circuit_cost_metric(cost.cosine_expected_cost,cost_metric),
                            shots['sine']*circuit_cost_metric(cost.sine_expected_cost,cost_metric)))
            if not math.isfinite(total): raise OverflowError("Round cost overflow")
    return dict(round_index=specification.round_index,q=specification.q_m,delta_time=specification.delta_time,
        context_hash=context_hash,arm_hash=arm_hash,cost_metric=cost_metric,compiler_hash=compiler_hash,
        backend_hash=backend_hash,beta_rpe=beta_rpe,
        feasible=not reasons,reasons=reasons,shots=shots or None,round_total_cost=total,
        attenuation=attenuation,normalization_product=1/attenuation if attenuation>0 else None,
        reference_radius_lower_bound=reference_radius_lower_bound,rho_observed_lower_bound=rho,
        pf_phase_bound=pf_phase_bound,model_phase_bound=model_phase_bound,
        implementation_phase_bound=implementation_phase_bound,finite_rte_phase_bound=beta_rte,
        radius_provenance=radius_provenance,alpha_bound=allocation.round_failure_probability_bound,
        cost_scope=RPE_HADAMARD_INTERROGATION_SCOPE,guarantee_status="not_certified",
        qualification="conditional_shot_arithmetic_only; energy/alias/PF-state and IID law require separate qualification")


def aggregate_registered_rounds(rounds, *, required_max_round, total_alpha_budget):
    """Reject missing rounds/prices/mixed contexts; no implicit RPE horizon."""
    required_max_round=require_integer_count(required_max_round,name="required_max_round")
    if not math.isfinite(total_alpha_budget) or not 0<total_alpha_budget<1:
        raise ValueError("Explicit total failure budget in (0,1) required")
    rounds=tuple(rounds)
    if sorted(r['round_index'] for r in rounds)!=list(range(required_max_round+1)):
        raise ValueError("Every registered round exactly once is required")
    keys=('context_hash','arm_hash','cost_metric','compiler_hash','backend_hash','cost_scope','delta_time','beta_rpe')
    if any(tuple(r[k] for k in keys)!=tuple(rounds[0][k] for k in keys) for r in rounds):
        raise ValueError("Mixed physical/arm/compiler/scope context")
    if any(not r['feasible'] or r['round_total_cost'] is None or r['compiler_hash'] is None for r in rounds):
        raise ValueError("Infeasible or unpriced round; missing prices are not zero")
    alpha=math.fsum(r['alpha_bound'] for r in rounds)
    if alpha>total_alpha_budget: raise ValueError("Total failure budget exceeded")
    return dict(scope="registered_rounds_only",round_count=len(rounds),maximum_round=required_max_round,
        total_cost=math.fsum(r['round_total_cost'] for r in rounds),alpha_bound=alpha,
        guarantee_status="not_certified",context={k:rounds[0][k] for k in keys})
