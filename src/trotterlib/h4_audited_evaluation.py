"""H4 opt-in evaluation path, copied from the frozen fixed-signal evaluator.

Only adapter selection and observation/join calls differ; old H2 SOURCE bytes
and entrypoints are not edited. Numeric accounting/compiler order is preserved.
The copied implementation is checked against the old AST by regression tests.
"""
from dataclasses import asdict
from fractions import Fraction
import importlib.metadata
import math
import platform
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply
from .fixed_signal_accounting import AXES, fixed_signal_shots, nonnegative
from .fixed_signal_evaluation import (DFSignalAdapter, default_compiler, hadamard_wrapper,
    _statistics, validate_fixed_signal_result)
from .pauli_signal_adapter import array_hash, digest, sparse_mean_action
from .rte_compiled_cost import TranspiledCircuitCostCache
from .rte_event_audit import audited_adapter


def evaluate_audited_signal(hamiltonian, state, config, *, epsilon_sig,
                          numerical_allowance=None, evidence_kind="UNKNOWN", evidence_ref=None,
                          log_B_margin=None, epsilon_axes=None, alpha_axes=None,
                          state_metadata=None, compiler=None, cost_cache=None,
                          source_provenance=None, verify_compiled=False, event_audit=None):
    """Evaluate one supplied model/configuration, with no quantum shot execution.

    Bias is the corrected law-mean signal minus exp(-iHT) on the supplied state.
    Numerical allowances and log-B margin are explicit caller declarations.
    Missing numerical/normalization evidence leaves N and G undefined.
    Cost MC and quantum shots are separate; both axes share trajectory seeds.
    """
    adapter = audited_adapter(hamiltonian, config, event_audit)
    state = np.array(state, dtype=complex, copy=True)
    if state.shape != (1 << adapter.num_qubits,) or not np.isfinite(state).all():
        raise ValueError("STATE_SHAPE_OR_FINITENESS")
    if abs(np.linalg.norm(state) - 1) > 1e-12:
        raise ValueError("STATE_MUST_BE_NORMALIZED")
    compiler = compiler or default_compiler()
    cache = cost_cache or TranspiledCircuitCostCache(retain_transpiled_circuits=verify_compiled)
    full = adapter.operators()[0]
    reference = complex(np.vdot(state, expm_multiply(-1j * config.T * full, state)))
    raw = complex(np.vdot(state, sparse_mean_action(adapter, state, q=config.q)))
    log_B = config.q * adapter.log_B_step
    B = math.exp(log_B) if log_B < 709 else None
    corrected = raw * B if B is not None else None
    bias = corrected - reference if corrected is not None else None
    allowance = dict({a: None for a in AXES} if numerical_allowance is None else numerical_allowance)
    if log_B_margin is not None:
        log_B_margin = nonnegative(log_B_margin, "log_B_margin")
    log_B_upper = log_B + log_B_margin if log_B_margin is not None else None
    accounting = fixed_signal_shots(epsilon_sig=epsilon_sig,
        bias=dict(cosine=abs(bias.real) if bias is not None else None,
                  sine=abs(bias.imag) if bias is not None else None),
        allowance=allowance, log_B_upper=log_B_upper, evidence_kind=evidence_kind,
        evidence_ref=evidence_ref, epsilon_axes=epsilon_axes, alpha_axes=alpha_axes)
    count = config.cost_samples if adapter.randomized else 1
    children = np.random.SeedSequence(config.seed).spawn(count)
    seeds = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
    prices = {a: [] for a in AXES}
    rows = []
    compiled_errors = []
    for trajectory_index, seed in enumerate(seeds):
        if event_audit is not None:
            event_audit.begin(seed, trajectory_index)
        native_controlled = isinstance(adapter, DFSignalAdapter)
        evolution = (adapter.trajectory(q=config.q, seed=seed, controlled=True) if native_controlled
                     else adapter.trajectory(q=config.q, seed=seed))
        if event_audit is not None:
            event_audit.validate_evolution(adapter, config, evolution)
        fingerprints = {}
        row = dict(seed=seed, RZ={})
        for axis in AXES:
            wrapped = hadamard_wrapper(evolution, axis, already_controlled=native_controlled)
            cost, key, hit = cache.get_or_transpile(wrapped, compiler,
                circuit_fingerprint=digest(dict(config=asdict(config), seed=seed, axis=axis,
                                               hamiltonian=adapter.metadata()["hamiltonian_hash"])))
            fingerprints[axis] = cost.actual_circuit_fingerprint
            prices[axis].append(cost.rz_count)
            row["RZ"][axis] = cost.rz_count
            if verify_compiled:
                if cost.transpiled_circuit is None:
                    raise ValueError("Compiled verification requires a cache with retain_transpiled_circuits=True")
                initial = Statevector(np.concatenate((state, np.zeros_like(state))))
                measured = cost.transpiled_circuit.remove_final_measurements(inplace=False)
                if measured.num_qubits != adapter.num_qubits + 1 or measured.layout is not None:
                    raise NotImplementedError("Compiled-signal check requires unchanged wire layout")
                before = initial.evolve(wrapped.remove_final_measurements(inplace=False)).probabilities([adapter.num_qubits])
                after = initial.evolve(measured).probabilities([adapter.num_qubits])
                error = float(np.max(np.abs(before-after)))
                compiled_errors.append(error)
                if error > 1e-9:
                    raise ValueError("COMPILED_WRAPPER_SIGNAL_MISMATCH")
        if event_audit is not None:
            joined = event_audit.join(row['RZ'], compiler, fingerprints)
            row.update(trajectory_id=joined['trajectory_id'], event_record_hash=joined['event_record_hash'],
                       trajectory_record_hash=joined['trajectory_record_hash'])
        rows.append(row)
    costs = {a: _statistics(prices[a]) for a in AXES}
    if not adapter.randomized:
        for value in costs.values():
            value.update(standard_error=0., nominal95_halfwidth=0.)
    G = None
    if accounting["usable"]:
        weighted = [sum(accounting["axes"][a]["shots"] * prices[a][i] for a in AXES) for i in range(count)]
        G = _statistics(weighted)
        if not adapter.randomized:
            G.update(standard_error=0., nominal95_halfwidth=0.)
    signal = lambda z: None if z is None else dict(real=float(z.real), imag=float(z.imag))
    metadata = adapter.metadata()
    result = dict(schema="fixed_signal_resource_evaluation_v1", model=metadata,
        config=asdict(config), state=dict(hash=array_hash(state), generation=state_metadata or {},
                                        basis="qiskit_little_endian"),
        reference=dict(hamiltonian_hash=metadata["hamiltonian_hash"], T=config.T,
                       method="scipy_sparse_expm_multiply", signal=signal(reference)),
        signal=dict(raw=signal(raw), corrected=signal(corrected), signed_bias=signal(bias),
                    absolute_error=abs(bias) if bias is not None else None,
                    mean_method="independent_sparse_paired_polynomial_state_action"),
        normalization=dict(law="finite_rte_even_paired" if adapter.randomized else "deterministic",
                           lambda_r=adapter.lambda_r, log_B=log_B, B=B, log_B_upper=log_B_upper,
                           log_B_margin=log_B_margin, normalization_certificate_verified=False,
                           r=config.r if adapter.randomized else None, K=config.K if adapter.randomized else None,
                           attenuation=math.exp(-log_B)),
        shots=accounting, circuit_cost=dict(axes=costs, sample_count=count,
            kind="finite_MC_mean" if adapter.randomized else "deterministic",
            scope="full_cosine_sine_hadamard_wrapper_no_state_preparation",
            paired_axes=True, mc_samples_are_quantum_shots=False, samples=rows,
            uncertainty="nominal_Student_t_95_not_rare_event_or_population_guarantee",
            cache_hits=cache.hit_count, cache_misses=cache.miss_count,
            compiled_signal_max_error=max(compiled_errors) if compiled_errors else None),
        total_RZ=G, compiler=asdict(compiler),
        provenance=dict(source=source_provenance or {}, python=platform.python_version(),
                        environment={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "qiskit", "qiskit-nature", "openfermion", "openfermionpyscf", "pyscf"]}),
        validation=dict(status="TECHNICAL_CONDITIONAL_ACCOUNTING", molecular_validation=False,
                        resource_status="CONDITIONAL_CNG_AVAILABLE" if G is not None else "N_G_UNDEFINED",
                        scientific_superiority_claimed=False, RPE_rounds_evaluated=False))
    validate_fixed_signal_result(result)
    return result
