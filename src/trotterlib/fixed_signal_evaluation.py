"""Adapters for fixed-time PF/PR signals, full-wrapper C, sufficient N and G.

No molecular generation, candidate search, fitting or RPE scheduling. Existing
DF APIs are used unchanged. Callers explicitly supply the Hamiltonian/state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import importlib.metadata
import math
import platform

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.sparse import eye
from scipy.sparse.linalg import expm_multiply

from .fixed_signal_accounting import AXES, fixed_signal_shots, nonnegative
from .pauli_signal_adapter import PauliHamiltonian, PauliSignalAdapter, array_hash, digest, sparse_mean_action
from .rte import CompilerSettings, make_rte_config, require_integer_count
from .rte_compiled_cost import TranspiledCircuitCostCache


@dataclass(frozen=True)
class FixedSignalConfig:
    hamiltonian_type: str
    method: str = "PF"
    T: float = .2
    q: int = 1
    L_D: int | None = None
    r: int = 1
    K: int = 2
    seed: int = 17
    cost_samples: int = 8
    product_formula: str = "S2"
    randomization_law: str = "finite_rte_even_paired"

    def __post_init__(self):
        if self.hamiltonian_type not in {"df_native", "pauli_jw"}:
            raise ValueError("Unknown Hamiltonian representation")
        if self.method not in {"PF", "PR", "B0", "B0I"}:
            raise NotImplementedError("Supported methods are PF, PR, B0, B0I")
        if self.product_formula != "S2":
            raise NotImplementedError("Only S2 is connected in this implementation")
        if self.randomization_law != "finite_rte_even_paired":
            raise NotImplementedError("Only the existing paired finite-RTE law is connected")
        if isinstance(self.T, bool) or not math.isfinite(self.T):
            raise ValueError("T must be finite")
        for name in ("q", "r", "cost_samples"):
            object.__setattr__(self, name, require_integer_count(getattr(self, name), name=name, minimum=1))
        for name in ("seed", "K"):
            object.__setattr__(self, name, require_integer_count(getattr(self, name), name=name))
        if self.L_D is not None:
            object.__setattr__(self, "L_D", require_integer_count(self.L_D, name="L_D"))
        object.__setattr__(self, "T", float(self.T))
        if self.K % 2:
            raise ValueError("K must be nonnegative and even")


def default_compiler():
    return CompilerSettings(basis_gates=("rz", "sx", "x", "cx"), backend_name=None,
                            coupling_map=None, optimization_level=1, layout_method=None,
                            routing_method=None, transpiler_seed=17,
                            qiskit_version=importlib.metadata.version("qiskit"))


class DFSignalAdapter:
    representation = "df_native"

    def __init__(self, hamiltonian, method, ld, step_time, r, K):
        from .df_hamiltonian import DFHamiltonian
        from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
        from .df_partial_s2 import prepare_df_partial_s2
        if not isinstance(hamiltonian, DFHamiltonian):
            raise TypeError("df_native requires DFHamiltonian")
        self.hamiltonian = hamiltonian
        self.num_qubits = hamiltonian.n_qubits
        self.ld = hamiltonian.n_blocks if ld is None and method == "PF" else (0 if ld is None else ld)
        require_integer_count(self.ld, name="L_D")
        if self.ld > hamiltonian.n_blocks or (method == "PF" and self.ld != hamiltonian.n_blocks):
            raise ValueError("PF retains all DF fragments; use B0/B0I for truncation")
        self.method, self.step_time, self.r, self.K = method, step_time, r, K
        partition = split_df_hamiltonian_by_ld(hamiltonian, self.ld)
        self.original_preparation = prepare_df_partial_s2(hamiltonian, partition, coefficient_atol=0.)
        self.preparation = self.original_preparation
        if method in {"B0", "B0I"}:
            selected = hamiltonian.select_blocks(self.original_preparation.deterministic_fragment_indices)
            if method == "B0I":
                selected = DFHamiltonian(selected.constant + self.original_preparation.extracted_identity_coefficient,
                                         selected.one_body, selected.lambdas, selected.g_matrices, selected.metadata)
            self.preparation = prepare_df_partial_s2(selected, split_df_hamiltonian_by_ld(selected, selected.n_blocks))
        self.randomized = method == "PR" and not self.preparation.is_deterministic_only
        self.lambda_r = self.preparation.exact_rte_lambda_r if self.randomized else 0.
        self.config = self.distribution = None
        if self.randomized:
            self.config, self.distribution = make_rte_config(self.preparation.rte_preparation.symbolic_tail,
                evolution_time=step_time, rte_steps=r, finite_taylor_order=K, truncation_tolerance=1.)
        self._operators = None

    @property
    def log_B_step(self):
        return self.r * math.log(self.distribution.exact_finite_distribution) if self.randomized else 0.

    def trajectory(self, *, q, seed, controlled=False):
        from .df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder, make_df_partial_s2_repeated_request
        request = make_df_partial_s2_repeated_request(self.preparation, step_time=self.step_time,
            repetition_count=q, rte_config=self.config, rte_distribution=self.distribution,
            seed=seed, controlled=controlled, ancilla_qubit=self.num_qubits if controlled else None,
            construction_policy="raw_concatenation")
        return QiskitDFPartialS2RepeatedCircuitBuilder().build(request).circuit

    def operators(self):
        if self._operators is None:
            from openfermion import FermionOperator, jordan_wigner

            def one_body(matrix):
                fermion = FermionOperator()
                for p, q in np.argwhere(np.asarray(matrix) != 0):
                    fermion += FermionOperator(((int(p), 1), (int(q), 0)), complex(matrix[p,q]))
                return PauliHamiltonian.from_openfermion(jordan_wigner(fermion), num_qubits=self.num_qubits).sparse_operator()

            h = self.hamiltonian
            identity = eye(1 << self.num_qubits, dtype=complex, format="csr")
            one = one_body(h.one_body)
            fragments = []
            for w, g in zip(h.lambdas, h.g_matrices, strict=True):
                a = one_body(g)
                fragments.append(float(w) * (a @ a))
            full = h.constant * identity + one
            for op in fragments:
                full = full + op
            p = self.original_preparation
            retained = [one] if np.any(h.one_body != 0) else []
            retained += [fragments[i] for i in p.deterministic_fragment_indices]
            tail = full - h.constant * identity
            for op in retained:
                tail = tail - op
            extracted = p.extracted_identity_coefficient
            if self.randomized:
                tail = tail - extracted * identity
            scalar = h.constant + (extracted if self.randomized or self.method == "B0I" else 0.)
            self._operators = full.tocsr(), retained, tail.tocsr(), scalar
        return self._operators

    def metadata(self):
        p = self.original_preparation
        return dict(hamiltonian_type=self.representation, hamiltonian_hash=p.hamiltonian_hash,
                    generation=self.hamiltonian.metadata, DF_rank=self.hamiltonian.n_blocks,
                    split_unit="DF_fragment", L_D=self.ld,
                    deterministic_fragment_indices=list(p.deterministic_fragment_indices),
                    deterministic_block_order=list(p.deterministic_block_order),
                    preparation_hash=p.preparation_hash, identity_policy=p.identity_policy,
                    extracted_identity_coefficient=p.extracted_identity_coefficient,
                    B0I_equivalent_to_B0=p.extracted_identity_coefficient == 0.)


def prepare_signal_adapter(hamiltonian, config):
    cls = PauliSignalAdapter if config.hamiltonian_type == "pauli_jw" else DFSignalAdapter
    return cls(hamiltonian, config.method, config.L_D, config.T / config.q, config.r, config.K)


def hadamard_wrapper(evolution, axis, *, measure=True, already_controlled=False):
    """Ordinary diag(I,U), ancilla n, H / Sdg-H / measurement; no preparation."""
    if axis not in AXES:
        raise ValueError("Hadamard axis must be cosine or sine")
    n = evolution.num_qubits - int(already_controlled)
    circuit = QuantumCircuit(n + 1, 1 if measure else 0)
    circuit.h(n)
    if already_controlled:
        circuit.compose(evolution, inplace=True)
    else:
        circuit.append(evolution.to_gate().control(1), [n, *range(n)])
    if axis == "sine":
        circuit.sdg(n)
    circuit.h(n)
    if measure:
        circuit.measure(n, 0)
    return circuit


def _statistics(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return dict(mean=None, standard_error=None, nominal95_halfwidth=None)
    se = float(np.std(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else None
    if se is not None:
        from scipy.stats import t
        half = float(t.ppf(.975, len(a)-1)) * se
    else:
        half = None
    return dict(mean=float(np.mean(a)), standard_error=se, nominal95_halfwidth=half,
                mean_exact=str(Fraction(sum(values), len(values))) if all(isinstance(x, int) for x in values) else None)


def evaluate_fixed_signal(hamiltonian, state, config, *, epsilon_sig,
                          numerical_allowance=None, evidence_kind="UNKNOWN", evidence_ref=None,
                          log_B_margin=None, epsilon_axes=None, alpha_axes=None,
                          state_metadata=None, compiler=None, cost_cache=None,
                          source_provenance=None, verify_compiled=False):
    """Evaluate one supplied model/configuration, with no quantum shot execution.

    Bias is the corrected law-mean signal minus exp(-iHT) on the supplied state.
    Numerical allowances and log-B margin are explicit caller declarations.
    Missing numerical/normalization evidence leaves N and G undefined.
    Cost MC and quantum shots are separate; both axes share trajectory seeds.
    """
    adapter = prepare_signal_adapter(hamiltonian, config)
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
    for seed in seeds:
        native_controlled = isinstance(adapter, DFSignalAdapter)
        evolution = (adapter.trajectory(q=config.q, seed=seed, controlled=True) if native_controlled
                     else adapter.trajectory(q=config.q, seed=seed))
        row = dict(seed=seed, RZ={})
        for axis in AXES:
            wrapped = hadamard_wrapper(evolution, axis, already_controlled=native_controlled)
            cost, key, hit = cache.get_or_transpile(wrapped, compiler,
                circuit_fingerprint=digest(dict(config=asdict(config), seed=seed, axis=axis,
                                               hamiltonian=adapter.metadata()["hamiltonian_hash"])))
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


def validate_fixed_signal_result(result):
    """Validate the shared output contract and N/C/G consistency on load/save."""
    import json
    json.dumps(result, allow_nan=False)
    if result["schema"] != "fixed_signal_resource_evaluation_v1":
        raise ValueError("UNSUPPORTED_RESULT_SCHEMA")
    model = result["model"]
    if model["hamiltonian_type"] not in {"df_native", "pauli_jw"}:
        raise ValueError("RESULT_REPRESENTATION")
    if model["hamiltonian_type"] == "pauli_jw" and model["DF_rank"] is not None:
        raise ValueError("PAULI_HAS_NO_DF_RANK")
    if result["reference"]["hamiltonian_hash"] != model["hamiltonian_hash"]:
        raise ValueError("REFERENCE_HAMILTONIAN_BINDING")
    costs = result["circuit_cost"]
    if set(costs["axes"]) != set(AXES) or costs["sample_count"] != len(costs["samples"]):
        raise ValueError("COST_AXIS_OR_SAMPLE_SCHEMA")
    for axis in AXES:
        nonnegative(costs["axes"][axis]["mean"], "C_RZ")
    shots, total = result["shots"], result["total_RZ"]
    if not shots["usable"]:
        if shots["N_total"] is not None or total is not None:
            raise ValueError("UNAVAILABLE_N_G_MUST_BE_NULL")
    else:
        ns = [require_integer_count(shots["axes"][a]["shots"], name="N_axis", minimum=1) for a in AXES]
        if shots["N_total"] != sum(ns) or total is None:
            raise ValueError("N_TOTAL_OR_G_SCHEMA")
        exact_costs = [costs["axes"][a].get("mean_exact") for a in AXES]
        if all(x is not None for x in exact_costs):
            expected = sum(n*Fraction(c) for n,c in zip(ns, exact_costs, strict=True))
            if total.get("mean_exact") is None or Fraction(total["mean_exact"]) != expected:
                raise ValueError("G_RZ_IDENTITY")
        nonnegative(total["mean"], "G_RZ")
    return result
