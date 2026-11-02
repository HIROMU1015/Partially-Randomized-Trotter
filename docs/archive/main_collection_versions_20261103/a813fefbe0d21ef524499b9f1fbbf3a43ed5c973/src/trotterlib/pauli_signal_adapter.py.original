"""Non-DF JW Hamiltonians and sparse Pauli state/circuit primitives.

Labels are Qiskit strings: rightmost character is qubit 0, state index is
sum(occupation[q] * 2**q). OpenFermion indices are mapped explicitly.
No SCF, DF, eigensolver or molecular I/O runs in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .rte import RTEComponent, finite_rte_distribution, require_integer_count, sample_rte_events


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value, dtype="<c16")
    return digest(dict(shape=list(a.shape), dtype="<c16", bytes=hashlib.sha256(a.tobytes()).hexdigest()))


def real_coefficient(value):
    z = complex(value)
    if not math.isfinite(z.real) or not math.isfinite(z.imag) or abs(z.imag) > 1e-12:
        raise ValueError("Hamiltonian coefficients must be finite and Hermitian (imaginary tolerance 1e-12)")
    return float(z.real)


@dataclass(frozen=True)
class PauliHamiltonian:
    num_qubits: int
    constant: float
    terms: tuple[tuple[str, float], ...]
    metadata_json: str

    def __post_init__(self):
        object.__setattr__(self, "num_qubits", require_integer_count(self.num_qubits, name="num_qubits", minimum=1))
        object.__setattr__(self, "constant", real_coefficient(self.constant))
        labels = []
        for label, coefficient in self.terms:
            if len(label) != self.num_qubits or set(label) - set("IXYZ") or set(label) == {"I"}:
                raise ValueError("Expected nonidentity Qiskit Pauli label")
            if real_coefficient(coefficient) == 0:
                raise ValueError("Zero terms must be consolidated before construction")
            labels.append(label)
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate Pauli labels")
        object.__setattr__(self, "terms", tuple(sorted(((s, real_coefficient(c)) for s,c in self.terms),
                                                       key=lambda item: (-abs(item[1]), item[0]))))
        if not isinstance(json.loads(self.metadata_json), dict):
            raise ValueError("metadata must be a JSON object")

    @classmethod
    def from_terms(cls, terms, *, num_qubits=None, model_id="artificial", metadata=None):
        terms = list(terms)
        if num_qubits is None:
            if not terms:
                raise ValueError("num_qubits is required for an empty Hamiltonian")
            num_qubits = len(terms[0][0])
        num_qubits = require_integer_count(num_qubits, name="num_qubits", minimum=1)
        grouped = {}
        for label, coefficient in terms:
            if not isinstance(label, str) or len(label) != num_qubits or set(label) - set("IXYZ"):
                raise ValueError("Invalid Pauli label/qubit count")
            grouped.setdefault(label, []).append(complex(coefficient))
        merged = {label: real_coefficient(complex(math.fsum(z.real for z in values),
                                                math.fsum(z.imag for z in values)))
                  for label, values in grouped.items()}
        constant = merged.pop("I" * num_qubits, 0.)
        # Match existing descending-|h| prefix semantics; label resolves ties
        # canonically, independently of input insertion order. No cutoff.
        ordered = tuple(sorted(((s, c) for s, c in merged.items() if c != 0.),
                               key=lambda item: (-abs(item[1]), item[0])))
        md = dict(metadata or {})
        md.update(model_id=model_id, qubit_order="qiskit_little_endian",
                  label_order="q[n-1]...q[0]", coefficient_cutoff=0.,
                  split_order="descending_abs_then_label", df_rank=None)
        return cls(num_qubits, constant, ordered, json.dumps(md, sort_keys=True, allow_nan=False))

    @classmethod
    def from_openfermion(cls, operator, *, num_qubits, model_id="supplied_jw", metadata=None):
        from openfermion import QubitOperator
        if not isinstance(operator, QubitOperator):
            raise TypeError("Expected OpenFermion QubitOperator after JW")
        labels = []
        for term, coefficient in operator.terms.items():
            label = ["I"] * num_qubits
            for q, axis in term:
                if q >= num_qubits:
                    raise ValueError("Pauli index exceeds declared num_qubits")
                label[num_qubits - 1 - q] = axis
            labels.append(("".join(label), coefficient))
        return cls.from_terms(labels, num_qubits=num_qubits, model_id=model_id, metadata=metadata)

    @classmethod
    def from_spatial_integrals(cls, one_body, eri_chemist, *, nuclear_repulsion,
                               model_id="supplied_mo_integrals", metadata=None):
        """Real orthonormal spatial MOs; eri[p,q,r,s]=(pq|rs), not AO integrals.

        Follows openfermionpyscf.compute_integrals: transpose(0,2,3,1),
        interleaved alpha/beta spin orbitals, 1/2 on two-body coefficients.
        Spin lifting is explicit to avoid spinorb_from_spatial's hidden cutoff.
        """
        from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
        one = np.asarray(one_body)
        eri = np.asarray(eri_chemist)
        if one.ndim != 2 or one.shape[0] != one.shape[1] or one.shape[0] < 1:
            raise ValueError("one_body must be a nonempty square MO matrix")
        n = len(one)
        if eri.shape != (n,) * 4 or not np.isfinite(one).all() or not np.isfinite(eri).all():
            raise ValueError("Integral shape or finiteness")
        if np.iscomplexobj(one) or np.iscomplexobj(eri):
            raise NotImplementedError("Complex spatial orbital integral convention is not supported")
        if not np.allclose(one, one.T, rtol=0, atol=1e-12):
            raise ValueError("one_body must be Hermitian")
        two = eri.transpose(0, 2, 3, 1)
        spin_one = np.zeros((2*n, 2*n))
        spin_two = np.zeros((2*n,) * 4)
        for a in range(2):
            spin_one[a::2, a::2] = one
            for b in range(2):
                spin_two[a::2, b::2, b::2, a::2] = two / 2
        interaction = InteractionOperator(real_coefficient(nuclear_repulsion), spin_one, spin_two)
        jw = jordan_wigner(get_fermion_operator(interaction))
        md = dict(metadata or {})
        md.update(orbital_basis="real_orthonormal_spatial_MO", input_eri="chemist_(pq|rs)",
                  openfermion_eri="h[p,q,r,s]=(ps|qr)", spin_order="2*p+alpha(0)/beta(1)",
                  two_body_prefactor=.5, nuclear_repulsion=float(nuclear_repulsion),
                  one_body_hash=array_hash(one), eri_chemist_hash=array_hash(eri))
        return cls.from_openfermion(jw, num_qubits=2*n, model_id=model_id, metadata=md)

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    @property
    def hamiltonian_hash(self):
        return digest(dict(num_qubits=self.num_qubits, constant=self.constant, terms=self.terms,
                           qubit_order="qiskit_little_endian"))

    def sparse_operator(self, terms=None, *, constant=None):
        selected = self.terms if terms is None else tuple(terms)
        scalar = self.constant if constant is None else constant
        labels = [(s, c) for s, c in selected]
        labels.append(("I" * self.num_qubits, scalar))
        return csr_matrix(SparsePauliOp.from_list(labels).to_matrix(sparse=True))


def append_pauli_evolution(circuit, label, coefficient, time):
    circuit.append(PauliEvolutionGate(SparsePauliOp.from_list([(label, coefficient)]), time=time),
                   range(len(label)))


def append_pauli_event(circuit, event, labels):
    """Honor generic event application order, signs and (-1)^(order/2) phase."""
    for item in event.application_sequence:
        label = labels[item.component_id]
        if item.role == "product":
            for q, axis in enumerate(reversed(label)):
                if axis != "I":
                    getattr(circuit, axis.lower())(q)
            if item.coefficient_sign < 0:
                circuit.global_phase += math.pi
        else:
            append_pauli_evolution(circuit, label, item.coefficient_sign, event.rotation_angle)
    circuit.global_phase += float(np.angle(event.phase))


class PauliSignalAdapter:
    representation = "pauli_jw"

    def __init__(self, hamiltonian, method, ld, step_time, r, K):
        if not isinstance(hamiltonian, PauliHamiltonian):
            raise TypeError("pauli_jw requires PauliHamiltonian")
        self.hamiltonian = hamiltonian
        self.num_qubits = hamiltonian.num_qubits
        self.ld = len(hamiltonian.terms) if ld is None and method == "PF" else (0 if ld is None else ld)
        require_integer_count(self.ld, name="L_D")
        if self.ld > len(hamiltonian.terms) or (method == "PF" and self.ld != len(hamiltonian.terms)):
            raise ValueError("PF retains all terms; use B0/B0I for truncation")
        self.method, self.step_time, self.r, self.K = method, step_time, r, K
        self.retained, self.tail = hamiltonian.terms[:self.ld], hamiltonian.terms[self.ld:]
        self.randomized = method == "PR" and bool(self.tail)
        self.lambda_r = math.fsum(abs(c) for _, c in self.tail) if self.randomized else 0.
        self.distribution = finite_rte_distribution(self.lambda_r * step_time / r, K) if self.randomized else None
        self.components = tuple(RTEComponent(str(j), abs(c) / self.lambda_r, abs(c), 1 if c > 0 else -1)
                                for j, (_, c) in enumerate(self.tail)) if self.randomized else ()
        self.labels = {str(j): s for j, (s, _) in enumerate(self.tail)}
        self._operators = None

    @property
    def log_B_step(self):
        return self.r * math.log(self.distribution.exact_finite_distribution) if self.randomized else 0.

    def trajectory(self, *, q, seed):
        circuit = QuantumCircuit(self.num_qubits)
        events = iter(sample_rte_events(self.components, self.distribution, sample_count=q*self.r, seed=seed)) if self.randomized else None
        for _ in range(q):
            for label, c in self.retained:
                append_pauli_evolution(circuit, label, c, self.step_time/2)
            if self.randomized:
                for _ in range(self.r):
                    append_pauli_event(circuit, next(events), self.labels)
            for label, c in reversed(self.retained):
                append_pauli_evolution(circuit, label, c, self.step_time/2)
            circuit.global_phase -= self.hamiltonian.constant * self.step_time
        return circuit

    def operators(self):
        if self._operators is None:
            h = self.hamiltonian
            full = h.sparse_operator()
            blocks = [h.sparse_operator([(s,c)], constant=0.) for s,c in self.retained]
            tail = h.sparse_operator(self.tail, constant=0.)
            self._operators = full, blocks, tail, h.constant
        return self._operators

    def metadata(self):
        return dict(hamiltonian_type=self.representation, hamiltonian_hash=self.hamiltonian.hamiltonian_hash,
                    generation=self.hamiltonian.metadata, DF_rank=None, split_unit="nonidentity_Pauli_term",
                    L_D=self.ld, retained_labels=[s for s,_ in self.retained], tail_labels=[s for s,_ in self.tail],
                    identity_policy="constant_always_retained_once", B0I_equivalent_to_B0=True,
                    B0I_reason="nonidentity_Pauli_tail_has_zero_identity_coefficient")


def sparse_mean_action(adapter, state, *, q):
    """Independent paired-polynomial mean; no sampled signal or dense U."""
    _, blocks, tail, scalar = adapter.operators()
    out = np.asarray(state, dtype=complex).copy()
    for _ in range(q):
        for op in blocks:
            out = expm_multiply(-.5j * adapter.step_time * op, out)
        if adapter.randomized:
            # Paired degrees 0..K+1 divided by the finite event L1 norm.
            for _ in range(adapter.r):
                term = out.copy()
                result = out.copy()
                for degree in range(1, adapter.K + 2):
                    term = (-1j * adapter.step_time / adapter.r / degree) * (tail @ term)
                    result += term
                out = result / adapter.distribution.exact_finite_distribution
        for op in reversed(blocks):
            out = expm_multiply(-.5j * adapter.step_time * op, out)
        out *= np.exp(-1j * scalar * adapter.step_time)
    return out
