"""Fixed N3-B1b-1 vacuum-word provider, not a synthesis algorithm.

Written chunk order is ascending j=-36..6.  Emission is delegated unchanged
to lower_native; negative targets invert the COMPLETE positive circuit.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import math
from types import MappingProxyType

B = 36
EXPONENTS = tuple(range(-36, 7))
EMITTER_UPPER = F(45, 140737488355328)
CACHE_EMITTER_UPPER = F(64, 2**53)
# Reused verbatim from pr_reference_ct1.word_matrices.
ADJOINT = {"H": "H", "T": "t", "t": "T", "S": "SSS",
           "X": "X", "W": "WWWWWWW"}


def round_index(theta: float) -> int:
    """Exact nearest dyadic index, with ties to even and sign symmetry."""
    if not isinstance(theta, float):
        raise TypeError("The target must be an actual binary64 float")
    if not math.isfinite(theta):
        raise ValueError("Nonfinite target")
    n, d = abs(theta).as_integer_ratio()
    q, r = divmod(n << B, d)
    q += 2 * r > d or (2 * r == d and q % 2 == 1)
    return -q if theta < 0 else q


def actual_adjoint(word: str) -> str:
    return "".join(ADJOINT[g] for g in reversed(word))


@dataclass(frozen=True)
class WordCertificate:
    theta_hex: str
    route: str
    positive_sequence: str
    negative: bool
    index: int | None
    basic_indices: tuple[int, ...]
    quantization_upper: F
    word_error_upper: F
    emitter_upper: F
    T: int
    Tdagger: int
    global_W_signed: int

    @property
    def strict_upper(self) -> F:
        # Actual emitter error is STRICTLY below its declared bound.
        return self.quantization_upper + self.word_error_upper + self.emitter_upper

    def signed_sequence(self) -> str:
        return actual_adjoint(self.positive_sequence) if self.negative else self.positive_sequence

    def summary(self):
        return dict(theta_hex=self.theta_hex, route=self.route,
            positive_sequence_sha256=hashlib.sha256(self.positive_sequence.encode()).hexdigest(),
            negative_whole_adjoint=self.negative, rounded_index=self.index,
            basic_indices=list(self.basic_indices),
            quantization_upper=str(self.quantization_upper),
            word_error_upper=str(self.word_error_upper), emitter_upper=str(self.emitter_upper),
            strict_operator_Frobenius_upper=str(self.strict_upper),
            T=self.T, Tdagger=self.Tdagger, global_W_signed=self.global_W_signed)


class DictionaryProvider:
    """Immutable selected strings/metadata; no mutable-cache acquisition.

    Inputs have already passed the reused CT-0 record validator.  Constructor
    checks the fixed scientific specification and retained emitter headroom.
    """
    def __init__(self, parameters, budget, basic_records, cache_records):
        assert parameters["b"] == B and parameters["M_b"] == 43
        assert (parameters["j_min"], parameters["j_max"]) == (-36, 6)
        assert F(parameters["uniform_basic_Frobenius_allocation"]) == F(1, 10**13)
        self.limit = float.fromhex(parameters["L_hex"])
        self.N = int(parameters["N_b"])
        self.epsilon = F(budget["epsilon"])
        self.sqrt2_upper = F(budget["sqrt2"]["upper"])
        self.Q_upper = F(budget["quantization_Frobenius"]["upper"])
        assert self.epsilon == F(1, 10**11)
        assert EMITTER_UPPER < F(budget["emitter_remaining"]["lower"])
        assert set(basic_records) == set(EXPONENTS)
        self.basic = {}
        for j in EXPONENTS:
            r = basic_records[j]
            assert r["theta_hex"] == float(2.0**j).hex() and r["precision_exponent"] == -13
            assert r["guard_method"] == "exact cyclotomic word matrix + outward interval strict Frobenius"
            assert F(r["strict_error_upper"]) <= F(1, 10**13)
            self.basic[j] = self._retain(r)
        self.cache = {}
        for key, r in cache_records.items():
            assert key == r["theta_hex"] and r["precision_exponent"] == -11
            assert r["guard_method"] == "exact cyclotomic word matrix + outward interval strict Frobenius"
            assert F(r["strict_error_upper"]) + CACHE_EMITTER_UPPER < self.epsilon
            self.cache[key] = self._retain(r)
        assert self.Q_upper + 43 * F(1, 10**13) + EMITTER_UPPER < self.epsilon
        self.basic = MappingProxyType(self.basic)
        self.cache = MappingProxyType(self.cache)

    @staticmethod
    def _retain(r):
        # Strings/integers/Fractions are immutable; do not expose old records.
        return MappingProxyType(dict(sequence=r["sequence"], guard=F(r["strict_error_upper"]),
                    T=r["T"], Tdagger=r["Tdagger"], W=r["global_W"]))

    def provide(self, theta: float) -> WordCertificate:
        if not isinstance(theta, float) or not math.isfinite(theta):
            raise ValueError("Finite binary64 target required")
        if abs(theta) > self.limit:
            raise ValueError("Outside the fixed vacuum angle domain")
        negative = theta < 0
        key = abs(theta).hex()
        if key in self.cache:
            r = self.cache[key]
            return WordCertificate(theta.hex(), "fixed_C0_priority", r["sequence"],
                negative, None, (), F(0), r["guard"], CACHE_EMITTER_UPPER,
                r["T"], r["T"] - r["Tdagger"] if negative else r["Tdagger"],
                -r["W"] if negative else r["W"])
        n = round_index(theta)
        assert abs(n) <= self.N and abs(n).bit_length() <= 43
        indices = tuple(j for j in EXPONENTS if abs(n) & (1 << (j + B)))
        selected = [self.basic[j] for j in indices]
        sequence = "".join(r["sequence"] for r in selected)
        delta = abs(F.from_float(theta) - F(n, 2**B))
        q_upper = self.sqrt2_upper * delta / 2
        g_upper = sum((r["guard"] for r in selected), F(0))
        assert q_upper <= self.Q_upper
        total = sum(r["T"] for r in selected)
        td = sum(r["Tdagger"] for r in selected)
        w = sum(r["W"] for r in selected)
        cert = WordCertificate(theta.hex(), "dyadic_composite_certificate",
            sequence, negative, n, indices, q_upper, g_upper, EMITTER_UPPER,
            total, total - td if negative else td, -w if negative else w)
        assert cert.strict_upper < self.epsilon
        return cert

    def emit(self, theta: float, lower_native):
        """Return a fresh 1-qubit circuit and separate composite certificate.

        The sequence-only carrier is an interface to lower_native, NOT a
        backend record or a fictitious pygridsynth spec/strict guard.
        """
        from qiskit import QuantumCircuit
        cert = self.provide(theta)
        qc = QuantumCircuit(1)
        qc.rz(theta, 0)
        key = abs(theta).hex()
        allocation = [dict(role="dictionary", angle=key, exponent=-11)]
        carrier = {(key, -11): dict(sequence=cert.positive_sequence)}
        # lower_native does the whole inverse itself. Never pass the signed
        # sequence here: that would invert a negative word twice.
        return lower_native(qc, allocation, carrier, "dictionary"), cert
