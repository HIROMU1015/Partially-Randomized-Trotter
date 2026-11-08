"""B1 fixture-only overlay of ONE existing strict record.

No default dictionary/cache mutation, candidate selection, guard, or synthesis.
Backend provenance remains distinct from the consumer's -11 transport key.
"""
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType

from .pr_reference_dyadic_dictionary import actual_adjoint, CACHE_EMITTER_UPPER


@dataclass(frozen=True)
class SavedRecordCertificate:
    theta_hex: str
    positive_sequence: str
    negative: bool
    record_id: str
    backend_spec: str
    precision_exponent: int
    sequence_sha256: str
    saved_guard: str
    guard_method: str
    positive_T_quoted: int
    positive_Tdagger_quoted: int
    positive_W_quoted: int

    def signed_sequence(self):
        return actual_adjoint(self.positive_sequence) if self.negative else self.positive_sequence

    def summary(self):
        return dict(theta_hex=self.theta_hex, route="single_saved_S2_fixture_overlay",
            record_id=self.record_id, backend_spec=self.backend_spec,
            backend_precision_exponent=self.precision_exponent,
            positive_sequence_sha256=self.sequence_sha256,
            saved_strict_guard=self.saved_guard, guard_method=self.guard_method,
            positive_record_T_quoted=self.positive_T_quoted,
            positive_record_Tdagger_quoted=self.positive_Tdagger_quoted,
            positive_record_W_quoted=self.positive_W_quoted,
            negative_whole_adjoint=self.negative,
            emitter_allowance_from_existing_SOURCE=str(CACHE_EMITTER_UPPER),
            combined_strict_guard_newly_evaluated=False,
            transport_exponent=-11, transport_is_backend_spec=False)


class SingleSavedRecordOverlay:
    """Serve only +/- the configured binary64 literal; no arm/role selector."""

    def __init__(self, record):
        r = record
        theta = float.fromhex(r["theta_hex"])
        expected_spec = r["theta_hex"] + ":10^-13:pygridsynth2.0.0:strict:seed17"
        if (not math.isfinite(theta) or theta <= 0 or theta.hex() != r["theta_hex"]
                or r["status"] != "PASS" or r["precision_exponent"] != -13
                or r["spec"] != expected_spec
                or hashlib.sha256(r["spec"].encode()).hexdigest() != r["key"]
                or r["guard_method"] != "exact cyclotomic word matrix + outward interval strict Frobenius"
                or not isinstance(r["strict_error_upper"], str) or not r["strict_error_upper"]
                or not isinstance(r["sequence"], str)
                or set(r["sequence"]) - set("HTtSXW")
                or hashlib.sha256(r["sequence"].encode()).hexdigest() != r["sequence_sha256"]
                or r["sequence"].count("W") != r["global_W"]):
            raise ValueError("Saved record provenance/specification mismatch")
        # Only immutable values; modifying the caller's dict cannot change policy.
        self._record = MappingProxyType({k: r[k] for k in (
            "theta_hex", "sequence", "key", "spec", "precision_exponent",
            "sequence_sha256", "strict_error_upper", "guard_method", "T", "Tdagger", "global_W")})

    def provide(self, theta):
        if (not isinstance(theta, float) or not math.isfinite(theta)
                or abs(theta).hex() != self._record["theta_hex"]):
            raise ValueError("Outside the single fixed B1 literal")
        r = self._record
        return SavedRecordCertificate(theta.hex(), r["sequence"], theta < 0,
            r["key"], r["spec"], r["precision_exponent"], r["sequence_sha256"],
            r["strict_error_upper"], r["guard_method"], r["T"], r["Tdagger"], r["global_W"])

    def emit(self, theta, lower_native):
        from qiskit import QuantumCircuit
        cert = self.provide(theta)
        native = QuantumCircuit(1)
        native.rz(theta, 0)
        key = abs(theta).hex()
        # Always positive: unchanged lower_native performs the whole inverse once.
        return lower_native(native,
            [dict(role="B1_fixture", angle=key, exponent=-11)],
            {(key, -11): dict(sequence=cert.positive_sequence)}, "B1_fixture"), cert
