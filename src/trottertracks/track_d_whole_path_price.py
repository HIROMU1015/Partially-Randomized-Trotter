"""Opt-in saved whole-path requests and qualified finite-fixture prices.

No sampler, solver, compiler or scientific execution is called on import.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

from . import track_d_c3_whole_path as c3

SCIENCE_SOURCE = "ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7"
CARRIER = "3649b75af191916daf44abb91e14d34c43b83e8f"
LAW = "c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5"
INPUT_PACKET = "1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db"
SCOPE = "single_hadamard_interrogation_without_state_preparation"
OPTIMIZATION = "strong_whole_path_half_sweep_relative_basis_scalar_aggregation_v1"
METHODS = ("symmetric_directional", "reference")
AXES = ("X", "Y")
METRICS = ("rz_count", "cx_count", "rz_depth", "cx_depth", "total_depth", "circuit_size")


@dataclass(frozen=True)
class OrderedRequest:
    q: int
    ordered_samples: tuple[int, ...]
    cut_after_complete_step: int
    science_source: str = SCIENCE_SOURCE
    carrier: str = CARRIER
    endpoint: str = c3.paired.ENDPOINT
    cohort: str = c3.paired.COHORT
    cohort_source: str = c3.paired.COHORT_SOURCE
    law_sha256: str = LAW
    input_packet_sha256: str = INPUT_PACKET

    def __post_init__(self):
        if type(self.q) is not int or self.q < 2:
            raise ValueError("ORDERED_Q")
        object.__setattr__(self, "ordered_samples", tuple(self.ordered_samples))
        if len(self.ordered_samples) != self.q or any(type(x) is not int or not 0 <= x < 32 for x in self.ordered_samples):
            raise ValueError("ORDERED_SAMPLE_COUNT_OR_ID")
        if type(self.cut_after_complete_step) is not int or not 0 < self.cut_after_complete_step < self.q:
            raise ValueError("COMPLETE_INTERNAL_STEP_CUT")
        if (self.science_source != SCIENCE_SOURCE or self.carrier != CARRIER
                or self.endpoint != c3.paired.ENDPOINT or self.cohort != c3.paired.COHORT
                or self.cohort_source != c3.paired.COHORT_SOURCE or self.law_sha256 != LAW
                or self.input_packet_sha256 != INPUT_PACKET):
            raise ValueError("SOURCE_ENDPOINT_COHORT_LAW_BINDING")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(context, saved, request):
    if not isinstance(request, OrderedRequest):
        raise TypeError("EXPLICIT_ORDERED_REQUEST")
    model = context.arm.model
    if (model.source_identity != request.endpoint or model.input_identity != request.input_packet_sha256
            or model.modes != c3.base.N):
        raise ValueError("CONTEXT_SOURCE_ENDPOINT_INPUT")
    bundle = c3.prepare_ordered(context, saved, request.ordered_samples,
        request.cut_after_complete_step,
        fixture_selection="explicit saved ordered request; fixed cut; not fresh-IID law sample")
    bundle.identity.update(request=asdict(request), request_fingerprint=request.fingerprint,
                           optimization_policy=OPTIMIZATION)
    return bundle


def wrapper_for(bundle, request, *, method, axis):
    if method not in METHODS or axis not in AXES:
        raise ValueError("METHOD_AXIS")
    if (bundle.identity.get("request_fingerprint") != request.fingerprint
            or bundle.identity["ordered_samples"] != list(request.ordered_samples)
            or bundle.identity["cut_after_complete_step"] != request.cut_after_complete_step):
        raise ValueError("SAME_ORDERED_REQUEST_REQUIRED")
    return bundle.wrappers[method, axis]


def price_record(request, *, method, axis, postcheck, raw_status, paired_status,
                 execution_source, compiler, cost_scope=SCOPE):
    """C5 prices only its qualified fixed fixture; no q/law extrapolation."""
    if (request.q, request.ordered_samples, request.cut_after_complete_step) != (2, (0, 1), 1):
        raise ValueError("UNVALIDATED_PRICE_FIXTURE_OR_Q")
    return _checked_price(request, method=method, axis=axis, postcheck=postcheck,
        raw_status=raw_status, paired_status=paired_status, execution_source=execution_source,
        compiler=compiler, cost_scope=cost_scope)


def _checked_price(request, *, method, axis, postcheck, raw_status, paired_status,
                   execution_source, compiler, cost_scope=SCOPE):
    """Shared full-wrapper validation; entry points retain their own domains."""
    if method not in METHODS or axis not in AXES:
        raise ValueError("METHOD_AXIS")
    if cost_scope != SCOPE:
        raise ValueError("FULL_WRAPPER_NOT_KERNEL_SHELL")
    if (raw_status != "PASS" or paired_status != "PASS" or postcheck.get("status") != "PASS"
            or postcheck.get("arm") != method or postcheck.get("axis") != axis
            or postcheck.get("global_phase_fit") is not False):
        raise ValueError("UNQUALIFIED_PRICE")
    expected = dict(qiskit="1.3.0", optimization_level=1, seed_transpiler=17,
        basis_gates=["rz", "sx", "x", "cx"], approximation_degree=1.0,
        qubits_initially_zero=False)
    if compiler != expected:
        raise ValueError("P0_PROFILE")
    if len(execution_source) != 40 or any(c not in "0123456789abcdef" for c in execution_source):
        raise ValueError("EXECUTION_SOURCE")
    metrics = postcheck["metrics"]
    if any(type(metrics[k]) is not int or metrics[k] < 0 for k in (*METRICS, "qubit_count")):
        raise ValueError("FINITE_INTEGER_CIRCUIT_METRICS")
    if metrics['qubit_count'] != (17 if method == 'reference' else 9):
        raise ValueError("FIXED_H4_WRAPPER_WIDTH")
    return dict(estimate_kind="finite_fixture_compiled", method=method, axis=axis,
        q=request.q, request=asdict(request), request_fingerprint=request.fingerprint,
        execution_source=execution_source, optimization_policy=OPTIMIZATION,
        compiler=compiler, topology="free", precision=dict(component_abs=1e-10,
            observable_frobenius=2e-10, rtol=0, global_phase_fit=False),
        cost_scope=SCOPE, depth_semantics="compiled_full_wrapper_DAG",
        includes=["whole ordered path", "legal step fusion", "scalar/identity once per step",
                  "ancilla H/readout/measurement", "Reference register/CSWAP/actual inverse/chi/cut loss when applicable"],
        state_preparation="EXCLUDED", metrics={k: metrics[k] for k in METRICS},
        peak_qubits=metrics["qubit_count"], width_scope="compiled logical wrapper only",
        qualification=dict(raw="PASS", postcheck="PASS", paired_XY_phase="PASS",
            scope="fixed request; reference/ancilla zero; all256 system inputs; local binary64",
            all_law=False, long_q=False), population_expectation=None,
        confidence_interval=None, model_estimate=None, routing_cost=None,
        RPE_total=None)
