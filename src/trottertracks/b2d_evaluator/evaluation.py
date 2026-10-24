"""Shared, memoized evaluation DAG and approval boundary; stdlib only.

    build_graph: metadata only
    evaluate: separate explicit authorization, then injected backend calls

No default scientific grid, input state, tolerances, approval or grant exists.
This module does not issue/consume a grant, create a fixture, or write results.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math

from trottertracks.b2d_adapter.adapter import DContext, TailOccurrence
from .plans import ARMS, make_arm_plan

SCHEMA = "b2d_five_arm_evaluation_v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Settings:
    source_commit: str
    fixture_definition_commit: str
    fixture_definition_blob: str
    hamiltonian_id: str
    sector_id: str
    state_id: str
    input_data_sha256: str
    context: DContext
    deltas: tuple[Fraction, ...]
    q_values: tuple[int, ...]
    n_d_values: tuple[int, ...]
    r_rte: int
    cutoff: int
    seed: int
    truncation_tolerance: float
    coefficient_atol: Fraction
    identity_policy: str
    finite_bit_policy: str
    precision: str
    numerical_atol: float | None
    radius_floor: float | None
    branch_margin: float | None
    decision_policy: str

    def validate(self):
        for name in ("source_commit", "fixture_definition_commit", "fixture_definition_blob",
                     "hamiltonian_id", "sector_id", "state_id", "input_data_sha256",
                     "decision_policy"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("Unresolved scientific condition: " + name)
        if not isinstance(self.context, DContext) or self.context.controlled:
            raise ValueError("Stage 1 uses system operators, no controlled circuit.")
        reserved = {"exact_D", "exact_R", "exact_physical", "physical", "tail", "state"}
        if reserved.intersection(self.context.block_ids):
            raise ValueError("D block labels collide with reference identifiers.")
        for name in ("deltas", "q_values", "n_d_values"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values or len(set(values)) != len(values):
                raise ValueError("An explicit unique tuple is required: " + name)
        if any(not isinstance(d, Fraction) or d == 0 for d in self.deltas):
            raise ValueError("Explicit nonzero signed Fraction deltas required.")
        for value in (*self.q_values, *self.n_d_values, self.r_rte):
            if type(value) is not int or value < 1:
                raise ValueError("Positive integer q/n_D/r_RTE required.")
        if type(self.cutoff) is not int or self.cutoff < 0 or self.cutoff % 2:
            raise ValueError("Explicit nonnegative even Taylor cutoff required.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Explicit nonnegative metadata seed required.")
        if self.coefficient_atol != 0 or self.identity_policy != "extract_identity_phase":
            raise ValueError("This minimal stage-1 path requires no coefficient dropping.")
        if self.precision != "complex128" or self.finite_bit_policy != "analytic_moments_no_sampling":
            raise ValueError("Only the explicit analytic-moment complex128 path is prepared.")
        for name in ("numerical_atol", "radius_floor", "branch_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or value is None or not math.isfinite(value) or value <= 0:
                raise ValueError("Unresolved positive numerical condition: " + name)
        if not math.isfinite(self.truncation_tolerance) or self.truncation_tolerance <= 0:
            raise ValueError("Explicit finite positive RTE tolerance metadata required.")
        if self.branch_margin >= math.pi:
            raise ValueError("Branch margin must be smaller than pi.")
        if self.decision_policy != "descriptive_only_no_scientific_go":
            raise ValueError("No scientific GO threshold is implemented or adopted.")

    @property
    def fingerprint(self):
        return hashlib.sha256(_json(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class Node:
    kind: str
    dependencies: tuple[int, ...]
    parameters: tuple


@dataclass(frozen=True)
class Graph:
    settings: Settings
    nodes: tuple[Node, ...]
    rows: tuple[dict, ...]
    correctness: dict

    @property
    def call_counts(self):
        return dict(sorted(Counter(n.kind for n in self.nodes).items()))

    @property
    def fingerprint(self):
        payload = (self.settings.fingerprint, [asdict(n) for n in self.nodes], self.rows, self.correctness)
        return hashlib.sha256(_json(payload).encode()).hexdigest()


def build_graph(settings: Settings) -> Graph:
    """Compile all registered rows and shared references without any science."""
    settings.validate()
    nodes, lookup, rows = [], {}, []

    def add(kind, dependencies=(), parameters=()):
        node = Node(kind, tuple(dependencies), tuple(parameters))
        if node not in lookup:
            lookup[node] = len(nodes)
            nodes.append(node)
        return lookup[node]

    def exponential(label, time):
        return add("exponential", parameters=(label, time))

    def product_append(operations):
        return add("product_append", operations)

    def word(ops):
        return product_append(exponential(op.block_id, op.evolution_time) for op in ops)

    def power(matrix, q):
        return add("power", (matrix,), (q,))

    def phased(matrix, time):
        return add("physical_scalar_phase", (matrix,), (time,))

    def error(a, b):
        return add("operator_error", (a, b))

    correctness = {"input_validation": add("validate_input")}
    for delta in settings.deltas:
        tail = TailOccurrence(settings.context, "common-tail", delta,
                              "existing_finite_RTE", "retain_event_phase", "retain_normalization")
        half_ref = exponential("exact_D", delta / 2)
        d_ref = exponential("exact_D", delta)
        tail_ref = exponential("exact_R", delta)
        moments = add("rte_moments", parameters=(delta,))
        corrected_tail = add("moment_field", (moments,), ("corrected_operator",))
        raw_tail = add("moment_field", (moments,), ("attenuated_event_mean_operator",))
        outer_step = phased(product_append((half_ref, tail_ref, half_ref)), delta)
        # Put numerical mathematical-order gates before full arm evaluations.
        for n_d in settings.n_d_values:
            a4 = make_arm_plan("A4", settings.context, delta, n_d, tail)
            correctness[f"A4:{delta}:{n_d}"] = add(
                "a4_reference_check", (word(a4.before_tail),), (delta / 2, n_d)
            )
        registrations = [("A0", None)] + [(a, n) for a in ARMS[1:] for n in settings.n_d_values]
        for arm, n_d in registrations:
            plan = make_arm_plan(arm, settings.context, delta, n_d, tail)
            before, after = word(plan.before_tail), word(plan.after_tail)
            d_pair_error = error(product_append((before, after)), d_ref)
            d_half_error = None if arm == "A0" else error(before, half_ref)
            ideal_step = phased(product_append((before, tail_ref, after)), delta)
            corrected_step = phased(product_append((before, corrected_tail, after)), delta)
            raw_step = phased(product_append((before, raw_tail, after)), delta)
            for q in settings.q_values:
                physical = exponential("exact_physical", delta * q)
                outer = power(outer_step, q)
                ideal, corrected, raw = (power(x, q) for x in (ideal_step, corrected_step, raw_step))
                normalization = add("normalization", (moments,), (q,))
                rescaled = add("rescale_raw", (raw, normalization))
                rows.append({
                    "arm": arm, "family": plan.family, "coefficient_source": plan.coefficient_source,
                    "delta": str(delta), "q": q, "T": str(delta * q), "n_D": n_d,
                    "input_fingerprint": settings.fingerprint, "tail_identity": settings.context.tail_id,
                    "outputs": {
                        "D_half_operator_error": d_half_error,
                        "D_pair_operator_error": d_pair_error,
                        "outer_operator_error": error(outer, physical),
                        "ideal_tail_physical_error": error(ideal, physical),
                        "corrected_physical_error": error(corrected, physical),
                        "finite_RTE_error": error(corrected, ideal),
                        "normalization_consistency": error(rescaled, corrected),
                        "normalization": normalization,
                        "signals": add("signals", (physical, ideal, corrected, raw)),
                    },
                })
    return Graph(settings, tuple(nodes), tuple(rows), correctness)


@dataclass(frozen=True)
class ExecutionApproval:
    approval_reference: str
    grant_reference: str
    source_commit: str
    graph_fingerprint: str
    approved_call_counts: tuple[tuple[str, int], ...]
    output_identity: str
    retry_policy: str


def _check_approval(graph, approval):
    if not isinstance(approval, ExecutionApproval):
        raise PermissionError("Separate explicit science approval and fresh grant are required.")
    if not all((approval.approval_reference, approval.grant_reference,
                approval.output_identity, approval.retry_policy)):
        raise PermissionError("Incomplete science approval metadata.")
    if approval.source_commit != graph.settings.source_commit or approval.graph_fingerprint != graph.fingerprint:
        raise PermissionError("Science source or registered grid differs from approval.")
    if dict(approval.approved_call_counts) != graph.call_counts:
        raise PermissionError("Scientific call scope differs from approval.")


def evaluate(graph: Graph, backend_factory, approval: ExecutionApproval | None = None):
    """Dormant science entry: reject missing approval before backend creation.

    Approval fields are caller-supplied attestations, not grant issuance or
    verification of external ledger state. A future launcher must validate
    source/input/unused-grant/fresh-output and ledger policy before calling.
    """
    graph.settings.validate()
    _check_approval(graph, approval)
    backend = backend_factory(graph.settings)
    values = []
    for node in graph.nodes:
        args = tuple(values[i] for i in node.dependencies)
        values.append(backend.perform(node.kind, args, node.parameters))
    return {
        "schema": SCHEMA,
        "mode": "MOCK_ONLY" if backend.is_mock else "SCIENCE_STAGE1",
        "source_commit": graph.settings.source_commit,
        "settings": json.loads(_json(asdict(graph.settings))),
        "input_fingerprint": graph.settings.fingerprint,
        "graph_fingerprint": graph.fingerprint,
        "call_counts": graph.call_counts,
        "approval_reference": approval.approval_reference,
        "grant_reference": approval.grant_reference,
        "output_identity": approval.output_identity,
        "correctness": {key: values[node] for key, node in graph.correctness.items()},
        "rows": [{**row, "outputs": {k: None if v is None else values[v]
                                    for k, v in row["outputs"].items()}} for row in graph.rows],
        "decision": "DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO",
        "native_compile_shots": "OUT_OF_SCOPE_STAGE2",
        "stop": "MANDATORY_STOP_NO_AUTOMATIC_EXTENSION",
    }
