"""Fixed 22-row analytic outer/RTE study. Building scope runs metadata only."""
from dataclasses import replace
from fractions import Fraction as F
import math

from . import outer_execution as previous
from .outer import OuterCase, build_outer_graph
from .scope import Stage1Failure

OUTPUT_PARENT = "artifacts/research_b_outer_rte22"
# ID, method, inner, q, f, r (time subdivisions), K (even Taylor cutoff).
CANDIDATES = (
    ("B1", "outer-S2", "A2", 2, None, 2, 2),
    ("B2", "refined-S2", "A2", 2, 2, 2, 2),
    ("B3", "outer-Yoshida4", "A2", 2, None, 2, 2),
    ("B4", "SABA2", "A2", 2, None, 2, 2),
    ("D1", "outer-Yoshida4", "A3", 2, None, 2, 2),
    ("R1", "outer-Yoshida4", "A2", 2, None, 4, 2),
    ("R2", "outer-Yoshida4", "A3", 2, None, 4, 2),
    ("R3", "outer-Yoshida4", "A2", 2, None, 2, 4),
    ("R4", "outer-Yoshida4", "A3", 2, None, 2, 4),
    ("S1", "SABA2", "A2", 4, None, 2, 2),
    ("F1", "refined-S2", "A2", 2, 8, 2, 2),
)


def candidate_graph(source_commit, *, settings=None):
    if settings is None:
        settings = replace(previous.candidate_graph(source_commit).settings,
            q_values=(2, 4), deltas=(F(9,100), F(-9,100), F(9,200), F(-9,200)))
    cases, parameters, ids = [], [], []
    for time in (F(9,50), F(-9,50)):
        for name, method, arm, q, f, r, k in CANDIDATES:
            cases.append(OuterCase(method, time, q, arm, 1, f))
            parameters.append((r, k)); ids.append(name)
    return build_outer_graph(settings, cases, independent_occurrences=True,
        rte_parameters=parameters, candidate_ids=ids, analytic_diagnostics=True)


def budget_metadata():
    return dict(epsilons=[1e-8, 1e-9], target="physical_complex_coherent_signal_absolute_error",
        bias_plus_numerical_fraction=0.5, implementation_fraction=0.1, statistical_fraction=0.4,
        confidence=0.95, alpha=0.05, X_alpha=0.025, Y_alpha=0.025,
        allocation="equal_union_bound_X_Y_per_candidate_and_T",
        familywise_status="UNALLOCATED_NO_SIMULTANEOUS_CLAIM", u_num=None,
        numerical_status="UNQUALIFIED_COMPLEX128_ATOL_IS_NOT_UNCERTAINTY",
        implementation_gamma=None, statistical_error=None, shots=None,
        end_to_end_status="NOT_CERTIFIED_NO_TOTAL_RESOURCE_CLAIM")


def bias_qualification(signals):
    """Describe analytic bias; no floating error certificate is available here."""
    bias = signals["corrected_physical_signal_error"]
    if not math.isfinite(bias) or bias < 0:
        raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
    direct_errors = {}
    physical = signals["physical"]
    for name in ("exact_outer", "ideal_tail", "corrected", "raw"):
        record = signals[name]
        direct_errors[name] = dict(X=abs(record["X"] - physical["X"]),
            Y=abs(record["Y"] - physical["Y"]), radius=abs(record["radius"] - physical["radius"]),
            complex_signal=signals[name + "_physical_signal_error"])
    corrected, outer = signals["corrected"], signals["exact_outer"]
    return dict(observed_bias=bias, u_num=None, physical_signal_errors=direct_errors,
        corrected_outer_signal_error=abs(complex(corrected["X"], corrected["Y"]) - complex(outer["X"], outer["Y"])),
        targets=[dict(epsilon=e, bias_plus_numerical_budget=e/2,
            implementation_budget=e/10, statistical_budget=2*e/5,
            observed_bias_within_budget=bias <= e/2,
            deterministic_qualified=None, status="UNRESOLVED_NUMERICAL_UNCERTAINTY")
            for e in budget_metadata()["epsilons"]],
        end_to_end_status="NOT_CERTIFIED_NO_TOTAL_RESOURCE_CLAIM")


def registry(graph):
    scope = previous.registry(graph, graph_factory=candidate_graph, schema="b2d_outer_rte22_scope_v1")
    scope["boundaries"].update({"rte.truncation": graph.call_counts["outer_taylor_bound"],
                                "bound.compose": graph.call_counts["outer_taylor_total"]})
    scope["max_event_records"] = 2 * sum(scope["boundaries"].values()) + 32
    scope["error_budget"] = budget_metadata()
    scope["rte_identity_policy"] = "tail/H/array identity, signed duration, r, K, tolerance, seed, precision and analytic law"
    scope["taylor_bound_assumptions"] = (
        "Hermitian normalized tail is a convex sum of signed involutions, norm<=1; "
        "finite paired Taylor through K+1; independent ordered R occurrences and unitary D factors; "
        "nominal analytic bound excludes floating-point uncertainty")
    return scope


def validate_registry(scope, graph):
    if scope != registry(graph):
        raise Stage1Failure("OUTER_RTE22_SCOPE_MISMATCH")


def resolved_graph(template, context, inputs, provenance):
    return previous.resolved_graph(template, context, inputs, provenance,
        graph_factory=candidate_graph, registry_factory=registry, validator=validate_registry)


class OuterRTE22ScopeLedger(previous.OuterScopeLedger):
    validate_registry = staticmethod(validate_registry)
    comparison_kinds = previous.OuterScopeLedger.comparison_kinds | frozenset(("outer_bias_qualification",))
    result_kinds = previous.OuterScopeLedger.result_kinds | frozenset((
        "outer_tail_bound_check", "outer_rte_check", "outer_rte_records", "outer_taylor_bound",
        "outer_taylor_total", "outer_bias_qualification"))


class OuterRTE22RunContext(previous.OuterRunContext):
    graph_factory = staticmethod(candidate_graph)
    registry_validator = staticmethod(validate_registry)


def verify_execution(source_root, output, source_commit, *, graph=None, call_scope=None):
    return previous.verify_execution(source_root, output, source_commit, graph=graph,
        call_scope=call_scope, graph_factory=candidate_graph, registry_factory=registry,
        validator=validate_registry, context_type=OuterRTE22RunContext, output_parent=OUTPUT_PARENT)
