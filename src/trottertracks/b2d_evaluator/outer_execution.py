"""Fixed 16-row outer SOURCE/scope/input binding, reusing Stage-1 execution IO.

Graph/registry functions are metadata-only. Chat permission is an operational
requirement for Codex; this module does not authenticate it.
"""
from dataclasses import asdict, replace
from fractions import Fraction as F
from pathlib import Path
import re

from . import scope as stage1
from .environment import inspect_environment
from .input_package import InputRecipe, fingerprint, read_fixture_definition
from .preflight import RunContext, _check_source
from .outer import METHODS, OuterCase, build_outer_graph

OUTPUT_PARENT = "artifacts/research_b_outer_splitting"


def fixed_cases():
    return tuple(OuterCase(method, time, 2, arm, 1, 2 if method == "refined-S2" else None)
                 for arm in ("A2", "A3") for time in (F(9, 50), F(-9, 50)) for method in METHODS)


def candidate_graph(source_commit, *, settings=None):
    if settings is None:
        settings = replace(stage1.candidate_graph(source_commit).settings,
                           q_values=(2,), n_d_values=(1,))
    return build_outer_graph(settings, fixed_cases(), independent_occurrences=True)


def _conditions(settings):
    result = asdict(settings)
    for key in ("source_commit", "hamiltonian_id", "sector_id", "state_id",
                "input_data_sha256", "context"):
        result.pop(key)
    return result


def graph_layout(graph):
    """Normalize unresolved input slots only; full input binding is separate."""
    slots = {label: f"D_SLOT_{i}" for i, label in enumerate(graph.settings.context.block_ids)}
    def normalized(value):
        if isinstance(value, dict):
            return {k: ("BOUND_INPUT_CONTEXT" if k == "context" else normalized(v))
                    for k, v in value.items() if k not in
                    ("scalar_phases", "input_fingerprint", "tail_identity")}
        if isinstance(value, (list, tuple)):
            return [normalized(v) for v in value]
        return value
    nodes = []
    for node in graph.nodes:
        record = asdict(node)
        if node.kind == "exponential":
            label, time = node.parameters
            record["parameters"] = (slots.get(label, label), time)
        if node.kind in ("outer_rte_moments", "outer_rte_check", "outer_taylor_bound") and len(node.parameters) == 2:
            spec, binding = record["parameters"]
            record["parameters"] = (spec, ("BOUND_TAIL", "BOUND_H", "BOUND_ARRAYS", *binding[3:]))
        nodes.append(record)
    return fingerprint(normalized(dict(settings=_conditions(graph.settings),
        nodes=nodes, rows=graph.rows, correctness=graph.correctness)))


def registry(graph, *, graph_factory=candidate_graph, schema="b2d_outer_stage1_scope_v1"):
    graph.settings.validate()
    template = graph_factory(graph.settings.source_commit)
    if (_conditions(graph.settings) != _conditions(template.settings)
            or len(graph.settings.context.block_ids) != 3
            or graph.settings.context.num_system_qubits != 2):
        raise stage1.Stage1Failure("OUTER_FIXED_CONDITIONS_MISMATCH")
    if graph != graph_factory(graph.settings.source_commit, settings=graph.settings):
        raise stage1.Stage1Failure("OUTER_GRAPH_LAYOUT_MISMATCH")
    keys = sorted({stage1.exp_key(graph, *n.parameters) for n in graph.nodes if n.kind == "exponential"})
    specs = [str(n.parameters[0]) for n in graph.nodes if n.kind == "outer_rte_moments"]
    boundaries = {**{"dag." + k: v for k, v in graph.call_counts.items()},
                  **stage1.INPUT_BOUNDARIES, "backend.expm":len(keys),
                  "rte.config":len(specs), "rte.moments":len(specs),
                  "signal.expectation":5 * len(graph.rows)}
    return dict(schema=schema, graph_layout=graph_layout(graph),
        recipe_fingerprint=InputRecipe().fingerprint, rows=len(graph.rows),
        dispatch_counts=graph.call_counts, dispatch_total=len(graph.nodes), boundaries=boundaries,
        exponential_keys=[list(k) for k in keys], rte_stage_specs=specs,
        outputs=list(stage1.OUTPUTS), output_limits=dict(stage1.OUTPUT_LIMITS),
        max_event_records=2 * sum(boundaries.values()) + 32,
        input_domain=dict(n_modes=2, N=1, basis=[1,2], occupied_modes=[0], D_blocks=3,
            selected_DF_factors=2, residual_DF_factors=1, retained_components=[1,3], retained_bases=1),
        input_expansion="Existing 10 boundaries; Gaussian preparation and tail basis/sector conversion enclosed; no native wrapper compilation",
        numeric_expansion="Independent D/outer products, normalization gates, norms and diagnostics enclosed by DAG nodes; exp cache shared",
        sampling="ANALYTIC_MOMENTS_INDEPENDENT_OCCURRENCES_NO_SAMPLING",
        retry_allowed=False, resume_allowed=False, mandatory_stop=True)


def validate_registry(scope, graph):
    if scope != registry(graph):
        raise stage1.Stage1Failure("OUTER_SCOPE_MISMATCH")


def resolved_graph(template, context, inputs, provenance, *, graph_factory=candidate_graph,
                   registry_factory=registry, validator=validate_registry):
    settings = replace(template.settings, context=context, hamiltonian_id=inputs.hamiltonian_id,
        sector_id=inputs.sector_id, state_id=inputs.state_id,
        input_data_sha256=provenance["input_data_sha256"])
    graph = graph_factory(settings.source_commit, settings=settings)
    validator(registry_factory(template), graph)
    return graph


class OuterScopeLedger(stage1.ScopeLedger):
    validate_registry = staticmethod(validate_registry)
    comparison_kinds = frozenset(("operator_error", "outer_signals"))
    result_kinds = stage1.ScopeLedger.result_kinds | frozenset(("outer_signals", "outer_normalization",
        "outer_diagnostics", "outer_d_check", "outer_order_check", "outer_rescale_check"))

    def check_rte_identity(self, identity):
        if identity not in self.scope["rte_stage_specs"]:
            raise stage1.Stage1Failure("UNREGISTERED_RTE_STAGE_SPEC")


class OuterRunContext(RunContext):
    graph_factory = staticmethod(candidate_graph)
    registry_validator = staticmethod(validate_registry)

    def check_settings(self, settings):
        self.check_input(self.source_root, InputRecipe())
        if self._input_binding != (settings.context, settings.hamiltonian_id,
                settings.sector_id, settings.state_id, settings.input_data_sha256):
            raise stage1.Stage1Failure("INPUT_IDENTITY_ORDER_MISMATCH")
        if settings.source_commit != self.source_commit:
            raise stage1.Stage1Failure("SOURCE_MISMATCH")
        graph = self.graph_factory(self.source_commit, settings=settings)
        self.registry_validator(self.call_scope, graph)
        if self.scope is not None:
            self.scope.bind_graph(graph)

    def begin_evaluation(self, graph):
        self.registry_validator(self.call_scope, graph)
        super().begin_evaluation(graph)


def verify_execution(source_root, output, source_commit, *, graph=None, call_scope=None,
                     graph_factory=candidate_graph, registry_factory=registry,
                     validator=validate_registry, context_type=OuterRunContext,
                     output_parent=OUTPUT_PARENT):
    root = Path(source_root).resolve()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("Full scientific SOURCE commit required")
    _check_source(root, source_commit, output=output)
    read_fixture_definition(root)
    template = graph_factory(source_commit)
    graph = template if graph is None else graph
    scope = registry_factory(template) if call_scope is None else call_scope
    if graph.settings.source_commit != source_commit:
        raise stage1.Stage1Failure("SOURCE_MISMATCH")
    validator(scope, graph)
    environment = inspect_environment()
    out, parent = Path(output).resolve(), root / output_parent
    if not out.is_relative_to(parent) or out == parent or out.exists():
        raise ValueError("Fresh dedicated outer output required")
    return context_type(root, out, source_commit, scope, environment)
