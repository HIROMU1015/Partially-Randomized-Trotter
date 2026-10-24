"""Stage-1 scientific boundaries, fixed recipe expansion and execution receipts.

Nested boundaries overlap. Their counts must NEVER be summed as a job total.
BLAS/NumPy internals stay inside their frozen library function boundary.
"""
from collections import Counter
from dataclasses import replace
from fractions import Fraction as F

from trottertracks.b2d_adapter.adapter import DContext, published_math_sequences
from .evaluation import Settings, build_graph
from .input_package import InputRecipe
from .errors import Stage1Failure

SCHEMA = "b2d_stage1_scope_v2"
STOP = "MANDATORY_STOP_NO_RETRY_NO_RESUME"
OUTPUTS = ("binding.json", "inputs.json", "inputs.npz", "graph.json", "events.jsonl",
           "partial_results.json", "result.json", "status.json", "STOP.json")
OUTPUT_LIMITS = dict(total_bytes=16*1024*1024, json_document_bytes=1024*1024,
                     event_record_bytes=64*1024, files=len(OUTPUTS), stop_reserve_bytes=256*1024)
REQUIRED_DEPENDENCIES = ("numpy", "scipy", "qiskit", "qiskit-nature", "openfermion",
                         "openfermionpyscf", "pyscf", "numba", "matplotlib")


def candidate_graph(source_commit):
    """Unresolved-input TEMPLATE only; no generated identity/hash is claimed."""
    recipe = InputRecipe()
    return build_graph(Settings(
        source_commit=source_commit, fixture_definition_commit=recipe.fixture_commit,
        fixture_definition_blob=recipe.fixture_blob, hamiltonian_id="UNRESOLVED_H",
        sector_id="UNRESOLVED_SECTOR", state_id="UNRESOLVED_STATE",
        input_data_sha256="UNRESOLVED_ARRAYS",
        context=DContext("UNRESOLVED_PREPARATION", "UNRESOLVED_TAIL", ("D0", "D1", "D2"), 2),
        deltas=(F(9,100), F(-9,100)), q_values=(2,3), n_d_values=(1,2), r_rte=2,
        cutoff=2, seed=41, truncation_tolerance=1.0, coefficient_atol=F(0),
        identity_policy="extract_identity_phase", finite_bit_policy="analytic_moments_no_sampling",
        precision="complex128", numerical_atol=1e-12, radius_floor=None,
        branch_margin=None, phase_diagnostics=False,
        decision_policy="descriptive_only_no_scientific_go"))


def exp_key(graph, label, time):
    slots = {name: f"D{i}" for i,name in enumerate(graph.settings.context.block_ids)}
    return (slots.get(label, label), str(time))


def exponential_keys(graph):
    keys = {exp_key(graph, *n.parameters) for n in graph.nodes if n.kind == "exponential"}
    kernel, processor = published_math_sequences()
    for delta in graph.settings.deltas:
        for n in graph.settings.n_d_values:
            for weight in kernel + processor:
                keys.update(exp_key(graph, b, delta * weight / (4*n))
                            for b in graph.settings.context.block_ids)
    return [list(key) for key in sorted(keys)]


def registry(graph):
    from .preflight import validate_stage1_graph, graph_layout
    validate_stage1_graph(graph)
    s = graph.settings
    if s.phase_diagnostics or s.numerical_atol != 1e-12:
        raise Stage1Failure("UNREGISTERED_NUMERIC_POLICY")
    boundaries = dict(
        **{f"dag.{kind}": count for kind,count in graph.call_counts.items()},
        **{"input.fixture":1, "input.split":1, "input.preparation":1,
           "input.sector":1, "input.full":1, "input.block":3, "input.tail":1,
           "input.state":1, "backend.expm":len(exponential_keys(graph)),
           "rte.config":2, "rte.moments":2, "signal.expectation":144})
    return dict(schema=SCHEMA, status="FIXED_RECIPE_BOUNDED_RUNTIME_EXPANSION",
        recipe_fingerprint=InputRecipe().fingerprint, graph_layout=graph_layout(graph),
        dispatch_counts=graph.call_counts, dispatch_total=len(graph.nodes), rows=36,
        boundaries=boundaries, exponential_keys=exponential_keys(graph),
        runtime_domain=dict(n_modes=2, N=1, sector_basis=[1,2], D_blocks=3,
            deterministic_factors=2, residual_factors=1, retained_components=[1,3],
            retained_bases=1, state_occupied_modes=[0]),
        library_expansion=dict(
            preparation="one-body Gaussian(1) + DF Gaussian(3) + residual DF Gaussian(1); frozen prepare_df_partial_s2 closure",
            gaussian_metadata="record each retained basis hash and operation count; no native PF pricing",
            materialization="full H and 3 isolated D: dense_df_operator_in_sector; dimension 2, two matvec columns each",
            tail="one retained basis: basis_change_unitary QuantumCircuit/Operator, diagonal grouping, sector permutation and leakage",
            input_gate="existing DenseBackend._validate: 5 Hermiticity, state norm, H identity, 3 D-pair and 1 DR commutator diagnostics",
            a4="4 independent P K^n P-adjoint references, actual inverse; shared exponential cache",
            rte="make_rte_config and finite_rte_operator_moments, r=2,K=2; Taylor/power/normalization enclosed, no sampling",
            signals="144 common-state expectations; X/Y, radius, direct complex errors; phase disabled"),
        outputs=list(OUTPUTS),
        output_limits=dict(OUTPUT_LIMITS),
        required_dependency_versions=list(REQUIRED_DEPENDENCIES),
        max_event_records=2*sum(boundaries.values())+32,
        retry_allowed=False, mandatory_stop=True,
        exclusions=["controlled_Hadamard_wrapper", "native_compile_transpile", "native_processor_pricing",
                    "shot_sampling_certification", "RPE_total_resources", "precision_or_grid_extension"],
        unresolved_scientific_identities="RESOLVE_AFTER_CLAIM_THEN_CORRECTNESS_GATES")


def validate_registry(scope, graph):
    if scope != registry(graph):
        raise Stage1Failure("SCOPE_MISMATCH")


def resolved_graph(template, context, inputs, provenance):
    settings = replace(template.settings, context=context, hamiltonian_id=inputs.hamiltonian_id,
        sector_id=inputs.sector_id, state_id=inputs.state_id,
        input_data_sha256=provenance["input_data_sha256"])
    graph = build_graph(settings)
    validate_registry(registry(template), graph)
    return graph


class ScopeLedger:
    """Local to one launch; records attempted, completed, failed boundaries."""
    def __init__(self, scope, graph, sink=lambda event: None):
        validate_registry(scope, graph)
        self.scope, self.graph, self.sink = scope, graph, sink
        self.attempted, self.completed, self.failed = Counter(), Counter(), Counter()
        self.seen, self.node_results, self.resolutions = set(), {}, {}
        self.resolution_candidates = {}
        self.events = 0

    def emit(self, event):
        self.events += 1
        if self.events > self.scope["max_event_records"]:
            raise Stage1Failure("LOG_SCOPE_EXCEEDED")
        self.sink(event)

    def call(self, name, function, *, identity=None, summary=None):
        if name not in self.scope["boundaries"] or self.attempted[name] >= self.scope["boundaries"][name]:
            raise Stage1Failure("CALL_SCOPE_EXCEEDED")
        key = (name, identity)
        if identity is not None and key in self.seen:
            raise Stage1Failure("DUPLICATE_CALL_NO_RETRY")
        if name == "backend.expm" and list(identity or ()) not in self.scope["exponential_keys"]:
            # Registry made directly by Python uses tuples, JSON packet uses lists.
            if tuple(identity or ()) not in map(tuple, self.scope["exponential_keys"]):
                raise Stage1Failure("UNREGISTERED_EXPONENTIAL")
        if name == "input.block" and identity not in (0,1,2):
            raise Stage1Failure("UNREGISTERED_D_POSITION")
        if name.startswith("rte.") and identity not in ("9/100", "-9/100"):
            raise Stage1Failure("UNREGISTERED_RTE_TIME")
        if identity is not None:
            self.seen.add(key)
        self.attempted[name] += 1
        self.emit(dict(event="start", boundary=name, identity=identity))
        try:
            value = function()
            payload = summary(value) if summary else None
            self.completed[name] += 1
            self.emit(dict(event="complete", boundary=name, identity=identity, result=payload))
            return value
        except BaseException as error:
            self.failed[name] += 1
            self.emit(dict(event="failed", boundary=name, identity=identity,
                           error_type=type(error).__name__, reason=getattr(error,"code","BOUNDARY_EXCEPTION")))
            raise

    def bind_graph(self, graph):
        validate_registry(self.scope, graph)
        self.graph = graph

    def record_resolution(self, name, value):
        if name in self.resolutions:
            raise Stage1Failure("DUPLICATE_RESOLUTION")
        self.resolution_candidates[name] = value
        self.emit(dict(event="resolution_candidate",name=name,value=value))
        if name == "preparation":
            blocks, selected, residual = value["D_blocks"], value["selected"], value["residual"]
            if (len(blocks) != 3 or blocks[0]["kind"] != "one_body"
                or [b["kind"] for b in blocks[1:]] != ["df_fragment"]*2
                or len(selected) != 2 or len(set(selected)) != 2 or len(residual) != 1
                or set(selected).intersection(residual) or set(selected+residual) != {0,1,2}
                or [b["factor_index"] for b in blocks[1:]] != selected):
                raise Stage1Failure("INPUT_SPLIT_ORDER_MISMATCH")
            components, bases = value["tail_components"], value["tail_bases"]
            if not 1 <= len(components) <= 3 or len(bases) != 1:
                raise Stage1Failure("INCONCLUSIVE_RESIDUAL_DOMAIN")
            if any(c["fragment"] != f"df-fragment-{residual[0]}" or
                   c["basis_id"] != bases[0]["basis_id"] or
                   tuple(c["support"]) not in ((0,), (1,), (0,1)) for c in components):
                raise Stage1Failure("RETAINED_COMPONENT_DOMAIN_MISMATCH")
            if len({tuple(c["support"]) for c in components}) != len(components):
                raise Stage1Failure("DUPLICATE_RETAINED_COMPONENT")
        elif name == "sector":
            if value != dict(n_modes=2, N=1, basis_indices=[1,2], occupied_modes=[0]):
                raise Stage1Failure("SECTOR_BASIS_STATE_MISMATCH")
        elif name != "input":
            raise Stage1Failure("UNREGISTERED_RESOLUTION")
        self.resolutions[name] = value
        self.emit(dict(event="resolution", name=name, value=value))

    def dispatch(self, index, node, function):
        if index < 0 or index >= len(self.graph.nodes) or self.graph.nodes[index] != node:
            raise Stage1Failure("UNREGISTERED_DAG_NODE")
        gate_indices = set(self.graph.correctness.values())
        comparison = node.kind in ("operator_error", "signals")
        if comparison and not gate_indices.issubset(self.node_results):
            raise Stage1Failure("COMPARISON_BEFORE_CORRECTNESS")
        kinds = {"validate_input", "a4_reference_check", "operator_error", "signals", "normalization"}
        value = self.call("dag."+node.kind, function, identity=index,
                          summary=(lambda v:v) if node.kind in kinds else None)
        self.node_results[index] = value if node.kind in kinds else None
        return value

    def partial_rows(self):
        rows = []
        for row in self.graph.rows:
            outputs = row["outputs"]
            if all(i is None or i in self.node_results for i in outputs.values()):
                rows.append({**row, "outputs":{k:None if i is None else self.node_results[i]
                                              for k,i in outputs.items()}})
        return rows

    def snapshot(self):
        return dict(attempted=dict(self.attempted), completed=dict(self.completed), failed=dict(self.failed),
                    resolutions=self.resolutions, resolution_candidates=self.resolution_candidates,
                    event_records=self.events)

    def finish(self):
        if dict(self.completed) != self.scope["boundaries"] or self.failed:
            raise Stage1Failure("INCOMPLETE_REGISTERED_SCOPE")
        if set(self.resolutions) != {"preparation", "sector", "input"}:
            raise Stage1Failure("MISSING_INPUT_RESOLUTION")
