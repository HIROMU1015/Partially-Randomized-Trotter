"""Scientific SOURCE, fixed recipe/grid and fresh-output checks; no approval inputs."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess

from .input_package import InputRecipe, read_fixture_definition, fingerprint


def graph_layout(graph):
    """Stable pre-generation graph identity; not a resolved-input fingerprint.

    Slot mapping binds ordered D positions. Actual identities/hashes are only
    resolved by the registered loader and numerically checked by DenseBackend.
    """
    settings = asdict(graph.settings)
    for name in ("source_commit", "fixture_definition_commit", "fixture_definition_blob",
                 "hamiltonian_id", "sector_id", "state_id", "input_data_sha256", "context"):
        settings.pop(name)
    slots = {label: f"D_SLOT_{i}" for i, label in enumerate(graph.settings.context.block_ids)}
    nodes = []
    for node in graph.nodes:
        params = list(node.parameters)
        if node.kind == "exponential":
            params[0] = slots.get(params[0], params[0])
        nodes.append(dict(kind=node.kind, dependencies=node.dependencies, parameters=params))
    rows = [{k: v for k, v in row.items() if k not in ("input_fingerprint", "tail_identity")}
            for row in graph.rows]
    return fingerprint(dict(settings=settings, D_slots=len(slots), nodes=nodes,
                            rows=rows, correctness=graph.correctness))


def validate_stage1_graph(graph):
    s = graph.settings
    s.validate()
    from fractions import Fraction as F
    if (s.deltas, s.q_values, s.n_d_values, s.r_rte, s.cutoff, s.seed,
        s.truncation_tolerance, len(s.context.block_ids)) != (
            (F(9, 100), F(-9, 100)), (2, 3), (1, 2), 2, 2, 41, 1.0, 3):
        raise PermissionError("Unregistered stage-1 grid/input structure; STOP.")
    if len(graph.rows) != 36:
        raise PermissionError("Stage-1 row inventory differs.")


def validate_scope(scope, graph):
    from .scope import validate_registry
    validate_registry(scope, graph)


@dataclass
class RunContext:
    """Input identity and scope state for one run, not an approval certificate."""
    source_root: Path
    output: Path
    source_commit: str
    call_scope: dict
    environment: dict
    _input_started: bool = False
    _input_binding: object = None
    _evaluation_started: bool = False
    scope: object = None

    def attach_scope(self, scope):
        if self.scope is not None or scope.scope != self.call_scope:
            raise ValueError("Scientific scope binding differs.")
        self.scope = scope

    def science_call(self, name, function, *, identity=None):
        if self.scope is None:
            raise PermissionError("Registered science scope recorder required.")
        return self.scope.call(name, function, identity=identity)

    def record_resolution(self, name, value):
        if self.scope is None:
            raise PermissionError("Registered input resolution recorder required.")
        self.scope.record_resolution(name, value)

    def check_input(self, root, recipe):
        if Path(root).resolve() != self.source_root or recipe.fingerprint != InputRecipe().fingerprint:
            raise PermissionError("Input recipe or SOURCE root changed.")
        _check_source(self.source_root, self.source_commit, output=self.output)

    def begin_input(self, root, recipe):
        self.check_input(root, recipe)
        if self._input_started:
            raise PermissionError("Input generation already attempted; no retry.")
        self._input_started = True

    def bind_input(self, context, inputs, provenance):
        if not self._input_started or self._input_binding is not None:
            raise PermissionError("Input generation sequence differs.")
        self._input_binding = (context, inputs.hamiltonian_id, inputs.sector_id,
                               inputs.state_id, provenance["input_data_sha256"])

    def check_settings(self, settings):
        from .evaluation import build_graph
        self.check_input(self.source_root, InputRecipe())
        if self._input_binding != (settings.context, settings.hamiltonian_id,
                settings.sector_id, settings.state_id, settings.input_data_sha256):
            raise PermissionError("Settings must bind the generated common input.")
        graph = build_graph(settings)
        validate_stage1_graph(graph)
        if settings.source_commit != self.source_commit or graph_layout(graph) != self.call_scope["graph_layout"]:
            raise PermissionError("SOURCE, numeric conditions or graph changed.")
        if (settings.fixture_definition_commit, settings.fixture_definition_blob) != (
                InputRecipe().fixture_commit, InputRecipe().fixture_blob):
            raise PermissionError("Fixture SOURCE changed.")
        validate_scope(self.call_scope, graph)
        if self.scope is not None:
            self.scope.bind_graph(graph)

    def begin_evaluation(self, graph):
        self.check_settings(graph.settings)
        if self._evaluation_started:
            raise ValueError("Evaluation already attempted; no retry/resume.")
        self._evaluation_started = True


def _check_source(root, source_commit, *, output=None):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    command = ["git", "status", "--porcelain", "--untracked-files=normal", "--", "."]
    if output is not None:
        command.append(":(exclude)" + str(Path(output).relative_to(root)))
    dirty = subprocess.check_output(command, cwd=root, text=True)
    if head != source_commit or dirty:
        raise ValueError("Science SOURCE must be the fixed clean commit.")


def verify_execution(source_root, output, source_commit, *, graph=None, call_scope=None):
    """Read-only scientific preflight. Chat permission is a Codex operating rule."""
    from .scope import candidate_graph, registry
    from .environment import ENVIRONMENT_PATH, verify_environment
    import json
    root = Path(source_root).resolve()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("Full scientific SOURCE commit required.")
    _check_source(root, source_commit)
    read_fixture_definition(root)
    template = candidate_graph(source_commit)
    graph = template if graph is None else graph
    validate_stage1_graph(graph)
    if graph.settings.source_commit != source_commit or graph_layout(graph) != graph_layout(template):
        raise ValueError("Science SOURCE, numeric conditions or graph changed.")
    scope = registry(template) if call_scope is None else call_scope
    validate_scope(scope, graph)
    frozen = json.loads((root/ENVIRONMENT_PATH).read_bytes())
    environment = verify_environment(frozen)
    out = Path(output).resolve()
    parent = root/"artifacts/research_b_b2d_stage1"
    if not out.is_relative_to(parent) or out == parent or out.exists():
        raise ValueError("Fresh dedicated scientific output required.")
    return RunContext(root, out, source_commit, scope, environment)
