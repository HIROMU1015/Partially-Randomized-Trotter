"""B2-D read-only preflight, with existing one-shot packet/marker convention.

No issuance API, implicit approval, default authority or science import exists.
An operations authority must check the actual user approval record, not merely
trust authorization.json/user_approval.txt written by a caller. Its production
binding is a prerequisite still to be adopted by research B.
"""
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess

from trottertracks.pr_dictionary_validation.authorization import _canonical, _under, file_sha
from .input_package import InputRecipe, read_fixture_definition, fingerprint

STAGE = "B2D_STAGE1"
SCOPE_GROUPS = (
    "fixture_identity", "split_basis_sector", "materialization", "state",
    "input_correctness", "exact_references", "a4_correctness", "five_arm_operators",
    "finite_rte_normalization", "complex_signals", "save_provenance",
)


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
    """Reject incomplete scopes; dispatch counts alone are insufficient."""
    if scope.get("fully_resolved") is not True or scope.get("unresolved") != []:
        raise PermissionError("Full scientific call scope is unresolved; no new grant.")
    if scope.get("dispatch_counts") != graph.call_counts:
        raise PermissionError("DAG call scope differs.")
    groups = scope.get("groups", {})
    if set(groups) != set(SCOPE_GROUPS):
        raise PermissionError("Missing external/internal science call group.")
    for group in groups.values():
        if not isinstance(group, list) or not group:
            raise PermissionError("Empty science group.")
        for entry in group:
            if (not isinstance(entry.get("operation"), str) or not entry["operation"]
                or type(entry.get("count")) is not int or entry["count"] < 0
                or not isinstance(entry.get("dependencies"), list)):
                raise PermissionError("Unresolved low-level operation/count/dependency.")


def scope_proposal(graph):
    """No resolved scientific total. Counts are static dispatch/interface calls."""
    validate_stage1_graph(graph)
    from trottertracks.b2d_adapter.adapter import published_math_sequences
    kernel, processor = published_math_sequences()
    dag_exp = {node.parameters for node in graph.nodes if node.kind == "exponential"}
    reference_exp = set()
    for delta in graph.settings.deltas:
        for n in graph.settings.n_d_values:
            tau = delta / (2 * n)
            for weight in kernel + processor:
                reference_exp.update((block, tau * weight / 2)
                                     for block in graph.settings.context.block_ids)
    checks, blocks = graph.call_counts["a4_reference_check"], len(graph.settings.context.block_ids)
    def operation(name, count, *dependencies):
        return dict(operation=name, count=count, dependencies=list(dependencies))
    groups = dict(
        fixture_identity=[operation("fixture_H", 1), operation("hamiltonian_hash", 1, "fixture_H")],
        split_basis_sector=[operation("weight_ranked_split", 1, "fixture_H"),
            operation("preparation", 1, "weight_ranked_split"),
            operation("Gaussian_gate_basis_internals", None, "preparation"),
            operation("number_sector", 1, "fixture_H")],
        materialization=[operation("full_H_sector", 1, "number_sector"),
            operation("D_block_sectors", 3, "preparation", "number_sector"),
            operation("grouped_normalized_tail", 1, "preparation", "number_sector"),
            operation("basis_circuit_and_sector_matvec_internals", None, "grouped_normalized_tail")],
        state=[operation("occupation_state", 1, "number_sector")],
        input_correctness=[operation("validate_input", 1, "full_H_sector", "D_block_sectors", "grouped_normalized_tail", "occupation_state"),
            operation("D_pair_commutators", 3, "validate_input"),
            operation("DR_commutator", 1, "validate_input"),
            operation("input_norm_and_identity_internals", None, "validate_input")],
        exact_references=[operation("backend_unique_expm_cache_keys", len(dag_exp | reference_exp), "validate_input")],
        a4_correctness=[operation("a4_reference_check", 4, "backend_unique_expm_cache_keys"),
            operation("independent_reference_matmul", checks*((len(kernel)+len(processor))*(2*blocks+1)+3), "a4_reference_check"),
            operation("independent_reference_matrix_power", 4, "a4_reference_check"),
            operation("independent_reference_error_norm", 8, "a4_reference_check")],
        five_arm_operators=[operation("DAG_product_matmul", sum(len(n.dependencies) for n in graph.nodes if n.kind == "product_append"), "a4_reference_check"),
            operation("DAG_power", 112, "a4_reference_check"),
            operation("DAG_operator_error", 182, "a4_reference_check"),
            operation("DAG_scalar_phase", 56, "a4_reference_check")],
        finite_rte_normalization=[operation("finite_RTE_config", 2, "validate_input"),
            operation("finite_RTE_moments", 2, "finite_RTE_config"),
            operation("normalization", 4, "finite_RTE_moments"),
            operation("rescale_raw", 36, "normalization"),
            operation("RTE_Taylor_power_internals", None, "finite_RTE_moments")],
        complex_signals=[operation("expectation", 144, "a4_reference_check"),
            operation("principal_phase", None if graph.settings.phase_diagnostics else 0, "expectation")],
        save_provenance=[operation("output_log_provenance_inventory", None, "validate_input", "a4_reference_check")])
    return dict(fully_resolved=False, dispatch_counts=graph.call_counts,
        dispatch_total=len(graph.nodes), rows=len(graph.rows),
        graph_layout=graph_layout(graph),
        groups=groups,
        known_interface_counts=dict(fixture_H=1, split=1, preparation=1,
            number_sector=1, sector_matrix_materializations=4,
            grouped_normalized_tail=1, physical_state=1,
            input_validation=1, D_pair_commutators=3, DR_commutator=1,
            a4_mathematical_checks=4, signal_expectations=144,
            backend_unique_expm_keys=len(dag_exp | reference_exp),
            A4_extra_expm_cache_keys=len(reference_exp - dag_exp),
            DAG_product_matrix_multiplications=sum(len(n.dependencies) for n in graph.nodes if n.kind == "product_append"),
            A4_reference_matrix_multiplications=checks*((len(kernel)+len(processor))*(2*blocks+1)+3)),
        unresolved=["Gaussian preparation and tail basis circuit/gate internals",
            "retained component/basis counts and sector matvec work",
            "input validation, RTE and library-internal norm/product work beyond enumerated keys",
            "RTE moment/power internal work and numerical diagnostics",
            "runtime environment, output/log/provenance inventory and call enforcement"])


def source_inventory(root):
    """Bind the local Python library closure; read bytes only, no imports."""
    root = Path(root).resolve()
    names = [str(p.relative_to(root)) for p in (root / "src").rglob("*.py")]
    names.append(InputRecipe().fixture_path)
    return {name: file_sha(_under(root, name).read_bytes()) for name in sorted(names)}


@dataclass
class VerifiedRun:
    contract: dict
    authorization: dict
    authorization_digest: str
    contract_digest: str
    source_root: Path
    output: Path
    consumption_path: Path
    is_mock: bool
    _handle_issued: bool = False

    def for_existing_claim(self):
        # Future launcher uses authorization.claim_packet; NOT called in prep.
        if self.is_mock:
            raise PermissionError("Mock evidence cannot claim a scientific grant.")
        return dict(consumption_path=self.consumption_path, output=self.output,
                    authorization_digest=self.authorization_digest)

    def after_existing_claim(self):
        if self.is_mock or self._handle_issued or not self.output.is_dir():
            raise PermissionError("Actual atomic one-shot claim required.")
        if self.consumption_path.read_text() != self.authorization_digest + "\n":
            raise PermissionError("Existing marker does not match this authorization.")
        self._handle_issued = True
        return _ClaimedRun(self)


class _ClaimedRun:
    """Internal handle: only obtained after the existing operations claim."""
    def __init__(self, verified):
        self.verified = verified
        self.contract = verified.contract
        self._input_started = False
        self._input_binding = None
        self._evaluation_started = False

    def check_input(self, root, recipe):
        if Path(root).resolve() != self.verified.source_root or recipe.fingerprint != self.contract["recipe_fingerprint"]:
            raise PermissionError("Input recipe or SOURCE root changed.")
        _check_source(self.verified.source_root, self.contract, output=self.verified.output)

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
        self.check_input(self.verified.source_root, InputRecipe())
        if self._input_binding != (settings.context, settings.hamiltonian_id,
                settings.sector_id, settings.state_id, settings.input_data_sha256):
            raise PermissionError("Settings must bind the generated common input.")
        graph = build_graph(settings)
        validate_stage1_graph(graph)
        if settings.source_commit != self.contract["source_commit"] or graph_layout(graph) != self.contract["graph_layout"]:
            raise PermissionError("SOURCE, numeric conditions or graph changed.")
        if (settings.fixture_definition_commit, settings.fixture_definition_blob) != (
                InputRecipe().fixture_commit, InputRecipe().fixture_blob):
            raise PermissionError("Fixture SOURCE changed.")
        validate_scope(self.contract["call_scope"], graph)

    def check_graph(self, graph, approval):
        self.check_settings(graph.settings)
        a = self.verified.authorization
        if (approval.approval_reference, approval.grant_reference, approval.output_identity,
            approval.retry_policy) != (a["approval_reference"], a["grant_id"],
                                       str(self.verified.output), "NO_RETRY_NO_RESUME"):
            raise PermissionError("Execution attestation differs from operations packet.")

    def begin_evaluation(self, graph, approval):
        self.check_graph(graph, approval)
        if self._evaluation_started:
            raise PermissionError("Evaluation already attempted; no retry/resume.")
        self._evaluation_started = True


def require_claimed_run(value):
    if not isinstance(value, _ClaimedRun):
        raise PermissionError("Verified operations approval and claimed fresh grant required.")
    return value


def _check_source(root, contract, *, output=None):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    command = ["git", "status", "--porcelain", "--untracked-files=normal", "--", "."]
    if output is not None:
        command.append(":(exclude)" + str(Path(output).relative_to(root)))
    dirty = subprocess.check_output(command, cwd=root, text=True)
    if head != contract.get("source_commit") or dirty:
        raise PermissionError("Science SOURCE must be the fixed clean commit.")
    if source_inventory(root) != contract.get("source_files"):
        raise PermissionError("Science SOURCE closure differs.")


def verify_execution(packet_dir, source_root, output, graph, *, authority=None):
    """Read-only. Existing ledgers/markers are never changed by preflight.

    authority.verify_approval(packet, approval_bytes, authorization_digest,
    contract_digest) must consult the actual operations/user record. No default
    implementation or text/string substitute is accepted. Mock authorities are
    explicitly marked and cannot produce a science claim.
    """
    validate_stage1_graph(graph)
    if authority is None or not callable(getattr(authority, "verify_approval", None)) or type(getattr(authority, "is_mock", None)) is not bool:
        raise PermissionError("Actual approval-record verification is missing.")
    packet, root = Path(packet_dir).resolve(), Path(source_root).resolve()
    grant_bytes = (packet / "authorization.json").read_bytes()
    contract_bytes = (packet / "contract.json").read_bytes()
    approval_bytes = (packet / "user_approval.txt").read_bytes()
    grant, contract = json.loads(grant_bytes), json.loads(contract_bytes)
    if (contract.get("stage"), contract.get("status")) != (STAGE, "USER_ADOPTED_STAGE1"):
        raise PermissionError("Unadopted/draft contract.")
    if (grant.get("stage"), grant.get("authority"), grant.get("status")) != (
            STAGE, "USER_EXPLICIT_B2D_STAGE1", "APPROVED_BOUND_ONE_SHOT"):
        raise PermissionError("A separate new B2-D grant is required.")
    for document in (contract, grant):
        if document.get("retry_allowed") is not False or document.get("mandatory_stop") is not True:
            raise PermissionError("No retry/resume and mandatory STOP required.")
    if not approval_bytes.strip() or grant.get("approval_sha256") != file_sha(approval_bytes):
        raise PermissionError("Approval evidence not bound.")
    if grant.get("contract_sha256") != file_sha(contract_bytes):
        raise PermissionError("Contract not bound to grant.")
    if authority.verify_approval(packet, approval_bytes, file_sha(grant_bytes), file_sha(contract_bytes)) is not True:
        raise PermissionError("Operations record does not confirm actual approval.")
    if contract.get("recipe_fingerprint") != InputRecipe().fingerprint:
        raise PermissionError("Input resolution recipe differs.")
    if contract.get("input_resolution") != "REGISTERED_RECIPE_THEN_BACKEND_CORRECTNESS":
        raise PermissionError("Runtime input identity resolution must be explicitly adopted.")
    if contract.get("graph_layout") != graph_layout(graph):
        raise PermissionError("Registered grid/graph/numeric conditions differ.")
    if contract.get("numerical_atol") != graph.settings.numerical_atol:
        raise PermissionError("Unadopted numerical correctness tolerance.")
    validate_scope(contract.get("call_scope", {}), graph)
    if contract.get("source_commit") != graph.settings.source_commit:
        raise PermissionError("Graph SOURCE commit mismatch.")
    read_fixture_definition(root)
    _check_source(root, contract)
    if not contract.get("dependency_versions"):
        raise PermissionError("Runtime dependencies unresolved.")
    for name, version in contract["dependency_versions"].items():
        if importlib.metadata.version(name) != version:
            raise PermissionError("Runtime dependency changed: " + name)
    out = Path(output).resolve()
    parent = root / "artifacts/research_b_b2d_stage1"
    if not out.is_relative_to(parent) or out == parent or out.exists() or not out.parent.is_dir():
        raise PermissionError("Fresh dedicated output with existing parent required.")
    if grant.get("fresh_output_namespace") != str(out):
        raise PermissionError("Unbound output.")
    if grant.get("source_commit") != contract["source_commit"] or grant.get("source_closure_sha256") != file_sha(_canonical(contract["source_files"])):
        raise PermissionError("Grant SOURCE binding differs.")
    if re.fullmatch(r"b2d_[A-Za-z0-9_-]+", grant.get("grant_id", "")) is None or not grant.get("approval_reference"):
        raise PermissionError("New stage-specific grant/reference required.")
    marker = packet / ("consumed-" + file_sha(grant_bytes))
    if marker.exists():
        raise PermissionError("Grant already used; no retry/resume.")
    return VerifiedRun(contract, grant, file_sha(grant_bytes), file_sha(contract_bytes),
                       root, out, marker, authority.is_mock)
