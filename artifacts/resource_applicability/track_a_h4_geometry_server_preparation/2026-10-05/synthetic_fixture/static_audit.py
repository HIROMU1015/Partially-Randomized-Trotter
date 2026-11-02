"""Static source and explicitly allowed JSON identity checks only."""
import ast
import collections
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

BASE = "4c23453c541700c6a41ba71fc5ec9323b53858d6"
HANDOFF = "48f9ac3b756bcf572aeb7c594a0c88791a458725"
EVIDENCE = [
    ("artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json", "1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086"),
    ("artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json", "71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4"),
    ("artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result.json", "9305857873602d6bc4f45fbc78c4903911d083156620df01e9b23f00e7fdf05b"),
    ("artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json", "f41a92beb57e59cddc8c063b061c40acd4da50cb76ac0698efc2bce004937931"),
]


def main():
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    records = []
    for path, expected in EVIDENCE:
        current = (root / path).read_bytes()
        earlier, committed = git("show", f"{BASE}:{path}"), git("show", f"{HANDOFF}:{path}")
        digest = hashlib.sha256(current).hexdigest()
        assert current == earlier == committed and digest == expected, path
        records.append({"path": path, "sha256": digest, "worktree_base_handoff_byte_identical": True})
    directory = "artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/"
    for name in ["candidate_inventory_v1.json", "input_identity_v1.json"]:
        path = directory + name
        current = (root / path).read_bytes()
        assert current == git("show", f"{BASE}:{path}") == git("show", f"{HANDOFF}:{path}")
        records.append({"path": path, "sha256": hashlib.sha256(current).hexdigest(), "worktree_base_handoff_byte_identical": True})
    inventory = json.loads((root / (directory + "candidate_inventory_v1.json")).read_text())
    proposal = json.loads((root / "docs/research/track_a_geometry_precision_extension_proposal_v0.json").read_text())
    expected = set()
    for name, method in [("B0", "B0"), ("B1", "B1"), ("B2_regular", "B2"), ("B3_regular", "B3")]:
        config = proposal["candidate_templates"][name]
        rr = config["r"] if isinstance(config["r"], list) else [config["r"]]
        kk = config["K"] if isinstance(config["K"], list) else [config["K"]]
        expected.update((method, rank, q, r, k) for rank, q, r, k in itertools.product(config["deterministic_prefix_ranks"], config["q"], rr, kk))
    expected.update((x["method"], x["deterministic_prefix_rank"], x["q"], x["r"], x["K"]) for x in proposal["candidate_templates"]["inherited_boundary_templates"])
    actual = [tuple(row["parameters"][key] for key in ["method", "rank", "q", "r", "K"]) for row in inventory["development"]]
    assert len(actual) == len(set(actual)) == len(expected) == 218 and set(actual) == expected
    paths = [p for p in git("ls-tree", "-r", "--name-only", HANDOFF, "--", "src", "scripts").decode().splitlines() if p.endswith(".py")]
    failures, sources = [], []
    for path in paths:
        source = (root / path).read_bytes()
        try:
            ast.parse(source, filename=path)
        except SyntaxError as error:
            failures.append({"path": path, "error": str(error)})
        sources.append({"path": path, "sha256": hashlib.sha256(source).hexdigest()})
    assert not failures, failures
    environment = json.loads((output / "environment_inventory_v0.json").read_text())
    static = {
        "schema_version": "track_a_server_static_audit_v0", "status": "STATIC_AND_IDENTITY_CHECKS_PASS",
        "handoff_commit": HANDOFF, "evidence_base_commit": BASE, "allowed_json_identity": records,
        "template_set_exact_match": True, "template_count": len(actual),
        "template_method_counts": dict(collections.Counter(row[0] for row in actual)),
        "python312_syntax_parsed_source_count": len(sources), "source_hashes": sources,
        "runtime_compatibility_claimed": False,
        "source_port_required": True,
        "findings": [
            {"path": "src/trottertracks/resource_applicability/pm1_discard_contract.py", "line": 42,
             "issue": "source-bound environment requires Python 3.11.0rc1; server Python is 3.12.3",
             "resolution": "future new Track A source and contract bound to server dependencies; keep old guard/source unchanged"},
            {"path": "src/trotterlib/rte_compiled_cost.py", "line": 861,
             "issue": "old helper records three explicit compiler kwargs but inherits additional defaults",
             "resolution": "future new compiler identity must freeze effective defaults, plugin inventory and per-process internal concurrency"},
            {"path": "src/trotterlib/rpe_hadamard_interrogation.py", "line": 145,
             "issue": "system register precedes ancilla; cosine H/CU/H and sine H/CU/Sdg/H preserve relative/global phase",
             "resolution": "new synthetic fixture uses highest-index ancilla and exact diag(I,U) tests"},
        ],
        "environment_difference": {"old_python": "3.11.0rc1", "server_python": "3.12.3", "qiskit_version": environment["dependencies"]["qiskit"]["version"], "same_complete_compiler_identity_claimed": False},
        "api_presence": environment["api_presence"],
        "science_module_imports": 0, "molecular_or_runtime_access": 0, "gpu_access": 0,
        "science_runner_executions": 0, "old_source_mutations": 0,
    }
    with (output / "static_audit_v0.json").open("x") as handle:
        json.dump(static, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: static[key] for key in ["status", "template_count", "template_method_counts", "python312_syntax_parsed_source_count", "source_port_required", "science_module_imports"]}))


if __name__ == "__main__":
    main()
