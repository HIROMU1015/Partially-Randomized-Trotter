"""Metadata-only checks. No scientific module imports or numerical evaluation."""
from pathlib import Path
import hashlib
import json
import posixpath
import re
import subprocess


def verify(worktree, root):
    worktree, root = Path(worktree), Path(root)
    artifact = worktree / "artifacts/qc_m_fixed_circuit_k0/2026-10-15"
    base = "0e923f73754d1ebd168bf21a1b9e73971946e603"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=worktree, stderr=subprocess.PIPE)

    def identity(path):
        data = Path(path).read_bytes()
        return {"SHA256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    def read(name):
        return json.loads((artifact / name).read_text())

    assert git("branch", "--show-current").decode().strip() == "qc-m-fixed-circuit-k0-audit-20261015"
    assert git("remote", "get-url", "origin").decode().strip() == "git@github.com:HIROMU1015/Partially-Randomized-Trotter.git"
    review = read("review_identity.json")
    assert identity(worktree / review["saved_path"]) == identity(review["source_path"]) == review["saved_identity"]
    assert (worktree / review["saved_path"]).read_bytes() == Path(review["source_path"]).read_bytes()
    protected = read("preservation_index.json")["files"]
    assert len(protected) == 5177
    for path, expected in protected.items():
        assert identity(worktree / path) == expected, path
        assert hashlib.sha256(git("show", base + ":" + path)).hexdigest() == expected["SHA256"], path
    old = json.loads(git("show", base + ":artifacts/validation_manifest.json"))
    new = json.loads((worktree / "artifacts/validation_manifest.json").read_text())
    assert len(old["result_sets"]) == 164
    assert new["result_sets"][:164] == old["result_sets"]
    assert {k: v for k, v in old.items() if k != "result_sets"} == {k: v for k, v in new.items() if k != "result_sets"}
    assert "description" not in new["relevant_commits"][7]
    method = read("method_status.json")
    assert method["K0"]["status"] == "UNDETERMINED_NOT_READY" and not method["K0"]["gate_pass"]
    assert method["K1"] == "NOT_EXECUTED_K0_NOT_PASSED"
    assert method["pilot_or_production_preparation"] == "NOT_STARTED"
    assert method["M_b_rescue_search"] == "NOT_STARTED"
    assert method["technical_conclusion"]["counterterm_projection_parameter_fit_core"] == "REDUCES_TO_KNOWN"
    assert all(v == 0 for v in method["scientific_calculation_counts"].values())
    assert method["formal_first_candidate_status"] == "AWAITING_USER_ADOPTION_DECISION"
    assert method["mandatory_stop"] and not method["M_formal_adoption"] and not method["M_formal_scientific_closure"]
    references = read("reference_identity.json")["sources"]
    assert len(references) == 12
    for row in references:
        expected = {k: row[k] for k in ("SHA256", "bytes")}
        assert identity(worktree / row["path"]) == expected
        assert hashlib.sha256(git("show", row["commit"] + ":" + row["path"])).hexdigest() == row["SHA256"]
        assert not row["imported_or_executed"] and not row["new_numeric_aggregation"]
    literature = read("literature_registry.json")
    assert len(literature["papers"]) == 6 and not literature["missing_important_body"]
    for row in literature["papers"]:
        assert not row["numerical_code_executed"]
        local = row.get("local_file", row.get("local_reference_copy"))
        if Path(local).is_file():
            assert identity(local) == {k: row[k] for k in ("SHA256", "bytes")}
    assert all(row["id"].startswith("qc_m_fixed_circuit_k0") for row in new["result_sets"][164:])
    changed = {p.decode() for p in git("diff", "--name-only", "-z", base).split(b"\0") if p}
    changed |= {p.decode() for p in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if p}
    links = []
    for path in sorted(changed):
        assert path.startswith(("docs/", "artifacts/qc_m_fixed_circuit_k0/")) or path in (
            "PROJECT_MAP.md", "VALIDATION_STATUS.md", "artifacts/validation_manifest.json"
        ), path
        if not path.endswith(".md"):
            continue
        text = (worktree / path).read_text()
        try:
            prior = git("show", base + ":" + path).decode()
        except subprocess.CalledProcessError:
            prior = ""
        if prior:
            assert text.endswith(prior.split("\n", 1)[1]), path
        for target in set(re.findall(r"\]\(([^)]+)\)", text)) - set(re.findall(r"\]\(([^)]+)\)", prior)):
            if target.startswith(("https:", "http:", "#", "mailto:")):
                continue
            full = posixpath.normpath(posixpath.join(posixpath.dirname(path), target.split("#")[0]))
            assert (worktree / full).exists(), (path, full)
            links.append({"document": path, "target": full})
    baseline = read("root_worktree_baseline.json")
    current_raw = subprocess.check_output(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=root)
    parts = current_raw.split(b"\0")
    current = {}
    i = 0
    while i < len(parts) and parts[i]:
        s = parts[i].decode()
        i += 1
        current[s[3:]] = s[:2]
        if "R" in s[:2] or "C" in s[:2]:
            i += 1
    for row in baseline["records"]:
        path = row["path"]
        assert current.get(path) == row["status"], path
        assert (identity(root / path) if (root / path).is_file() else None) == row["identity"], path
    added = sorted(set(current) - {r["path"] for r in baseline["records"]})
    return {
        "status": "DOCUMENT_IDENTITY_PRESERVATION_AND_SCOPE_PASS",
        "scope": "Metadata/byte/hash/JSON/link checks only; not scientific revalidation or current CI",
        "review_byte_identical": True,
        "protected_files_matched": len(protected),
        "old_manifest_records_preserved": 164,
        "non_result_fields_and_known_global_FAIL_preserved": True,
        "root_baseline_paths_preserved": len(baseline["records"]),
        "root_added_paths_outside_this_task": added,
        "root_full_status_unchanged": hashlib.sha256(current_raw).hexdigest() == baseline["status_SHA256"],
        "new_relative_links": links,
        "fixed_reference_bodies_checked": 12,
        "primary_literature_body_identities_checked": 6,
        "new_scientific_calculations": 0,
        "new_saved_numeric_aggregation": 0,
        "mandatory_stop": True,
    }


if __name__ == "__main__":
    import sys
    result = verify(sys.argv[1], sys.argv[2])
    Path(sys.argv[3]).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "new_relative_links"}, ensure_ascii=False))
