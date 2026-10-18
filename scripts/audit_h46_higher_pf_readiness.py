"""Read fixed Git evidence and metadata only; never run PF optimizers/PR kernels."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from h46_higher_pf_adapter import ROOT, inventory, git_blob

PF_COMMIT = "20d64c2fea1ed2d2188777df4f3d962ee94df486"
PF_FILES = (
    "src/trotterlib/product_formula.py", "src/trotterlib/optimal_trotter.py",
    "review_response/search_pf_cost_predictability_m2_m3.py",
    "review_response/run_full_electron_nh3_higher_term_diagnosis.py",
    "review_tests/test_morales_2025_product_formula.py",
    "docs/current_research_status.md", "docs/repository_guide.md", "docs/pf_data_use_ledger.md",
)


def external_evidence(repo):
    contents = {p: git_blob(repo, PF_COMMIT, p) for p in PF_FILES}
    optimal = ast.parse(contents[PF_FILES[1]])
    defined = {n.name for n in ast.walk(optimal) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    helper_names = ("fourth_order_moment_residual_float64", "solve_nonprocessed_4th_moment_coefficients")
    missing = [name for name in helper_names if name not in defined]
    search = ast.parse(contents[PF_FILES[2]])
    imports = [n for n in ast.walk(search) if isinstance(n, ast.ImportFrom) and n.module == "trotterlib.optimal_trotter"]
    requested = {alias.name for n in imports for alias in n.names}
    if not set(helper_names) <= requested:
        raise ValueError("EXTERNAL_PF_AUDIT_EXPECTATION_CHANGED")
    diagnosis = ast.parse(contents[PF_FILES[3]])
    saved_literal = None
    for node in diagnosis.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PAPER_NEW4" for t in node.targets):
            # Read literal text/identity only; no optimizer or moment evaluation.
            saved_literal = list(ast.literal_eval(node.value))
    if saved_literal is None:
        raise ValueError("SAVED_PF_LITERAL_UNAVAILABLE")
    return {"repository": "HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae",
            "commit": PF_COMMIT,
            "files": [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                       "url": f"https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/{PF_COMMIT}/{path}"}
                      for path, data in contents.items()],
            "projector_dependency_names_missing_from_target_module": missing,
            "projector_executed": False, "higher_precision_original_optimum_recovered": False,
            "saved_PAPER_NEW4_compact_literal": saved_literal,
            "saved_numeric_candidate_adopted_in_PR": False,
            "interpretation": "saved numerical candidate and order tests reusable; source gap prevents claiming full projector reproduction; energy/QPE costs do not replace PR signal/native costs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"metadata": inventory(), "external_PF": external_evidence(args.pf_repo),
              "real_signal_analysis": 0, "real_cost_analysis": 0, "native_compile": 0,
              "trajectory_sampling": 0, "coefficient_optimization": 0}
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    print(json.dumps({"status": "METADATA_VERIFIED", "output": str(args.output),
                      "counts": result["metadata"]["counts"]}))


if __name__ == "__main__":
    main()
