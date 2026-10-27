"""Plan or run the fixed non-scientific P1/P2 regression slice.

Uses the caller's isolated Python environment and ordinary package imports.
Never installs dependencies or dispatches scientific validation/old runners.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DF_BASE = "2c865c66171fc9d1972b67c1257c57e5684fe26c"
PAULI_BASE = "f0a035961583d976e8967511a78f9da66758f0bb"
INTEGER_TESTS = ("test_integer_count_rejects_bool_float_and_string",
                 "test_integer_count_accepts_python_and_numpy_integers")
EXECUTOR_TESTS = ("test_task_fingerprint_is_deterministic_and_binds_inputs",
                  "test_atomic_json_write_leaves_no_temporary_file",
                  "test_deterministic_aggregate_ignores_completion_and_runtime_order",
                  "test_resource_guard_limits_workers_memory_and_gpu_use",
                  "test_resume_skips_completed_task",
                  "test_corrupt_checkpoint_is_rejected_without_overwrite",
                  "test_mismatched_manifest_checkpoint_is_rejected")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   env={**os.environ, "GIT_OPTIONAL_LOCKS": "0",
                                        "GIT_NO_LAZY_FETCH": "1"})


def check_pinned_sources(root, base, scopes):
    # Git compares only these existing implementation scopes; no raw traversal,
    # alternate package loader, or whole-repository SOURCE audit is needed.
    changed = git(root, "diff", "--name-only", base, "--", *scopes).decode().splitlines()
    if changed:
        raise ValueError("Pinned implementation differs: " + ", ".join(changed))


def selections(home):
    return {
        "df_current": ["tests/test_df_research_pipeline.py"],
        "df_legacy13": ["tests/test_rte_integer_counts.py::" + name for name in INTEGER_TESTS]
                       + ["tests/test_parallel_validation_executor.py::" + name for name in EXECUTOR_TESTS],
        "pauli_current": ["tests/test_pauli_jw_research_pipeline.py"],
        "pauli_legacy_selected13": [str(home / "scripts/integration/validation_cases/pauli_legacy_selected.py")],
        "df_writer_import": [str(home / "tests/test_pipeline_real_writer_validation.py"),
                             str(home / "tests/test_pipeline_eager_import_validation.py")],
        "pauli_writer_import": [str(home / "tests/test_pipeline_real_writer_validation.py"),
                                str(home / "tests/test_pipeline_eager_import_validation.py")],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--df-checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pauli-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="run only the listed technical tests")
    args = parser.parse_args(argv)
    home = Path(__file__).resolve().parents[2]
    groups = selections(home)
    if not args.execute:
        print(json.dumps({"mode": "PLAN_ONLY", "fixed_df": DF_BASE,
                          "fixed_pauli": PAULI_BASE, "groups": groups,
                          "science": "NONE; separate H4 input qualification not implemented here"}, indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires a fresh --output directory")
    roots = {"df": args.df_checkout.resolve(), "pauli": args.pauli_checkout.resolve()}
    check_pinned_sources(roots["df"], DF_BASE,
                         ("src/trotterlib", "src/trottertracks/resource_applicability"))
    check_pinned_sources(roots["pauli"], PAULI_BASE,
                         ("src/trotterlib", "src/trottertracks/pauli_jw"))
    output = args.output.resolve()
    if any(output.is_relative_to(root) for root in roots.values()):
        parser.error("output must be outside the tested source checkouts")
    output.mkdir(parents=True, exist_ok=False)
    for name, nodes in groups.items():
        kind = name.split("_", 1)[0]
        root = roots[kind]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0",
               "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
               "PRT_VALIDATION_SCOPE": kind, "MPLCONFIGDIR": str(output / "mpl-cache")}
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   *nodes, "--junitxml=" + str(output / (name + ".xml"))]
        with (output / (name + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
