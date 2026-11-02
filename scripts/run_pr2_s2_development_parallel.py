#!/usr/bin/env python3
"""Run the frozen PR-2 S2 comparison with bounded CPU parallelism."""

from __future__ import annotations

import os

# Set thread caps before importing NumPy, SciPy, or Qiskit in spawned workers.
for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.pr2_v4_s2_development_validation import (  # noqa: E402
    AUTHORIZATION_COMMIT,
    V4_PASS_STATUS,
    file_sha256,
    validate_s2_payload,
    validate_v4_payload,
    write_json_artifact,
)
from trotterlib.pr2_v4_s2_parallel_execution import (  # noqa: E402
    DEFAULT_PARALLEL_WORKERS,
    MAXIMUM_PARALLEL_WORKERS,
    PROCESS_START_METHOD,
    run_s2_development_parallel,
)


DEFAULT_V4 = (
    ROOT
    / "artifacts"
    / "pr2_v4_s2_development"
    / "2026-09-28"
    / "pr2_v4_correctness_result_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "pr2_v4_s2_development"
    / "2026-09-28"
    / "pr2_s2_development_resource_result_parallel_v1.json"
)
DEFAULT_CACHE = (
    ROOT
    / "artifacts"
    / "pr2_v4_s2_development"
    / "2026-09-28"
    / "cache"
    / "pr2_s2_parallel_compile.sqlite"
)
EXECUTABLE_SOURCE_PATHS = (
    "src/trotterlib/pr2_v4_s2_development_validation.py",
    "src/trotterlib/pr2_v4_s2_parallel_execution.py",
    "scripts/run_pr2_s2_development_parallel.py",
    "tests/test_pr2_v4_s2_parallel_execution.py",
)
SOURCE_PATHS = EXECUTABLE_SOURCE_PATHS + (
    "docs/research/pr2_v4_s2_development_authorization_v5.md",
    "artifacts/pr2_v4_s2_development/2026-09-28/"
    "pr2_v4_s2_authorization_v1.json",
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _assert_source_frozen() -> str:
    head = _git("rev-parse", "HEAD")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", AUTHORIZATION_COMMIT, head),
        cwd=ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("The authorization commit is not an ancestor of HEAD.")
    for path in EXECUTABLE_SOURCE_PATHS:
        subprocess.run(
            ("git", "ls-files", "--error-unmatch", path),
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    for mode in ((), ("--cached",)):
        if subprocess.run(
            (
                "git",
                "diff",
                *mode,
                "--quiet",
                "HEAD",
                "--",
                *EXECUTABLE_SOURCE_PATHS,
            ),
            cwd=ROOT,
            check=False,
        ).returncode:
            raise RuntimeError("Executable parallel PR-2 source differs from HEAD.")
    return head


def _junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if summary["tests"] <= 0 or summary["failures"] or summary["errors"]:
        raise RuntimeError("Dedicated parallel S2 test log is not passing.")
    return summary


def _provenance(
    test_log: Path,
    v4_path: Path,
    *,
    workers: int,
    cache_path: Path,
) -> dict[str, Any]:
    head = _assert_source_frozen()
    resolved_log = test_log.resolve(strict=True)
    resolved_cache = cache_path.resolve()
    status = _git("status", "--porcelain=v1")
    return {
        "stage": "S2-development-parallel-execution",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "git_branch": _git("branch", "--show-current"),
        "authorization_commit": AUTHORIZATION_COMMIT,
        "source_hashes": {
            relative: file_sha256(ROOT / relative) for relative in SOURCE_PATHS
        },
        "V4_artifact": {
            "path": str(v4_path.resolve()),
            "sha256": file_sha256(v4_path),
        },
        "worktree_dirty": bool(status),
        "worktree_status_porcelain_v1": status.splitlines(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "execution": {
            "workers": workers,
            "maximum_workers_allowed": MAXIMUM_PARALLEL_WORKERS,
            "process_start_method": PROCESS_START_METHOD,
            "persistent_compile_cache": str(resolved_cache),
            "thread_environment": {
                name: os.environ[name]
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "dedicated_test_log": {
            "path": str(resolved_log),
            "sha256": file_sha256(resolved_log),
            "junit": _junit_summary(resolved_log),
            "evidence_class": "local_validation",
            "independent_external_reproduction": False,
        },
    }


def _write_failure_report(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite failure report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary report: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen PR-2 S2 comparison with bounded cell-level CPU "
            "parallelism."
        )
    )
    parser.add_argument("--test-log", type=Path, required=True)
    parser.add_argument("--v4-artifact", type=Path, default=DEFAULT_V4)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        choices=range(1, MAXIMUM_PARALLEL_WORKERS + 1),
    )
    parser.add_argument("--persistent-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--failure-report",
        type=Path,
        help="Non-overwriting JSON report written if the run raises an exception.",
    )
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    cache_path = args.persistent_cache.resolve()
    failure_report = (
        args.failure_report.resolve()
        if args.failure_report is not None
        else artifact.with_name(f"{artifact.stem}_failure.json")
    )
    if failure_report.exists():
        raise FileExistsError(
            f"Refusing to overwrite failure report: {failure_report}"
        )
    if failure_report in (artifact, cache_path):
        raise ValueError("Failure report, result artifact, and cache must differ.")

    print(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "parallel_s2_run_started",
                "pid": os.getpid(),
                "workers": args.workers,
                "artifact": str(artifact),
                "failure_report": str(failure_report),
                "persistent_cache": str(cache_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {artifact}")
        if cache_path == artifact:
            raise ValueError("Persistent cache and result artifact must differ.")

        v4_path = args.v4_artifact.resolve(strict=True)
        v4_payload = json.loads(v4_path.read_text(encoding="utf-8"))
        validate_v4_payload(v4_payload)
        if v4_payload["status"] != V4_PASS_STATUS or v4_payload["deviations"]:
            raise RuntimeError("S2 requires a deviation-free V4 PASS artifact.")

        payload = run_s2_development_parallel(
            ROOT,
            v4_payload,
            provenance=_provenance(
                args.test_log,
                v4_path,
                workers=args.workers,
                cache_path=cache_path,
            ),
            workers=args.workers,
            persistent_cache_path=cache_path,
        )
        write_json_artifact(payload, artifact, validator=validate_s2_payload)
    except BaseException as exc:
        failure_payload = {
            "schema_version": "pr2_s2_parallel_failure_v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "workers": args.workers,
            "artifact": str(artifact),
            "persistent_cache": str(cache_path),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "command": [sys.executable, *sys.argv],
            "cwd": str(ROOT),
        }
        _write_failure_report(failure_report, failure_payload)
        print(
            json.dumps(
                {
                    "timestamp_utc": failure_payload["timestamp_utc"],
                    "event": "parallel_s2_run_failed",
                    "exception_type": failure_payload["exception_type"],
                    "exception_message": failure_payload["exception_message"],
                    "failure_report": str(failure_report),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise

    print(
        json.dumps(
            {
                "stage": "S2-development-parallel-execution",
                "status": payload["status"],
                "artifact": str(artifact),
                "result_fingerprint": payload["result_fingerprint"],
                "selected_B2": payload["decision"]["selected_B2"],
                "workers": args.workers,
                "S3_authorized": False,
                "automatic_next_stage": None,
                "mandatory_stop_reached": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
