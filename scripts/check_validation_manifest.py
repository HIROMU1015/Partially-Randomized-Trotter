#!/usr/bin/env python3
"""Validate the repository's machine-readable validation evidence manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any


OVERALL_STATUSES = {
    "reproducible_from_repository",
    "partially_reproducible_from_repository",
    "not_reproducible_from_repository",
}
RESULT_STATUSES = {
    "historical_evidence",
    "invalidated_pending_regeneration",
    "prose_only_missing_raw_artifacts",
    "source_present_no_current_ci",
    "reproducible_from_repository",
}
ARTIFACT_SECTIONS = ("present", "quarantined", "missing")
ARTIFACT_KINDS = {"file", "directory"}
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class ManifestError(ValueError):
    """Raised when the validation manifest is structurally inconsistent."""


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{location} must be an object")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{location} must be an array")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location} must be a non-empty string")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ManifestError(f"{location} has invalid keys: {', '.join(details)}")


def _unique_strings(value: Any, location: str, *, nonempty: bool = True) -> list[str]:
    items = _array(value, location)
    strings = [
        _string(item, f"{location}[{index}]") for index, item in enumerate(items)
    ]
    if nonempty and not strings:
        raise ManifestError(f"{location} must not be empty")
    if len(strings) != len(set(strings)):
        raise ManifestError(f"{location} contains duplicate values")
    return strings


def _sha(value: Any, location: str) -> str:
    sha = _string(value, location)
    if SHA_RE.fullmatch(sha) is None:
        raise ManifestError(f"{location} must be a lowercase 40-character Git SHA")
    return sha


def _safe_repo_path(raw_path: Any, location: str, repo_root: Path) -> tuple[str, Path]:
    path_text = _string(raw_path, location)
    if "\\" in path_text:
        raise ManifestError(f"{location} must use repository-relative POSIX separators")
    posix_path = PurePosixPath(path_text)
    if posix_path.is_absolute() or path_text == "." or ".." in posix_path.parts:
        raise ManifestError(f"{location} must be a safe repository-relative path")
    candidate = repo_root.joinpath(*posix_path.parts)
    try:
        candidate.resolve(strict=False).relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ManifestError(f"{location} escapes the repository root") from exc
    return path_text, candidate


def _validate_counts(value: Any, location: str) -> None:
    counts = _object(value, location)
    if not counts:
        raise ManifestError(f"{location} must not be empty")
    for name, count in counts.items():
        _string(name, f"{location} key")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ManifestError(f"{location}.{name} must be a non-negative integer")


def validate_manifest(manifest: Any, repo_root: Path) -> dict[str, int | str]:
    root = _object(manifest, "manifest")
    _exact_keys(
        root,
        {
            "schema_version",
            "repository",
            "audit",
            "relevant_commits",
            "artifact_inventory",
            "result_sets",
            "repository_completion_criteria",
        },
        "manifest",
    )
    if root["schema_version"] != 1:
        raise ManifestError("manifest.schema_version must be 1")
    _string(root["repository"], "manifest.repository")

    audit = _object(root["audit"], "manifest.audit")
    _exact_keys(
        audit,
        {"base_commit", "performed_on", "scope", "overall_status", "statement"},
        "manifest.audit",
    )
    base_commit = _sha(audit["base_commit"], "manifest.audit.base_commit")
    performed_on = _string(audit["performed_on"], "manifest.audit.performed_on")
    try:
        date.fromisoformat(performed_on)
    except ValueError as exc:
        raise ManifestError(
            "manifest.audit.performed_on must be an ISO calendar date"
        ) from exc
    _string(audit["scope"], "manifest.audit.scope")
    overall_status = _string(audit["overall_status"], "manifest.audit.overall_status")
    if overall_status not in OVERALL_STATUSES:
        raise ManifestError(
            "manifest.audit.overall_status is not an allowed status: "
            f"{overall_status!r}"
        )
    _string(audit["statement"], "manifest.audit.statement")

    commits = _array(root["relevant_commits"], "manifest.relevant_commits")
    if not commits:
        raise ManifestError("manifest.relevant_commits must not be empty")
    commit_shas: set[str] = set()
    commit_roles: set[str] = set()
    for index, raw_commit in enumerate(commits):
        location = f"manifest.relevant_commits[{index}]"
        commit = _object(raw_commit, location)
        _exact_keys(commit, {"sha", "role", "description"}, location)
        commit_sha = _sha(commit["sha"], f"{location}.sha")
        role = _string(commit["role"], f"{location}.role")
        _string(commit["description"], f"{location}.description")
        if commit_sha in commit_shas:
            raise ManifestError(f"duplicate relevant commit SHA: {commit_sha}")
        if role in commit_roles:
            raise ManifestError(f"duplicate relevant commit role: {role}")
        commit_shas.add(commit_sha)
        commit_roles.add(role)
    if base_commit not in commit_shas:
        raise ManifestError(
            "manifest.audit.base_commit must appear in relevant_commits"
        )

    result_sets = _array(root["result_sets"], "manifest.result_sets")
    if not result_sets:
        raise ManifestError("manifest.result_sets must not be empty")
    result_ids: set[str] = set()
    completion_criteria_count = 0
    for index, raw_result in enumerate(result_sets):
        location = f"manifest.result_sets[{index}]"
        result = _object(raw_result, location)
        _exact_keys(
            result,
            {"id", "status", "summary", "counts", "completion_criteria"},
            location,
        )
        result_id = _string(result["id"], f"{location}.id")
        if result_id in result_ids:
            raise ManifestError(f"duplicate result-set id: {result_id}")
        result_ids.add(result_id)
        status = _string(result["status"], f"{location}.status")
        if status not in RESULT_STATUSES:
            raise ManifestError(
                f"{location}.status is not an allowed status: {status!r}"
            )
        _string(result["summary"], f"{location}.summary")
        _validate_counts(result["counts"], f"{location}.counts")
        criteria = _unique_strings(
            result["completion_criteria"], f"{location}.completion_criteria"
        )
        completion_criteria_count += len(criteria)

    inventory = _object(root["artifact_inventory"], "manifest.artifact_inventory")
    _exact_keys(inventory, set(ARTIFACT_SECTIONS), "manifest.artifact_inventory")
    seen_paths: dict[str, str] = {}
    inventory_counts: dict[str, int] = {}
    for section in ARTIFACT_SECTIONS:
        entries = _array(inventory[section], f"manifest.artifact_inventory.{section}")
        inventory_counts[section] = len(entries)
        for index, raw_entry in enumerate(entries):
            location = f"manifest.artifact_inventory.{section}[{index}]"
            entry = _object(raw_entry, location)
            expected_keys = {"path", "kind", "supports", "reason"}
            if section == "quarantined":
                expected_keys.add("invalidated_by_commit")
            _exact_keys(entry, expected_keys, location)
            path_text, candidate = _safe_repo_path(
                entry["path"], f"{location}.path", repo_root
            )
            if path_text in seen_paths:
                raise ManifestError(
                    f"artifact path {path_text!r} appears in both "
                    f"{seen_paths[path_text]} and {section}"
                )
            seen_paths[path_text] = section
            kind = _string(entry["kind"], f"{location}.kind")
            if kind not in ARTIFACT_KINDS:
                raise ManifestError(f"{location}.kind is not allowed: {kind!r}")
            supports = _unique_strings(entry["supports"], f"{location}.supports")
            unknown_result_ids = set(supports) - result_ids
            if unknown_result_ids:
                raise ManifestError(
                    f"{location}.supports references unknown result sets: "
                    f"{sorted(unknown_result_ids)}"
                )
            _string(entry["reason"], f"{location}.reason")
            if section == "quarantined":
                invalidating_sha = _sha(
                    entry["invalidated_by_commit"],
                    f"{location}.invalidated_by_commit",
                )
                if invalidating_sha not in commit_shas:
                    raise ManifestError(
                        f"{location}.invalidated_by_commit is not in relevant_commits"
                    )

            if section in {"present", "quarantined"}:
                if not candidate.exists():
                    raise ManifestError(
                        f"artifact asserted {section} does not exist: {path_text}"
                    )
                if kind == "file" and not candidate.is_file():
                    raise ManifestError(
                        f"artifact asserted as file is not a file: {path_text}"
                    )
                if kind == "directory" and not candidate.is_dir():
                    raise ManifestError(
                        "artifact asserted as directory is not a directory: "
                        f"{path_text}"
                    )
            elif candidate.exists():
                raise ManifestError(f"artifact asserted missing exists: {path_text}")

    repository_criteria = _unique_strings(
        root["repository_completion_criteria"],
        "manifest.repository_completion_criteria",
    )
    completion_criteria_count += len(repository_criteria)

    return {
        "base_commit": base_commit,
        "overall_status": overall_status,
        "result_sets": len(result_sets),
        "present": inventory_counts["present"],
        "quarantined": inventory_counts["quarantined"],
        "missing": inventory_counts["missing"],
        "completion_criteria": completion_criteria_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        help="Manifest to validate (default: artifacts/validation_manifest.json).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        args.manifest
        or repo_root / "artifacts" / "validation_manifest.json"
    )
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path

    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        summary = validate_manifest(manifest, repo_root)
    except (ManifestError, json.JSONDecodeError, OSError) as exc:
        print(f"Validation manifest FAILED: {exc}", file=sys.stderr)
        return 1

    print("Validation manifest OK")
    print(f"  audit base: {summary['base_commit']}")
    print(f"  overall status: {summary['overall_status']}")
    print(f"  result sets: {summary['result_sets']}")
    print(
        "  artifacts: "
        f"present={summary['present']}, "
        f"quarantined={summary['quarantined']}, "
        f"missing={summary['missing']}"
    )
    print(f"  completion criteria: {summary['completion_criteria']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
