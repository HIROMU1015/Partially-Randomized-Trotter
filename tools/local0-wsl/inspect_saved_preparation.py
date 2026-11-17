"""Validate and clone a saved preparation request without invoking its provider."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_bytes())


def inspect(request_path, profile_path, output, source_root, source_commit):
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if actual != source_commit:
        raise ValueError("Inspection transport SOURCE differs")
    sys.path.insert(0, str(source_root / "src"))
    from trottertracks import rpe_preparation_worker as worker
    from trottertracks.rpe_q3_coordinator import digest

    request = read(request_path)
    profile = deepcopy(read(profile_path))
    if profile["host_id"] != "local0-wsl":
        raise ValueError("Expected a local0-wsl profile")
    output.mkdir(parents=True, exist_ok=False)
    for rule in profile.get("preparation_profiles", []):
        rule["profile"]["preparation_metadata_cache_directory"] = str(output / "metadata")
    started = time.monotonic()
    checked = worker.dry_check(request, profile)
    dry_seconds = time.monotonic() - started
    fingerprint = checked["fingerprint"]
    spec = request["preparation"]
    assignment = dict(
        request=request,
        request_sha256=digest(request),
        job_id=request["request_id"],
        source_manifest=spec["source"],
    )
    selected = worker.select_preparation_profile(profile, assignment)
    cache = worker.worker_cache(selected)
    artifact_inputs = worker.assigned_artifacts(assignment)
    started = time.monotonic()
    cache.materialize(artifact_inputs, output / "inputs")
    root = Path(fingerprint["views"][spec["source"]["commit"]])
    owned = {}
    if spec.get("source_workspace"):
        target = worker._clone(
            root, output / "workspace", spec.get("input_git_bundles", []),
            cache, source_files=spec["source"]["files"],
        )
        owned["source"] = (target, spec["source"])
    for alias, definition in spec.get("workspaces", {}).items():
        target = worker._clone(
            fingerprint["views"][alias], output / "workspaces" / alias,
            definition.get("input_git_bundles", []), cache,
            definition.get("checkout_input_head", True), definition["source"]["files"],
        )
        owned[alias] = (target, definition["source"])
    worker._include_declared_outputs(owned, output, [
        *spec["outputs"], *spec.get("optional_outputs", []),
    ])
    destinations = spec.get("materialize_inputs")
    if destinations is None:
        destinations = {
            name: name for name in artifact_inputs
            if name.startswith(("workspace/", "workspaces/", "private/"))
        }
    for name, destination in destinations.items():
        data = (output / "inputs" / worker.artifact_path(name)).read_bytes()
        target = output / worker.artifact_path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Saved input differs from existing fixed workspace bytes")
        if not target.exists():
            target.write_bytes(data)
    for path, manifest in owned.values():
        worker._guard_owned_source(path, manifest)
    result = dict(
        request_id=request["request_id"],
        request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
        transport_source=source_commit,
        fixed_source=spec["source"]["commit"],
        dry_status=checked["status"],
        fixed_input_count=len(spec.get("input_hashes", [])),
        materialized_artifacts=len(artifact_inputs),
        working_input_destinations=len(destinations),
        fresh_clone_count=len(owned),
        runtime=fingerprint["runtime"],
        preflight_seconds=round(dry_seconds, 3),
        workspace_seconds=round(time.monotonic() - started, 3),
        scientific_calls=0,
        queue_operations=0,
    )
    (output / "REPORT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("request", "profile", "output", "transport-root"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--transport-source", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(
        args.request, args.profile, args.output.resolve(),
        args.transport_root.resolve(), args.transport_source,
    )))


if __name__ == "__main__":
    main()
