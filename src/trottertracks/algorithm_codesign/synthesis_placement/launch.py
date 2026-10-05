"""Source S -> authorization-only child A, with a fresh SP05 one-shot marker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate_binding(auth, contract_hash, head, parents, changed, dirty, allowed):
    source = auth.get("source_commit")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise PermissionError("source commit is not fixed")
    if (auth.get("status") != "APPROVED_FOR_ONE_SP05_RUN"
            or auth.get("science_execution_authorized") is not True
            or auth.get("runs") != 1 or auth.get("retries") != 0
            or auth.get("mandatory_STOP") is not True):
        raise PermissionError("separate one-shot authorization is required")
    instruction = auth.get("explicit_execution_instruction")
    if not isinstance(instruction, str) or len(instruction.strip()) < 10:
        raise PermissionError("explicit execution instruction receipt missing")
    if auth.get("contract_sha256") != contract_hash:
        raise PermissionError("contract identity mismatch")
    if head == source or parents != [source] or dirty:
        raise PermissionError("clean direct authorization-only child required")
    if not changed or not set(changed).issubset(allowed):
        raise PermissionError("authorization commit changed a forbidden path")


def verify_launch(root, contract_path):
    root, contract_path = Path(root), Path(contract_path)
    contract = json.loads(contract_path.read_text())
    auth_path = root / contract["authorization_path"]
    auth = json.loads(auth_path.read_text())
    head = git(root, "rev-parse", "HEAD")
    # Pending auth fails before resolving any user-supplied Git revision.
    source = auth.get("source_commit")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise PermissionError("source review / authorization remains pending")
    parents = git(root, "show", "-s", "--format=%P", head).split()
    changed = git(root, "diff", "--name-only", source, head).splitlines()
    dirty = bool(git(root, "status", "--porcelain", "--untracked-files=all"))
    allowed = {contract["authorization_path"], contract["optional_receipt_path"]}
    validate_binding(auth, sha(contract_path), head, parents, changed, dirty, allowed)
    if contract["authorization_path"] not in changed:
        raise PermissionError("authorization JSON must be committed in child A")
    return contract, auth, head


def consume_marker(directory, receipt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "one_shot_consumed.json"
    # Never overwrite, remove, or retry after this succeeds.
    with marker.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    return marker
