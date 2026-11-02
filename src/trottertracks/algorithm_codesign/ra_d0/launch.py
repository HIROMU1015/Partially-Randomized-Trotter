"""Future direct-child authorization gate. No authorization is provided here."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

SOURCE_DIR = "artifacts/track_b_ra_d0_source_review_v3/2026-10-07"
AUTH_PATH = SOURCE_DIR+"/authorization.json"
RECEIPT_PATH = "docs/tracks/algorithm_codesign/ra_d0_execution_authorization_receipt.md"
MANIFEST_PATH = SOURCE_DIR+"/source_manifest_v3.json"
OUTPUT_PATH = "artifacts/track_b_ra_d0_development_result/2026-10-07/v3"


@dataclass
class ExecutionPermit:
    root: Path
    source_commit: str
    authorization_commit: str
    authorization_sha256: str
    active: bool = False

    def assert_active(self):
        if not self.active:
            raise PermissionError("one-shot marker not exclusively consumed")


def validate_authorization(auth, head, parents, changed_paths, contract_sha256):
    if (auth.get("status") != "APPROVED_FOR_ONE_RA_D0_RUN"
            or auth.get("development_execution_authorized") is not True
            or auth.get("science_execution_authorized") is not False
            or auth.get("runs") != 1 or auth.get("retries") != 0
            or auth.get("mandatory_STOP") is not True
            or not isinstance(auth.get("explicit_execution_instruction"), str)
            or not auth["explicit_execution_instruction"].strip()
            or auth.get("contract_sha256") != contract_sha256):
        raise PermissionError("separate explicit one-shot authorization required")
    source = auth.get("source_commit")
    if not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{40}", source) is None or parents != [source] or head == source:
        raise PermissionError("HEAD must be an authorization-only direct child of source S")
    if AUTH_PATH not in changed_paths or set(changed_paths)-{AUTH_PATH, RECEIPT_PATH}:
        raise PermissionError("authorization-only path violation")


def verify_launch(root):
    root = Path(root)
    # This fails before marker/table/minima access in the review-only source.
    raw = (root/AUTH_PATH).read_bytes()
    auth = json.loads(raw)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    remote = git("remote", "get-url", "origin").decode().strip()
    if not (remote.startswith("git@github.com:HIROMU1015/")
            or remote.startswith("https://github.com/HIROMU1015/")):
        raise PermissionError("repository owner is not HIROMU1015")
    head = git("rev-parse", "HEAD").decode().strip()
    parents = git("show", "-s", "--format=%P", "HEAD").decode().strip().split()
    changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", "HEAD").decode().split("\0")
    changed = [p for p in changed if p]
    contract = root/SOURCE_DIR/"execution_contract_v3.json"
    validate_authorization(auth, head, parents, changed, hashlib.sha256(contract.read_bytes()).hexdigest())
    if git("status", "--porcelain", "--untracked-files=normal").strip():
        raise PermissionError("execution HEAD must start clean")
    manifest = json.loads((root/MANIFEST_PATH).read_text())
    for path, expected in manifest["critical_sha256"].items():
        if path.lower().endswith(".npz"):
            raise PermissionError("forbidden molecular identity path")
        if hashlib.sha256((root/path).read_bytes()).hexdigest() != expected:
            raise PermissionError("source/input/runtime identity mismatch: "+path)
    from importlib import metadata
    import platform
    runtime = manifest["runtime_identity"]
    if platform.python_version() != runtime["python"]:
        raise PermissionError("runtime Python mismatch")
    for package, identity in runtime["packages"].items():
        dist = metadata.distribution(package)
        if (dist.version != identity["version"] or
                hashlib.sha256(dist.read_text("RECORD").encode()).hexdigest() != identity["RECORD_sha256"]):
            raise PermissionError("runtime package identity mismatch")
    return ExecutionPermit(root, auth["source_commit"], head, hashlib.sha256(raw).hexdigest())


def consume_marker(permit):
    output = permit.root/OUTPUT_PATH
    output.mkdir(parents=True, exist_ok=True)
    value = {"source_commit": permit.source_commit, "authorization_commit": permit.authorization_commit,
             "authorization_sha256": permit.authorization_sha256, "runs": 1, "retries": 0,
             "mandatory_STOP": True, "next_stage_authorized": False}
    with (output/"one_shot_consumed.json").open("x") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
    permit.active = True
    return output
