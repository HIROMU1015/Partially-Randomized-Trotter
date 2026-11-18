"""Add verified missing bundle bases to a WSL worker's private Git view."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time


def git(directory, *args, **kwargs):
    return subprocess.check_output(["git", *args], cwd=directory, **kwargs)


def missing(root, object_ids):
    result = git(root, "cat-file", "--batch-check",
                 input="\n".join(object_ids) + "\n", text=True)
    return [line.split()[0] for line in result.splitlines() if line.endswith(" missing")]


def install(root, source, supplement, pack, declaration):
    started = time.monotonic()
    if git(root, "rev-parse", "HEAD", text=True).strip() != source:
        raise ValueError("Fixed checkout SOURCE differs")
    ids = declaration["ids"]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch("[0-9a-f]{40}", x) for x in ids):
        raise ValueError("Invalid supplement object list")
    common = Path(git(root, "rev-parse", "--path-format=absolute",
                      "--git-common-dir", text=True).strip()).resolve()
    alternates = common / "objects/info/alternates"
    alternates.parent.mkdir(exist_ok=True)
    with alternates.with_name("local0-object-pack.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = missing(root, ids)
        if not before:
            return dict(status="REUSED", objects=len(ids), imported_objects=0,
                        seconds=round(time.monotonic() - started, 3))
        if set(before) != set(ids):
            raise ValueError("Pack contains already present objects; regenerate the minimal supplement")
        if hashlib.sha256(pack.read_bytes()).hexdigest() != declaration["pack_sha256"]:
            raise ValueError("Supplement pack SHA256 differs")
        supplement.mkdir(parents=True, exist_ok=True)
        if not (supplement / "HEAD").is_file():
            git(supplement, "init", "--bare", "-q")
        indexes = list((supplement / "objects/pack").glob("*.idx"))
        if not indexes:
            with pack.open("rb") as stream:
                git(supplement, "index-pack", "--stdin", "--strict", stdin=stream)
            indexes = list((supplement / "objects/pack").glob("*.idx"))
        index, = indexes
        git(supplement, "verify-pack", str(index))
        with index.open("rb") as stream:
            actual = git(supplement, "show-index", stdin=stream, text=True)
        if {line.split()[1] for line in actual.splitlines()} != set(ids):
            raise ValueError("Supplement contains unexpected objects")
        previous = alternates.read_text() if alternates.exists() else ""
        target = str(supplement.resolve() / "objects")
        with tempfile.NamedTemporaryFile(mode="w", dir=alternates.parent, delete=False) as update:
            update.write(previous.rstrip("\n") + ("\n" if previous else "") + target + "\n")
            temporary = Path(update.name)
        temporary.replace(alternates)
    if missing(root, ids):
        raise ValueError("Fixed view does not see the verified supplement")
    return dict(status="INSTALLED", source=source, checkout=str(root),
                supplement=str(supplement), objects=len(ids), imported_objects=len(ids),
                seconds=round(time.monotonic() - started, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "supplement", "pack", "declaration"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    declaration = json.loads(args.declaration.read_bytes())
    print(json.dumps(install(args.root, args.source, args.supplement, args.pack, declaration)))


if __name__ == "__main__":
    main()
