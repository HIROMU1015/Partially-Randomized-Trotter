"""Read-only snapshot of pre-existing Python environments, profiles and SOURCE.

Print JSON to stdout; the caller chooses a new task-owned output file.
Tree hashes cover names, modes, sizes, mtime_ns and symlink destinations.
Critical files additionally have content SHA256 hashes.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = Path('/home/AbeHiromu/projects/partially-randomized-trotter')
STATE = REPO / '.server-preparation'
RUNTIME = STATE / 'rpe-distributed-q3'
PLAN = STATE / 'rpe-distributed-h4-plan-20261117'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(path):
    rows = []
    for current, directories, files in os.walk(path, followlinks=False):
        for name in sorted(directories + files):
            p = Path(current) / name
            st = p.lstat()
            rows.append([str(p.relative_to(path)), st.st_mode, st.st_size,
                         st.st_mtime_ns, os.readlink(p) if p.is_symlink() else None])
    rows.sort(key=lambda row: row[0])
    return dict(entries=len(rows), metadata_sha256=hashlib.sha256(
        json.dumps(rows, separators=(',', ':')).encode()).hexdigest())


def snapshot():
    environments = [Path('/home/AbeHiromu/venvs/trotter-common'),
                    RUNTIME / 'runtime/python311rc1', RUNTIME / 'runtime-acceptance']
    files = [Path('/usr/bin/python3.12'),
             Path('/home/AbeHiromu/venvs/trotter-common/pyvenv.cfg'),
             RUNTIME / 'runtime/python311rc1/bin/python3.11',
             RUNTIME / 'runtime-acceptance/bin/python3']
    for directory in ('profiles', 'assignments', 'dry'):
        files.extend(sorted((PLAN / directory).rglob('*')))
    # Saved successful Full PF/B0I receipts and original PR-DF failure stay intact.
    files = [p for p in files if p.is_file()]
    views = [STATE / 'worktrees/rpe-distributed-q3-source-28d976b99',
             STATE / 'worktrees/rpe-q3-route-23160cc87',
             Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees') /
             'd2-pr-df-ld-split-q3-source-20261115']
    source = {}
    for view in views:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=view, text=True).strip()
        dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=view, text=True).splitlines()
        tracked = subprocess.check_output(['git', 'ls-files', '-z', 'scripts', 'src'], cwd=view).decode().split('\0')
        contents = {name:sha(view / name) for name in tracked if name and (view / name).is_file()}
        source[str(view)] = dict(realpath=str(view.resolve()), head=head, tracked_dirty=dirty,
            files_count=len(contents), files_manifest_sha256=hashlib.sha256(
                json.dumps(contents, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    return dict(environments={str(p):tree(p) for p in environments},
                files={str(p):dict(realpath=str(p.resolve()), sha256=sha(p)) for p in files},
                fixed_views=source)


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2, sort_keys=True))
