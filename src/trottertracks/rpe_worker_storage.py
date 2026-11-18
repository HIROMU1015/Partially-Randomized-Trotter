"""Host-wide intake limits for persistent optimization worker files."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

from .rpe_q3_coordinator import atomic_json


SAMPLE_SECONDS = 5


def _used_bytes(roots):
    existing = set()
    for path in roots:
        try:
            root = Path(path).resolve(strict=True)
        except FileNotFoundError:
            continue  # A worker may not have created this cache yet.
        if not root.is_dir():
            raise ValueError('Worker storage root is not a directory: ' + path)
        existing.add(root)
    ordered = sorted(existing, key=lambda root: (len(root.parts), str(root)))
    distinct = [root for i, root in enumerate(ordered)
                if not any(root.is_relative_to(parent) for parent in ordered[:i])]
    if not distinct:
        return 0
    output = subprocess.check_output(
        ['du', '--summarize', '--total', '--block-size=1', '--', *map(str, distinct)],
        text=True, stderr=subprocess.PIPE, timeout=10)
    return int(output.splitlines()[-1].split('\t', 1)[0])


def report(policy):
    """Measure allocated blocks once per interval across the host's workers."""
    if policy is None:
        return None
    limit, roots = policy['limit_bytes'], policy['roots']
    sample = Path(policy['sample_path'])
    if (type(limit) is not int or limit <= 0 or not roots
            or not isinstance(roots, list) or any(not Path(p).is_absolute() for p in roots)
            or not sample.is_absolute()):
        raise ValueError('Worker storage requires a positive byte limit and absolute paths')
    identity = dict(limit_bytes=limit, roots=roots)

    def recent():
        try:
            saved = json.loads(sample.read_bytes())
            if saved['policy'] == identity and 0 <= time.time() - saved['sampled_epoch'] < SAMPLE_SECONDS:
                return saved['report']
        except (OSError, KeyError, TypeError, ValueError):
            pass
        return None

    try:
        cached = recent()
        if cached is not None:
            return cached
        sample.parent.mkdir(parents=True, exist_ok=True)
        with sample.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = recent()
            if cached is not None:
                return cached
            try:
                used = _used_bytes(roots)
            except (OSError, ValueError, subprocess.SubprocessError):
                measured = dict(limit_bytes=limit, used_bytes=None, accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
            else:
                measured = dict(limit_bytes=limit, used_bytes=used, accepting_new_jobs=used < limit,
                                reason='STORAGE_LIMIT_REACHED' if used >= limit else None)
            atomic_json(sample, dict(policy=identity, sampled_epoch=time.time(), report=measured))
            return measured
    except OSError:
        return dict(limit_bytes=limit, used_bytes=None, accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
