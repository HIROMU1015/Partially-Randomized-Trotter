"""Host-wide intake limits and reclamation of acknowledged worker copies."""
from contextlib import ExitStack
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time

from .rpe_q3_coordinator import atomic_json
from .rpe_preparation_artifacts import worker_cache
from .rpe_dp_prefix_reuse import PrefixStore


SAMPLE_SECONDS = 5
PREPARATION_COPIES = ('inputs', 'workspace', 'workspaces')
GIT_TEMP_PACK_MIN_AGE = 300


def reclaim_watermarks(limit, *, proactive):
    """Use an earlier high-water mark in the independent storage watcher."""
    high, low = (80, 70) if proactive else (95, 90)
    return limit * high // 100, limit * low // 100


def duplicate_reclaim_target(usage):
    """Free only enough verified copies to restore the existing storage headroom."""
    if usage['reason'] not in (None, 'STORAGE_LIMIT_REACHED', 'FILESYSTEM_SPACE_LOW'):
        return 0
    limit, used = usage['limit_bytes'], usage['used_bytes']
    if used is None:
        return 0
    high, low = reclaim_watermarks(limit, proactive=True)
    underway = usage.get('reclaim_start_bytes') == high and usage.get('reclaim_pending') is True
    quota = used - low if used >= high or underway else 0
    free, minimum = usage.get('filesystem_free_bytes'), usage.get('filesystem_min_free_bytes', 0)
    pressure = 2 * minimum - free if free is not None and free < minimum else 0
    return max(0, quota, pressure)


def duplicate_inventory(profile):
    policy = profile.get('storage_limit') or {}
    cache_path = profile.get('artifact_cache_directory')
    if not cache_path or cache_path not in policy.get('roots', []):
        return {}
    cache = Path(cache_path)
    return {path.name: path.stat().st_size for path in cache.glob('*/*')
            if path.parent.name == path.name[:2] and re.fullmatch('[a-f0-9]{64}', path.name)
            and path.resolve() == path and path.is_file() and path.stat().st_nlink == 1}


def _referenced_hashes(value):
    if isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value):
        yield value
    elif isinstance(value, (dict, list)):
        for item in value.values() if isinstance(value, dict) else value:
            yield from _referenced_hashes(item)


def compute_storage_observation(policy, samples, *, max_age=330):
    """Read authoritative compute-node measurements without a login-node du.

    Use the shared filesystem modification time for age (node clocks differ).
    Reports are maintenance observations, not new worker admission offers.
    """
    if not policy:
        return None
    observations = []
    for value in samples:
        path = Path(value)
        try:
            if path.is_symlink() or (path.parent/'OFFLINE_ACTIVE').exists():
                continue
            data = json.loads(path.read_bytes())
            age = time.time() - path.stat().st_mtime
            measured = data['report']
            if (data['policy'] != policy_identity(policy) or not 0 <= age <= max_age
                    or measured['limit_bytes'] != policy['limit_bytes']
                    or type(measured['used_bytes']) is not int or measured['used_bytes'] < 0):
                continue
            observations.append(dict(report=measured, age_seconds=age, path=str(path)))
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return min(observations, key=lambda value: value['age_seconds'], default=None)


def _compute_quota_hold(policy, samples):
    """Only an actual hard limit observation fences a legacy live spool."""
    if not samples:
        return False
    for path in samples:
        observation = compute_storage_observation(policy, [path])
        if observation is None:
            return False
        measured = observation['report']
        if (measured['accepting_new_jobs'] is not False or
                measured['reason'] != 'STORAGE_LIMIT_REACHED' or
                measured['used_bytes'] < policy['limit_bytes']):
            return False
    return True


def reclaim_verified_duplicates(profile, proof, *, spools=None, compute_samples=(), diagnostic=None):
    """Reclaim transport cache copies verified from the central CAS."""
    policy = profile['storage_limit']
    diagnostic = diagnostic if diagnostic is not None else {}
    sample = Path(policy['sample_path'])
    spools = sorted(set(map(Path, spools or [profile['work_dir']])))
    cache = Path(profile.get('artifact_cache_directory') or '.')
    if str(cache) not in policy['roots'] or not spools:
        diagnostic['reason'] = 'CACHE_OR_SPOOL_OUTSIDE_POLICY'
        return {}
    with ExitStack() as locks:
        fenced_live_spool = False
        for spool in spools:
            if spool.resolve() != spool or not spool.is_dir() or spool.stat().st_uid != os.getuid():
                diagnostic.update(reason='SPOOL_UNAVAILABLE', spool=str(spool))
                return {}
            worker_lock = locks.enter_context((spool/'WORKER.lock').open('a+b'))
            try:
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # The active allocation daemon retains WORKER.lock even when
                # its real quota prevents intake. Its already-owned inputs
                # remain pinned; a verified unowned CAS copy is refetchable.
                if spool != Path(profile['work_dir']) or not _compute_quota_hold(policy, compute_samples):
                    diagnostic.update(reason='WORKER_LOCKED_NO_CONFIRMED_QUOTA_HOLD', spool=str(spool))
                    return {}
                fenced_live_spool = True
        storage_lock = locks.enter_context(sample.with_suffix('.lock').open('r+b'))
        fcntl.flock(storage_lock, fcntl.LOCK_EX)
        observation = compute_storage_observation(policy, compute_samples)
        if observation is None:
            observation = json.loads(sample.read_bytes())
            if observation['policy'] != policy_identity(policy):
                return {}
        report = observation['report']
        target = duplicate_reclaim_target(report)
        if target <= 0:
            return {}
        retained = set()
        for marker in (marker for spool in spools for marker in spool.glob('*/*/WORKER_STATE.json')):
            status = json.loads(marker.read_bytes())['state']
            if status == 'TRANSFERRED':
                continue
            if status == 'COMPLETED_PENDING_TRANSFER':
                # An ended task without its final ACK retains all referenced
                # inputs and outputs. It need not pin unrelated cache copies.
                if not all(marker.with_name(name).is_file() for name in
                           ('ASSIGNMENT.json', 'OUTCOME.json', 'PROCESS_RETURN.json')):
                    return {}
            elif status == 'ASSIGNED':
                if not marker.with_name('ASSIGNMENT.json').is_file():
                    return {}
            elif status in ('STARTING', 'RUNNING'):
                # A stale historical RUNNING marker does not pin the entire
                # host. Hold its individual execution lock for this cleanup,
                # retain its referenced blobs, and leave its state untouched.
                process = marker.with_name('WORKER_PROCESS.lock')
                if (not marker.with_name('ASSIGNMENT.json').is_file()
                        or process.is_symlink() or not process.is_file()):
                    diagnostic.update(reason='ATTEMPT_LOCK_UNAVAILABLE', attempt=str(marker.parent))
                    return {}
                attempt_lock = locks.enter_context(process.open('rb'))
                try:
                    fcntl.flock(attempt_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    diagnostic.update(reason='WORKER_HAS_ACTIVE_ATTEMPT', attempt=str(marker.parent), state=status)
                    return {}
            elif status != 'UNKNOWN':
                diagnostic.update(reason='WORKER_HAS_ACTIVE_ATTEMPT', attempt=str(marker.parent), state=status)
                return {}
            for name in ('ASSIGNMENT.json', 'ARTIFACT_MANIFEST.json', 'OUTCOME.json'):
                path = marker.with_name(name)
                if path.is_file():
                    retained.update(_referenced_hashes(json.loads(path.read_bytes())))
        # Older metadata caches record only imported hashes. Keep their ref
        # selector receipts before removing the last local raw bundle copy.
        imported = {}
        metadata_root = Path(profile.get('preparation_metadata_cache_directory') or
                             Path(profile['work_dir']).parent/'preparation-metadata-cache')
        try:
            from .rpe_preparation_worker import remember_imported_bundle
        except ImportError:  # A maintenance process on an older fixed worker.
            remember_imported_bundle = None
        if remember_imported_bundle is not None:
            for marker in metadata_root.glob('*/IMPORTED.json'):
                try:
                    repo = marker.parent/'objects.git'
                    for sha in json.loads(marker.read_bytes()):
                        if sha in proof and repo.is_dir():
                            imported.setdefault(sha, []).append(repo)
                except (OSError, ValueError, TypeError):
                    continue
        inventory = duplicate_inventory(profile)
        removed = {}
        for sha, size in proof.items():
            if fenced_live_spool and not _compute_quota_hold(policy, compute_samples):
                diagnostic['reason'] = 'QUOTA_HOLD_CLEARED'
                break
            if sha in retained or inventory.get(sha) != size:
                continue
            path = cache/sha[:2]/sha
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != sha:
                    continue
            try:
                if any(not remember_imported_bundle(repo, path, sha) for repo in imported.get(sha, ())):
                    continue
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
            removed[sha] = path.stat().st_blocks * 512
            path.unlink()
            if sum(removed.values()) >= target:
                break
        diagnostic['removed'] = len(removed)
        diagnostic.setdefault('reason', 'REMOVED' if removed else 'NO_UNPINNED_MATCHING_COPY')
        return removed


def prefix_caches(profile):
    profiles = [profile, *(rule['profile'] for rule in profile.get('route_profiles', []))]
    return tuple(dict.fromkeys(p['prefix_cache_directory'] for p in profiles
                               if p.get('prefix_cache_directory')))


def storage_exclusion_file(policy):
    return Path(policy.get('exclusion_file', Path(policy['sample_path']).with_name('STORAGE_EXCLUSIONS.json')))


def policy_identity(policy):
    """Account for explicitly designated, retained legacy paths only.

    Exclusions are exact absolute descendants, not glob patterns. They affect
    quota accounting, never the actual filesystem-free check or raw retention.
    A removed config immediately invalidates a prior discounted sample.
    """
    identity = dict(limit_bytes=policy['limit_bytes'], roots=policy['roots'])
    try:
        config = json.loads(storage_exclusion_file(policy).read_bytes())
    except FileNotFoundError:
        return identity
    paths = config.get('excluded_paths')
    if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
        raise ValueError('Invalid storage exclusions')
    roots = [Path(p).resolve() for p in policy['roots']]
    checked = []
    for name in paths:
        path = Path(name)
        if (not path.is_absolute() or path.resolve() != path or any(c in name for c in '*?[]\\')
                or not any(path != root and path.is_relative_to(root) for root in roots)):
            raise ValueError('Storage exclusion is not an exact monitored descendant')
        checked.append(str(path))
    if checked:
        identity['excluded_paths'] = sorted(set(checked))
    return identity


def _used_bytes(roots, *, timeout=10, excluded_paths=()):
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
    command = ['du', '--summarize', '--total', '--block-size=1',
               *('--exclude='+path for path in excluded_paths), '--', *map(str, distinct)]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.CalledProcessError as error:
        # A worker can retire files during du. Retry only that ordinary race;
        # permission errors and a real timeout must remain unavailable.
        errors = [line for line in (error.stderr or '').splitlines() if line.strip()]
        if not errors or not all('No such file or directory' in line for line in errors):
            raise
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=timeout)
    return int(output.splitlines()[-1].split('\t', 1)[0])


def _attempt_copies(marker):
    attempt = marker.parent
    if attempt.resolve() != attempt or marker.is_symlink():
        return
    try:
        state = json.loads(marker.read_bytes())
        if state.get('state') != 'TRANSFERRED':
            return
        assignment = json.loads((attempt/'ASSIGNMENT.json').read_bytes())
        if (assignment['job_id'], assignment['attempt_id']) != (attempt.parent.name, attempt.name):
            return
        paths = [(0, attempt/'_dry_check')]
        if assignment['request']['route']['kind'] == 'preparation':
            outcome = json.loads((attempt/'OUTCOME.json').read_bytes())
            receipt = json.loads((attempt/'PROCESS_RETURN.json').read_bytes())
            matches = (outcome['request_id'] == assignment['job_id']
                       and outcome['request_sha256'] == assignment['request_sha256'])
            if (matches and outcome['status'] == 'PREPARATION_COMPLETE'
                    and receipt.get('returncode') == 0 and receipt.get('signal') is None):
                artifacts = outcome['artifacts']
                profile = json.loads((attempt/'WORKER_PROFILE.json').read_bytes())
                cache = worker_cache(profile)
                if all(cache.path(sha).is_file() for sha in set(artifacts.values())):
                    paths += [(1, attempt/name) for name in PREPARATION_COPIES]
                    paths += [(2, path) for path in attempt.glob('*-output.bundle') if path.name in artifacts]
            elif (matches and outcome['status'] == 'PREPARATION_FAILED'
                    and outcome.get('workspace_error', {}).get('scientific_body_started') is False
                    and outcome.get('artifacts') == {} and outcome.get('git_bundles') == []
                    and type(receipt.get('returncode')) is int and receipt['returncode'] > 0
                    and receipt.get('signal') is None and not (attempt/'PREPARATION_BODY.log').exists()):
                paths += [(1, attempt/name) for name in PREPARATION_COPIES]
        for priority, path in paths:
            if path.exists() and not path.is_symlink():
                yield priority, marker.stat().st_mtime_ns, path
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return


def _acknowledged_copies(spools, roots):
    monitored = {Path(root).resolve() for root in roots}
    for spool in sorted({Path(path).resolve() for path in spools}):
        if any(spool.is_relative_to(root) for root in monitored):
            for marker in spool.glob('*/*/WORKER_STATE.json'):
                yield from _attempt_copies(marker)


def _remove_copy(path):
    lock = path.parent/'WORKER_PROCESS.lock'
    try:
        if not stat.S_ISREG(lock.lstat().st_mode):
            return False
        with lock.open('r+b') as process_lock:
            fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if path.is_symlink():
                return False
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return True
    except OSError:
        return False


def compact_failed_preparation_workspaces(attempt):
    """Drop only clean, reproducible source files after a failed result is ACKed.

    All scientific output, modified/untracked (including ignored) files, input
    paths and Git objects remain. A running or uncertain owner is never touched.
    """
    attempt = Path(attempt)
    lock = attempt/'WORKER_PROCESS.lock'
    layout = attempt/'FAILED_WORKSPACE_LAYOUT.json'
    try:
        if attempt.resolve() != attempt or not stat.S_ISREG(lock.lstat().st_mode):
            return 0
        with lock.open('r+b') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = json.loads((attempt/'WORKER_STATE.json').read_bytes())
            assignment = json.loads((attempt/'ASSIGNMENT.json').read_bytes())
            if (state.get('state') != 'TRANSFERRED' or assignment['request']['route']['kind'] != 'preparation'
                    or (assignment['job_id'], assignment['attempt_id']) != (attempt.parent.name, attempt.name)):
                return 0
            outcome = json.loads((attempt/'OUTCOME.json').read_bytes())
            receipt = json.loads((attempt/'PROCESS_RETURN.json').read_bytes())
            if (outcome['status'] != 'PREPARATION_FAILED'
                    or outcome['request_id'] != assignment['job_id']
                    or outcome['request_sha256'] != assignment['request_sha256']
                    or (receipt.get('returncode') in (None, 0) and receipt.get('signal') is None)):
                return 0
            spec = assignment['request']['preparation']
            declared = list(spec.get('outputs', [])) + list(spec.get('optional_outputs', []))
            declared += list((spec.get('materialize_inputs') or {}).values())
            declared += list(assignment.get('artifact_manifest', {}))
            declared += list(outcome.get('artifacts', {}))
            declared += [row['path'] for row in spec.get('working_input_hashes', [])]
            if spec.get('materialize_inputs') is None:
                inputs = spec.get('inputs', {})
                declared += list(inputs) if isinstance(inputs, dict) else [row['path'] for row in inputs]
            records = json.loads(layout.read_bytes()) if layout.is_file() else {}
            count = 0
            for workspace in [attempt/'workspace', *sorted((attempt/'workspaces').glob('*'))]:
                if (not workspace.is_dir() or workspace.resolve() != workspace or
                        not (workspace/'.git').is_dir()):
                    continue
                prefix = workspace.relative_to(attempt).as_posix()
                source = spec['source'] if prefix == 'workspace' else spec['workspaces'][workspace.name]['source']
                def git(*args):
                    return subprocess.check_output(['git', *args], cwd=workspace,
                                                   stderr=subprocess.PIPE, timeout=30)
                head = git('rev-parse', 'HEAD').decode().strip()
                saved = records.get(prefix, {})
                if saved and saved['head'] != head:
                    continue
                # Only source files identical to the immutable base qualify.
                # Leave tracked provider changes, partial outputs and all
                # untracked files, even if Git would consider them clean.
                def blobs(revision):
                    raw = git('ls-tree', '-r', '-z', revision, '--', 'src', 'scripts',
                              'configs', 'schemas', 'references')
                    return {os.fsdecode(name): header.split()[2] for row in raw.split(b'\0') if row
                            for header, name in [row.split(b'\t',1)]
                            if header.split()[0] in (b'100644', b'100755')}
                base_blobs = blobs(source['commit'])
                current_blobs = blobs(head)
                candidates = {}
                for name, oid in base_blobs.items():
                    if current_blobs.get(name) != oid or '\n' in name or '\r' in name:
                        continue
                    rel = Path(prefix)/name
                    if any(rel.is_relative_to(Path(value)) for value in declared):
                        continue
                    path = workspace/name
                    if path.is_symlink() or not path.is_file() or path.resolve() != path:
                        continue
                    candidates[name] = oid
                if not candidates:
                    continue
                # A damaged metadata cache may be missing the only recoverable
                # Git object. Verify source availability and actual file bytes.
                names = sorted(candidates)
                oids = set(candidates.values())
                checked = subprocess.check_output(['git','cat-file','--batch-check'],
                    cwd=workspace, input=b'\n'.join(sorted(oids))+b'\n',
                    stderr=subprocess.PIPE,timeout=30).splitlines()
                known = {oid for oid,row in zip(sorted(oids),checked) if not row.endswith(b' missing')}
                if len(checked) != len(oids):
                    continue
                names = [name for name in names if candidates[name] in known]
                if not names:
                    continue
                actual = subprocess.check_output(['git','hash-object','--no-filters','--stdin-paths'],
                    cwd=workspace, input=os.fsencode('\n'.join(names)+'\n'),
                    stderr=subprocess.PIPE,timeout=30).splitlines()
                if len(actual) != len(names):
                    continue
                removable = [name for name,oid in zip(names,actual) if candidates[name] == oid]
                if not removable:
                    continue
                # Do not run sparse-checkout set here: it can unexpectedly
                # materialize other cones and increase the storage footprint.
                # The terminal workspace is for inspection only; record exact
                # removed paths before unlinking, and restore only those paths.
                records[prefix] = dict(saved, head=head,
                    files=sorted(set(saved.get('files', [])) | set(removable)))
                atomic_json(layout, records)
                for name in removable:
                    (workspace/name).unlink()
                count += 1
            records['_cleanup_complete'] = True
            atomic_json(layout, records)
            return count
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return 0


def restore_failed_workspace(attempt, relative='workspace'):
    """Restore a compacted terminal workspace for inspection, not execution."""
    attempt = Path(attempt).resolve()
    workspace = attempt/relative
    if not workspace.resolve().is_relative_to(attempt) or workspace.resolve() != workspace:
        raise ValueError('Invalid retained workspace')
    with (attempt/'WORKER_PROCESS.lock').open('r+b') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads((attempt/'FAILED_WORKSPACE_LAYOUT.json').read_bytes())[relative]
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip() != record['head']:
            raise ValueError('Retained workspace HEAD changed')
        if record.get('cones'):  # Layout from the earlier sparse-cone implementation.
            subprocess.run(['git', 'sparse-checkout', 'add', '--stdin'], cwd=workspace,
                input='\n'.join(record['cones'])+'\n', text=True, check=True, capture_output=True)
        missing = [name for name in record.get('files', [])
                   if not (workspace/name).exists() and not (workspace/name).is_symlink()]
        if missing:
            subprocess.run(['git','--literal-pathspecs','restore','--source='+record['head'],
                            '--worktree','--ignore-skip-worktree-bits',
                            '--pathspec-from-file=-','--pathspec-file-nul'],
                cwd=workspace, input=b'\0'.join(map(os.fsencode,missing))+b'\0',
                check=True,capture_output=True)



def reclaim_transferred_attempt(policy, attempt):
    """Discard acknowledged working copies, retaining receipts and result blobs."""
    if policy is None or not any(attempt.is_relative_to(Path(root).resolve()) for root in policy['roots']):
        return
    for _, _, path in _attempt_copies(attempt/'WORKER_STATE.json'):
        _remove_copy(path)
    # Failed workspaces are compacted by the existing storage maintenance
    # pass, not synchronously in ACK/claim; Git and I/O must not delay the
    # next job.


def pending_pbs_completions(spool):
    """Inventory completed attempts whose worker session has not recorded an ACK."""
    candidates = []
    for marker in Path(spool).glob('*/*/WORKER_STATE.json'):
        try:
            state = json.loads(marker.read_bytes())
            if state['state'] == 'COMPLETED_PENDING_TRANSFER':
                candidates.append(dict(job_id=marker.parent.parent.name, attempt_id=marker.parent.name))
        except (OSError, KeyError, ValueError):
            continue
    return candidates


def acknowledge_pbs_completions(profile, spool, receipts):
    """Retire exactly the completion bytes already stored by the controller."""
    from .rpe_q3_worker import digest, event, result_bundle, safe_id, save_state

    spool = Path(spool)
    policy = profile.get('storage_limit')
    lock = spool/'WORKER.lock'
    if (not policy or not spool.is_dir() or spool.resolve() != spool
            or spool.stat().st_uid != os.getuid() or not lock.is_file() or lock.is_symlink()
            or not any(spool.is_relative_to(Path(root).resolve()) for root in policy['roots'])):
        return dict(acknowledged=0, reason='PBS_SPOOL_UNAVAILABLE')
    worker_profile = dict(profile, work_dir=str(spool))
    acknowledged = mismatch = deferred = 0
    with lock.open('rb') as worker_lock:
        try:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(acknowledged=0, reason='PBS_WORKER_OWNS_SPOOL')
        for receipt in receipts:
            try:
                job, attempt_id = (safe_id(receipt[key]) for key in ('job_id', 'attempt_id'))
                attempt = spool/job/attempt_id
                if attempt.resolve() != attempt or not attempt.is_dir():
                    raise ValueError('Unexpected PBS attempt path')
                paths = [attempt/name for name in ('ASSIGNMENT.json', 'WORKER_STATE.json', 'WORKER_PROCESS.lock')]
                if any(path.is_symlink() for path in paths):
                    raise ValueError('Unexpected PBS attempt metadata path')
                assignment = json.loads(paths[0].read_bytes())
                state = json.loads(paths[1].read_bytes())['state']
                request_sha = receipt['request_sha256']
                identity = (assignment.get('host_id', profile['host_id']), assignment['job_id'],
                            assignment['attempt_id'], assignment['request_sha256'])
                if identity != (profile['host_id'], job, attempt_id, request_sha):
                    raise ValueError('PBS attempt owner differs from central receipt')
                if state != 'COMPLETED_PENDING_TRANSFER':
                    continue
                with paths[2].open('rb') as execution_lock:
                    fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    completion_sha = receipt['completion_sha256']
                    if digest(assignment['request']) != request_sha or digest(result_bundle(attempt)) != completion_sha:
                        raise ValueError('PBS request or completion digest differs')
                    save_state(attempt, 'TRANSFERRED', via='PBS_HARVEST_ACK_RECOVERY',
                               request_sha256=request_sha, completion_sha256=completion_sha)
                    acknowledged += 1
                event(worker_profile, 'TRANSFER_ACK', job_id=job, attempt_id=attempt_id,
                      via='PBS_HARVEST_ACK_RECOVERY')
                reclaim_transferred_attempt(policy, attempt)
            except (KeyError, ValueError, TypeError):
                mismatch += 1
            except OSError:
                deferred += 1
    return dict(acknowledged=acknowledged, mismatch=mismatch, deferred=deferred)


def _reclaim(spools, roots, used, target, *, du_timeout=10, excluded_paths=()):
    copies = sorted(_acknowledged_copies(spools, roots))
    removed = 0
    # Recheck the host total after each batch. Separate copies can share blocks.
    for offset in range(0, len(copies), 32):
        paths = [str(path) for _, _, path in copies[offset:offset+32]]
        output = subprocess.check_output(
            ['du', '--summarize', '--block-size=1', '--', *paths],
            text=True, stderr=subprocess.PIPE, timeout=du_timeout)
        sizes = {name: int(size) for size, name in (line.split('\t', 1) for line in output.splitlines())}
        remaining = used
        for filename in paths:
            if _remove_copy(Path(filename)):
                removed += 1
                remaining -= sizes.get(filename, 0)
                if remaining <= target:
                    break
        used = _used_bytes(roots, timeout=du_timeout, excluded_paths=excluded_paths)
        if used <= target:
            break
    if used > target:
        monitored = [Path(root).resolve() for root in roots]
        compacted = 0
        for spool in sorted({Path(path).resolve() for path in spools}):
            if not any(spool.is_relative_to(root) for root in monitored):
                continue
            for marker in spool.glob('*/*/WORKER_STATE.json'):
                layout = marker.parent/'FAILED_WORKSPACE_LAYOUT.json'
                if layout.is_file():
                    try:
                        if json.loads(layout.read_bytes()).get('_cleanup_complete'):
                            continue
                    except (OSError, ValueError):
                        continue
                if compact_failed_preparation_workspaces(marker.parent):
                    compacted += 1
                    removed += 1
                    if compacted >= 8:
                        break
            if compacted >= 8:
                break
        if compacted:
            used = _used_bytes(roots, timeout=du_timeout, excluded_paths=excluded_paths)
    return used, removed


def _share_transferred_git_packs(spools, roots, bytes_needed):
    """Share byte-identical immutable Git packs in acknowledged workspaces."""
    monitored = [Path(root).resolve() for root in roots]
    reclaimed = 0
    for spool in sorted({Path(path).resolve() for path in spools}):
        if not any(spool.is_relative_to(root) for root in monitored):
            continue
        for marker in spool.glob('*/*/WORKER_STATE.json'):
            if reclaimed >= bytes_needed:
                return reclaimed
            attempt = marker.parent
            if marker.resolve() != marker or not (attempt/'workspaces').is_dir():
                continue
            lock = attempt/'WORKER_PROCESS.lock'
            try:
                if not stat.S_ISREG(lock.lstat().st_mode):
                    continue
                with lock.open('r+b') as process_lock:
                    fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if json.loads(marker.read_bytes()).get('state') != 'TRANSFERRED':
                        continue
                    reclaimed += _share_attempt_git_packs(attempt)
            except (OSError, ValueError):
                continue
    return reclaimed


def _share_attempt_git_packs(attempt):
    copies = sorted(attempt.glob('workspaces/*/.git/objects/pack/pack-*'))
    original = attempt/'workspace/.git/objects/pack'
    reclaimed = 0
    for path in copies:
        source = original/path.name
        if (not re.fullmatch(r'pack-[a-f0-9]{40,64}\.(pack|idx)', path.name)
                or source.resolve() != source or path.resolve() != path):
            continue
        try:
            left, right = source.lstat(), path.lstat()
        except FileNotFoundError:
            continue
        if (not stat.S_ISREG(left.st_mode) or left.st_mode != right.st_mode
                or left.st_mode & 0o222 or left.st_size != right.st_size
                or left.st_dev != right.st_dev or left.st_ino == right.st_ino
                or (left.st_uid, left.st_gid) != (right.st_uid, right.st_gid)):
            continue
        with source.open('rb') as a, path.open('rb') as b:
            if hashlib.file_digest(a, 'sha256').digest() != hashlib.file_digest(b, 'sha256').digest():
                continue
        temporary = path.with_name('.shared-'+path.name)
        os.link(source, temporary)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        if right.st_nlink == 1:
            reclaimed += right.st_blocks * 512
    return reclaimed


def _packed_objects(directory):
    objects = set()
    for index in sorted(directory.glob('*.idx')):
        with index.open('rb') as stream:
            output = subprocess.check_output(['git', 'show-index'], stdin=stream, timeout=30)
        objects.update(line.split()[1] for line in output.splitlines())
    return objects


def _abandoned_pack_candidates(directory):
    cutoff = time.time() - GIT_TEMP_PACK_MIN_AGE
    for path in directory.glob('tmp_pack_*'):
        if not re.fullmatch(r'tmp_pack_[A-Za-z0-9]+', path.name):
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and info.st_uid == os.getuid() and info.st_mtime < cutoff):
            yield path, info


def _reclaim_abandoned_packs(directory):
    """Drop unindexed files left by interrupted Git imports under the import lock."""
    candidates = dict(_abandoned_pack_candidates(directory))
    if not candidates:
        return 0
    proc = Path('/proc')
    if not (proc/'self/fd').is_dir():
        return 0
    open_paths = set()
    for process in proc.iterdir():
        try:
            if not process.name.isdigit() or process.stat().st_uid != os.getuid():
                continue
            for descriptor in (process/'fd').iterdir():
                try:
                    path = descriptor.readlink()
                except FileNotFoundError:
                    continue
                if path in candidates:
                    open_paths.add(path)
        except FileNotFoundError:
            continue  # Process exited during the scan.
        except PermissionError:
            # Unrelated protected processes do not write Git temporary
            # packs. An unreadable Git process leaves this cache unchanged.
            if (process/'comm').read_text().strip().startswith(('git', 'index-pack')):
                return 0
    reclaimed = 0
    for path, info in candidates.items():
        if path in open_paths:
            continue
        current = path.lstat()
        if (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size, current.st_nlink) == (
                info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size, info.st_nlink):
            path.unlink()
            reclaimed += info.st_blocks * 512
    return reclaimed


def _compact_metadata_packs(policy, spools, pbs_runs):
    """Reclaim temporary files and compact an idle worker's derived Git store."""
    roots = [Path(root).resolve() for root in policy['roots']]
    scratch_root = Path(policy['sample_path']).resolve().parent
    if any(scratch_root.is_relative_to(root) for root in roots):
        return 0, 0
    spools = set(spools)
    if pbs_runs:
        spools.update(str(spool) for spool in Path(pbs_runs).glob('*/workers/*')
                      if (spool/'WORKER.lock').is_file())
    candidates = {}
    for spool_name in spools:
        spool = Path(spool_name).resolve()
        pool = spool.parent/'preparation-metadata-cache'
        for repository in pool.glob('*/objects.git'):
            if repository.resolve() == repository and any(repository.is_relative_to(root) for root in roots):
                candidates.setdefault(repository, [spool])
    if pbs_runs:
        shared = Path(pbs_runs).resolve()/'preparation-metadata-cache'
        for repository in shared.glob('*/objects.git'):
            if repository.resolve() == repository and any(repository.is_relative_to(root) for root in roots):
                candidates[repository] = sorted({Path(name).resolve() for name in spools})
    packed = []
    for repository, owners in candidates.items():
        directory = repository/'objects/pack'
        if directory.resolve() != directory:
            continue
        packs = list(directory.glob('*.pack'))
        indexed = sum(p.stat().st_blocks * 512 for p in packs) if len(packs) >= 8 else 0
        temporary = sum(info.st_blocks * 512 for _, info in _abandoned_pack_candidates(directory))
        if temporary or indexed >= 256 * 1024**2:
            packed.append((temporary, indexed, directory, owners))
    for temporary, indexed, directory, owners in sorted(packed, reverse=True):
        try:
            with ExitStack() as stack:
                for spool in owners:
                    worker_lock = stack.enter_context((spool/'WORKER.lock').open('rb'))
                    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if any(json.loads(marker.read_bytes()).get('state') in
                           ('ASSIGNED', 'STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER')
                           for marker in spool.glob('*/*/WORKER_STATE.json')):
                        raise BlockingIOError('Active worker attempts still own inputs')
                import_lock = stack.enter_context((directory.parents[2]/'IMPORT.lock').open('rb'))
                fcntl.flock(import_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                abandoned = _reclaim_abandoned_packs(directory) if temporary else 0
                if abandoned:
                    return 0, abandoned
                if indexed >= 256 * 1024**2:
                    compacted = _compact_pack_directory(directory, scratch_root)
                    if compacted:
                        return compacted, 0
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    return 0, 0


def _compact_pack_directory(directory, scratch_root):
    entries = list(directory.iterdir())
    if any(path.resolve() != path or not path.is_file() or path.suffix not in ('.pack', '.idx', '.rev')
           for path in entries):
        return 0
    reverse_indexes = [path for path in entries if path.suffix == '.rev']
    if any(not path.with_suffix('.pack').is_file() or not path.with_suffix('.idx').is_file()
           for path in reverse_indexes):
        return 0
    for path in reverse_indexes:
        subprocess.run(['git', '-C', str(directory.parent.parent), 'index-pack', '--verify', '--rev-index',
                        str(path.with_suffix('.pack'))], check=True, capture_output=True, timeout=60)
    if directory.stat().st_dev != scratch_root.stat().st_dev:
        return 0
    before = sum(path.stat().st_blocks * 512 for path in entries)
    if shutil.disk_usage(scratch_root).free < before:
        return 0
    originals = _packed_objects(directory)
    if not originals:
        return 0
    repo = directory.parent.parent
    with tempfile.TemporaryDirectory(prefix='rpe-cache-compact-', dir=scratch_root) as temp:
        scratch = Path(temp)/'objects'
        stage = scratch/'pack'
        stage.mkdir(parents=True, mode=directory.stat().st_mode & 0o777)
        env = dict(os.environ, GIT_OBJECT_DIRECTORY=str(scratch),
                   GIT_ALTERNATE_OBJECT_DIRECTORIES=str(repo/'objects'))
        git = ['git', '-C', str(repo)]
        name = subprocess.check_output([*git, '-c', 'pack.threads=2', 'pack-objects',
                                        str(stage/'pack')],
            input=b'\n'.join(sorted(originals)) + b'\n',
            env=env, stderr=subprocess.PIPE, timeout=180).decode().strip()
        subprocess.run([*git, 'index-pack', '--verify', str(stage/f'pack-{name}.pack')],
                       check=True, capture_output=True, timeout=60)
        if _packed_objects(stage) != originals:
            raise ValueError('Compacted metadata pack has a different object set')
        after = sum(path.stat().st_blocks * 512 for path in stage.iterdir())
        if after >= before:
            return 0
        if not _exchange_pack_directories(stage, directory):
            return 0
        return before - after


def _exchange_pack_directories(left, right):
    rename = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
    if rename is None:
        return False
    if rename(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot atomically exchange metadata packs')
    return True


def report(policy, *, spools=(), pbs_runs=None, prefix_cache=None, cache_directories=(),
           refresh=False, compact_metadata=False, proactive=False, du_timeout=10):
    """Report intake state and reclaim copies independently of the hard limit."""
    if policy is None:
        return None
    limit, roots = policy['limit_bytes'], policy['roots']
    sample = Path(policy['sample_path'])
    if (type(limit) is not int or limit <= 0 or not roots
            or not isinstance(roots, list) or any(not Path(p).is_absolute() for p in roots)
            or not sample.is_absolute()):
        raise ValueError('Worker storage requires a positive byte limit and absolute paths')
    identity = policy_identity(policy)
    excluded_paths = identity.get('excluded_paths', ())
    start, target = reclaim_watermarks(limit, proactive=proactive)
    caches = list(dict.fromkeys([*([prefix_cache] if prefix_cache else []), *cache_directories]))
    cleanup = dict(spools=list(spools), pbs_runs=pbs_runs, prefix_caches=caches)

    def recent():
        if refresh:
            return None
        try:
            saved = json.loads(sample.read_bytes())
            if saved['policy'] == identity and 0 <= time.time() - saved['sampled_epoch'] < SAMPLE_SECONDS:
                report = saved['report']
                if 'filesystem_free_bytes' not in report:
                    return None
                same_watermarks = (
                    report.get('reclaim_start_bytes'), report.get('reclaim_target_bytes')
                ) == (start, target)
                if proactive and not same_watermarks:
                    return None
                if (report['used_bytes'] is None or report['used_bytes'] < start
                        or (same_watermarks and saved.get('cleanup') == cleanup)):
                    return report
        except (OSError, KeyError, TypeError, ValueError):
            pass
        return None

    try:
        cached = recent()
        if cached is not None:
            return cached
        sample.parent.mkdir(parents=True, exist_ok=True)
        compacted = abandoned = 0
        if compact_metadata:
            observation = report(policy, spools=spools, pbs_runs=pbs_runs,
                                 cache_directories=caches, refresh=refresh, proactive=proactive,
                                 du_timeout=du_timeout)
            if observation.get('reclaim_pending'):
                compacted, abandoned = _compact_metadata_packs(policy, spools, pbs_runs)
            if not (compacted or abandoned):
                return observation
            refresh = True
        with sample.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = recent()
            if cached is not None:
                return cached
            started = time.monotonic()
            try:
                used = _used_bytes(roots, timeout=du_timeout, excluded_paths=excluded_paths)
                before, reclaimed = used, 0
                cleanup_due = used >= start
                if cleanup_due and (spools or pbs_runs):
                    candidates = list(spools)
                    if pbs_runs:
                        candidates.extend(Path(pbs_runs).glob('*/workers/*'))
                    # Leave room for new attempts before the next cleanup pass.
                    used, reclaimed = _reclaim(candidates, roots, used, target, du_timeout=du_timeout, excluded_paths=excluded_paths)
                if cleanup_due and used > target and (spools or pbs_runs):
                    if _share_transferred_git_packs(candidates, roots, used-target):
                        used = _used_bytes(roots, timeout=du_timeout, excluded_paths=excluded_paths)
                evicted = 0
                if cleanup_due and used > target:
                    for path in caches:
                        cache = Path(path).resolve()
                        if cache.is_dir() and any(cache.is_relative_to(Path(root).resolve()) for root in roots):
                            cache_size = _used_bytes([str(cache)], timeout=du_timeout)
                            evicted += PrefixStore(cache).reclaim(cache_size - (used-target))
                            used = _used_bytes(roots, timeout=du_timeout, excluded_paths=excluded_paths)
                            if used <= target:
                                break
                filesystem_free = {}
                for name in roots:
                    root = Path(name)
                    while not root.exists():
                        root = root.parent
                    filesystem_free[name] = shutil.disk_usage(root).free
                min_free = limit // 10
                low_roots = [name for name,free in filesystem_free.items() if free < min_free]
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                measured = dict(limit_bytes=limit, used_bytes=None, accepting_new_jobs=False,
                                reason='STORAGE_USAGE_UNAVAILABLE',filesystem_free_bytes=None,
                                filesystem_min_free_bytes=limit//10, measurement_error=type(error).__name__)
            else:
                measured = dict(limit_bytes=limit, used_bytes=used, accepting_new_jobs=used < limit and not low_roots,
                                reason='STORAGE_LIMIT_REACHED' if used >= limit else 'FILESYSTEM_SPACE_LOW' if low_roots else None,
                                filesystem_free_bytes=min(filesystem_free.values()),
                                filesystem_min_free_bytes=min_free,filesystem_low_space_roots=low_roots,
                                reclaim_start_bytes=start, reclaim_target_bytes=target,
                                reclaim_checked=cleanup_due or bool(compacted or abandoned),
                                reclaim_pending=(cleanup_due or bool(compacted or abandoned)) and used > target)
                if used < before:
                    measured.update(reclaimed_copies=reclaimed, reclaimed_bytes=max(0, before-used))
                if evicted:
                    measured['evicted_prefix_checkpoints'] = evicted
                if compacted:
                    measured['compacted_metadata_bytes'] = compacted
                if abandoned:
                    measured['reclaimed_metadata_temporary_bytes'] = abandoned
            if excluded_paths:
                measured['excluded_paths'] = list(excluded_paths)
            measured['measurement_seconds'] = time.monotonic()-started
            atomic_json(sample, dict(policy=identity, cleanup=cleanup, sampled_epoch=time.time(), report=measured))
            return measured
    except OSError:
        return dict(limit_bytes=limit, used_bytes=None, accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')


_Q3_RESULT_COPY_NAMES = ('RESULT.json', 'API_RETURN.json')
_Q3_COPY_SWEEP_BYTES = 1024**3


def _q3_copy_attempt(spool, job, attempt):
    if not (re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', job)
            and re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', attempt)):
        return None
    directory = spool/job/attempt
    if directory.resolve() != directory:
        return None
    marker, assignment_path = (directory/name for name in ('WORKER_STATE.json', 'ASSIGNMENT.json'))
    if marker.is_symlink() or assignment_path.is_symlink():
        return None
    if json.loads(marker.read_bytes()).get('state') != 'TRANSFERRED':
        return None
    assignment = json.loads(assignment_path.read_bytes())
    if (assignment['job_id'], assignment['attempt_id']) != (job, attempt):
        return None
    if assignment['request']['route']['kind'] in ('preparation', 'artificial'):
        return None
    if not (directory/'OUTCOME.json').is_file() or not (directory/'PROCESS_RETURN.json').is_file():
        return None
    return directory, assignment['request_sha256']


def q3_result_copy_inventory(profile, spools):
    """List only known Q3 copies for comparison with the central ledger."""
    roots = [Path(root).resolve() for root in profile['storage_limit']['roots']]
    items = []
    for spool in sorted(set(map(Path, spools))):
        if spool.resolve() != spool or not any(spool.is_relative_to(root) for root in roots):
            continue
        for marker in spool.glob('*/*/WORKER_STATE.json'):
            job, attempt = marker.parent.parent.name, marker.parent.name
            try:
                result = _q3_copy_attempt(spool, job, attempt)
                if result is None:
                    continue
                directory, request_sha256 = result
                for name in _Q3_RESULT_COPY_NAMES:
                    path = directory/job/name
                    if path.resolve() != path or not path.is_file():
                        continue
                    info = path.stat()
                    if info.st_nlink == 1:
                        items.append(dict(spool=str(spool), job=job, attempt=attempt, name=name,
                                          size=info.st_size, blocks=info.st_blocks*512,
                                          request_sha256=request_sha256))
            except (OSError, KeyError, ValueError, TypeError):
                continue
    return items


def reclaim_q3_result_copies(profile, spools, proofs, target):
    """Retire only byte-identical copies once worker ownership has closed."""
    if target <= 0:
        return dict(files=0, bytes=0)
    policy = profile['storage_limit']
    roots = [Path(root).resolve() for root in policy['roots']]
    allowed = set(map(Path, spools))
    removed = size = 0
    lockfile = Path(policy['sample_path']).with_suffix('.lock')
    with lockfile.open('r+b') as storage_lock:
        fcntl.flock(storage_lock, fcntl.LOCK_EX)
        for row in proofs:
            spool = Path(row['spool'])
            name = row['name']
            if (spool not in allowed or spool.resolve() != spool or name not in _Q3_RESULT_COPY_NAMES
                    or not any(spool.is_relative_to(root) for root in roots)):
                continue
            try:
                result = _q3_copy_attempt(spool, row['job'], row['attempt'])
                if result is None:
                    continue
                directory, request_sha256 = result
                if request_sha256 != row['request_sha256']:
                    continue
                lock_path = directory/'WORKER_PROCESS.lock'
                if not stat.S_ISREG(lock_path.lstat().st_mode):
                    continue
                with lock_path.open('rb') as execution_lock:
                    fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if _q3_copy_attempt(spool, row['job'], row['attempt']) != result:
                        continue
                    path = directory/row['job']/name
                    if path.resolve() != path:
                        continue
                    before = path.lstat()
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != row['size']:
                        continue
                    with path.open('rb') as stream:
                        if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                            continue
                    after = path.lstat()
                    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                        continue
                    path.unlink()
                    removed += 1
                    size += before.st_blocks*512
                    if size >= target:
                        break
            except (OSError, KeyError, ValueError, TypeError):
                continue
    return dict(files=removed, bytes=size)


def _ended_pbs_rpc_pairs(namespace):
    """Find candidate completion replays in closed allocation directories."""
    root = Path(namespace)/'allocations'
    for session in sorted(root.glob('*')):
        terminal = session/'SESSION_END.json'
        if (not re.fullmatch(r'[a-f0-9]{32}', session.name) or session.resolve() != session
                or terminal.is_symlink() or not terminal.is_file()):
            continue
        for request in sorted((session/'requests').glob('*.json')):
            if not re.fullmatch(r'(?:complete-[a-f0-9]{20}-)?[a-f0-9]{32}', request.stem):
                continue
            response = session/'responses'/request.name
            try:
                if request.resolve() != request or response.resolve() != response:
                    continue
                incoming, outgoing = request.stat(), response.stat()
                if not (stat.S_ISREG(incoming.st_mode) and stat.S_ISREG(outgoing.st_mode)
                        and incoming.st_nlink == outgoing.st_nlink == 1 and outgoing.st_size <= 4096):
                    continue
                reply = json.loads(response.read_bytes())
                result = reply.get('result') or {}
                if reply.get('ok') is True and result.get('ok') is True and type(result.get('duplicate')) is bool:
                    yield request, response, incoming.st_blocks*512+outgoing.st_blocks*512
            except (OSError, ValueError, AttributeError):
                continue


def pbs_completion_copy_inventory(namespace):
    return [(request.parents[1].name, request.stem, size)
            for request, _, size in _ended_pbs_rpc_pairs(namespace)]


def reclaim_pbs_completion_copies(profile, namespace, proof, adopted, target):
    """Reclaim matched successful RPC pairs after the central result was adopted."""
    from .rpe_q3_worker import digest

    policy = profile['storage_limit']
    namespace = Path(namespace)
    if namespace.resolve() != namespace or not any(namespace.is_relative_to(Path(root).resolve())
                                                     for root in policy['roots']):
        return dict(pairs=0, bytes=0)
    removed = size = 0
    proof = {(row['token'],row['request_id']):row for row in proof}
    with Path(policy['sample_path']).with_suffix('.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for request, response, pair_size in _ended_pbs_rpc_pairs(namespace):
            key = request.parents[1].name, request.stem
            saved = proof.get(key)
            if saved is None:
                continue
            try:
                incoming = json.loads(request.read_bytes())
                outgoing = json.loads(response.read_bytes())
                value = incoming['payload']
                if (incoming['endpoint'] != '/v1/complete' or incoming['request_id'] != key[1]
                        or digest(incoming) != saved['request_sha256'] or outgoing != saved['response']
                        or value['host_id'] != profile['host_id']):
                    continue
                owner = '|'.join(value[k] for k in ('job_id','attempt_id','request_sha256'))
                if owner not in adopted or digest(value['files']) != adopted[owner]:
                    continue
                request.unlink()
                response.unlink()
                removed += 1
                size += pair_size
                if size >= target:
                    break
            except (OSError, KeyError, ValueError, TypeError):
                continue
    return dict(pairs=removed, bytes=size)
