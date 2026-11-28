#!/usr/bin/env python3
"""Archive and expire delivery copies from fully verified, inactive central runs."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess
import sys
import tempfile

from recover_stopped_rpe_run import load, live_driver, read_rows


def allocated(path):
    return path.stat().st_blocks * 512


def verify_result(run, archive, request_file, sha):
    from recover_stopped_rpe_run import fixed_modules
    fixed_modules(run)
    from trottertracks.rpe_preparation_bridge import DistributedPreparationBridge
    from trottertracks.rpe_preparation_artifacts import ArtifactCache, artifact_path

    code = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_preparation_retention.py'
    spec = importlib.util.spec_from_file_location('trottertracks._archive_verification_tools', code)
    retention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retention)

    request = load(request_file)
    preparation = request['preparation']
    if preparation.get('workspaces'):
        raise ValueError('Independent restore of auxiliary preparation workspaces is not configured')
    job = request_file.parent
    outcome = load(job/'OUTCOME.json')
    sources = load(run/'ROUND.json')['executor']['distributed']['preparation']['bundle_validation_sources']
    baseline = sources[preparation['source']['commit']]
    retained = ArtifactCache(run/'distributed/artifacts')
    with tempfile.TemporaryDirectory(prefix='rpe-archive-verification-') as scratch:
        temp = Path(scratch)
        cache = ArtifactCache(temp/'cache')
        retention.restore(request,sha,archive=archive,cache_directory=cache.root)
        needed = {*preparation['inputs'].values(), *outcome.get('artifacts', {}).values(),
                  *(item['sha256'] for item in preparation['input_git_bundles'])} - {sha}
        for key in needed:
            target = cache.path(key)
            original = retained.path(key)
            target.parent.mkdir(parents=True,exist_ok=True)
            if original.is_symlink():
                raise ValueError('Symlink in retained job inputs')
            shutil.copyfile(original,target)
        executor = SimpleNamespace(artifact_cache=cache,run_dir=temp)
        repo = temp/'provider'
        subprocess.run(['git','clone','--shared','--no-checkout',baseline,str(repo)],
                       capture_output=True,check=True)
        bridge = DistributedPreparationBridge(executor,repo,dict(bundle_validation_sources=sources))
        bridge.alias_path.parent.mkdir(parents=True,exist_ok=True)
        bridge._validate_bundle_inputs(preparation)
        outputs = temp/'saved'
        cache.materialize(outcome['artifacts'],outputs)
        request['preparation']['controller_root'] = str(repo)
        bindings = preparation.get('path_bindings', {})
        preparation['path_bindings'] = {str(temp/'private') if value == '@WORK@/private' else name:value
                                        for name,value in bindings.items()}
        bridge._import_result(dict(outcome,artifact_directory=str(outputs)),request)
        for name, key in outcome['artifacts'].items():
            if name.startswith('workspace/') and not name.endswith('.bundle') and preparation['inputs'].get(name) != key:
                if retention.file_sha256(repo/artifact_path(name.removeprefix('workspace/'))) != key:
                    raise ValueError('Restored preparation result differs')
    return dict(bundle=sha, input_check='PASS', saved_result_replay='PASS')


def run_catalog(state):
    keys = load(state/'RUNS.json')
    return {run_id:Path(row['run_directory']).resolve() for run_id,row in keys.items()}


def compact(state, batch):
    from archive_rpe_preparation_inputs import archive_inputs, file_sha256, sync_path
    from run_rpe_central_maintenance import archive_candidates, save

    current = load(state/'CURRENT_TARGET.json')['run_id']
    registry = run_catalog(state)
    if any(live_driver(run) for run in registry.values()):
        return {'status':'DEFERRED_ACTIVE_CONTROLLER'}
    catalog = archive_candidates(registry.values(),registry.get(current))
    candidates = catalog['candidates'][:batch]
    if len(candidates) < 2:
        return {'status':'NO_COMPACTION_BATCH','candidates':len(candidates)}
    shas = sorted(row['sha256'] for row in candidates)
    key = hashlib.sha256(''.join(shas).encode()).hexdigest()[:16]
    root = state.parent/'archives/preparation-inputs'
    root.mkdir(parents=True,exist_ok=True)
    name = 'automatic-'+key
    final = root/name
    marker = root/(name+'.reclaimed.json')
    last = state/'maintenance'/'ARCHIVE_COMPACTION.json'
    if marker.exists():
        return {'status':'ALREADY_ARCHIVED'}
    if last.is_file() and load(last).get('batch') == key and load(last)['status'] in ('NO_SAVINGS','INPUT_OR_RESULT_HOLD'):
        return load(last)
    with tempfile.TemporaryDirectory(prefix='.rpe-compact-',dir=root) as temporary:
        temporary = Path(temporary)
        built = temporary/name
        archive_inputs([Path(row['request']) for row in candidates],built)
        retained_bytes = allocated(built)+sum(allocated(p) for p in built.iterdir())
        expected = sum(row['allocated_bytes'] for row in candidates)
        if retained_bytes >= expected:
            return dict(status='NO_SAVINGS',batch=key,input_bytes=expected,archive_bytes=retained_bytes)
        checks = {}
        for row in candidates:
            command = [sys.executable,'-B',__file__,'--verify',str(built),'--state-directory',str(state),
                       '--request',row['request'],'--sha256',row['sha256']]
            try:
                completed = subprocess.run(command,text=True,capture_output=True,check=True,timeout=150)
                checks[row['sha256']] = json.loads(completed.stdout)
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                return dict(status='INPUT_OR_RESULT_HOLD',batch=key,job_id=row['job_id'],reason=str(error)[:350])
        with ExitStack() as stack:
            gate = stack.enter_context((state/'REGISTRY.lock').open('rb'))
            fcntl.flock(gate,fcntl.LOCK_EX)
            if registry != run_catalog(state) or any(live_driver(run) for run in registry.values()):
                return {'status':'DEFERRED_OWNERSHIP_CHANGED'}
            current = load(state/'CURRENT_TARGET.json')['run_id']
            now = archive_candidates(registry.values(),registry.get(current))
            if not all(row in now['candidates'] for row in candidates):
                return {'status':'DEFERRED_REFERENCES_CHANGED'}
            for run in {registry[row['run_id']] for row in candidates}:
                for lock_name in ('FAST_DRIVER.lock','distributed/pbs/MANAGER.lock'):
                    lock_path = run/lock_name
                    if lock_path.is_file():
                        fd = stack.enter_context(lock_path.open('rb'))
                        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            files = {}
            for row in candidates:
                sha = row['sha256']
                root_path = registry[row['run_id']]/'distributed/artifacts'
                blob = root_path/sha[:2]/sha
                if (blob.resolve() != blob or blob.stat().st_nlink != 1 or file_sha256(blob) != sha):
                    return {'status':'DEFERRED_FILE_CHANGED'}
                files[sha] = dict(row,source_path=str(blob),bytes=blob.stat().st_size)
            if final.exists():
                return {'status':'DEFERRED_EXISTING_ARCHIVE'}
            built.rename(final)
            sync_path(root)
            save(marker,dict(archive=str(final),bundles=files,replay_checks=checks,
                             removed_allocated_bytes=expected,archive_allocated_bytes=retained_bytes,
                             net_allocated_bytes=expected-retained_bytes))
            sync_path(marker)
            sync_path(root)
            for row in files.values():
                blob = Path(row['source_path'])
                blob.unlink()
                sync_path(blob.parent)
    return dict(status='ARCHIVED',batch=key,bundles=len(files),net_allocated_bytes=expected-retained_bytes,
                archive=str(final))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--verify',type=Path)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    state = args.state_directory.resolve(strict=True)
    if args.verify:
        run = args.request.resolve(strict=True).parents[3]
        print(json.dumps(verify_result(run,args.verify,args.request.resolve(),args.sha256)))
    else:
        if args.batch < 2:
            parser.error('--batch needs at least two candidate bundles')
        print(json.dumps(compact(state,args.batch)))


if __name__ == '__main__':
    main()
