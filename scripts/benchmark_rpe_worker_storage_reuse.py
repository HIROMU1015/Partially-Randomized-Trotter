#!/usr/bin/env python3
"""Artificial Git/worker storage benchmark; no scientific calculation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import rpe_preparation_worker as P
from trottertracks import rpe_q3_worker as W
from trottertracks import rpe_worker_storage as S


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE)


def size(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def measure(fn, repeats):
    results = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        results.append(time.perf_counter()-start)
    return dict(median_seconds=statistics.median(results), samples_seconds=results)


def run(root, mib, repeats):
    source = root/'source'
    source.mkdir()
    git(source, 'init', '-q')
    git(source, 'config', 'user.name', 'Synthetic storage benchmark')
    git(source, 'config', 'user.email', 'synthetic@example.invalid')
    rng = random.Random(20261126)
    (source/'src').mkdir()
    (source/'scripts').mkdir()
    (source/'scripts/leaf.py').write_text('pass\n')
    for i in range(mib*16):
        (source/'src'/f'{i:05d}.dat').write_bytes(rng.randbytes(65536))
    (source/'docs/research').mkdir(parents=True)
    (source/'docs/research/input.json').write_text('{"artificial": true}\n')
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'fixed artificial source')
    commit = P._head(source)
    manifest = dict(commit=commit, files={'scripts/leaf.py':hashlib.sha256(b'pass\n').hexdigest()})
    profile = dict(host_id='synthetic', root=str(source), interpreter=sys.executable,
                   work_dir=str(root/'spool'), slots=1, dp_workers=1)
    views = {commit:str(source)}
    spec = dict(source=manifest, source_workspace=True, outputs=['workspace/docs/research/result'])
    cache = P.worker_cache(profile)
    sizes = []
    def expanded(record_size=False):
        with tempfile.TemporaryDirectory(dir=root) as temp:
            directory = Path(temp)
            _, owned = P._clone_owned_workspaces(spec, profile, views, directory, cache)
            P._include_declared_outputs(owned, directory, spec['outputs'])
            if record_size: sizes.append(size(directory))
    def indexed():
        P._check_owned_workspaces(spec,profile,views)
    # Warm both routes; measure equivalent required cones with warm filesystem.
    expanded(record_size=True)
    indexed()
    old = measure(expanded, repeats)
    new = measure(indexed, repeats)

    producer = root/'producer'
    git(root, 'clone', '-q', str(source), str(producer))
    (producer/'docs/research/bundle.bin').write_bytes(rng.randbytes(mib*1024*1024//2))
    git(producer, 'add', '.')
    git(producer, 'commit', '-qm', 'artificial new input')
    bundle = root/'input.bundle'
    git(producer,'bundle','create',str(bundle),'HEAD','^'+commit)
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    bundle_spec = dict(spec,stage='audit',provider='synthetic',artificial=True,
                       input_git_bundles=[dict(sha256=sha,ref='HEAD')])
    assignment = dict(request=dict(preparation=bundle_spec,route=dict(kind='preparation')))
    cache.put_file(bundle,sha)
    metadata=P._metadata_input_repository(profile,source,bundle_spec)
    cache.path(sha).unlink()
    def transfer_again():
        # Local file staging isolates read/write/verification; SSH time excluded.
        assert cache.missing([sha]) == [sha]
        cache.put_file(bundle,sha)
        git(metadata,'bundle','verify',str(cache.path(sha)))
        git(metadata,'bundle','list-heads',str(cache.path(sha)))
        cache.path(sha).unlink()
    def use_imported():
        assert P.missing_metadata_bundle_hashes(assignment,profile) == []
        assert P._metadata_input_repository(profile,source,bundle_spec) == metadata
    transfer_again()
    use_imported()
    transfer = measure(transfer_again,repeats)
    retained = measure(use_imported,repeats)

    attempt = root/'spool'/'artificial-job'/'artificial-attempt'
    attempt.mkdir(parents=True)
    _, owned=P._clone_owned_workspaces(spec,profile,views,attempt,cache)
    P._include_declared_outputs(owned,attempt,spec['outputs'])
    output=attempt/'workspace/docs/research/result'
    output.mkdir()
    (output/'partial.json').write_text('artificial partial result')
    job=dict(job_id=attempt.parent.name,attempt_id=attempt.name,request_sha256='synthetic',
             request=dict(route=dict(kind='preparation'),preparation=spec))
    W.atomic(attempt/'ASSIGNMENT.json',job)
    W.atomic(attempt/'PROCESS_RETURN.json',dict(returncode=2))
    W.atomic(attempt/'OUTCOME.json',dict(status='PREPARATION_FAILED',request_id=job['job_id'],
                                        request_sha256='synthetic',artifacts={}))
    (attempt/'WORKER_PROCESS.lock').touch()
    W.save_state(attempt,'TRANSFERRED')
    before=size(attempt)
    start=time.perf_counter()
    assert S.compact_failed_preparation_workspaces(attempt) == 1
    compact_seconds=time.perf_counter()-start
    after=size(attempt)
    assert (output/'partial.json').read_text()=='artificial partial result'
    start=time.perf_counter()
    S.restore_failed_workspace(attempt)
    restore_seconds=time.perf_counter()-start
    assert (attempt/'workspace/src/00000.dat').read_bytes()==(source/'src/00000.dat').read_bytes()
    return dict(scientific_calls=0, fixture_source_mib=mib, repeats=repeats,
        readiness=dict(previous=old,new=new,previous_peak_logical_bytes=max(sizes),new_workspace_bytes=0),
        bundle=dict(previous=transfer,new=retained,previous_retransfer_bytes=bundle.stat().st_size,new_retransfer_bytes=0),
        failed_workspace=dict(previous_bytes=before,new_bytes=after,compaction_seconds=compact_seconds,
                              restore_seconds=restore_seconds,partial_preserved=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mib',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--directory',type=Path,default=None)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.directory,prefix='rpe-storage-artificial-') as temporary:
        print(json.dumps(run(Path(temporary),args.mib,args.repeats),indent=2))


if __name__=='__main__':
    main()
