"""Publish saved RomuTail controller metadata through C's existing manifest.

Artificial fixtures are excluded. This copies saved JSON, never evaluates it.
The execution SOURCE and the publication checkout are separate.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import os
import shutil
import sqlite3
from tempfile import NamedTemporaryFile
import subprocess
import sys
from publish_h4_search_run import prepare, FILES, _search_progress_summary


GIT_BLOB_LIMIT = 100 * 1024**2


def hashlib_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_metadata(source, target, *, limit=GIT_BLOB_LIMIT):
    """Strip only JSON line whitespace if needed. Preserve exact numeric tokens."""
    with source.open('rb') as reader, NamedTemporaryFile(dir=target.parent, delete=False) as writer:
        pending = Path(writer.name)
        try:
            if os.fstat(reader.fileno()).st_size <= limit:
                shutil.copyfileobj(reader, writer)
            else:
                for line in reader:
                    writer.write(line.strip(b' \n\r\t'))
                writer.write(b'\n')
            if writer.tell() > limit:
                raise ValueError(f'{source.name} exceeds publication size limit after compaction')
            writer.flush()
            os.fchmod(writer.fileno(), 0o644)
            os.replace(pending, target)
        finally:
            pending.unlink(missing_ok=True)


def execution_hosts(run):
    ledger = run/'distributed/QUEUE.sqlite3'
    if not ledger.is_file():
        return ()
    from contextlib import closing
    with closing(sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True)) as conn:
        return [host for host, in conn.execute(
            "SELECT DISTINCT host_id FROM jobs WHERE state='ACCEPTED' AND host_id IS NOT NULL")]


def publish(repository, run, *, branch, controller_source=None, operations_source=None, push=False):
    run = Path(run).resolve()
    for filename in ('PROBLEM.json', 'ROUND.json', 'ROUND.requested.json'):
        path = run/filename
        if not path.is_file():
            continue
        value = json.loads(path.read_bytes())
        if (value.get('artificial_only') or value.get('artificial') or value.get('artificial_fixture',{}).get('only')
                or str(value.get('problem',{}).get('system','')).startswith('ARTIFICIAL')):
            return dict(status='ARTIFICIAL_SKIPPED', production_publication=False)
    repository = Path(repository).resolve()
    git_dir = Path(subprocess.check_output(
        ['git', 'rev-parse', '--absolute-git-dir'], cwd=repository, text=True, timeout=15).strip())
    with (git_dir/'rpe-publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _publish(repository, run, branch=branch, controller_source=controller_source,
                        operations_source=operations_source, push=push)


def _publish(repository, run, *, branch, controller_source=None, operations_source=None, push=False):
    git=lambda *args:subprocess.check_output(['git',*args],cwd=repository,text=True,
                                            stderr=subprocess.PIPE,timeout=60).strip()
    if git('branch','--show-current')!=branch:raise ValueError('Publication branch differs')
    remote=git('remote','get-url','--push','origin')
    if not (remote.startswith('git@github.com:HIROMU1015/') or remote.startswith('https://github.com/HIROMU1015/')):
        raise ValueError('Publication target must be HIROMU1015/*')
    if git('status','--porcelain'):raise ValueError('Publication checkout has existing changes')
    target=repository/'docs/research/romutail_controller_runs'/run.name
    target.mkdir(parents=True,exist_ok=True)
    # Only changed metadata needs to be rewritten. The public checkout's
    # exact file identity is checked too, so a replaced target is not reused.
    stamp_path=run/'PUBLICATION_SOURCE_STAMPS.json'
    try:previous=json.loads(stamp_path.read_bytes())
    except (OSError, ValueError):previous={}
    stamps={};unchanged=set();deferred=[]
    def stamp(path):
        info=path.stat();return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]
    for name in FILES+('KAPPA_STATE.json','APPLIED_SCOPE.json','RELEASE.json'):
        source=run/name;destination=target/name
        if not source.is_file():continue
        source_id=stamp(source)
        prior=previous.get(name)
        if destination.is_file() and prior==dict(source=source_id,target=stamp(destination)):
            stamps[name]=prior;unchanged.add(name);continue
        try:
            copy_metadata(source,destination)
        except ValueError as error:
            if ('exceeds publication size limit' not in str(error) or name not in ('ROUND_RESULT.json','ROUND_RESULT_INTERRUPTED.json')
                    or not ((target/'SCOPE_METADATA.json').is_file() or destination.is_file() or (run/'SCOPE_METADATA.json').is_file())):
                raise
            # A growing full result must not stop live progress publication.
            # Keep the previously published exact result, if any, and mark it stale.
            deferred.append(name)
            continue
        if stamp(source)==source_id:
            stamps[name]=dict(source=source_id,target=stamp(destination))
    if (run/'KAPPA_STATE.json').is_file():
        state = json.loads((target/'KAPPA_STATE.json').read_bytes())
        if state.get('round_id') == run.name and isinstance(state.get('families'), dict):
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
            from trottertracks.rpe_width_checkpoint import replay
            seq = replay(state, run/'POINT_PROGRESS.jsonl')
            progress = dict(_search_progress_summary(state), point_checkpoint_seq=seq,
                            state_path=(target/'KAPPA_STATE.json').relative_to(repository).as_posix(),
                            state_sha256=hashlib_file(target/'KAPPA_STATE.json'))
            output = target/'LIVE_SEARCH_PROGRESS.json'
            raw = json.dumps(progress, ensure_ascii=False, separators=(',', ':'), default=str)+'\n'
            if not output.is_file() or output.read_text() != raw:
                with NamedTemporaryFile(mode='w', encoding='utf-8', dir=target, delete=False) as stream:
                    pending = Path(stream.name); stream.write(raw)
                os.replace(pending, output)
    status,manifest=prepare(repository,target,target/'SEARCH_PUBLICATION.json',run_id=run.name,
                            execution_hosts=execution_hosts(run), unchanged_files=unchanged, deferred_files=deferred)
    relative=target.relative_to(repository).as_posix()
    git('add','--sparse','--',relative)
    if git('diff','--cached','--name-only','--',relative):
        git('commit','-m','Publish saved RomuTail controller '+run.name+' [skip ci]','--',relative)
    result=dict(status='SAVED_METADATA_COMMITTED',production_publication=True,head=git('rev-parse','HEAD'),
        controller_source=controller_source,manifest=str(target/'SEARCH_PUBLICATION.json'),C_sync='ORDINARY_SEARCH_PUBLICATION_DISCOVERY')
    with NamedTemporaryFile(mode='w',encoding='utf-8',dir=run,delete=False) as stream:
        marker=Path(stream.name);json.dump(stamps,stream,separators=(',',':'))
    os.replace(marker,stamp_path)
    if operations_source:
        result['operations_source'] = operations_source
    if push:
        try:git('push','origin','HEAD:refs/heads/'+branch);result['status']='PUBLISHED'
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            result.update(status='PUBLICATION_PENDING',error=str(error)[-1500:],
                          push_command='git push origin HEAD:refs/heads/'+branch)
    return result


def main():
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--branch',required=True);p.add_argument('--controller-source');p.add_argument('--operations-source')
    p.add_argument('--push',action='store_true')
    a=p.parse_args()
    value=publish(a.repository,a.run_dir,branch=a.branch,controller_source=a.controller_source,
                  operations_source=a.operations_source,push=a.push)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=a.run_dir, delete=False) as stream:
        pending = Path(stream.name)
        stream.write(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    os.replace(pending, a.run_dir/'PUBLICATION_STATUS.json')
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
