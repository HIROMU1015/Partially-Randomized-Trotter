#!/usr/bin/env python3
"""Run collection-only checks and nominate recoverable central archive candidates."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sqlite3
import sys
import time

from recover_stopped_rpe_run import (OWNED, has_owned_attempts, load, live_driver, optional, read_rows,
                                     safe_name, read_preparation_readiness)
from publish_h4_search_run import FILES as PUBLICATION_FILES


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)

def error_summary(error):
    detail = getattr(error, 'stderr', None)
    if isinstance(detail, bytes):
        detail = detail.decode(errors='replace')
    return detail[-500:] if detail else str(error)[:500]


def latest_wait(path):
    if not path.is_file():
        return {}
    with path.open('rb') as stream:
        start = max(0, stream.seek(0,os.SEEK_END)-2*1024**2)
        stream.seek(start)
        if start:
            stream.readline()
        for line in reversed(stream.readlines()):
            if b'"QUEUE_WAIT_STATE"' in line:
                try:
                    return json.loads(line)
                except ValueError:
                    continue
    return {}


def wait_summary(run, recovery):
    wait = latest_wait(run/'PROGRESS.jsonl')
    relay = optional(run/'distributed/relay/STATUS.json')
    pbs_path=run/'distributed/pbs/STATE.json'
    if pbs_path.with_name('STATE_DELTAS.jsonl').is_file():
        from trottertracks.rpe_pbs_state import read_pbs_state
        pbs=read_pbs_state(pbs_path)
    else:
        pbs=optional(pbs_path)
    allocations = [row for row in pbs.get('allocations', {}).values() if not row.get('harvested')]
    sessions = Counter(row.get('session_status') or row.get('status', 'UNCONFIRMED') for row in allocations)
    reservations = {}
    for row in allocations:
        observed = row.get('observed') or {}
        identity = observed.get('job_id') or observed.get('pilot_token') or observed.get('token')
        if identity:
            reservations[identity] = (row.get('reservation_status') or (observed.get('scheduler') or {}).get('job_state')
                                      or row.get('status', 'UNCONFIRMED'))
    return dict(queue_wait_observed_at=wait.get('UTC'),
                live_controller=live_driver(run),
                queue_wait_reasons=wait.get('reason_counts', dict(Counter(wait.get('reasons', {}).values()))),
                hosts=wait.get('hosts', {}), preparation_budget=wait.get('preparation_budget_blocked', {}),
                relay={host:{key:value[key] for key in ('phase','error','last_error_epoch') if key in value}
                       for host,value in relay.get('workers', {}).items()},
                pbs_allocations=dict(sessions), pbs_reservations=dict(Counter(reservations.values())),
                recovery_status=recovery.get('status'),
                delivery_hold=recovery.get('delivery_hold'))


def _archive_db_signature(database):
    signatures=[]
    for path in (database, Path(str(database)+'-wal')):
        try:
            st=path.stat()
            signatures.append([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns])
        except FileNotFoundError:
            signatures.append(None)
    return signatures


def archive_snapshot(run, ended, cache=None):
    """Read bundle references alone; unchanged ended runs reuse the read-only snapshot."""
    database = run/'distributed/QUEUE.sqlite3'
    if not database.is_file():
        return set(), []
    signature = _archive_db_signature(database)
    key = str(run.resolve())
    saved = (cache or {}).get(key) if ended else None
    if saved is not None and saved.get('signature') == signature:
        return set(saved['hashes']), saved['candidates']
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        hashes = {row[0] for row in db.execute("SELECT DISTINCT json_extract(b.value,'$.sha256') "
            "FROM jobs j, json_each(j.request,'$.preparation.input_git_bundles') b")}
        candidates = []
        if ended and hashes and not db.execute("SELECT 1 FROM jobs WHERE state NOT IN ('ACCEPTED','CANCELLED') LIMIT 1").fetchone():
            candidates=[]
            for row in db.execute(
                "SELECT job_id,method,state,request_sha256,adopted_epoch,"
                "json_extract(request,'$.preparation.input_git_bundles') AS bundles,"
                "json_extract(request,'$.preparation.workspaces') AS workspaces FROM jobs "
                "WHERE state='ACCEPTED' AND adopted_epoch IS NOT NULL AND "
                "json_array_length(request,'$.preparation.input_git_bundles')>0"):
                item=dict(row);prep={'input_git_bundles':json.loads(item.pop('bundles')),
                    'workspaces':json.loads(item.pop('workspaces') or 'null')}
                candidates.append(dict(item,request={'preparation':prep},request_projected=True))
    if cache is not None:
        if ended and signature == _archive_db_signature(database):
            cache[key] = dict(signature=signature, hashes=sorted(hashes), candidates=candidates)
        else:
            cache.pop(key,None)
    return hashes, candidates


def archive_candidates(runs, current, rows=None, *, cache=None):
    """Nominate ended, adopted inputs which no other registered run references."""
    referenced = Counter()
    proposals = []
    for run in runs:
        launch = optional(run/'LAUNCH.json'); finished = optional(run/'EXIT.json')
        ended = (run != current and launch.get('invocation_id') and launch['invocation_id'] == finished.get('invocation_id')
                 and finished.get('returncode') == 0 and not live_driver(run))
        if rows is None:
            bundles, run_rows = archive_snapshot(run, ended, cache)
        else:
            run_rows = rows.get(run, [])
            bundles = {bundle['sha256'] for row in run_rows
                       for bundle in (row['request'].get('preparation') or {}).get('input_git_bundles', [])}
            ended = ended and all(row['state'] in ('ACCEPTED','CANCELLED') for row in run_rows)
        pending = read_preparation_readiness(run/'distributed')
        bundles.update(bundle['sha256'] for row in pending.values()
                       for bundle in (row['request'].get('preparation') or {}).get('input_git_bundles', []))
        referenced.update(bundles)
        if not ended:
            continue
        for row in run_rows:
            if row['state'] != 'ACCEPTED' or row['adopted_epoch'] is None:
                continue
            prep = row['request'].get('preparation') or {}
            bundles = [bundle for bundle in prep.get('input_git_bundles', []) if '/inputs-' in bundle['ref']
                       and not bundle.get('workspace_objects')]
            if not bundles or prep.get('workspaces'):
                continue
            job = run/'jobs'/safe_name(row['method'])/safe_name(row['job_id'])
            outcome = optional(job/'OUTCOME.json')
            receipt = optional(job/'PREPARATION_RECEIPT.json')
            if (outcome.get('request_sha256') != row['request_sha256']
                    or outcome.get('status') != 'PREPARATION_COMPLETE'
                    or receipt.get('returncode') != 0 or receipt.get('signal') is not None):
                continue
            saved_request=optional(job/'REQUEST.json')
            if row.get('request_projected'):
                from trottertracks.rpe_q3_worker import digest
                if digest(saved_request)!=row['request_sha256']:continue
                actual=saved_request.get('preparation') or {}
                if actual.get('input_git_bundles')!=prep['input_git_bundles'] or actual.get('workspaces')!=prep.get('workspaces'):
                    continue
            elif saved_request!=row['request']:continue
            for bundle in bundles:
                sha = safe_name(bundle['sha256'])
                path = run/'distributed/artifacts'/sha[:2]/sha
                if path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1:
                    proposals.append(dict(run_id=run.name, job_id=row['job_id'], sha256=sha,
                                          request=str(job/'REQUEST.json'), allocated_bytes=path.stat().st_blocks*512))
    unique = {row['sha256']:row for row in proposals if referenced[row['sha256']] == 1}
    result = sorted(unique.values(), key=lambda x:-x['allocated_bytes'])
    return dict(status='CANDIDATES_ONLY', candidates=result, count=len(result),
                allocated_bytes=sum(row['allocated_bytes'] for row in result))


def resume_interrupted_target(state, run):
    """Run only after collection, via the fixed run's ordinary Gateway."""
    if (run/'STOP_REQUESTED.json').exists() or (run/'SEARCH_TERMINATION.json').exists() or live_driver(run):
        return None
    launch, ended = optional(run/'LAUNCH.json'), optional(run/'EXIT.json')
    invocation = launch.get('invocation_id', '')
    if (not isinstance(invocation, str) or not re.fullmatch(r'[0-9a-f]{32}', invocation)
            or ended.get('invocation_id') != invocation or ended.get('returncode') != -signal.SIGKILL):
        return None
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code/'src'))
    from trottertracks.rpe_local0_search import Gateway
    try:
        config = load(run/'CONTROLLER_CONFIG.json')
        if Path(config['state_directory']).resolve() != state.resolve():
            raise ValueError('Fixed controller state directory differs')
        result = Gateway(config).resume_interrupted(run, invocation)
        return dict(status=result, previous_invocation=invocation)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, sqlite3.Error) as error:
        return dict(status='RECOVERY_HELD', previous_invocation=invocation, reason=str(error))


def publication_source(run, config):
    """Use the verified publisher selected by the run's current invocation."""
    launch = optional(run/'LAUNCH.json')
    command = launch.get('command', [])
    if '--schedule-resume' not in command:
        if config.get('periodic_publication'):
            return Path(__file__).with_name('publish_romutail_search.py'), None
        return None
    selection = json.loads(command[command.index('--schedule-resume')+1])
    hashes = selection.get('publisher_sha256')
    if not hashes:
        if config.get('periodic_publication'):
            return Path(__file__).with_name('publish_romutail_search.py'), None
        return None
    scripts = Path(selection['root']).resolve()/'scripts'
    if (selection['run_id'] != run.name or selection['base_source'] != config['code_source']
            or Path(command[2]).resolve() != scripts/'run_rpe_local0_search.py'
            or Path(command[command.index('--execute-run')+1]).resolve() != run.resolve()
            or command[command.index('--invocation')+1] != launch['invocation_id']):
        raise ValueError('Scheduled publisher does not match the current run invocation')
    for name in ('publish_romutail_search.py', 'publish_h4_search_run.py'):
        if hashlib.sha256((scripts/name).read_bytes()).hexdigest() != hashes.get(name):
            raise ValueError('Scheduled publisher differs from the selected deployment: '+name)
    return scripts/'publish_romutail_search.py', selection['source']


def publish_due(run, *, active, previous):
    """Publish completed snapshots through the run's ordinary C destination."""
    if time.time() - previous.get('checked_epoch', 0) < (600 if active else 120):
        return previous
    config = optional(run/'CONTROLLER_CONFIG.json')
    settings = config.get('publication')
    if not settings or config.get('artificial_controller_directory'):
        return previous
    if not (run/'ROUND_RESULT.json').is_file():
        return previous
    published = run/'PUBLICATION_STATUS.json'
    stamp = published.stat().st_mtime if published.is_file() else 0
    snapshot = PUBLICATION_FILES + ('KAPPA_STATE.json',)
    if (optional(published).get('status') == 'PUBLISHED'
            and all(not (run/name).is_file() or (run/name).stat().st_mtime <= stamp for name in snapshot)):
        return previous
    checked = time.time()
    try:
        selected = publication_source(run, config)
        if selected is None:
            return previous
        script, operations_source = selected
        command = [config['python'], '-B', str(script), '--repository', settings['repository'],
                   '--run-dir', str(run), '--controller-source', config['code_source'],
                   '--branch', settings['branch']]
        if operations_source:
            command.extend(['--operations-source', operations_source])
        if settings.get('push'):
            command.append('--push')
        response = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
        status = json.loads(response.stdout)
        return dict(checked_epoch=checked, status=status['status'], head=status.get('head'))
    except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as error:
        return dict(checked_epoch=checked, status='PUBLICATION_PENDING',
                    reason=error_summary(error))


def run_pass(state, *, offline=False, resume_interrupted=False, publication_only=False):
    report_dir = state/'maintenance'
    report_dir.mkdir(exist_ok=True)
    with (report_dir/'LOCK').open('a+b') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status':'ALREADY_RUNNING'}
        index = load(state/'RUNS.json')
        current = (state/'CURRENT_TARGET.json')
        current = optional(current).get('run_id')
        if publication_only:
            checks = {}
            for run_id, item in index.items():
                try:
                    run = Path(item['run_directory']).resolve(strict=True)
                    if run.name != run_id:
                        raise ValueError('Registered run path differs')
                    active = live_driver(run)
                    if run_id != current and not active:
                        continue
                    path = report_dir/(run_id+'.json')
                    report = optional(path)
                    result = publish_due(run, active=active, previous=report.get('publication_check', {}))
                    if result:
                        save(path, dict(report, publication_check=result))
                    checks[run_id] = result
                except (OSError, ValueError, KeyError, TypeError) as error:
                    checks[run_id] = dict(status='PUBLICATION_PENDING', reason=error_summary(error))
            return dict(status='CHECKED', publication_checks=checks, new_scientific_calls=0)
        runs, warnings = {}, []
        for run_id, item in index.items():
            try:
                run = Path(item['run_directory']).resolve(strict=True)
                if run.name != run_id:
                    raise ValueError('Registered run path differs')
                runs[run_id] = run
            except (OSError,ValueError,KeyError,sqlite3.Error) as error:
                warnings.append(dict(run_id=run_id,error=str(error)))
        archive_path = report_dir/'ARCHIVE_CANDIDATES.json'
        previous = optional(archive_path)
        if time.time()-previous.get('epoch',0) >= 300:
            try:
                cache_path = report_dir/'ARCHIVE_SNAPSHOTS.json'
                cache = optional(cache_path)
                cache = {key:value for key,value in cache.items() if key in {str(run.resolve()) for run in runs.values()}}
                candidates = archive_candidates(runs.values(), runs.get(current), cache=cache)
                save(cache_path,cache)
                save(archive_path,dict(candidates,epoch=time.time(),warnings=warnings))
            except (OSError,ValueError,KeyError,sqlite3.Error) as error:
                warnings.append(dict(archive_candidates_error=str(error)))
        # Remember runs seen active so a later target change need not orphan a
        # known outstanding attempt. A fresh deployment leaves historical
        # non-current terminal attempts available through the one-run utility.
        monitored = set(optional(report_dir/'WATCH.json').get('runs',[]))
        for run_id, run in runs.items():
            if run_id == current or live_driver(run):
                monitored.add(run_id)
        selected = [run_id for run_id in sorted(monitored) if run_id in runs]
        for run_id in selected:
            run = runs[run_id]
            report = report_dir/(run_id+'.json')
            previous = optional(report)
            source = Path(__file__).with_name('recover_stopped_rpe_run.py')
            cmd = [sys.executable,'-B',str(source),'--state-directory',str(state),'--run-id',run_id]
            if offline:
                cmd.append('--offline')
            try:
                completed = subprocess.run(cmd, capture_output=True,check=True,text=True,timeout=110)
                result = json.loads(completed.stdout)
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                result = dict(status='RECOVERY_PENDING', reason=error_summary(error),
                              jobs=previous.get('jobs', []),
                              repaired_inputs=[],new_scientific_calls=0)
            if resume_interrupted and not offline and run_id == current and result['status'] != 'RECOVERY_PENDING':
                resumed = resume_interrupted_target(state, run)
                if resumed is not None:
                    result['controller_resume'] = resumed
            if not offline:
                result['publication_check'] = publish_due(
                    run, active=live_driver(run), previous=previous.get('publication_check', {}))
            save(report,dict(result,wait=wait_summary(run,result),observed_epoch=time.time()))
            if (run_id != current and not live_driver(run)
                    and not has_owned_attempts(run)):
                monitored.discard(run_id)
        save(report_dir/'WATCH.json',dict(runs=sorted(monitored)))
        compaction_path = report_dir/'ARCHIVE_COMPACTION.json'
        previous_compaction = optional(compaction_path)
        if not offline and time.time()-previous_compaction.get('epoch',0) >= 300:
            command = [sys.executable,'-B',str(Path(__file__).with_name('compact_rpe_central_archives.py')),
                       '--state-directory',str(state)]
            try:
                completed = subprocess.run(command, capture_output=True,check=True,text=True,timeout=1250)
                compact = json.loads(completed.stdout)
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                compact = dict(status='COMPACTION_PENDING',reason=str(error)[:350])
            save(compaction_path,dict(compact,epoch=time.time()))
        return dict(status='CHECKED',runs=selected,warnings=warnings,new_scientific_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory',type=Path,required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--offline',action='store_true')
    mode.add_argument('--resume-interrupted',action='store_true')
    mode.add_argument('--publication-only',action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_pass(args.state_directory.resolve(strict=True), offline=args.offline,
                              resume_interrupted=args.resume_interrupted,
                              publication_only=args.publication_only), ensure_ascii=False))


if __name__ == '__main__':
    main()
