"""Reconcile explicitly selected ended preparation attempts for a new run.

Original attempts, STOPs, and partial scientific output stay in their old run.
This does not enable automatic retries or modify the selected scientific scope.
"""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

PROVIDERS = {'DF_DESCRIPTOR', 'DF_SIGNALS', 'DF_CALIBRATION', 'PF_SIGNALS'}


def read(path):
    return json.loads(Path(path).read_bytes())


def verify_attempts(parent, job_ids):
    parent = Path(parent).resolve()
    if (not job_ids or len(set(job_ids)) != len(job_ids)
            or read(parent/'STOP_REQUESTED.json').get('reason') != 'USER_REQUESTED_STOP'
            or (read(parent/'KAPPA_STATE.json').get('stop') or {}).get('reason') != 'USER_REQUESTED_STOP'
            or read(parent/'EXIT.json').get('invocation_id') != read(parent/'LAUNCH.json').get('invocation_id')
            or read(parent/'EXIT.json').get('returncode') not in (0, -2)
            or read(parent/'distributed/relay/STATUS.json').get('stopped') is not True):
        raise ValueError('Preparation repair requires the selected, ended normal-stop parent')
    verified = {}
    with sqlite3.connect((parent/'distributed/QUEUE.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
        for job_id in job_ids:
            row = db.execute('SELECT method,state,host_id,attempt_id,request_sha256,request '
                             'FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if not row or not row[0].startswith('PREPARATION:') or row[1] != 'ACCEPTED' or not row[2] or not row[3]:
                raise ValueError('Preparation repair has an unresolved attempt: '+job_id)
            job = parent/'jobs'/row[0]/job_id
            request, outcome = read(job/'REQUEST.json'), read(job/'OUTCOME.json')
            receipt, owner = read(job/'PREPARATION_RECEIPT.json'), read(job/'LAUNCH.json')
            from .rpe_release_registry import fingerprint
            if (json.loads(row[5]) != request or fingerprint(request) != row[4]
                    or request['preparation']['provider'] not in PROVIDERS
                    or outcome.get('status') != 'PREPARATION_FAILED'
                    or outcome.get('automatic_retry') is not False
                    or outcome.get('request_sha256') != row[4]
                    or (owner.get('host_id'), owner.get('attempt_id')) != row[2:4]
                    or type(receipt.get('returncode')) is not int or receipt['returncode'] == 0
                    or receipt.get('signal') is not None or receipt.get('automatic_retry') is not False):
                raise ValueError('Preparation repair differs from its terminal failed receipt: '+job_id)
            verified[job_id] = dict(provider=request['preparation']['provider'], job_directory=str(job),
                                   attempt_id=row[3], request_sha256=row[4], returncode=receipt['returncode'])
    return verified


def retry_records(records, repairs, *, pf=False):
    records = deepcopy(records)
    archived = {}
    for repair in repairs:
        state_file = Path(repair['state_file']).resolve()
        parent = state_file.parent.parent
        old = read(state_file)
        old = old if pf else old['records']
        attempts = verify_attempts(parent, repair['job_ids'])
        used = set()
        for key in repair['record_keys']:
            original = old[key]
            carried = records.get(key)
            if carried is None or {k:v for k,v in carried.items() if k != 'reused_from'} != {k:v for k,v in original.items() if k != 'reused_from'}:
                raise ValueError('Preparation repair must preserve the original selected record: '+key)
            directory = state_file.parent/'candidates'/key
            matches = []
            for marker in directory.rglob('*.DISTRIBUTED.json'):
                task = read(marker).get('job_id')
                if task in attempts:
                    original_request = deepcopy(read(marker).get('request'))
                    submitted_request = read(Path(attempts[task]['job_directory'])/'REQUEST.json')
                    if not isinstance(original_request, dict) or not isinstance(original_request.get('preparation'), dict):
                        raise ValueError('Preparation repair candidate/task binding differs')
                    # Submission assigns the task ID and the default empty
                    # dependency list after the bridge freezes its leaf.
                    if 'request_id' not in original_request:
                        original_request['request_id'] = task
                    original_request['preparation'].setdefault('dependencies', [])
                    submitted_request['preparation'].setdefault('dependencies', [])
                    if original_request != submitted_request:
                        raise ValueError('Preparation repair candidate/task binding differs')
                    matches.append(task)
            if not matches:
                raise ValueError('Preparation repair has no selected terminal task: '+key)
            from .pr_candidate_completion import preparation_retry_protected
            if not pf and (original.get('method') != 'PR_DF' or not preparation_retry_protected(original)):
                raise ValueError('Preparation repair only lifts the selected PR candidate failure')
            used.update(matches)
            archived[key] = dict(original_record=original, state_file=str(state_file),
                                 original_attempts={job:attempts[job] for job in matches})
            records.pop(key)
        if used != set(attempts):
            raise ValueError('Preparation repair includes an unrelated task')
    return records, archived
