#!/usr/bin/env python3
"""Build a new-release CPU policy from already accepted Q3 timing receipts.

No evaluation is run. The fixed profile selector compares only identical
workloads under observed alternative CPU footprints; unknown workloads retain
the original profile. This is an operational policy, not scientific evidence.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
from collections import defaultdict
from itertools import chain

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trottertracks.rpe_host_profiles import q3_cpu_measurement_key


def metadata_int(value):
    # Very large stored rational numerators are unrelated to resource timing.
    # Keep their text instead of triggering a Python integer-string limit.
    return int(value) if len(value) <= 1000 else value


def read(path):
    return json.loads(path.read_bytes(), parse_int=metadata_int)


def accepted_observations(run):
    run = Path(run).resolve()
    database = run/'distributed/QUEUE.sqlite3'
    if not database.is_file():
        return
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as db:
        rows = db.execute("select job_id,attempt_id,host_id,request_sha256,request,eligibility from jobs "
                          "where state='ACCEPTED' and method not like 'PREPARATION:%'").fetchall()
    for job, attempt, host, sha, raw, eligibility in rows:
        leaf = run/'distributed/transfers'/job/attempt
        try:
            sample, receipt, outcome = (read(leaf/name) for name in ('Q3_EXECUTION_RESOURCES.json','PROCESS_RETURN.json','OUTCOME.json'))
            req = json.loads(raw, parse_int=metadata_int)
            runtime = json.loads(eligibility, parse_int=metadata_int)['execution_profiles'][host]
            policy_host = sample.get('policy_host_id',host) if host=='local' else host
            if policy_host != host:
                runtime['q3_measurement_host_id'] = policy_host
            key = q3_cpu_measurement_key(req, runtime)
            if (sample.get('schema')!='RPE_Q3_EXECUTION_RESOURCES_V1' or not key or sample['key']!=key
                    or (sample['job_id'],sample['attempt_id'],sample['host_id'],sample['request_sha256'])!=(job,attempt,host,sha)
                    or sample.get('policy_host_id',host)!=policy_host
                    or sample['route_kind']!=req['route']['kind'] or sample['method']!=req['method']
                    or sample['dp_workers']!=runtime['dp_workers'] or sample['task_cpu_count']-runtime['dp_workers'] not in (0,1)
                    or sample['wall_seconds']!=receipt['wall_seconds']
                    or sample['samples']<2 or sample['peak_rss_bytes']<=0
                    or receipt['returncode']!=0 or receipt.get('signal') is not None
                    or outcome['request_id']!=job or outcome.get('status') not in
                    ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD','INCONCLUSIVE')):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        yield policy_host, sample['route_kind'], dict(key=key, job_id=job, task_cpu_count=sample['task_cpu_count'],
            dp_workers=sample['dp_workers'], wall_seconds=sample['wall_seconds'], completed=True)


def imported_observations(run):
    """Independent PBS blocks export the same optional metadata after acceptance."""
    from trottertracks.rpe_q3_resources import valid_q3_execution_receipt
    pbs = Path(run).resolve()/'distributed/pbs'
    state_path = pbs/'STATE.json'
    if not state_path.is_file():return
    allocations = read(state_path).get('allocations',{})
    for path in (pbs/'offline-history/jobs').glob('*/*/Q3_EXECUTION_RESOURCES.json'):
        try:
            leaf=path.parent; imported=read(leaf/'IMPORTED.json')
            host=allocations[imported['token']]['spec']['profile']['host_id']
            sample, req, receipt, outcome=(read(leaf/name) for name in ('Q3_EXECUTION_RESOURCES.json','REQUEST.json','PROCESS_RETURN.json','OUTCOME.json'))
            if not valid_q3_execution_receipt(sample,req,receipt,outcome,policy_host=host):continue
            yield host,sample['route_kind'],dict(key=sample['key'],job_id=sample['job_id'],task_cpu_count=sample['task_cpu_count'],
                dp_workers=sample['dp_workers'],wall_seconds=sample['wall_seconds'],completed=True)
        except (OSError,ValueError,KeyError,TypeError):continue


def build_policy(runs):
    observations = defaultdict(lambda: defaultdict(dict))
    routes = defaultdict(set)
    for run in runs:
        for host, route, row in chain(accepted_observations(run), imported_observations(run)):
            routes[host].add(route)
            # An accepted job repeated in a resumed/copied run is not a new measurement.
            observations[host][row['key']][row['job_id']] = row
    policies = {}
    for host, by_key in observations.items():
        measured = {}
        for key, jobs in by_key.items():
            counts = defaultdict(int)
            for row in jobs.values():
                counts[row['task_cpu_count']] += 1
            # Only keys with at least two already observed footprints can change a policy.
            matched = {cpu for cpu, count in counts.items() if count >= 2}
            if len(matched)>=2:
                measured[key] = [row for row in jobs.values() if row['task_cpu_count'] in matched]
        if measured:
            policies[host] = dict(route_kinds=sorted(routes[host]), measurements=measured)
    return policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_policy(args.run), indent=2)+'\n')


if __name__=='__main__':
    main()
