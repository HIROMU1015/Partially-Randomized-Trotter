"""Byte/native proof of consumed run03, independent of the next attempt budget."""
import json
from pathlib import Path

from .identity import require,sha,fingerprint
from .prelaunch_audit import private_path,streaming_sha

RUN='h4-newhost-signal-compile-20261009-run03'
SOURCE='6e68fd9bcc68e788db6f5d43eaa6a03866e53d3b'
RECEIPT_BYTES=12527
RECEIPT_SHA='420f270a65e9c178581bfaa5d61e4eeae281ffbfdc474022ecfca23a458cac4a'
BASE=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261009/h4-production-run03-20261009')


def validate_metadata(receipt,prior,carry):
    require(receipt['run_id']==RUN and receipt['source_commit']==SOURCE and
            receipt['status']=='FAIL_CLOSED_STOP' and receipt['automatic_retry'] is False and
            receipt['one_shot_retained'] is True and receipt['all_owned_processes_ended'] is True,
            'run03 native STOP lineage')
    require(receipt['new_reserved_invocations']==1 and receipt['completed_wrappers']==0 and
            receipt['signal_records']==0 and receipt['actual_compiler_start_confirmed'] is True and
            receipt['actual_invocations_consumed_or_reserved']==carry['actual_invocations']==prior['actual_invocations']+1 and
            receipt['cumulative_charged_bytes']==carry['charged_bytes'] and
            receipt['conservative_wall_upper_seconds']==carry['wall_seconds'] and
            receipt['driver_exit_exact_time'] is None,'run03 consumed reservation/charge/conservative wall')
    audits=receipt['native_identity_audits']
    require(len(audits)==2 and audits[1]['monotonic']-audits[0]['monotonic']>=2,'two separated native audits')
    expected_pids={receipt['driver_pid'],receipt['observer_pid'],*receipt['worker_pids']}
    require(len(receipt['worker_pids'])==12 and len(expected_pids)==14,'run03 exact14 role identities')
    maps=[]
    for audit in audits:
        rows=audit['identities']
        require(len(rows)==14 and not audit['same_owned_remaining'] and
                all(row['classification']=='ABSENT' for row in rows),'run03 native zero remaining')
        identities={row['expected']['pid']:row['expected'] for row in rows}
        require(set(identities)==expected_pids and
                all(set(row)=={'pid','start','parent','uid'} for row in identities.values()),'run03 identity fields')
        maps.append(identities)
    require(maps[0]==maps[1],'run03 native identities differ between passes')
    return maps[0]


def verify_run03_stop(evidence,prior,carry):
    """Bind the recorded upper wall; do not infer historical end/compiler times."""
    from .observer import process_sample
    ref=evidence['newhost_run03_predecessor']
    require(set(ref)=={'path','bytes','sha256'} and ref['path']==str(BASE/'RUNTIME_STOP_RECEIPT_v3.json') and
            ref['bytes']==RECEIPT_BYTES and ref['sha256']==RECEIPT_SHA,'run03 fixed receipt reference')
    require(streaming_sha(ref['path'])=={'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run03 receipt bytes/hash')
    receipt=json.loads(private_path(ref['path']).read_bytes())
    identities=validate_metadata(receipt,prior,carry)
    allowed={BASE/RUN/name for name in ('byte-budget.journal','ledger-000000.json','ledger-000001.json','ledger.lock','observer.jsonl','worker-log-first-stop.txt')}
    allowed|={BASE/'control'/RUN/name for name in ('one-shot.json','runner.log')}
    rows=receipt['files'];require(len(rows)==len(allowed) and {Path(r['path']) for r in rows}==allowed,'run03 exact proof files')
    for row in rows:
        require(streaming_sha(row['path'])=={'bytes':row['bytes'],'sha256':row['sha256']},'run03 proof file bytes/hash')
    root=private_path(BASE/RUN);data=(root/'byte-budget.journal').read_bytes()
    require(sha(data)==receipt['budget_journal_sha256'] and len(data)==128*receipt['budget_journal_rows'], 'run03 journal lineage')
    charges=[int(data[i:i+128].strip()) for i in range(0,len(data),128)]
    require(charges[0]==prior['charged_bytes']+128 and all(v>=128 for v in charges) and
            sum(charges)==carry['charged_bytes'],'run03 cumulative charge cannot be refunded')
    chain=None;reservations={}
    for version in range(2):
        row=json.loads((root/('ledger-%06d.json'%version)).read_bytes())
        require(row['schema_version']=='h4-completion-ledger-delta-v1' and row['mandatory_stop'] is True and
                row['version']==version and row['previous_digest']==chain and not row['entries'],'run03 ledger chain')
        reservations.update(row['reservations']);chain=fingerprint('h4-ledger-delta-v1',row)
    require(len(reservations)==1 and next(iter(reservations.values()))=={'invocation':'science-000022','status':'RESERVED'},
            'run03 failed actual reservation stays consumed')
    marker=json.loads((private_path(BASE/'control'/RUN)/'one-shot.json').read_bytes())
    require(marker['run_id']==RUN and marker['source']==SOURCE and marker['driver_pid']==receipt['driver_pid'] and
            marker['uid']==identities[receipt['driver_pid']]['uid'],'run03 consumed one-shot identity')
    trace=[json.loads(line) for line in (root/'observer.jsonl').read_bytes().splitlines()]
    stops=[row for row in trace if row.get('kind')=='first_stop']
    observed=[row for row in trace if row.get('kind')=='observation']
    require(len(stops)==1 and stops[0]['first_failure']==receipt['first_stop'] and observed,'run03 first STOP preserved')
    last=observed[-1]
    recorded={row['pid']:{key:row[key] for key in ('pid','start','parent','uid')}
              for row in [*last['processes'],last['observer']]}
    require(recorded==identities,'run03 observer/native identity binding')
    for pid,expected in identities.items():
        try:current=process_sample(pid)
        except FileNotFoundError:continue
        require(current['start']!=expected['start'],'run03 owned identity still exists')
    return receipt
