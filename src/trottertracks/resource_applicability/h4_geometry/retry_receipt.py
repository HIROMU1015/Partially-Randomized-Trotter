"""Byte/native proof of consumed run02. No arrays, resume, or budget refunds."""
import json
from pathlib import Path

from .identity import require,sha,fingerprint
from .prelaunch_audit import private_path,streaming_sha

RUN='h4-newhost-signal-compile-20261009-run02'
SOURCE='a7b617600cd7063f7870f2059d5694ef00283f0e'
RECEIPT_BYTES=10034
RECEIPT_SHA='b58388bea86bb81699fcf65da95478ecf05ff99b749ffd6f646e09aff44b96dc'
BASE=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261009/approved-relaunch')


def validate_metadata(receipt,prior,carry):
    require(receipt['run_id']==RUN and receipt['source_commit']==SOURCE and
            receipt['status']=='FAIL_CLOSED_STOP' and receipt['automatic_retry'] is False and
            receipt['one_shot_retained'] is True and receipt['all_owned_processes_ended'] is True,
            'run02 native STOP lineage')
    require(receipt['new_reserved_invocations']==1 and receipt['completed_wrappers']==0 and
            receipt['signal_records']==0 and receipt['actual_compiler_start_confirmed'] is False and
            receipt['actual_invocations_consumed_or_reserved']==carry['actual_invocations']==prior['actual_invocations']+1 and
            receipt['cumulative_charged_bytes']==carry['charged_bytes'] and
            receipt['conservative_wall_upper_seconds']==carry['wall_seconds'] and
            receipt['driver_exit_exact_time'] is None,'run02 consumed reservation/charge/conservative wall')
    audits=receipt['native_identity_audits']
    require(len(audits)==2 and audits[1]['monotonic']-audits[0]['monotonic']>=2,'two separated native audits')
    expected_pids={receipt['driver_pid'],receipt['observer_pid'],*receipt['worker_pids']}
    require(len(receipt['worker_pids'])==12 and len(expected_pids)==14,'run02 exact14 role identities')
    maps=[]
    for audit in audits:
        rows=audit['identities']
        require(len(rows)==14 and not audit['same_owned_remaining'] and
                all(row['classification']=='ABSENT' for row in rows),'run02 native zero remaining')
        identities={row['expected']['pid']:row['expected'] for row in rows}
        require(set(identities)==expected_pids and
                all(set(row)=={'pid','start','parent','uid'} for row in identities.values()),'run02 identity fields')
        maps.append(identities)
    require(maps[0]==maps[1],'run02 native identities differ between passes')
    return maps[0]


def verify_retry_stop(evidence,prior,carry):
    """Bind the recorded upper wall; do not infer historical end/compiler times."""
    from .observer import process_sample
    ref=evidence['newhost_retry_predecessor']
    require(set(ref)=={'path','bytes','sha256'} and ref['path']==str(BASE/'RUNTIME_STOP_RECEIPT_v2.json') and
            ref['bytes']==RECEIPT_BYTES and ref['sha256']==RECEIPT_SHA,'run02 fixed receipt reference')
    require(streaming_sha(ref['path'])=={'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run02 receipt bytes/hash')
    receipt=json.loads(private_path(ref['path']).read_bytes())
    identities=validate_metadata(receipt,prior,carry)
    allowed={BASE/RUN/name for name in ('byte-budget.journal','ledger-000000.json','ledger-000001.json','ledger.lock','observer.jsonl')}
    allowed|={BASE/'control'/RUN/name for name in ('one-shot.json','runner.log')}
    rows=receipt['files'];require(len(rows)==len(allowed) and {Path(r['path']) for r in rows}==allowed,'run02 exact proof files')
    for row in rows:
        require(streaming_sha(row['path'])=={'bytes':row['bytes'],'sha256':row['sha256']},'run02 proof file bytes/hash')
    root=private_path(BASE/RUN);data=(root/'byte-budget.journal').read_bytes()
    require(sha(data)==receipt['budget_journal_sha256'] and len(data)==128*receipt['budget_journal_rows'], 'run02 journal lineage')
    charges=[int(data[i:i+128].strip()) for i in range(0,len(data),128)]
    require(charges[0]==prior['charged_bytes']+128 and all(v>=128 for v in charges) and
            sum(charges)==carry['charged_bytes'],'run02 cumulative charge cannot be refunded')
    chain=None;reservations={}
    for version in range(2):
        row=json.loads((root/('ledger-%06d.json'%version)).read_bytes())
        require(row['schema_version']=='h4-completion-ledger-delta-v1' and row['mandatory_stop'] is True and
                row['version']==version and row['previous_digest']==chain and not row['entries'],'run02 ledger chain')
        reservations.update(row['reservations']);chain=fingerprint('h4-ledger-delta-v1',row)
    require(len(reservations)==1 and next(iter(reservations.values()))=={'invocation':'science-000021','status':'RESERVED'},
            'run02 failed actual reservation stays consumed')
    marker=json.loads((private_path(BASE/'control'/RUN)/'one-shot.json').read_bytes())
    require(marker['run_id']==RUN and marker['source']==SOURCE and marker['driver_pid']==receipt['driver_pid'] and
            marker['uid']==identities[receipt['driver_pid']]['uid'],'run02 consumed one-shot identity')
    trace=[json.loads(line) for line in (root/'observer.jsonl').read_bytes().splitlines()]
    stops=[row for row in trace if row.get('kind')=='first_stop']
    observed=[row for row in trace if row.get('kind')=='observation']
    require(len(stops)==1 and stops[0]['first_failure']==receipt['first_stop'] and observed,'run02 first STOP preserved')
    last=observed[-1]
    recorded={row['pid']:{key:row[key] for key in ('pid','start','parent','uid')}
              for row in [*last['processes'],last['observer']]}
    require(recorded==identities,'run02 observer/native identity binding')
    for pid,expected in identities.items():
        try:current=process_sample(pid)
        except FileNotFoundError:continue
        require(current['start']!=expected['start'],'run02 owned identity still exists')
    return receipt
