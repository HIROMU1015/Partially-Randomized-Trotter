"""Byte/native proof of stopped run04; historical cost is not the new budget."""
import json
from pathlib import Path
from .identity import require,sha,fingerprint
from .prelaunch_audit import private_path,streaming_sha
RUN='h4-newhost-signal-compile-20261009-run04'
SOURCE='d31b51080665a7eea806ff1adc17f3e0d19151fc'
BASE=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261009/h4-production-run04-20261009')
RECEIPT_BYTES=15988
RECEIPT_SHA='30e76ba2a7c69bd4ef57cb1c82d579f5a913df68286ff4f477f421dba412596f'

def validate_metadata(receipt):
    require(receipt['schema_version']=='h4-stopped-attempt-receipt-v1' and receipt['run_id']==RUN and
            receipt['source_commit']==SOURCE and receipt['status']=='FAIL_CLOSED_STOP' and
            receipt['one_shot_retained'] is True and receipt['automatic_retry'] is False and
            receipt['all_owned_processes_ended'] is True,'run04 stopped attempt lineage')
    require(receipt['first_stop']['reason']=='memory_pressure' and
            receipt['actual_invocations_consumed_or_reserved']==receipt['new_reserved_invocations']==12 and
            receipt['completed_wrappers']==receipt['signal_records']==0 and
            receipt['cumulative_charged_bytes']==4263792756 and
            receipt['conservative_wall_upper_seconds']==742.2718616949767 and
            receipt['driver_exit_exact_time'] is None,'run04 original stop/cost')
    audits=receipt['native_identity_audits'];expected_pids={receipt['driver_pid'],receipt['observer_pid'],*receipt['worker_pids']}
    require(len(expected_pids)==14 and len(receipt['worker_pids'])==12 and len(audits)==2 and
            audits[1]['monotonic']-audits[0]['monotonic']>=2,'run04 exact14 two native audits')
    identities=[]
    for audit in audits:
        rows=audit['identities'];require(len(rows)==14 and not audit['same_owned_remaining'] and
            all(r['classification']=='ABSENT' for r in rows),'run04 native remaining owners')
        current={r['expected']['pid']:r['expected'] for r in rows}
        require(set(current)==expected_pids and all(set(r)=={'pid','start','parent','uid'} for r in current.values()),'run04 native identities')
        identities.append(current)
    require(identities[0]==identities[1],'run04 native identity mismatch')
    return identities[0]

def verify_stopped_attempt(evidence):
    from .observer import process_sample
    ref=evidence['newhost_stopped_attempt']
    require(ref=={'path':str(BASE/'RUNTIME_STOP_RECEIPT_v4.json'),'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run04 fixed proof reference')
    require(streaming_sha(ref['path'])=={'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run04 receipt bytes/hash')
    receipt=json.loads(private_path(ref['path']).read_bytes());identities=validate_metadata(receipt)
    out=BASE/RUN;control=BASE/'control'/RUN
    allowed={out/('ledger-%06d.json'%i) for i in range(13)}
    allowed|={out/n for n in ('byte-budget.journal','ledger.lock','observer.jsonl')}
    allowed|={control/'one-shot.json',control/'runner.log'}
    rows=receipt['files'];require(len(rows)==len(allowed) and {Path(r['path']) for r in rows}==allowed,'run04 exact proof inventory')
    for row in rows:require(streaming_sha(row['path'])=={'bytes':row['bytes'],'sha256':row['sha256']},'run04 proof bytes/hash')
    raw=(out/'byte-budget.journal').read_bytes();require(sha(raw)==receipt['budget_journal_sha256'] and len(raw)==128*receipt['budget_journal_rows'],'run04 budget journal')
    charges=[int(raw[i:i+128].strip()) for i in range(0,len(raw),128)]
    require(charges[:3]==[16908416,4246814848,59760] and all(v>=128 for v in charges) and
            sum(charges)==receipt['cumulative_charged_bytes'],'run04 original reservations not rewritten')
    chain=None;reservations={}
    for version in range(13):
        row=json.loads((out/('ledger-%06d.json'%version)).read_bytes())
        require(row['schema_version']=='h4-completion-ledger-delta-v1' and row['version']==version and
                row['previous_digest']==chain and row['mandatory_stop'] is True and not row['entries'],'run04 immutable ledger chain')
        reservations.update(row['reservations']);chain=fingerprint('h4-ledger-delta-v1',row)
    require(chain==receipt['ledger_head_digest'] and len(reservations)==12 and
            {r['invocation'] for r in reservations.values()}=={'science-%06d'%i for i in range(1,13)} and
            all(r['status']=='RESERVED' for r in reservations.values()),'run04 original science reservations')
    marker=json.loads((control/'one-shot.json').read_bytes())
    require(marker['run_id']==RUN and marker['source']==SOURCE and marker['driver_pid']==receipt['driver_pid'] and
            marker['uid']==identities[receipt['driver_pid']]['uid'],'run04 consumed exclusive marker')
    trace=[json.loads(line) for line in (out/'observer.jsonl').read_bytes().splitlines()]
    stops=[r for r in trace if r.get('kind')=='first_stop'];observations=[r for r in trace if r.get('kind')=='observation']
    require(len(stops)==1 and stops[0]['first_failure']==receipt['first_stop'] and observations,'run04 first STOP retained')
    last=observations[-1]
    recorded={r['pid']:{k:r[k] for k in ('pid','start','parent','uid')} for r in [*last['processes'],last['observer']]}
    require(recorded==identities,'run04 observer/native identity binding')
    for pid,expected in identities.items():
        try:current=process_sample(pid)
        except FileNotFoundError:continue
        require(current['start']!=expected['start'],'run04 owned identity still exists')
    return receipt
