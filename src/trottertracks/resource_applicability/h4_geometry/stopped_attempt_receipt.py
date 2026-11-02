"""Byte/native proof of stopped run05; historical cost is not the new budget."""
import json
from pathlib import Path
from .identity import require,sha,fingerprint
from .prelaunch_audit import private_path,streaming_sha
RUN='h4-newhost-signal-compile-20261009-run05'
SOURCE='9d1471aff5840a76fa1579f4e14e71e62b8497a3'
BASE=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261009/h4-production-run05-20261009')
RECEIPT_BYTES=15688
RECEIPT_SHA='4cc89cb8bfb0034427b527af9a666680336961f313247c9ef1accc921b6cf64a'

def validate_metadata(receipt):
    require(receipt['run_id']==RUN and
            receipt['source_commit']==SOURCE and receipt['status']=='FAIL_CLOSED_STOP' and
            receipt['driver_exit_code']==143 and
            receipt['all_owned_processes_ended'] is True,'run05 stopped attempt lineage')
    require(receipt['first_stop']['reason']=='memory_pressure' and
            receipt['new_actual_reservations']==12 and
            receipt['completed_records']==receipt['signal_records']==0 and
            receipt['attempt_charged_bytes']==4263792756 and
            receipt['exact_exit_time'] is None and
            receipt['first_stop']['memory']['psi_full_by_scope']['host']==0.18 and
            all(v==0 for k,v in receipt['first_stop']['memory']['psi_full_by_scope'].items() if k!='host'),'run05 original stop/cost')
    audits=receipt['native_identity_audits'];expected_pids={receipt['driver_pid'],receipt['observer_pid'],*receipt['worker_pids']}
    require(len(expected_pids)==14 and len(receipt['worker_pids'])==12 and len(audits)==2 and
            audits[1]['monotonic']-audits[0]['monotonic']>=2,'run05 exact14 two native audits')
    identities=[]
    for audit in audits:
        rows=audit['identities'];require(len(rows)==14 and not audit['same_owned_remaining'] and
            all(r['classification']=='ABSENT' for r in rows),'run05 native remaining owners')
        current={r['expected']['pid']:r['expected'] for r in rows}
        require(set(current)==expected_pids and all(set(r)=={'pid','start','parent','uid'} for r in current.values()),'run05 native identities')
        identities.append(current)
    require(identities[0]==identities[1],'run05 native identity mismatch')
    return identities[0]

def verify_stopped_attempt(evidence):
    from .observer import process_sample
    ref=evidence['newhost_latest_stopped_attempt']
    require(ref=={'path':str(BASE/'RUNTIME_STOP_RECEIPT_v5.json'),'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run05 fixed proof reference')
    require(streaming_sha(ref['path'])=={'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run05 receipt bytes/hash')
    receipt=json.loads(private_path(ref['path']).read_bytes());identities=validate_metadata(receipt)
    out=BASE/RUN;control=BASE/'control'/RUN
    allowed={out/('ledger-%06d.json'%i) for i in range(13)}
    allowed|={out/n for n in ('byte-budget.journal','ledger.lock','observer.jsonl')}
    allowed|={control/'one-shot.json',control/'runner.log'}
    rows=receipt['files'];require(len(rows)==len(allowed) and {Path(r['path']) for r in rows}==allowed,'run05 exact proof inventory')
    for row in rows:require(streaming_sha(row['path'])=={'bytes':row['bytes'],'sha256':row['sha256']},'run05 proof bytes/hash')
    raw=(out/'byte-budget.journal').read_bytes();require(len(raw)%128==0,'run05 budget journal')
    charges=[int(raw[i:i+128].strip()) for i in range(0,len(raw),128)]
    require(charges[:3]==[16908416,4246814848,59760] and all(v>=128 for v in charges) and
            sum(charges)==receipt['attempt_charged_bytes'],'run05 original reservations not rewritten')
    chain=None;reservations={}
    for version in range(13):
        row=json.loads((out/('ledger-%06d.json'%version)).read_bytes())
        require(row['schema_version']=='h4-completion-ledger-delta-v1' and row['version']==version and
                row['previous_digest']==chain and row['mandatory_stop'] is True and not row['entries'],'run05 immutable ledger chain')
        reservations.update(row['reservations']);chain=fingerprint('h4-ledger-delta-v1',row)
    require(len(reservations)==12 and
            {r['invocation'] for r in reservations.values()}=={'science-%06d'%i for i in range(1,13)} and
            all(r['status']=='RESERVED' for r in reservations.values()),'run05 original science reservations')
    marker=json.loads((control/'one-shot.json').read_bytes())
    require(marker['run_id']==RUN and marker['source']==SOURCE and marker['driver_pid']==receipt['driver_pid'] and
            marker['uid']==identities[receipt['driver_pid']]['uid'],'run05 consumed exclusive marker')
    trace=[json.loads(line) for line in (out/'observer.jsonl').read_bytes().splitlines()]
    stops=[r for r in trace if r.get('kind')=='first_stop'];observations=[r for r in trace if r.get('kind')=='observation']
    require(len(stops)==1 and stops[0]['first_failure']==receipt['first_stop'] and observations,'run05 first STOP retained')
    last=observations[-1]
    recorded={r['pid']:{k:r[k] for k in ('pid','start','parent','uid')} for r in [*last['processes'],last['observer']]}
    require(recorded==identities,'run05 observer/native identity binding')
    for pid,expected in identities.items():
        try:current=process_sample(pid)
        except FileNotFoundError:continue
        require(current['start']!=expected['start'],'run05 owned identity still exists')
    return receipt
