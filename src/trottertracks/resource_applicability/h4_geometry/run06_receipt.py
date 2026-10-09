"""Byte/native proof of stopped run06; historical cost is not the new budget."""
import json
from pathlib import Path
from .identity import require,sha,fingerprint
from .prelaunch_audit import private_path,streaming_sha
RUN='h4-newhost-signal-compile-20261010-run06'
SOURCE='697843fbbd2a2aa7224261aa26f8da141ca57687'
BASE=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261010/h4-production-run06-20261010')
RECEIPT_BYTES=8476
RECEIPT_SHA='0448c31ab20416b47142c3877816e0e3aa4e66c20d9f0e9b0bf04fc69ac43fe6'

def validate_metadata(receipt):
    require(receipt['run_id']==RUN and
            receipt['source_commit']==SOURCE and receipt['status']=='FAIL_CLOSED_STOP' and
            receipt['driver_exit_code']==143 and
            receipt['all_owned_processes_ended'] is True,'run06 stopped attempt lineage')
    require(receipt['first_stop']['reason']=='memory_pressure' and
            receipt['new_actual_reservations']==4 and
            receipt['completed_records']==receipt['signal_records']==0 and
            receipt['attempt_charged_bytes']==4263786622 and
            receipt['exact_exit_time'] is None and
            receipt['first_stop']['memory']['psi_full_by_scope']['host']==0.18 and
            all(v==0 for k,v in receipt['first_stop']['memory']['psi_full_by_scope'].items() if k!='host'),'run06 original stop/cost')
    audits=receipt['native_identity_audits'];expected_pids={receipt['driver_pid'],receipt['observer_pid'],*receipt['worker_pids']}
    require(len(expected_pids)==6 and len(receipt['worker_pids'])==4 and len(audits)==2 and
            audits[1]['monotonic']-audits[0]['monotonic']>=2,'run06 exact6 two native audits')
    identities=[]
    for audit in audits:
        rows=audit['identities'];require(len(rows)==6 and not audit['same_owned_remaining'] and
            all(r['classification']=='ABSENT' for r in rows),'run06 native remaining owners')
        current={r['expected']['pid']:r['expected'] for r in rows}
        require(set(current)==expected_pids and all(set(r)=={'pid','start','parent','uid'} for r in current.values()),'run06 native identities')
        identities.append(current)
    require(identities[0]==identities[1],'run06 native identity mismatch')
    return identities[0]

def verify_run06_stop(evidence):
    from .observer import process_sample
    ref=evidence['newhost_run06_predecessor']
    require(ref=={'path':str(BASE/'RUNTIME_STOP_RECEIPT_v6.json'),'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run06 fixed proof reference')
    require(streaming_sha(ref['path'])=={'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run06 receipt bytes/hash')
    receipt=json.loads(private_path(ref['path']).read_bytes());identities=validate_metadata(receipt)
    out=BASE/RUN;control=BASE/'control'/RUN
    allowed={out/('ledger-%06d.json'%i) for i in range(5)}
    allowed|={out/n for n in ('byte-budget.journal','ledger.lock','observer.jsonl')}
    allowed|={control/'one-shot.json',control/'runner.log'}
    rows=receipt['files'];require(len(rows)==len(allowed) and {Path(r['path']) for r in rows}==allowed,'run06 exact proof inventory')
    for row in rows:require(streaming_sha(row['path'])=={'bytes':row['bytes'],'sha256':row['sha256']},'run06 proof bytes/hash')
    raw=(out/'byte-budget.journal').read_bytes();require(len(raw)%128==0,'run06 budget journal')
    charges=[int(raw[i:i+128].strip()) for i in range(0,len(raw),128)]
    require(charges[:3]==[16908416,4246814848,59760] and all(v>=128 for v in charges) and
            sum(charges)==receipt['attempt_charged_bytes'],'run06 original reservations not rewritten')
    chain=None;reservations={}
    for version in range(5):
        row=json.loads((out/('ledger-%06d.json'%version)).read_bytes())
        require(row['schema_version']=='h4-completion-ledger-delta-v1' and row['version']==version and
                row['previous_digest']==chain and row['mandatory_stop'] is True and not row['entries'],'run06 immutable ledger chain')
        reservations.update(row['reservations']);chain=fingerprint('h4-ledger-delta-v1',row)
    require(len(reservations)==4 and
            {r['invocation'] for r in reservations.values()}=={'science-%06d'%i for i in range(1,5)} and
            all(r['status']=='RESERVED' for r in reservations.values()),'run06 original science reservations')
    marker=json.loads((control/'one-shot.json').read_bytes())
    require(marker['run_id']==RUN and marker['source']==SOURCE and marker['driver_pid']==receipt['driver_pid'] and
            marker['uid']==identities[receipt['driver_pid']]['uid'],'run06 consumed exclusive marker')
    trace=[json.loads(line) for line in (out/'observer.jsonl').read_bytes().splitlines()]
    stops=[r for r in trace if r.get('kind')=='first_stop'];observations=[r for r in trace if r.get('kind')=='observation']
    require(len(stops)==1 and stops[0]['first_failure']==receipt['first_stop'] and observations,'run06 first STOP retained')
    last=observations[-1]
    recorded={r['pid']:{k:r[k] for k in ('pid','start','parent','uid')} for r in [*last['processes'],last['observer']]}
    require(recorded==identities,'run06 observer/native identity binding')
    for pid,expected in identities.items():
        try:current=process_sample(pid)
        except FileNotFoundError:continue
        require(current['start']!=expected['start'],'run06 owned identity still exists')
    return receipt
