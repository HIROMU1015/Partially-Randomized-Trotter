"""Exact native/byte proof of run08; retain unknown exits and historical cost."""
import json
from pathlib import Path
from .identity import require,sha
from .prelaunch_audit import private_path,streaming_sha

RUN='h4-newhost-signal-compile-20261010-run08'
SOURCE='8c917a5af7943969d4abede8b0bb012880efeee0'
BASE=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261010/h4-production-run08-20261010')
RECEIPT=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261010/h4-production-run09-20261010/RUNTIME_STOP_RECEIPT_v8.json')
RECEIPT_BYTES=17322
RECEIPT_SHA='d478ef6fbc47f2c02f0da273392c6f9f3c718343e5a05100ff0cc843eea1dfa6'


def validate_metadata(receipt):
    require(receipt['run_id']==RUN and receipt['source_commit']==SOURCE and
            receipt['artifact_commit']=='8ab82ef6f600a3497fa170149012d97904162c20' and
            receipt['status']=='FAIL_CLOSED_STOP' and receipt['all_owned_processes_ended'] is True,
            'run08 stopped lineage')
    require(receipt['historical_driver_exit_code'] is None and receipt['exact_driver_worker_exit_time'] is None and
            receipt['exit_data_not_inferred'] is True and receipt['first_stop']['reason']=='host_memory_pressure' and
            receipt['first_stop']['memory']['psi_full_by_scope']['host']==1.26 and
            all(v==0 for k,v in receipt['first_stop']['memory']['psi_full_by_scope'].items() if k!='host') and
            receipt['actual_invocations_consumed_or_reserved']==6 and receipt['completed_wrappers']==2 and
            receipt['signal_records']==1 and receipt['charged_bytes']==4263867374 and receipt['ledger_versions']==9 and
            receipt['next_attempt_carry']==dict(actual_invocations=0,charged_bytes=0,wall_seconds=0.0),
            'run08 original pressure/cost/unknown exits')
    audits=receipt['native_identity_audits'];values=[]
    require(len(audits)==2 and audits[1]['monotonic']-audits[0]['monotonic']>=2,'run08 two separated native audits')
    for audit in audits:
        rows=audit['identities']
        require(len(rows)==6 and not audit['same_owned_remaining'] and
                all(r['classification']=='ABSENT' and r['current'] is None for r in rows),'run08 native absence')
        identities={r['expected']['pid']:r['expected'] for r in rows}
        require(len(identities)==6 and all(set(r)=={'pid','start','parent','uid'} for r in identities.values()),'run08 exact6 identities')
        values.append(identities)
    last=receipt['last_observation']
    native={r['pid']:{k:r[k] for k in ('pid','start','parent','uid')} for r in [*last['processes'],last['observer']]}
    require(len(last['processes'])==5 and native==values[0]==values[1],'run08 observer/native identities')
    return values[0]


def verify_run08_stop(evidence):
    from .observer import process_sample
    ref={'path':str(RECEIPT),'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA}
    require(evidence['newhost_run08_predecessor']==ref,'run08 fixed proof reference')
    require(streaming_sha(RECEIPT)=={'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run08 proof bytes/SHA')
    receipt=json.loads(private_path(RECEIPT).read_bytes());identities=validate_metadata(receipt)
    origin=receipt['original_status_receipt']
    require(origin==dict(path=str(BASE/'RUNTIME_STOP_STATUS_RECEIPT_v8.json'),bytes=12213,
                        sha256='10cd139947c5497d970cc3620b25075e91fc6b10794421b563da47a2ca450e6c') and
            streaming_sha(origin['path'])=={'bytes':origin['bytes'],'sha256':origin['sha256']},'run08 original receipt retained')
    out=BASE/RUN;control=BASE/'control'/RUN
    allowed={out/('ledger-%06d.json'%i) for i in range(9)}
    allowed|={out/n for n in ('byte-budget.journal','ledger.lock','observer.jsonl',
        'record-0c8b47966c4e9bf5cee23e7f7be59f4820e31374b0b7e559fa8dff62719a3630.json',
        'record-b2e9097f87722b81e0179aec5fe4eb943f6fbc0b11fa79a1c17c39e0559c1539.json',
        'signal-0.70-B0-rank3-q1-r0-K0.json')}
    allowed|={control/'one-shot.json',control/'runner.log'}
    rows=receipt['files'];require(len(rows)==17 and {Path(r['path']) for r in rows}==allowed,'run08 exact original17 proof files')
    for row in rows:require(streaming_sha(row['path'])=={'bytes':row['bytes'],'sha256':row['sha256']},'run08 original file bytes/SHA')
    data=(out/'byte-budget.journal').read_bytes()
    require(len(data)%128==0 and sum(int(data[i:i+128].strip()) for i in range(0,len(data),128))==receipt['charged_bytes'],
            'run08 historical journal preserved')
    marker=json.loads((control/'one-shot.json').read_bytes());driver=receipt['last_observation']['processes'][0]['pid']
    require(marker['run_id']==RUN and marker['source']==SOURCE and marker['driver_pid']==driver and
            marker['uid']==identities[driver]['uid'],'run08 consumed one-shot')
    for pid,expected in identities.items():
        try:current=process_sample(pid)
        except FileNotFoundError:continue
        require(current['start']!=expected['start'],'run08 owned process still exists')
    return receipt
