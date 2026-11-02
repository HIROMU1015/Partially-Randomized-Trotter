"""Pinned native/byte proof of stopped run07; no scientific array reads."""
import json
from pathlib import Path
from .identity import require,sha
from .prelaunch_audit import private_path,streaming_sha

RUN='h4-newhost-signal-compile-20261010-run07'
SOURCE='1dbdd1be2133a59ab81c7b81a6074af0880f72a3'
ARTIFACT='84cea869cfb03b3312b52e23946256bd7ab62ba1'
BASE=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261010/h4-production-run07-20261010')
RECEIPT=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261010/h4-run07-stop-cause-audit-20261010/RUNTIME_STOP_RECEIPT_v7.json')
RECEIPT_BYTES=13094
RECEIPT_SHA='3156efe49c02a091067c95337a444c8b2ca9a391006f43b72299370a8ca0559f'


def validate_metadata(receipt):
    require(receipt['run_id']==RUN and receipt['source_commit']==SOURCE and
            receipt['artifact_commit']==ARTIFACT and receipt['status']=='FAIL_CLOSED_STOP' and
            receipt['driver_exit_code']==143 and receipt['all_owned_processes_ended'] is True,
            'run07 stopped attempt lineage')
    require(receipt['worker_exit_code'] is None and receipt['exact_driver_worker_exit_time'] is None and
            receipt['past_missing_exit_data_not_inferred'] is True and
            receipt['actual_invocations_consumed_or_reserved']==6 and receipt['completed_wrappers']==2 and
            receipt['signal_records']==1 and receipt['charged_bytes']==4263870068 and
            receipt['ledger_versions']==9 and receipt['next_attempt_carry']==dict(actual_invocations=0,charged_bytes=0,wall_seconds=0.0),
            'run07 original cost/unknown exits/per-attempt carry')
    audits=receipt['native_identity_audits']
    require(len(audits)==2 and audits[1]['monotonic']-audits[0]['monotonic']>=2,'run07 two native audits')
    identities=[]
    for audit in audits:
        rows=audit['identities']
        require(len(rows)==6 and not audit['same_owned_remaining'] and
                all(row['classification']=='ABSENT' for row in rows),'run07 exact6 absent identities')
        values={row['expected']['pid']:row['expected'] for row in rows}
        require(len(values)==6 and all(set(row)=={'pid','start','parent','uid'} for row in values.values()),
                'run07 native identity fields')
        identities.append(values)
    require(identities[0]==identities[1],'run07 native identity mismatch')
    observation=receipt['last_observation']
    recorded={row['pid']:{key:row[key] for key in ('pid','start','parent','uid')}
              for row in [*observation['processes'],observation['observer']]}
    require(len(observation['processes'])==5 and recorded==identities[0], 'run07 observer/native identity binding')
    return identities[0]


def verify_run07_stop(evidence):
    from .observer import process_sample
    expected={'path':str(RECEIPT),'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA}
    require(evidence['newhost_run07_predecessor']==expected,'run07 fixed proof reference')
    require(streaming_sha(RECEIPT)=={'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run07 receipt bytes/hash')
    receipt=json.loads(private_path(RECEIPT).read_bytes());identities=validate_metadata(receipt)
    out=BASE/RUN;control=BASE/'control'/RUN
    allowed={out/('ledger-%06d.json'%i) for i in range(9)}
    allowed|={out/name for name in ('byte-budget.journal','ledger.lock','observer.jsonl','worker-log-first-stop.txt')}
    allowed|={out/row['file'] for row in receipt['record_completion_digest_verification']}
    allowed|={out/'signal-0.70-B0-rank3-q1-r0-K0.json',control/'one-shot.json',control/'runner.log'}
    rows=receipt['files']
    require(len(rows)==18 and len(allowed)==18 and {Path(row['path']) for row in rows}==allowed,
            'run07 exact original proof inventory')
    for row in rows:
        require(streaming_sha(row['path'])=={'bytes':row['bytes'],'sha256':row['sha256']},'run07 original proof bytes/hash')
    raw=(out/'byte-budget.journal').read_bytes()
    require(len(raw)==128*receipt['journal_rows'] and sha(raw)==receipt['journal_sha256'] and
            sum(int(raw[i:i+128].strip()) for i in range(0,len(raw),128))==receipt['charged_bytes'],
            'run07 original charge journal retained')
    marker=json.loads((control/'one-shot.json').read_bytes())
    driver=receipt['last_observation']['processes'][0]['pid']
    require(marker['run_id']==RUN and marker['source']==SOURCE and marker['driver_pid']==driver and
            marker['uid']==identities[driver]['uid'],'run07 consumed one-shot/native driver')
    for pid,expected_identity in identities.items():
        try:current=process_sample(pid)
        except FileNotFoundError:continue
        require(current['start']!=expected_identity['start'],'run07 owned identity still exists')
    return receipt
