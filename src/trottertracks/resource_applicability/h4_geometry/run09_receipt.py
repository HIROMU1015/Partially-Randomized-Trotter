"""Pinned proof of user-approved run09 stop for the Gaussian cost switch."""
import json
from pathlib import Path
from .identity import require,sha
from .prelaunch_audit import private_path,streaming_sha

RUN='h4-newhost-signal-compile-20261010-run09'
SOURCE='251993785ef1dab2a3891bdbb5d079f5d2184f4d'
BASE=Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261010/h4-production-run09-20261010')
RECEIPT=BASE/'RUNTIME_STOP_RECEIPT_v9.json'
RECEIPT_BYTES=15424
RECEIPT_SHA='3e96fa11367c8b2c41c47e26dfdb6752e07dca9148b4059def6ab1a5d86fa0e2'


def validate_metadata(receipt):
    require(receipt['run_id']==RUN and receipt['source_commit']==SOURCE and
            receipt['artifact_commit']=='82d1f43e97ee924f737a7b43a50b82a1d076297e' and
            receipt['status']=='USER_APPROVED_STRUCTURE_SWITCH_STOP' and
            receipt['manual_stop_authority']=='新方式へ切替・再実行' and receipt['all_owned_processes_ended'] is True,
            'run09 explicit owned switch stop lineage')
    require(receipt['historical_driver_exit_code'] is None and receipt['exact_driver_worker_exit_time'] is None and
            receipt['exit_data_not_inferred'] is True and receipt['actual_invocations_consumed_or_reserved']==8 and
            receipt['completed_wrappers']==4 and receipt['signal_records']==2 and receipt['charged_bytes']==4263948142 and
            receipt['ledger_versions']==13 and receipt['old_cost_series']=='h4-full-gaussian-paired-wrapper-v1' and
            receipt['old_partial_results_reused'] is False and
            receipt['next_attempt_carry']==dict(actual_invocations=0,charged_bytes=0,wall_seconds=0.0),
            'run09 historical cost/unknown exits/new-series isolation')
    audits=receipt['native_identity_audits'];values=[]
    require(len(audits)==2 and audits[1]['monotonic']-audits[0]['monotonic']>=2,'run09 two native audits')
    for audit in audits:
        rows=audit['identities']
        require(len(rows)==6 and not audit['same_owned_remaining'] and all(r['classification']=='ABSENT' and
                r['current'] is None for r in rows),'run09 exact native6 absence')
        identities={r['expected']['pid']:r['expected'] for r in rows}
        require(len(identities)==6 and all(set(r)=={'pid','start','parent','uid'} for r in identities.values()),'run09 native identity fields')
        values.append(identities)
    last=receipt['last_observation'];native={r['pid']:{k:r[k] for k in ('pid','start','parent','uid')}
        for r in [*last['processes'],last['observer']]}
    require(len(last['processes'])==5 and native==values[0]==values[1],'run09 observer/native identity binding')
    submission=receipt['stop_submission']
    require(submission['authority']=='新方式へ切替・再実行' and submission['reason']=='USER_APPROVED_STRUCTURE_SWITCH' and
            submission['signal']=='SIGINT' and submission['only_driver_pidfd'] is True and
            {k:submission['target_driver_identity'][k] for k in ('pid','start','parent','uid')}==native[last['processes'][0]['pid']],
            'run09 owned manual stop authorization retained')
    return values[0]


def verify_run09_stop(evidence):
    from .observer import process_sample
    ref={'path':str(RECEIPT),'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA}
    require(evidence['newhost_run09_predecessor']==ref,'run09 fixed proof reference')
    require(streaming_sha(RECEIPT)=={'bytes':RECEIPT_BYTES,'sha256':RECEIPT_SHA},'run09 proof bytes/SHA')
    receipt=json.loads(private_path(RECEIPT).read_bytes());identities=validate_metadata(receipt)
    out=BASE/RUN;control=BASE/'control'/RUN;rows=receipt['files']
    require(len(rows)==26 and len({r['path'] for r in rows})==26,'run09 exact26 proof inventory')
    for row in rows:
        p=Path(row['path'])
        require((p.parent==out or p.parent==control or p in (BASE/'USER_APPROVED_STRUCTURE_SWITCH_STOP_v1.json',
                BASE/'NATIVE_ZERO_AUDITS_STRUCTURE_SWITCH_v1.json')), 'run09 proof scope')
        require(streaming_sha(p)=={'bytes':row['bytes'],'sha256':row['sha256']},'run09 original bytes/SHA')
        if p.name.startswith('record-'):
            record=json.loads(p.read_bytes())
            require(record['status']=='COMPLETE' and record['source_commit']==SOURCE and
                    record['wrapper_semantics']=='h4-full-gaussian-paired-wrapper-v1' and
                    p.name=='record-'+record['wrapper_key']+'.json','run09 old-cost completion isolation')
    needed={str(out/('ledger-%06d.json'%i)) for i in range(13)}
    needed|={str(out/n) for n in ('byte-budget.journal','ledger.lock','observer.jsonl',
        'signal-0.70-B0-rank3-q1-r0-K0.json','signal-0.70-B0-rank3-q2-r0-K0.json')}
    needed|={str(control/'one-shot.json'),str(control/'runner.log'),str(BASE/'USER_APPROVED_STRUCTURE_SWITCH_STOP_v1.json'),
        str(BASE/'NATIVE_ZERO_AUDITS_STRUCTURE_SWITCH_v1.json')}
    require(needed<={r['path'] for r in rows} and len([r for r in rows if Path(r['path']).name.startswith('record-')])==4,
            'run09 ledger/control/trace/old completion proof inventory')
    raw=(out/'byte-budget.journal').read_bytes()
    require(len(raw)%128==0 and sha(raw)==receipt['journal_sha256'] and
            sum(int(raw[i:i+128].strip()) for i in range(0,len(raw),128))==receipt['charged_bytes'],
            'run09 historical journal retained')
    marker=json.loads((control/'one-shot.json').read_bytes());driver=receipt['last_observation']['processes'][0]['pid']
    require(marker['run_id']==RUN and marker['source']==SOURCE and marker['driver_pid']==driver and
            marker['uid']==identities[driver]['uid'],'run09 consumed one-shot retained')
    for pid,expected in identities.items():
        try:current=process_sample(pid)
        except FileNotFoundError:continue
        require(current['start']!=expected['start'],'run09 owned process still exists')
    return receipt
