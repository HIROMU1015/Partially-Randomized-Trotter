"""Adopt an exact saved model store before activating a replacement controller.

Uses the existing ModelStore and its model IDs. Frozen requests, owners, partial
results and failures remain opaque. Conflicts require reconciliation, never a
second dispatch. No compiler, model evaluation or scientific job is invoked.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_price_validation import ModelStore, digest


def reconcile(source, target):
    source=Path(source).resolve();target=Path(target).resolve()
    if source==target:raise ValueError('Source snapshot and target must differ')
    old=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
    try:rows=old.execute('SELECT id,identity,record FROM models').fetchall()
    finally:old.close()
    store=ModelStore(target);imported=0;retained=0;source_states={};conflicts=[]
    with store.connect()as db:
        db.execute('BEGIN IMMEDIATE')
        for mid,identity,raw in rows:
            try:
                record=json.loads(raw)
                if record['model_id']!=mid or digest(record['identity'])!=mid or json.loads(identity)!=record['identity']:
                    raise ValueError('Model identity differs')
            except (ValueError,KeyError,TypeError)as error:
                conflicts.append(dict(model_id=mid,reason='INVALID_SOURCE_MODEL_IDENTITY',detail=str(error)))
                continue
            source_states[record['status']]=source_states.get(record['status'],0)+1
            previous=db.execute('SELECT record FROM models WHERE id=?',(mid,)).fetchone()
            if previous:
                current=json.loads(previous[0])
                frozen=('identity','selected_round','job_id','request','validation_request','owner_run','attempt_id','owner_host')
                differences=[key for key in frozen if record.get(key)is not None and current.get(key)!=record[key]]
                # External state may advance, but its actual owner/attempt must
                # not be exchanged for a different job during a transfer.
                for key in ('job_id','attempt_id','host_id','owner_run','request_sha256'):
                    value=(record.get('external_job')or{}).get(key)
                    if value is not None and(current.get('external_job')or{}).get(key)!=value:
                        differences.append('external_job.'+key)
                if differences:
                    conflicts.append(dict(model_id=mid,reason='FROZEN_MODEL_STATE_DIFFERS',fields=differences))
                    continue
                # A repeated transfer must not replace locally collected
                # results/parser corrections with its older saved snapshot.
                retained+=1
            else:
                db.execute('INSERT INTO models VALUES(?,?,?)',(mid,identity,raw));imported+=1
        states={};jobs=[]
        for(raw,)in db.execute('SELECT record FROM models ORDER BY id'):
            record=json.loads(raw)
            states[record['status']]=states.get(record['status'],0)+1
            if record.get('job_id')or record.get('external_job'):
                jobs.append({key:record.get(key)for key in ('model_id','status','owner_run','job_id','external_job','result_path')})
    return dict(status='HOLD'if conflicts else'PASS',source=str(source),target=str(target),models=len(rows),
        imported=imported,retained=retained,held=len(conflicts),conflicts=conflicts,
        states=states,source_states=source_states,owned_or_external_jobs=jobs,
        frozen_requests_and_owners_preserved=True,scientific_calls=0,compile_calls=0,new_queue_submissions=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=reconcile(a.source,a.target)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
