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
    store=ModelStore(target);imported=0;retained=0;states={};jobs=[]
    with store.connect()as db:
        db.execute('BEGIN IMMEDIATE')
        for mid,identity,raw in rows:
            record=json.loads(raw)
            if record['model_id']!=mid or digest(record['identity'])!=mid or json.loads(identity)!=record['identity']:
                raise ValueError('Invalid model identity in source snapshot: '+mid)
            previous=db.execute('SELECT record FROM models WHERE id=?',(mid,)).fetchone()
            if previous:
                if json.loads(previous[0])!=record:
                    raise ValueError('Model state differs; retain both stores and reconcile before activation: '+mid)
                retained+=1
            else:
                db.execute('INSERT INTO models VALUES(?,?,?)',(mid,identity,raw));imported+=1
            states[record['status']]=states.get(record['status'],0)+1
            if record.get('job_id')or record.get('external_job'):
                jobs.append({key:record.get(key)for key in ('model_id','status','owner_run','job_id','external_job','result_path')})
    return dict(status='PASS',source=str(source),target=str(target),models=len(rows),
        imported=imported,retained=retained,states=states,owned_or_external_jobs=jobs,
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
