"""Parallel saved-path publication only; never start or alter scientific workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

import run_pauli_ld2_m9 as M


def checked_raw(raw, manifest, binding, receipts, source):
    started=time.monotonic()
    M.same(len(raw),manifest['bytes'],'Raw path size differs')
    M.same(hashlib.sha256(raw).hexdigest(),manifest['sha256'],'Raw path hash differs')
    path=json.loads(gzip.decompress(raw))
    M.same(path['index'],manifest['index'],'Manifest path index differs')
    M.check_result_path(path,binding,receipts,source)
    # Event arrays need validation/preservation, not parent IPC or a second parse.
    compact=dict(index=path['index'],result={k:path['result'][k] for k in ('prices','scalar')})
    return compact,dict(index=path['index'],worker_pid=os.getpid(),seconds=time.monotonic()-started,
                        bytes=len(raw),sha256=manifest['sha256'],status='RAW_AND_RECEIPTS_CHECKED')


def load_path(task):
    commit,manifest,binding,receipts,source=task
    return checked_raw(M.blob(commit,manifest['path']),manifest,binding,receipts,source)


def parallel_paths(tasks, loader=load_path):
    M.same(sorted(t[1]['index'] for t in tasks),[0,1],'Incomplete/duplicate paths')
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('fork')) as pool:
        loaded=list(pool.map(loader,tasks))
    loaded.sort(key=lambda x:x[0]['index'])
    return [x[0] for x in loaded],[x[1] for x in loaded]


def publish(raw_commit):
    started=time.monotonic()
    here='scripts/publish_pauli_ld2_m9.py';commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M.ROOT,text=True).strip()
    M.same((M.ROOT/here).read_bytes(),M.blob(commit,here),'Commit publication SOURCE before use')
    B1=M.legacy();I,DC,G=B1.B0._meta();P=B1.B0.connection();binding=M.metadata()
    data=M.read(dict(commit=raw_commit,path=M.BASE+'/run01/RESULT.json'));source=data['source']
    M.same(M.blob(source['commit'],M.SCRIPT),(M.ROOT/M.SCRIPT).read_bytes(),'Scientific driver SOURCE differs')
    M.same(source['step_seeds'],binding['step_seeds'],'Result seed SOURCE differs')
    pre=M.read(data['predictions']);M.check_binding(pre['binding'],binding)
    receipts=M.read(dict(commit=raw_commit,path=M.BASE+'/run01/COMPILE_RECEIPTS.json'))
    M.same(sorted(receipts),['p0X','p0Y','p1X','p1Y'],'Four receipts required')
    tasks=[(raw_commit,x,binding,receipts,source) for x in data['raw_paths']]
    paths,checks=parallel_paths(tasks)
    M.same([p['index'] for p in paths],[0,1],'Incomplete m9 paths')
    M.same(paths[0]['result']['scalar'],paths[1]['result']['scalar'],'Mixed scalar quantities')
    req,packet,dictionary=B1.canonical(2);config=req['configuration'];B=I.providers(str(M.PIN_ROOT)).B
    metrics={a:{k:float(sum(Fraction(p['result']['prices'][a][k]) for p in paths)/2)
                  for k in paths[0]['result']['prices'][a]} for a in M.AXES.values()}
    proof_path=M.BASE+'/run01/IMPLEMENTATION_EVIDENCE.json'
    prices,sides,proofs=P.price_records([r for r in DC.pauli_requests(M.PIN_ROOT,req) if r['m']==9],config,
        trial_prices=metrics,scalar=paths[0]['result']['scalar'],native_source=dict(commit=M.PIN,path=P.MODULE,driver_source=source),
        cohort=M.COHORT,price_source=source,evidence_commit=None,evidence_path=proof_path,measurement=B.p.measurement)
    for p in prices:p['n_paths']=2;p['diagnostic'].update(pilot_role='M9_LONG_NORMAL',law_expectation=False)
    for name,value in (('PRICES.json',prices),('SIDECARS_UNBOUND.json',sides),('IMPLEMENTATION_EVIDENCE.json',proofs)):
        M.write(M.ROOT/M.BASE/'run01'/name,value)
    M.write(M.ROOT/M.BASE/'run01/RAW_VALIDATION.json',dict(status='FOUR_RECEIPTS_CHECKED',raw_source=raw_commit,
        paths=2,receipts=4,predictions=data['predictions'],gate_pass=True,model_qualified=False,workers=checks,
        wall_seconds=time.monotonic()-started,publication_source=dict(commit=commit,path=here),sampling=0,compiles=0,fit_calls=0))
    print('Four saved receipts checked in parallel; commit publication files before original --admit')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publish',required=True)
    publish(parser.parse_args().publish)
