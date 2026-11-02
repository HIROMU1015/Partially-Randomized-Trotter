#!/usr/bin/env python3
"""Recover completed checkpoints after pretty-JSON size stop; no scientific calls."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];CAP=64*1024**2
TASKS=[('N1',x) for x in ('planted_local_gauge','dense_group_frame','separated_spectrum')]+[
    ('N2',x) for x in ('signed_overlap','perturbed_signed','uniform_hamming','dense_real_rank2','sparse_no_collective')]+[('N3','coefficient_feasibility')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,x):
    data=(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(data)>CAP:raise ValueError('64 MiB cap remains in force')
    p.write_bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();parent=args.run.resolve();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'artifacts/hamiltonian_algorithm_design'):
        raise SystemExit('new dedicated output required; no overwrite')
    subprocess.run(['git','diff','--quiet'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT,check=True)
    audit=load(parent/'run_audit.json');failure=load(parent/'failure.json')
    if audit['status']!='TECHNICAL_STOP' or '64 MiB per-file cap' not in failure['traceback']:
        raise ValueError('only completed-checkpoint JSON-size recovery is authorized')
    for name,meta in audit['outputs'].items():
        if sha(parent/name)!=meta['sha256']:raise ValueError('parent output changed '+name)
    result=load(parent/'result.json');contexts=[];ir=[]
    for track,kind in TASKS:
        c=load(parent/f'checkpoint_{track}_{kind}.json');ir.extend(c.pop('native_ir'));contexts.append(c)
    if contexts!=result['contexts'] or len(ir)!=result['compiled_circuits']:
        raise ValueError('completed checkpoint/result mismatch')
    if len({r['label'] for r in ir})!=len(ir) or len(ir)>768:
        raise ValueError('IR labels or compilation cap')
    out.mkdir(parents=True);resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    save(out/'result.json',result);save(out/'native_ir.json',ir)
    recovery={'status':'COMPLETED_CHECKPOINT_EXPORT_RECOVERED','parent_run':str(parent.relative_to(ROOT)),
              'parent_audit_sha256':sha(parent/'run_audit.json'),
              'parent_files':{p.name:sha(p) for p in parent.iterdir() if p.is_file()},
              'scientific_source_commit':audit['source_commit'],
              'export_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'export_source_sha256':sha(Path(__file__)),
              'command':[sys.executable,*sys.argv],'new_scientific_runs':0,
              'serialization':'identical JSON objects; whitespace removed; float precision retained',
              'next_stage_authorized':False,'central_hypothesis_adopted':None}
    save(out/'export_recovery_audit.json',recovery)
    audit['status']=result['status'];audit['export_recovery']='export_recovery_audit.json'
    audit['outputs']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}
    save(out/'run_audit.json',audit)
    print(result['status'],'native IR',len(ir),'new science runs 0')


if __name__=='__main__':main()
