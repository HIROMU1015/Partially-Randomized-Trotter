"""Publish saved D7 holdout receipts and four exact comparisons. No sampling/fit."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import shutil
import subprocess
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_full_h_d7_holdout as H
P=H.P
spec=importlib.util.spec_from_file_location('_d7_receipt_reader',
    ROOT/'docs/research/d_full_h_p2_calibration_resume_20261115/READ_RECEIPT.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def write(path,value):P.P1.write(Path(path),value)


def evidence(result,axis):
    path=result['child']['paths'][axis['index']]
    return dict(price=axis,checks=path['checks'],xy_phase_gate=path['xy_phase_gate'],
        trial_checks=path['trial']['checks'],scalar=path['trial']['scalar'],
        preparation_hash=path['trial']['preparation_hash'],sampling=path['sampling'],
        resources={k:path['trial'][k] for k in ('wall_seconds','peak_RSS_bytes')},
        acceleration=path['acceleration'],
        measurement=dict(axis=axis['axis'],include_measurement=True,state_preparation_cost_included=False))


def prepare(cell_id,execution_commit):
    frozen=json.loads((ROOT/H.FROZEN).read_text());job=frozen['jobs'][cell_id]
    directory=ROOT/job['output'];cell=P.selected(job['plan'],'HOLDOUT',cell_id)
    result=json.loads((directory/'RESULT.json').read_text())
    if result['status']!='FINITE_CELL_COMPLETE':
        write(directory/'AXIS_STATUS.json',dict(status='ACQUISITION_INCOMPLETE_OR_INVALID',
            error=result.get('error'),child_status=result.get('child',{}).get('status') if result.get('child') else None,
            comparison_unavailable=True))
        return R.archive_result(directory,parallel_gzip=True)
    axes=H.axis_receipts(job['plan'],cell,result,execution_commit)
    proofs=[]
    for axis in axes:
        receipt=axis['compile_receipt'];raw=Path(receipt['locator']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=receipt['locator']['sha256'] or json.loads(raw)!=receipt['record']:
            raise ValueError('Original compile bytes differ')
        target=directory/'COMPILE_RECORDS'/('p'+str(axis['index'])+'_'+axis['axis']+'.json')
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:f.write(raw)
        proofs.append(evidence(result,axis))
    write(directory/'AXIS_PRICES.json',axes);write(directory/'AXIS_EVIDENCE.json',proofs)
    write(directory/'AXIS_STATUS.json',dict(status='ACQUISITION_AND_REQUIRED_GATES_PASS',path_axes=len(axes),
        new_compiles=sum(not a['compile_receipt']['reused'] for a in axes),model_verdict=H.VERDICT))
    del result
    return R.archive_result(directory,parallel_gzip=True)


def stage_when_finished(cell_id,execution_commit,checkout):
    """Overlap completed-cell publication work with the other ongoing cell.

    Wait for the existing timed parent to exit, then copy immutable outputs.
    This never launches, interrupts, resumes, samples, or compiles a path.
    """
    checkout=Path(checkout).resolve()
    if checkout==ROOT:raise ValueError('Separate fixed scientific checkout required')
    current=lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    if current()!=execution_commit:raise ValueError('Scientific checkout SOURCE differs')
    frozen=json.loads((ROOT/H.FROZEN).read_text());job=frozen['jobs'][cell_id]
    run=checkout/H.STUDY/'run01';timer=run/(cell_id+'.time.txt')
    if json.loads((run/('JOB_'+cell_id+'.json')).read_text())!=job:raise ValueError('Scientific job inputs differ')
    if (ROOT/job['output']).exists():raise FileExistsError('Publication output already exists')
    print('WAITING_EXISTING_JOB_ONLY '+cell_id,flush=True)
    while True:
        if timer.exists() and 'Exit status:' in timer.read_text():
            if not (checkout/job['output']/'RESULT.json').exists():raise ValueError('Parent exited without final receipt; preserve console/time and STOP')
            break
        time.sleep(10)
    if current()!=execution_commit:raise ValueError('Scientific SOURCE changed while waiting')
    destination=ROOT/job['output'];destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(checkout/job['output'],destination)
    for name in (cell_id+'.time.txt',cell_id+'.console.txt','JOB_'+cell_id+'.json'):
        shutil.copy2(run/name,destination.parent/name)
    monitoring=destination/'MONITORING';monitoring.mkdir()
    private=Path(job['private'])
    for source in [*private.glob('*PROGRESS.jsonl'),private/'CHILD_EXIT.json',private/'OBSERVATIONS.jsonl']:
        if source.exists():shutil.copy2(source,monitoring/source.name)
    print('STAGED_FINISHED_CELL '+cell_id,flush=True)
    return prepare(cell_id,execution_commit)


def bind_cell(task):
    commit,execution_commit,cid,job=task;started=time.monotonic()
    directory=job['output'];cell=P.selected(job['plan'],'HOLDOUT',cid)
    result,manifest=R.read_receipt(commit,directory+'/RAW_RECEIPT.json')
    if result['status']!='FINITE_CELL_COMPLETE':
        return [],dict(cell_id=cid,status='ACQUISITION_OR_REQUIRED_GATE_INCOMPLETE'), 'ACQUISITION_OR_REQUIRED_GATE_INCOMPLETE'
    axes=H.axis_receipts(job['plan'],cell,result,execution_commit)
    read=lambda name:json.loads(P.P1.blob(commit,directory+'/'+name))
    if read('AXIS_PRICES.json')!=axes or read('AXIS_EVIDENCE.json')!=[evidence(result,a) for a in axes]:
        raise ValueError('Public per-axis evidence differs from raw path')
    produced=P.records(job['plan'],cell,result['child']['paths'],source=result['source'],
        evidence_path=directory+'/IMPLEMENTATION_EVIDENCE.json')
    for filename,key in (('C_INPUTS.json','C_inputs'),('PRICES.json','prices'),
        ('SIDECARS_UNBOUND.json','sidecars'),('IMPLEMENTATION_EVIDENCE.json','implementation_evidence')):
        if read(filename)!=produced[key]:raise ValueError('Original holdout caller publication differs: '+filename)
    for ordinal,axis in enumerate(axes):
        p=directory+'/COMPILE_RECORDS/p'+str(axis['index'])+'_'+axis['axis']+'.json'
        raw=P.P1.blob(commit,p)
        if hashlib.sha256(raw).hexdigest()!=axis['compile_receipt']['locator']['sha256'] or json.loads(raw)!=axis['compile_receipt']['record']:
            raise ValueError('Published individual compiler record differs')
        locator=lambda path,selector:dict(commit=commit,path=path,selector=selector)
        axis['locators']=dict(price=locator(directory+'/AXIS_PRICES.json',[ordinal]),
            implementation_evidence=locator(directory+'/AXIS_EVIDENCE.json',[ordinal]),
            compile_record=locator(p,[]),compile_record_sha256=axis['compile_receipt']['locator']['sha256'],
            raw_receipt=locator(directory+'/RAW_RECEIPT.json',[]),decoded_path_selector=axis['raw_path_selector'],
            decoder='docs/research/d_full_h_p2_calibration_resume_20261115/READ_RECEIPT.py:read_receipt',
            execution_source=axis['source'])
    checked=dict(cell_id=cid,status='PASS',path_axes=len(axes),raw_sha256=manifest['raw_sha256'],
        actual_path_and_compile_evidence_checked=True,worker_pid=os.getpid(),wall_seconds=time.monotonic()-started)
    del result
    return axes,checked,None


def bind(commit,execution_commit):
    frozen=json.loads(P.P1.blob(commit,H.FROZEN));bound=frozen['prediction_bindings']
    if bound!=H.prediction_bindings(frozen['jobs']['M5_ROUND']['plan']):raise ValueError('Fixed prediction binding differs')
    observations=[];invalid={};checks=[]
    tasks=[(commit,execution_commit,cid,job) for cid,job in frozen['jobs'].items()]
    with ProcessPoolExecutor(max_workers=len(tasks),mp_context=mp.get_context('fork')) as pool:
        for axes,checked,reason in pool.map(bind_cell,tasks):
            observations.extend(axes);checks.append(checked)
            if reason:invalid[checked['cell_id']]=reason
    comparison=H.compare(bound,observations,invalid)
    comparison.update(rule_source=H.RULE,frozen_prediction_source=H.MODEL,publication_commit=commit,
        individual_axis_receipts=observations,publication_checks=checks,
        counts=dict(planned_fresh_paths=3,planned_compile_axes=6,preexisting_reused_compile_axes=0,
            checked_compile_axes=len(observations),new_compiles=sum(not a['compile_receipt']['reused'] for a in observations),
            missing_or_invalid_axes=6-len(observations)))
    out=ROOT/H.STUDY
    write(out/'COMPARISON.json',comparison)
    with (out/'COMPARISON.csv').open('x',newline='') as f:
        fields=['cell_id','axis','status','P','O','signed_error','signed_relative_error','display_signed_percent','reason','relative_error_reason','model_verdict']
        csvwriter=csv.DictWriter(f,fieldnames=fields);csvwriter.writeheader()
        for row in comparison['rows']:
            def display(v):return str(v['numerator'])+'/'+str(v['denominator']) if isinstance(v,dict) and set(v)=={'numerator','denominator'} else v
            csvwriter.writerow({k:display(row[k]) for k in fields})
    write(out/'PUBLICATION_CHECK.json',dict(status='D1_SAVED_PUBLICATION_LINKS_CHECKED',commit=commit,cells=checks,
        checked_path_axes=len(observations),prediction_recomputations=0,fit_calls=0,
        C_scientific_admission='NOT_PERFORMED',model_verdict=H.VERDICT))
    return dict(status=comparison['status'],valid_comparisons=comparison['valid_comparisons'],path_axes=len(observations))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',choices=('M5_ROUND','M6_ROUND'));mode.add_argument('--bind')
    mode.add_argument('--stage-when-finished',choices=('M5_ROUND','M6_ROUND'))
    parser.add_argument('--execution-checkout')
    parser.add_argument('--execution-source',required=True)
    args=parser.parse_args()
    if len(args.execution_source)!=40:parser.error('Full fixed execution SOURCE required')
    if args.stage_when_finished:
        if not args.execution_checkout:parser.error('Execution checkout required for completed-cell staging')
        result=stage_when_finished(args.stage_when_finished,args.execution_source,args.execution_checkout)
    else:result=prepare(args.prepare,args.execution_source) if args.prepare else bind(args.bind,args.execution_source)
    print(json.dumps(result))


if __name__=='__main__':main()
