#!/usr/bin/env python3
"""Fixed saved H6 C_i/native/compiled diagnostic, at most fifteen state actions."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from datetime import datetime
from fractions import Fraction as F
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import run_h6_pauli_p2_cost as P2

ROOT=Path(__file__).resolve().parents[1]
BASE='b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1'
DESIGN='docs/research/h6_pure_rte_hybrid_implementation_evidence_inventory_20261102.md'
INVENTORY='artifacts/h6_pure_rte_hybrid_evidence_audit/2026-11-02/INVENTORY.json'
OUT=ROOT/'artifacts/h6_selected_implementation/2026-11-02/run01'
OWN=('scripts/run_h6_selected_implementation.py','src/trotterlib/h6_selected_implementation.py',
    'tests/test_h6_selected_implementation.py','docs/research/h6_selected_implementation_execution_20261102.md',
    'scripts/run_h6_pauli_p2_cost.py','scripts/restore_pauli_h2_h6_evidence.py',
    'src/trotterlib/pauli_mechanism_cost.py','src/trotterlib/h4_d1_adapter.py',
    'src/trotterlib/h4_d2_pauli_control.py','src/trotterlib/h4_r0_empirical.py',
    'src/trotterlib/pauli_vector_enclosure.py','src/trotterlib/h4_conditional_cng.py',
    'src/trotterlib/pauli_signal_adapter.py','src/trotterlib/rte.py',
    'src/trotterlib/fixed_signal_evaluation.py','src/trotterlib/rte_compiled_cost.py')
_CTX=None


def modules():
    P2.modules()
    return importlib.import_module('p2_study.h6_selected_implementation')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def write(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def raw_state(model):
    import numpy as np
    pairs=np.asarray(model['state'],dtype=float);raw=np.empty(len(pairs),dtype=np.complex128)
    raw.real=pairs[:,0];raw.imag=pairs[:,1]
    return raw


def initialize(model,nu):
    global _CTX
    _CTX=dict(kernel=modules(),raw=raw_state(model),nu=F(nu))


def action(task):
    start=time.monotonic();k=_CTX['kernel'];raw=_CTX['raw'];nu=_CTX['nu']
    if task['role']=='selected':
        spec=P2.modules().recipe(task['law'],task['result'])
        record=k.selected_action(spec,raw,nu)
    else:
        circuit=P2.decode(task['QPY'])
        record=k.saved_action(circuit,raw,nu,compiled=task['role']=='compiled')
    return task['key'],dict(record,pid=os.getpid(),elapsed_seconds=time.monotonic()-start)


def fixed_inputs():
    bindings={}
    for path in (DESIGN,INVENTORY):
        raw=P2.gitbytes(BASE,path)
        if (ROOT/path).read_bytes()!=raw:raise ValueError('FIXED_AUDIT_CHANGED:'+path)
        bindings[path]=dict(ref=BASE,sha256=sha(raw))
    inventory=json.loads(P2.gitbytes(BASE,INVENTORY))
    laws,results,members,prefixes,provenance=P2.inputs()
    modelraw=P2.gitbytes(P2.RTE,P2.MODEL);model=json.loads(modelraw)
    if sha(modelraw)!=P2.MODEL_HASH:raise ValueError('RAW_MODEL_BINDING')
    for tag,(cid,ld,r) in P2.CANDIDATES.items():
        old=inventory['candidates'][cid]
        if (laws[tag]['config']!=old['law']['config'] or
            laws[tag]['B_saved_binary64']!=old['law']['B_saved_binary64']):raise ValueError('FROZEN_LAW:'+tag)
        if results[tag]['orders']!=old['fixed_trajectory']['orders']:raise ValueError('FROZEN_EVENT_ORDERS')
    nu=inventory['candidates'][P2.CANDIDATES['R64'][0]]['saved_target']['raw_norm_exact']
    return model,F(nu),laws,results,members,prefixes,inventory,dict(local=bindings,original=provenance)


def jobs(laws,results,members,prefixes):
    tasks=[]
    for tag in P2.CANDIDATES:
        tasks.append(dict(key=tag+'/selected',tag=tag,role='selected',law=laws[tag],result=results[tag]))
        for axis in ('cosine','sine'):
            for role,kind in [('native','input'),('compiled','compiled')]:
                key=prefixes[tag]+'prices/confirmation/0000/'+axis+'_'+kind+'.qpy.gz'
                tasks.append(dict(key=tag+'/'+axis+'/'+role,tag=tag,axis=axis,role=role,QPY=members[key]))
    if len(tasks)!=15 or len({x['key'] for x in tasks})!=15:raise ValueError('FIFTEEN_FIXED_ACTIONS')
    return tasks


def reduce_results(kernel,records,inventory):
    rows=[]
    for tag,(cid,_,_) in P2.CANDIDATES.items():
        old=inventory['candidates'][cid]
        account=next(x for x in old['confirmation_rows'] if x['epsilon_sig']=='0.0005')
        for axis in ('cosine','sine'):
            comparison=kernel.reduce_axis(records[tag+'/selected'],records[tag+'/'+axis+'/native'],
                records[tag+'/'+axis+'/compiled'],axis,account['B_exact'],account['axes'][axis]['rho_guard_exact'])
            rows.append(dict(tag=tag,candidate_id=cid,axis=axis,cohort='confirmation',index=0,
                epsilon_sig_budget='0.0005',**comparison))
    return rows


def csv_output(rows):
    fields=['tag','candidate_id','axis','epsilon_sig_budget','eta','delta','d','rho_budget','engineering_d_upper_candidate','engineering_comparison']
    with (OUT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            value={k:row[k] for k in fields if k!='engineering_d_upper_candidate'}
            value['engineering_d_upper_candidate']=row['engineering_intervals']['d']['upper_candidate'] if row['engineering_intervals'] else None
            writer.writerow(value)


def execute():
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();source={}
    counts=dict(scientific_batches=0,state_actions_submitted=0,state_actions_completed=0,
        selected_actions=0,native_input_actions=0,compiled_actions=0,comparisons=0,
        new_sampling=0,RNG_replay=0,Full_regenerated=0,new_compile=0,new_law_mean=0,
        new_full_H_reference=0,new_N_G=0,new_price_tail=0,automatic_retries=0)
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        code={path:sha((ROOT/path).read_bytes()) for path in OWN}
        for path,digest in code.items():
            if sha(P2.gitbytes(commit,path))!=digest:raise ValueError('UNFIXED_SOURCE:'+path)
        model,nu,laws,results,members,prefixes,inventory,bindings=fixed_inputs()
        runtime={name:importlib.metadata.version(name) for name in ('numpy','scipy','qiskit')}
        if runtime!=dict(numpy='1.26.4',scipy='1.14.1',qiskit='1.3.0'):raise ValueError('FIXED_RUNTIME')
        kernel=modules();raw=raw_state(model)
        from p2_study.pauli_signal_adapter import array_hash
        nu_actual=sum((F(x)**2+F(y)**2 for x,y in model['state']),F(0))
        if (nu_actual!=nu or array_hash(raw)!=model['state_hash'] or model['n_qubits']!=12
            or model['Hamiltonian_hash']!=P2.HAM_HASH):raise ValueError('RAW_STATE_HAMILTONIAN_NORM')
        specs,full,checks=P2.verify_full(laws,results,members,prefixes)
        write('preflight.json',dict(status='SAVED_FULL_SEMANTIC_AND_METRIC_BINDING_PASS',checks=checks,
            QPY_structural_decodes=12,state_actions=0,saved_full=full))
        tasks=jobs(laws,results,members,prefixes);workers=min(len(tasks),len(os.sched_getaffinity(0)))
        source=dict(commit=commit,design_commit=BASE,code_sha256=code,input_bindings=bindings,
            runtime=runtime,python=sys.version,executable=sys.executable,
            execution_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            conditions=dict(molecule='linear H6',distance_Angstrom=1.,basis='STO-3G',system_qubits=12,
                ancilla=12,T=.8,q=8,K=2,product_formula='S2',state_preparation_cost='P=0',candidates=P2.CANDIDATES),
            state_hash=model['state_hash'],raw_norm_exact=str(nu),workers=workers,
            cpu_affinity=len(os.sched_getaffinity(0)),load=os.getloadavg(),
            OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
            process_policy='spawn; one state action per fresh worker; all joined',
            state_action_keys=[t['key'] for t in tasks],precision='complex128; engineering resolution; no certified numerical error upper',
            QPY_policy='12 preflight decodes plus12 action-worker restores; no regenerated program/compile')
        write('source_and_inputs.json',source)
        records={};failures={};counts['scientific_batches']=1
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                max_tasks_per_child=1,initializer=initialize,initargs=(model,str(nu))) as pool:
            futures={pool.submit(action,t):t for t in tasks};counts['state_actions_submitted']=len(futures)
            for future in as_completed(futures):
                task=futures[future]
                try:
                    key,record=future.result();records[key]=record;counts['state_actions_completed']+=1
                    counts[dict(selected='selected_actions',native='native_input_actions',compiled='compiled_actions')[task['role']]]+=1
                except Exception:failures[task['key']]=traceback.format_exc()
                write('actions.json',records)
                write('execution_status.json',dict(state='RUNNING',counts=counts,completed=list(records),failures=failures))
        if failures:
            write('failures.json',failures);raise ValueError('ACTION_FAILURE_NO_RETRY')
        if counts['state_actions_completed']!=15:raise ValueError('FIFTEEN_ACTIONS_NOT_COMPLETE')
        rows=reduce_results(kernel,records,inventory);counts['comparisons']=len(rows)
        result=dict(rows=rows,scope='fixed three trajectories, two axes; no whole-law or population transfer',
            actual_compiled_rho=None,whole_law_rho=None,certified_upper=None,
            price_tail_bound=None,old_N_G_changed=False,all_workers_joined=True,
            global_phase_policy='full-wrapper observable; no posterior phase alignment')
        write('results.json',result);csv_output(rows)
        for path,digest in code.items():
            if sha((ROOT/path).read_bytes())!=digest:raise ValueError('SOURCE_CHANGED:'+path)
        for path,row in bindings['local'].items():
            if sha((ROOT/path).read_bytes())!=row['sha256']:raise ValueError('FIXED_INPUT_CHANGED:'+path)
        write('verification.json',dict(status='FIXED_SCOPE_TECHNICAL_CHECKS_PASS',source_files_unchanged=len(code),
            local_input_bindings_unchanged=len(bindings['local']),state_actions_completed=15,
            selected_actions=3,native_actions=6,compiled_actions=6,comparison_rows=6,
            unique_worker_pids=len({x['pid'] for x in records.values()}),all_workers_joined=True,
            engineering_only=True,whole_law_rho_measured=False,new_N_G=False))
        good=all(row['engineering_comparison']=='ENGINEERING_UPPER_WITHIN_SAVED_LOCAL_BUDGET' for row in rows)
        finish=dict(state='COMPLETED',counts=counts,elapsed_seconds=time.monotonic()-start,
            local_engineering_budget_comparisons='ALL_WITHIN' if good else 'NOT_ALL_RESOLVED_WITHIN',
            all_workers_joined=True)
        write('execution_status.json',finish)
        stop=('STOP_H6_SELECTED_IMPLEMENTATION_LOCAL_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP' if good else
              'STOP_H6_SELECTED_IMPLEMENTATION_UNRESOLVED_OR_EXCEEDS_BUDGET_NO_AUTOMATIC_FOLLOWUP')
        write('STOP.json',dict(status=stop,**finish,actual_compiled_rho=None,automatic_followup=False))
        print(json.dumps(dict(source=commit,**finish)))
    except Exception:
        failure=dict(state='FAILED_NO_RETRY',counts=counts,source_commit=source.get('commit'),
            error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        write('execution_status.json',failure)
        write('STOP.json',dict(status='STOP_H6_SELECTED_IMPLEMENTATION_FAILURE_NO_RETRY',**failure,automatic_followup=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',required=True,action='store_true')
    parser.parse_args();execute()
