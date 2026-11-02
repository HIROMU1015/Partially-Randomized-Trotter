#!/usr/bin/env python3
"""One user-authorized fixed-law H4 batch; no automatic retries or circuits."""
import argparse
import csv
from datetime import datetime
from fractions import Fraction as F
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import run_h4_p3_stage2 as previous

ROOT=Path(__file__).resolve().parents[1]
BASE='c52bbf2a76708079075f01f57af76676a6d5e7ff'
CONTRACT='docs/research/h4_p3_finite_bit_law_design_20261102.md'
INPUTS='docs/research/h4_p3_finite_bit_inputs_20261102.json'
OUT=ROOT/'artifacts/df_pauli_matched_partition_validation/2026-11-02/finite_bit/run01'
OWN=('scripts/run_h4_p3_finite_bit.py','src/trotterlib/matched_partition_finite_bit.py',
     'tests/test_matched_partition_finite_bit.py','docs/research/h4_p3_finite_bit_execution_20261102.md',INPUTS,
     'src/trotterlib/matched_partition_stage2.py','src/trotterlib/matched_partition_stage1.py',
     'src/trotterlib/matched_partition_stage4.py','src/trotterlib/h4_ball_evidence.py',
     'src/trotterlib/h4_enclosure_prototype.py','src/trotterlib/h4_conditional_cng.py',
     'src/trotterlib/df_hamiltonian.py','src/trotterlib/h2_non_df_pilot.py','src/trotterlib/pauli_signal_adapter.py',
     'scripts/run_h4_p3_stage2.py','scripts/run_h4_p3_stage1.py')


def modules():
    previous.modules()
    return importlib.import_module('stage1.matched_partition_finite_bit')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def gitbytes(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)


def write(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def fixed_inputs(commit):
    raw=gitbytes(commit,INPUTS)
    if (ROOT/INPUTS).read_bytes()!=raw:raise ValueError('FIXED_INPUT_DEFINITION')
    table=json.loads(raw);saved={};bindings={}
    for row in table['files']:
        data=gitbytes(row['ref'],row['path'])
        if sha(data)!=row['sha256'] or (ROOT/row['path']).read_bytes()!=data:
            raise ValueError('FIXED_INPUT_BYTES:'+row['path'])
        saved[row['key']]=data
        bindings[row['path']]=dict(ref=row['ref'],sha256=row['sha256'])
    return saved,bindings


def validate(saved,kernel):
    import numpy as np
    values={key:json.loads(raw) for key,raw in saved.items() if key.endswith('.json')}
    s2=values['stage2/source_and_inputs.json'];s3=values['run04/source_and_inputs.json']
    s4=values['stage4/source_and_inputs.json']
    c2,c3=s2['condition'],s3['conditions']
    for key in ('molecule','basis','charge','multiplicity','system_qubits','T_exact','q','r','K','arms'):
        if c2[key]!=c3[key]:raise ValueError('SCIENTIFIC_CONDITION:'+key)
    if (c2['system_qubits']!=8 or c2['dimension']!=256 or c2['distance_Angstrom']!=1.
        or c3['partition']!='A' or [c2[k] for k in ('q','r','K')]!=[8,4,4]
        or c2['T_exact']!=str(F(.8)) or 'rank10' not in c2['DF']):raise ValueError('FIXED_H4_TUPLE')
    if (s3['pilot_result_commit']!='90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5'
        or s3['native_SOURCE']!='9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c'):
        raise ValueError('SAVED_NATIVE_LINEAGE')
    for key in ('source_and_inputs.json','law_bindings.json','native_bases.json','preflight.json','STOP.json'):
        path='artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run02/'+key
        if sha(saved['run02/'+key])!=s3['saved_native_files_sha256'][path]:raise ValueError('SAVED_NATIVE_BYTES:'+key)
    for path,row in s2['inputs']['files'].items():
        if s3['input_files'].get(path)!=row:raise ValueError('INPUT_LINEAGE:'+path)
    for key,status in [('stage2/STOP.json','STOP_STAGE2_UNRESOLVED_COMPONENTS'),
        ('run04/STOP.json','STOP_STAGE3_PRICE_CONFIRMATION_COMPLETE_NO_AUTOMATIC_FOLLOWUP'),
        ('stage4/STOP.json','STOP_STAGE4_CONDITIONAL_ACCOUNTING_COMPLETE_NO_AUTOMATIC_FOLLOWUP')]:
        if values[key]['status']!=status:raise ValueError('HISTORICAL_STOP:'+key)
    def statuses(obj):
        if isinstance(obj,dict):
            if 'status' in obj:yield obj['status']
            else:
                for value in obj.values():yield from statuses(value)
    checks=list(statuses(values['run02/preflight.json']))
    if len(checks)!=81 or set(checks)!={'PASS'}:raise ValueError('SAVED_NATIVE_81_CHECKS')
    descriptors=values['stage2/dictionaries.json'];s2result=values['stage2/results.json'];laws=values['run02/law_bindings.json']
    for arm,count in [('P',184),('DF',360)]:
        if descriptors[arm]['component_count']!=count or len(descriptors[arm]['components'])!=count:
            raise ValueError('DICTIONARY_COUNT:'+arm)
        if laws[arm]['saved_Stage2_normalization']!=s2result['normalization'][arm]:raise ValueError('SAVED_NORMALIZATION')
    if len(descriptors['DF']['basis'])!=10:raise ValueError('SAVED_DF_BASES')
    state=previous.previous.complex_array(values['state.json']['state'])
    nu=F(s2['inputs']['raw_norm_exact'])
    from stage1.pauli_signal_adapter import array_hash
    if (array_hash(state)!=s2['inputs']['state_hash'] or kernel.K.exact_norm(state)!=nu
        or values['state.json']['state_hash']!=s2['inputs']['state_hash']):raise ValueError('RAW_STATE_NU')
    with np.load(io.BytesIO(saved['stage2/common_operators.npz']),allow_pickle=False) as archive:
        t=archive['T'].copy()
    if t.shape!=(256,256) or not np.isfinite(t).all():raise ValueError('SAVED_T')
    stage4=values['stage4/results.json']
    if len(stage4['rows'])!=10 or s4['scientific_lineage']['raw_norm_exact']!=str(nu):raise ValueError('STAGE4_LINEAGE')
    return t,descriptors,laws,s2result,stage4,state,nu,dict(conditions=c2,state_hash=s2['inputs']['state_hash'],
        raw_norm_exact=str(nu),Stage2_SOURCE=s2['commit'],Stage3_SOURCE=s3['commit'],Stage4_SOURCE=s4['commit'])


def csv_outputs(result):
    with (OUT/'signals.csv').open('w',newline='') as stream:
        columns=['arm','variant','real','imag','u_num','u_bind']
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for arm,data in result['arms'].items():
            for variant,row in data['signals'].items():
                writer.writerow(dict(arm=arm,variant=variant,**{key:row[key] for key in columns[2:]}))
    with (OUT/'headroom.csv').open('w',newline='') as stream:
        rows=result['headroom_preview'];writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def execute():
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();source={}
    counts=dict(scientific_batches=0,dictionaries=0,law_tables=0,event_angles=0,complex_signals=0,
        headroom_axes=0,new_sampling=0,RNG_replay=0,native_circuit_generation=0,new_compile=0,
        Gaussian_synthesis=0,DF_eigendecompositions=0,physical_reference_generation=0,
        finite_Taylor_bias_refinement=0,new_N_G=0,automatic_retries=0)
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        code={p:sha((ROOT/p).read_bytes()) for p in OWN}
        for path,digest in code.items():
            if sha(gitbytes(commit,path))!=digest:raise ValueError('UNFIXED_SOURCE:'+path)
        contract=gitbytes(BASE,CONTRACT)
        if (ROOT/CONTRACT).read_bytes()!=contract:raise ValueError('CHANGED_DESIGN_CONTRACT')
        saved,bindings=fixed_inputs(commit)
        runtime={name:importlib.metadata.version(name) for name in ('numpy','scipy','qiskit','openfermion')}
        if runtime['numpy']!='1.26.4':raise ValueError('NUMPY_CDF_SOURCE_VERSION')
        kernel=modules()
        t,descriptors,laws,s2,s4,phi,nu,lineage=validate(saved,kernel)
        source=dict(commit=commit,design_commit=BASE,design_path=CONTRACT,design_sha256=sha(contract),
            code_sha256=code,input_files=bindings,scientific_lineage=lineage,runtime=runtime,
            python=sys.version,executable=sys.executable,execution_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            workers=2,CPU_affinity=len(os.sched_getaffinity(0)),load=os.getloadavg(),
            OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
            assumptions=dict(IID_uniform_53bit_words=True,native_bridge_zero_for_headroom_preview=True,
                compiled_rho_zero_for_headroom_preview=True),
            precision='complex128 matrices/vectors; exact rational CDF/scalar rotation intervals; engineering norm/propagation bounds')
        write('source_and_inputs.json',source)
        kernel.K.OPERATIONS.clear();counts['scientific_batches']=1
        def checkpoint(arms,common):
            counts.update(dictionaries=len(arms),law_tables=2*len(arms),event_angles=3*len(arms),complex_signals=3*len(arms))
            write('partial_results.json',dict(arms=arms,common_T_half=common))
            write('execution_status.json',dict(state='RUNNING',counts=counts))
        result=kernel.evaluate(t,descriptors,laws,s2,s4,phi,nu,workers=2,checkpoint=checkpoint)
        counts['headroom_axes']=len(result['headroom_preview'])
        operations=dict(kernel.K.OPERATIONS)
        if operations.get('physical_complex_signals')!=6 or counts['complex_signals']!=6 or counts['headroom_axes']!=20:
            raise ValueError('SIX_SIGNALS_TWENTY_PREVIEWS')
        if operations.get('common_expm')!=1 or operations.get('common_spectral_eigendecompositions')!=1:
            raise ValueError('ONE_COMMON_T_HALF')
        write('results.json',result);csv_outputs(result)
        for path,digest in code.items():
            if sha((ROOT/path).read_bytes())!=digest:raise ValueError('SOURCE_CHANGED:'+path)
        for path,row in bindings.items():
            if sha((ROOT/path).read_bytes())!=row['sha256']:raise ValueError('INPUT_CHANGED:'+path)
        write('verification.json',dict(status='TECHNICAL_CHECKS_PASS',inputs_unchanged=len(bindings),
            source_files_unchanged=len(code),operations=operations,common_T_half_only=True,
            engineering_not_certified=True,native_full_mean_not_evaluated=True,
            old_N_G_unchanged=True,old_finite_bias_unresolved_preserved=True))
        finish=dict(state='COMPLETED',counts=counts,elapsed_seconds=time.monotonic()-start,
            result_status=result['status'],positive_conditional_headroom_axes=sum(row['status']=='POSITIVE_CONDITIONAL' for row in result['headroom_preview']))
        write('execution_status.json',finish)
        status=('STOP_FINITE_BIT_MODEL_UNRESOLVED_NO_AUTOMATIC_FOLLOWUP' if result['status']=='MODEL_UNRESOLVED'
                else 'STOP_FINITE_BIT_ENGINEERING_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP')
        write('STOP.json',dict(status=status,**finish,automatic_followup=False,implemented_accuracy_guaranteed=False))
        print(json.dumps(dict(source=commit,**finish)))
    except Exception:
        failure=dict(state='FAILED_NO_RETRY',counts=counts,source_commit=source.get('commit'),
            error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        write('execution_status.json',failure)
        write('STOP.json',dict(status='STOP_FINITE_BIT_EXECUTION_FAILURE_NO_RETRY',**failure,automatic_followup=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',required=True,action='store_true')
    parser.parse_args();execute()
