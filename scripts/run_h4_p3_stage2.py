#!/usr/bin/env python3
"""One fixed H4 Stage 2 batch; fresh output, no scientific retry."""
import argparse
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import run_h4_p3_stage1 as previous

ROOT=Path(__file__).resolve().parents[1]
BASE='495ed110ad8379663145791b788a63b64f05eedd'
S1='5708e37080738c4ebf74590d799a3741246483b2'
OUT=ROOT/'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01'
CONTRACT='docs/research/h4_p3_stage2_nominal_signal_contract_20261102.md'
READY='docs/research/h4_p3_stage2_execution_readiness_20261102.md'
C=previous.CBASE
N='artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/'
S='artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/'
FILES=('scripts/run_h4_p3_stage2.py','src/trotterlib/matched_partition_stage2.py',
    'tests/test_matched_partition_stage2.py','docs/research/h4_p3_stage2_nominal_validation_20261102.md',
    *previous.FILES,'src/trotterlib/df_rte_tail.py','src/trotterlib/rte.py',
    'src/trotterlib/pauli_mechanism_bias.py','src/trotterlib/df_trotter/decompose.py')


def modules():
    previous.modules()
    return importlib.import_module('stage1.matched_partition_stage2')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def write(name,data):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def fixed_inputs(kernel):
    # Reuse the corrected Stage 1 input validator, never its scientific runner.
    data,provenance=previous.inputs()
    prefixes={'I':previous.IBASE,'C':C,'N':N,'S':S}
    ready=previous.gitbytes(BASE,READY).decode()
    rows=re.findall(r'\| `([ICNS])/([^`]+)` \| (H4 anchor|Stage 1結果) \| `([0-9a-f]{64})`',ready)
    if len(rows)!=15:raise ValueError('CONTRACT_INPUT_TABLE')
    bound={};values={}
    for key,suffix,kind,digest in rows:
        ref=previous.INPUT_REF if kind=='H4 anchor' else S1
        path=prefixes[key]+suffix;raw=previous.gitbytes(ref,path)
        if sha(raw)!=digest or (ROOT/path).read_bytes()!=raw:raise ValueError('FIXED_INPUT_BYTES:'+path)
        bound[path]=dict(ref=ref,sha256=digest)
        if path.endswith('.json'):values[path]=json.loads(raw)
    model=values[C+'pauli_jw_model.json'];state=previous.complex_array(model['state'])
    nu=Fraction(values[N+'run01/signal_64.json']['raw_norm_exact'])
    # An input-integrity comparison, with no state rescaling.
    if kernel.exact_norm(state)!=nu:raise ValueError('EXACT_RAW_NU_BINDING')
    if values[N+'input_binding.json']['state_hash']!=model['state_hash']:raise ValueError('NORM_STATE_BINDING')
    expected=values[S+'run01/operator_comparisons.json']['scalar_identity']
    if data['constant']!=2.29310124732 or expected['DF_square_total']!=kernel.VBAR:
        raise ValueError('COMMON_SCALAR_BINDING')
    ref=values[C+'df_native_result.json']
    provenance.update(files=bound,state_used_for='six physical signals, exact saved nu once; no renormalization',
        raw_norm_exact=str(nu),reference='saved DF C RAW interval divided by exact nu once',
        contracts=dict(commit=BASE,files={p:sha(previous.gitbytes(BASE,p)) for p in (CONTRACT,READY)}))
    return data,state,nu,ref,provenance


def execute():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();counts=dict(scientific_batches_started=0,dictionaries=0,
        common_operator_bundles=0,signals=0,normalization_bundles=0,automatic_retries=0,
        sampling=0,RNG_replay=0,finite_bit_law=0,circuit_build_compile=0,
        SCF_FCI_DF_factorization=0,new_state_reference=0,M_A=0,N_C_G=0)
    kernel=None;source={}
    def progress(result):
        counts['signals']=len(result.get('signals',{}))
        counts['normalization_bundles']=len(result.get('normalization',{}))
        write('results.json',result)
        write('execution_status.json',dict(state='RUNNING',counts=counts))
    try:
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        code={p:sha((ROOT/p).read_bytes()) for p in FILES}
        if any(sha(previous.gitbytes(source_sha,p))!=v for p,v in code.items()):raise ValueError('UNFIXED_SOURCE')
        for p in (CONTRACT,READY):
            if (ROOT/p).read_bytes()!=previous.gitbytes(BASE,p):raise ValueError('CONTRACT_CHANGED')
        versions={x:importlib.metadata.version(x) for x in ('numpy','scipy','qiskit','openfermion')}
        kernel=modules();kernel.OPERATIONS.clear()
        from openfermion.circuits.low_rank import get_chemist_two_body_coefficients
        correction_file=Path(inspect.getsourcefile(get_chemist_two_body_coefficients))
        if sha(correction_file.read_bytes())!=previous.LOW_RANK_SHA:raise ValueError('CORRECTION_SOURCE')
        available=max(1,len(os.sched_getaffinity(0))-math.ceil(os.getloadavg()[0]))
        workers=min(10,available) # Ten independent saved factors; workload, not a resource cap.
        source=dict(commit=source_sha,base=BASE,code_sha256=code,runtime=versions,python=sys.version,
            correction_SOURCE=dict(path=str(correction_file),sha256=previous.LOW_RANK_SHA),
            executable=sys.executable,execution_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            condition=dict(molecule='linear H4',distance_Angstrom=1.,basis='STO-3G',charge=0,
                multiplicity=1,system_qubits=8,dimension=256,DF='tol-only 1e-8, saved rank10',
                split='A: T_eff / V_square, residual identity extracted',T_exact=str(kernel.T),
                delta_exact=str(kernel.DELTA),h_exact=str(kernel.H),q=8,r=4,K=4,degree=5,arms=['M_B','M_C']),
            resources=dict(affinity_CPUs=len(os.sched_getaffinity(0)),load=os.getloadavg(),
                meminfo={s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/meminfo').read_text().splitlines()
                    if s.startswith(('MemTotal:','MemAvailable:'))},DF_factor_workers=workers,
                OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
                reason='small sparse/dense matrices, independent fragment workers; no shared settings changed'),
            uncertainty='contract section 7 engineering estimates; not rigorous certification')
        write('source_and_inputs.json',source)
        data,phi,nu,reference,inputs=fixed_inputs(kernel)
        source['inputs']=inputs;write('source_and_inputs.json',source)
        counts['scientific_batches_started']=1
        t,v0,arms,checks=kernel.build(data,workers=workers)
        counts.update(dictionaries=2,common_operator_bundles=1)
        write('operator_preflight.json',dict(checks=checks,full_space_256=True,
            state_projected=False,common_scalar=dict(nuclear=data['constant'],residual_identity=kernel.VBAR,
                sum_exact=str(Fraction(data['constant'])+Fraction(kernel.VBAR))),
            identity_defects={name:dict(extracted=a.identity,common=kernel.VBAR,
                defect=a.identity-kernel.VBAR) for name,a in arms.items()}))
        descriptors={name:dict(component_count=len(a.coefficients),components=a.descriptors,
            identity=a.identity,identity_policy='all residual I outside, T identity inside',
            basis=[dict(fragment=b['fragment'],lambda_factor=b['lambda_factor'],
                eigenvalues=b['eta'].tolist(),U_real=b['U'].real.tolist(),U_imag=b['U'].imag.tolist())
                for b in a.basis]) for name,a in arms.items()}
        write('dictionaries.json',descriptors)
        import numpy as np
        np.savez_compressed(OUT/'common_operators.npz',T=t,V0=v0)
        if any(x['status']!='PASS' for x in checks.values()):raise ValueError('OPERATOR_PREFLIGHT_NONPASS')
        results=kernel.evaluate(t,v0,arms,phi,nu,data['constant'],data['snapshot_H_DF'],
            reference['reference'],Fraction(reference['operator_snapshot_residual_upper']['upper']),checkpoint=progress)
        progress(results)
        if counts['signals']!=6 or counts['normalization_bundles']!=2:raise ValueError('SCIENTIFIC_SCOPE_COUNT')
        if any(sha((ROOT/p).read_bytes())!=v for p,v in code.items()):raise ValueError('SOURCE_CHANGED')
        finish=dict(state='COMPLETED',scientific_status=results['numerical_status'],counts=counts,
            operations=dict(kernel.OPERATIONS),workers_joined=True,elapsed_seconds=time.monotonic()-started)
        write('execution_status.json',finish)
        write('STOP.json',dict(status=results['stop'],**finish,automatic_followup=False))
        print(json.dumps(dict(status=results['stop'],counts=counts,elapsed_seconds=finish['elapsed_seconds'])))
    except Exception:
        error=traceback.format_exc()
        finish=dict(state='FAILED_NO_RETRY',counts=counts,error=error,source_commit=source.get('commit'),
            operations=dict(kernel.OPERATIONS) if kernel else {},workers_joined=True,
            elapsed_seconds=time.monotonic()-started)
        write('execution_status.json',finish)
        write('STOP.json',dict(status='STOP_STAGE2_FAILURE_NO_RETRY',**finish,automatic_followup=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',required=True,action='store_true')
    parser.parse_args();execute()
