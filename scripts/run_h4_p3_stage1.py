#!/usr/bin/env python3
"""Fixed H4, partition A, operator-only Stage 1. No scientific retry."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/run01'
INPUT_REF='4591b8d75a2a6a55c74a51225f0c61648023c8ee'
ORIGINAL_SOURCE='0c329b30a81accb35c212be8defa66fee0dd0f12'
BASE='eb20e81074aaae3cb365b348f6c3ab9d217d3697'
IBASE='artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/'
CBASE='artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/'
INPUTS={IBASE+'receipt.json':'6a2304db448845aef6f86da06a7bffd5d5c708273964e7e797c3d8ec71fcb6bf',
    IBASE+'snapshot.npz':'8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50',
    IBASE+'pauli.json':'49e2f0b86cba9933eee20a03a05934d950caafb5044401bca50d3aa79a518175',
    CBASE+'pauli_jw_model.json':'9ae8973b457cab26bb685a8fac9887dbc5e3be5db9059355822893d26cae2aa4',
    CBASE+'df_native_model.json':'cc60f6dc3de885974cfe74ef822e49380edb6b9ca44777bf12e9654256cd322a'}
FILES=('scripts/run_h4_p3_stage1.py','src/trotterlib/matched_partition_stage1.py',
    'tests/test_matched_partition_stage1.py','docs/research/h4_p3_stage1_operator_validation_20261102.md',
    'src/trotterlib/df_hamiltonian.py','src/trotterlib/pauli_signal_adapter.py',
    'src/trotterlib/h4_molecular_inputs.py','src/trotterlib/h2_non_df_pilot.py')
LOW_RANK_SHA='de74fb802550417788362845515ed9e56a180e4ff85ab21ab43ba8445156dbbe'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def gitbytes(ref,path):
    return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)


def modules():
    for name,folder in [('stage1','src/trotterlib'),('stage1.df_trotter','src/trotterlib/df_trotter')]:
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(ROOT/folder)];sys.modules[name]=module
    from stage1 import matched_partition_stage1
    return matched_partition_stage1


def write(name,value):
    path=OUT/name;temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def complex_array(raw):
    import numpy as np
    a=np.asarray(raw,dtype=float)
    # Arithmetic construction can change signed zero and falsify raw-byte hashes.
    result=np.empty(a.shape[:-1],dtype=np.complex128)
    result.real=a[...,0];result.imag=a[...,1]
    return result


def inputs():
    import numpy as np
    from stage1.h4_molecular_inputs import spin_lift_unthresholded
    from stage1.pauli_signal_adapter import array_hash
    raw={p:gitbytes(INPUT_REF,p) for p in INPUTS}
    if any(sha(raw[p])!=digest for p,digest in INPUTS.items()):raise ValueError('INPUT_SHA')
    receipt=json.loads(raw[IBASE+'receipt.json'])
    if (receipt['snapshot_sha256']!=INPUTS[IBASE+'snapshot.npz']
            or receipt['pauli_json_sha256']!=INPUTS[IBASE+'pauli.json']):raise ValueError('RECEIPT_BINDING')
    pm,dm=(json.loads(raw[CBASE+n+'_model.json']) for n in ('pauli_jw','df_native'))
    with np.load(io.BytesIO(raw[IBASE+'snapshot.npz']),allow_pickle=False) as saved:
        arrays={k:saved[k] for k in ('one_mo','eri_chemist','spin_one','spin_two','df_one_body',
            'df_lambdas','df_g_matrices','nuclear_repulsion','state',
            'non_DF_matrix','DF_matrix','physical_qiskit_indices')}
    state=complex_array(pm['state'])
    if (pm['state']!=dm['state'] or not np.array_equal(state,arrays['state'])
            or array_hash(state)!=receipt['state_hash'] or pm['state_hash']!=receipt['state_hash']
            or dm['state_hash']!=receipt['state_hash']):raise ValueError('COMMON_RAW_STATE')
    if (pm['old_scientific_SOURCE']!=ORIGINAL_SOURCE or dm['old_scientific_SOURCE']!=ORIGINAL_SOURCE
            or pm['n_qubits']!=8 or dm['n_qubits']!=8
            or len(pm['hamiltonian']['terms'])!=184):raise ValueError('MODEL_IDENTITY')
    md=receipt['DF_metadata']
    if (md['df_rank_requested'] is not None or md['df_rank_actual']!=10
            or md['df_tol_requested']!=1e-8 or md['df_truncation_value']!=0.
            or md['charge']!=0 or md['multiplicity']!=1 or md['basis']!='STO-3G'
            or receipt['geometry']!=[['H',[0.,0.,float(j)]] for j in range(4)]):
        raise ValueError('FIXED_H4_TOL_ONLY')
    if arrays['df_g_matrices'].shape!=(10,8,8) or arrays['spin_two'].shape!=(8,)*4:
        raise ValueError('INPUT_SHAPE')
    lifted_one,lifted_two=spin_lift_unthresholded(arrays['one_mo'],arrays['eri_chemist'])
    if not np.array_equal(lifted_one,arrays['spin_one']) or not np.array_equal(lifted_two/2,arrays['spin_two']):
        raise ValueError('ORIGINAL_SPIN_INDEX_OR_HALF_CONVENTION')
    spec=dm['hamiltonian']
    if (not np.array_equal(complex_array(spec['one_body']),arrays['df_one_body'])
            or not np.array_equal(complex_array(spec['g_matrices']),arrays['df_g_matrices'])
            or not np.array_equal(np.asarray(spec['lambdas']),arrays['df_lambdas'])
            or spec['constant']!=float(arrays['nuclear_repulsion'])):raise ValueError('DF_DESCRIPTOR_SNAPSHOT')
    indices=arrays['physical_qiskit_indices']
    if indices.tolist()!=receipt['sector_indices'] or len(set(indices.tolist()))!=36:
        raise ValueError('SECTOR_DESCRIPTOR')
    for index in indices:
        if any(sum((int(index)>>q)&1 for q in range(spin,8,2))!=2 for spin in (0,1)):
            raise ValueError('SAVED_SPIN_SECTOR')
    if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('FINITE_SAVED_INPUTS')
    data=dict(spin_one=arrays['spin_one'].astype(complex),spin_two=arrays['spin_two'].astype(complex),
        df_one_body=arrays['df_one_body'],lambdas=arrays['df_lambdas'],g_matrices=arrays['df_g_matrices'],
        constant=spec['constant'],pauli=pm['hamiltonian'],sector_indices=indices,
        snapshot_H_P=arrays['non_DF_matrix'],snapshot_H_DF=arrays['DF_matrix'])
    provenance=dict(ref=INPUT_REF,files={p:dict(sha256=h,git_blob=subprocess.check_output(
        ['git','rev-parse',INPUT_REF+':'+p],cwd=ROOT,text=True).strip()) for p,h in INPUTS.items()},
        original_SOURCE=ORIGINAL_SOURCE,state_hash=receipt['state_hash'],state_used_for='identity audit only; no state action',
        stored_full_H_check=receipt['gates']['DF_full_H_agreement'],DF_metadata=md,
        tensor_convention_verified=True,sector_indices=indices.tolist(),
        source_code_sha256={p:sha(gitbytes(ORIGINAL_SOURCE,p)) for p in FILES if p.startswith('src/')
            and not p.endswith('matched_partition_stage1.py')})
    return data,provenance


def execute():
    OUT.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    counts=dict(scientific_batches_started=0,Pauli_route_started=0,DF_route_started=0,
        routes_completed=0,operator_constructions_completed=0,native_basis_column_actions=0,
        metric_jobs_completed=0,time_evolution=0,signal_bias_B=0,SCF_FCI_DF_factorization=0,
        new_state_reference=0,RTE_law_sampling=0,circuit_build_compile=0,N_C_G=0,automatic_retries=0)
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        codes={p:sha((ROOT/p).read_bytes()) for p in FILES}
        if any(sha(gitbytes(commit,p))!=digest for p,digest in codes.items()):
            raise ValueError('SOURCE_NOT_COMMITTED_OR_CHANGED')
        runtime={n:importlib.metadata.version(n) for n in ('numpy','scipy','qiskit','openfermion')}
        if runtime!=dict(numpy='1.26.4',scipy='1.14.1',qiskit='1.3.0',openfermion='1.6.1'):
            raise ValueError('PLANNED_RUNTIME')
        kernel=modules()
        from openfermion.circuits.low_rank import get_chemist_two_body_coefficients
        low_rank=Path(inspect.getsourcefile(get_chemist_two_body_coefficients))
        if sha(low_rank.read_bytes())!=LOW_RANK_SHA:raise ValueError('CORRECTION_SOURCE')
        source=dict(commit=commit,code_sha256=codes,runtime=runtime,python=sys.version,
            execution_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),scope='H4_P3_STAGE1_PARTITION_A_ONLY',
            condition=dict(molecule='linear H4',distance_Angstrom=1.,basis='STO-3G',charge=0,multiplicity=1,
                spatial_orbitals=4,spin_orbitals=8,full_dimension=256,Pauli_nonidentity_terms=184,
                DF_rank_policy='tol-only 1e-8, saved returned rank10',partition='c_N I + T_eff + V_square',
                q_r_K_T='not used; historical T=.8 only provenance'),
            numerical_policy=kernel.POLICY,low_rank=dict(path=str(low_rank),sha256=LOW_RANK_SHA),
            affinity_CPUs=len(os.sched_getaffinity(0)),load_average=os.getloadavg(),
            meminfo={s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/meminfo').read_text().splitlines()
                if s.startswith(('MemTotal:','MemAvailable:'))},
            BLAS=dict(OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'),
                      reason='small dense matrices; independent metric jobs provide parallelism'),
            prior_technical_preflight=dict(source='d57742f182bc4c2ae68f1df5b54dfb074a7ea7e4',
                artifacts='artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1',
                operator_constructions=0,reason='decoder changed signed-zero bits; original inputs intact',
                original_STOP_preserved=True,scientific_retry=False))
        write('source_and_inputs.json',source)
        data,provenance=inputs();source['inputs']=provenance;write('source_and_inputs.json',source)
        results={};matrices={}
        counts['scientific_batches_started']=1
        counts['Pauli_route_started']=1;counts['DF_route_started']=1
        write('execution_status.json',dict(state='CONSTRUCTING_INDEPENDENT_ROUTES',counts=counts))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(fn,data):tag for tag,fn in [('Pauli',kernel.pauli_route),('DF',kernel.df_route)]}
            for future in as_completed(futures):
                tag=futures[future];m,info=future.result();matrices.update(m);results[tag]=info
                counts['routes_completed']+=1
                counts['operator_constructions_completed']+=info['full_Fock_matrix_constructions']
                counts['native_basis_column_actions']+=info.get('native_operator_basis_column_actions',0)
                write('matrix_fingerprints.json',{k:kernel.matrix_fingerprint(v) for k,v in matrices.items()})
                write('execution_status.json',dict(state='CONSTRUCTING_INDEPENDENT_ROUTES',counts=counts))
        available=max(1,len(os.sched_getaffinity(0))-math.ceil(os.getloadavg()[0]))
        workers=min(available,34) # Actual fixed metric-job count; no CPU resource cap.
        source['metric_workers']=workers;source['route_workers']=2
        source['route_worker_reason']='two independent construction paths'
        write('source_and_inputs.json',source)
        metrics=kernel.diagnose(data,matrices,results['Pauli'],results['DF'],workers=workers)
        counts['metric_jobs_completed']=metrics['metric_jobs']
        write('operator_comparisons.json',metrics)
        import numpy as np
        tensor_info={tag:dict(source=info['path'],correction_tensor=np.stack(
            [info['correction_tensor'].real,info['correction_tensor'].imag],axis=-1).tolist())
            for tag,info in results.items()}
        write('construction_paths.json',dict(routes=tensor_info,
            independent_T_and_V=True,V_from_H_minus_T_used_only_for_closure=True,
            Pauli='bare W -> independent reorder D -> JW(T_bare+D), JW(V_normal-D)',
            DF='saved h_eff and lambda/G -> native occupation T and sum lambda A^2',
            identity_policy='physical operators include their own JW identity; nuclear scalar once',
            full_space_256=True,sector_projection_for_primary_comparison=False))
        if any(sha((ROOT/p).read_bytes())!=digest for p,digest in codes.items()):raise ValueError('SOURCE_CHANGED')
        write('execution_status.json',dict(state='COMPLETED',scientific_status=metrics['overall'],
            counts=counts,workers_joined=True,elapsed_seconds=time.monotonic()-started))
        write('STOP.json',dict(status='STOP_STAGE1_'+metrics['overall']+'_NO_AUTOMATIC_FOLLOWUP',
            scientific_status=metrics['overall'],counts=counts,
            Stage2_3_4_authorized=False,P1_P2_STOP_unchanged=True))
    except Exception:
        write('execution_status.json',dict(state='FAILED_NO_RETRY',counts=counts,error=traceback.format_exc()))
        write('STOP.json',dict(status='STOP_STAGE1_FAILURE_NO_RETRY',counts=counts,error=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True)
    parser.parse_args();execute()
