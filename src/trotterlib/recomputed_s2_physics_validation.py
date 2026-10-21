"""Supplemental one-shot native Gaussian/DF validation on saved inputs."""
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import recomputed_s2_runtime_v2 as r
CTX=None
FULL=None
TIMES=None


def check_block(index):
    import numpy as np
    from qiskit import QuantumCircuit
    from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives,df_squared_diagonal_primitives,append_diagonal_primitives
    from trotterlib.df_trotter.circuit import simulate_statevector
    from trotterlib.pr2_s0_s1_validation import _to_qiskit_state
    from trottertracks.resource_applicability.h8_sparse_reference_v1 import SpinPrimitive
    block=FULL.deterministic_blocks[index];ham=CTX['ham'];n=ham.n_qubits
    checks=[]
    definitions=[('one',ham.one_body,None,block.diagonal_eigenvalues)] if index==0 else [
        ('one_G',ham.g_matrices[index-1],None,block.diagonal_eta),
        ('squared_G',ham.g_matrices[index-1],block.lam,block.diagonal_eta)]
    for kind,matrix,weight,diagonal in definitions:
        target=SpinPrimitive(matrix,CTX['sector'],weight=weight)
        for t in TIMES:
            circuit=QuantumCircuit(n)
            QiskitDFPartialS2CircuitBuilder._append_basis(circuit,block,inverse=True)
            primitives=one_body_diagonal_primitives(diagonal,t) if weight is None else df_squared_diagonal_primitives(diagonal,weight,t)
            append_diagonal_primitives(circuit,primitives)
            QiskitDFPartialS2CircuitBuilder._append_basis(circuit,block,inverse=False)
            actual=simulate_statevector(circuit,CTX['qstate'])
            full=np.zeros(1<<n,complex);full[CTX['sector'].basis_indices]=target(CTX['state'],t)
            expected=_to_qiskit_state(full,n)
            error=float(np.linalg.norm(actual-expected))
            checks.append(dict(kind=kind,time=t,error=error,threshold=1e-9,PASS=error<1e-9,phase_alignment=False))
    return dict(block=index,basis_hash=block.basis_hash,checks=checks,PASS=all(x['PASS'] for x in checks))


def run(system):
    global CTX,FULL,TIMES
    source=r.read(r.BASE/'SOURCE_physics_validation.json')
    for p,h in source['files'].items():r.require(r.sha(ROOT/p)==h,'PHYSICS_SOURCE_CHANGED:'+p)
    dest=r.unit_path(system,'SYSTEM','native_physics_validation')
    r.require(not dest.exists(),'PHYSICS_UNIT_ALREADY_CONSUMED')
    dest.mkdir(parents=True)
    inp=r.unit_path(system,'SYSTEM','input_preparation')/'context.pkl'
    plan=dict(SOURCE=source['commit'],cohort=r.COHORT,system=system,candidate='SYSTEM',stage='native_physics_validation',
        source_manifest_sha256=r.sha(r.BASE/'SOURCE_physics_validation.json'),input_hashes={'context':r.sha(inp)},
        seed=None,conditions='Every registered full-H Gaussian basis; single-G and lambda*G^2; actual saved state; largest/smallest registered half-step times; no phase alignment',
        one_shot=True,automatic_retry=False)
    grant=dict(SOURCE=source['commit'],grant_id=r.digest(plan),plan_sha256=r.digest(plan),approved_by_user=True,
        authorization_sha256=r.sha(r.BASE/'user_authorization.md'),one_shot=True,automatic_retry=False)
    r.save(dest/'plan.json',plan);r.save(dest/'grant.json',grant)
    r.save(dest/'marker.json',dict(state='CONSUMED_NONREFUNDABLE',grant_id=grant['grant_id'],time=time.time()))
    started=time.monotonic()
    try:
        CTX=r.context(system)
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        FULL=CTX['preps'].get(CTX['ham'].n_blocks) or _prepare(CTX['ham'],CTX['ham'].n_blocks)
        registered=[c for c in r.read(r.BASE/'candidates.json') if c['system']==system]
        times=sorted({.8/c['q']/2 for c in registered})
        TIMES=sorted({min(times),max(times)})
        bindings=[]
        for p,prep in CTX['preps'].items():
            for block,reference in zip(prep.deterministic_blocks,FULL.deterministic_blocks):
                r.require(block.basis_hash==reference.basis_hash,'FULL_PREFIX_GAUSSIAN_BINDING')
            for comp in prep.rte_preparation.symbolic_tail.components:
                if comp.is_identity:continue
                j=int(comp.df_fragment_id.rsplit('-',1)[1])+1
                r.require(comp.basis_hash==FULL.deterministic_blocks[j].basis_hash,'TAIL_GAUSSIAN_BINDING')
            bindings.append(dict(prefix=p,preparation_hash=prep.preparation_hash,full_basis_match=True))
        nw=r.workers(len(FULL.deterministic_blocks))
        with mp.get_context('fork').Pool(nw) as pool:rows=list(pool.imap_unordered(check_block,range(len(FULL.deterministic_blocks))))
        rows.sort(key=lambda x:x['block'])
        result=dict(status='PASS' if all(x['PASS'] for x in rows) else 'FAIL',system=system,blocks=rows,
            maximum_error=max(x['error'] for row in rows for x in row['checks']),times=TIMES,workers=nw,
            bindings=bindings,full_preparation_hash=FULL.preparation_hash,phase_alignment=False,independent_scientific_review=False)
        r.save(dest/'result.json',result)
        r.require(result['status']=='PASS','NATIVE_PHYSICAL_PRIMITIVE_MISMATCH')
        r.save(dest/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP_UNIT',SOURCE=source['commit'],grant_id=grant['grant_id'],
            elapsed_seconds=time.monotonic()-started,result_files={p.name:r.sha(p) for p in dest.iterdir() if p.is_file()}))
        return dict(system=system,status='PASS',maximum_error=result['maximum_error'],checks=sum(len(x['checks']) for x in rows))
    except BaseException as exc:
        r.save(dest/'STOP.json',dict(status='FAILED_NO_RETRY',SOURCE=source['commit'],grant_id=grant['grant_id'],
            reason=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started))
        raise
