"""Enclose run02's literal reconstruction model, not native circuit semantics.

The total-N=4 block is used only as a sparse full-column representation of
the same sector36 embedding. No state, gap or signal in another sector is
evaluated. Individual sampled atoms are neither projected nor tested again.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def frame_interval(job):
    from trotterlib.calpha_interval_bounds import IntervalBounds
    started=time.monotonic()
    basis, diagonal, head=job
    b=IntervalBounds(40);c=b.c;u=b.matrix(basis)
    if head:
        d=c.diag([c.exp(-c.j*b.real(.02)*b.real(float(x))/2) for x in diagonal])
        defect=b.norm(b.dagger(u)*u-c.eye(u.cols))
    else:
        d=c.diag([b.real(float(x)) for x in diagonal]);defect=None
    result=u*d*b.dagger(u)
    return dict(entries=[[result[i,j]._mpci_ for j in range(result.cols)] for i in range(result.rows)],
                defect=None if defect is None else defect._mpi_,
                elapsed=time.monotonic()-started)


def evaluate(private_dir):
    import numpy as np
    from trotterlib import b2_p1_signal_validation as p1
    from trotterlib import b2_p1_remaining_rounds as saved
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.calpha_interval_bounds import IntervalBounds,endpoint
    from trotterlib.calpha_macrostep_bounds import rational_inertia
    data=p1.load_private_inputs(private_dir)
    prior=json.loads((ROOT/'artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected/result.json').read_text())
    core=json.loads((ROOT/'artifacts/calpha_h4_step1/2026-11-10/run06_reference_interval/result.json').read_text())
    _,screen=saved.checked_prior();old=prior['operator_quantities']
    all_n4=np.array([x for x in range(256) if int(x).bit_count()==4])
    excluded=np.setdiff1d(np.arange(256),all_n4)
    positions=np.array([int(np.flatnonzero(all_n4==x)[0]) for x in data['indices']])
    occupation=((all_n4[:,None]>>np.arange(8))&1)
    jobs=[]
    for k,matrix in enumerate((data['ham'].one_body,*data['ham'].g_matrices)):
        orbital,eigenvalues=diag_hermitian(matrix,sort='descending_abs',assume_hermitian=True)
        basis=data['ref'].fock_lift(orbital)
        if np.count_nonzero(basis[np.ix_(excluded,all_n4)]):
            raise ValueError('TOTAL_NUMBER_BLOCK_SPARSE_IDENTITY_FAILED')
        if k<=3:
            diagonal=occupation@eigenvalues
            if k:
                diagonal=float(data['ham'].lambdas[k-1])*diagonal**2
        else:
            diagonal=np.zeros(len(all_n4))
            for atom in screen['coefficient_projection']['components']:
                if atom['fragment']!=k-1:
                    continue
                signs=np.ones(len(all_n4))
                for mode in atom['support']:
                    signs*=1-2*occupation[:,mode]
                diagonal+=atom['coefficient']*signs
        jobs.append((basis[np.ix_(all_n4,all_n4)],diagonal,k<=3))
    workers=min(len(jobs),len(os.sched_getaffinity(0)))
    print('parallel independent interval frames: '+str(workers),flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(frame_interval,x) for x in jobs]
        results=[f.result() for f in futures]
    b=IntervalBounds(40);c=b.c;n=len(all_n4);ident=c.eye(n)
    matrices=[c.matrix([[c.make_mpc(x) for x in row] for row in item['entries']]) for item in results]
    tail=sum(matrices[4:],c.zeros(n))
    hermitian=lambda x:(x+x.conj().T)/2
    full_reference=hermitian(sum(data['fragments'][3:],np.zeros((256,256),complex))-old['identity']*np.eye(256))
    reference=full_reference[np.ix_(all_n4,all_n4)]
    radius=Fraction.from_float(float(np.linalg.norm(reference,2)))+Fraction(1,1<<36)
    if [rational_inertia(reference,x)['negative'] for x in [-radius,radius]]!=[0,n]:
        raise ValueError('FULL_COLUMN_REFERENCE_NORM_NOT_ENCLOSED')
    tail_difference=b.norm(tail-b.matrix(reference))
    tail_norm=b.real(radius)+tail_difference
    hermitian_defect=b.norm(tail-b.dagger(tail))
    delta=b.real(.02);theta=delta/4;lam=b.real(old['lambda_R'])
    normalization=c.sqrt(1+(theta*lam)**2)
    mean=(ident-c.j*theta*tail)/normalization
    tail_norm_bound=c.sqrt(1+theta**2*tail_norm**2+theta*hermitian_defect)/normalization
    head_norm_factors=[1+c.make_mpf(item['defect']) for item in results[:4]]
    macro_norm=tail_norm_bound**4
    for value in head_norm_factors:
        macro_norm*=value**2
    if endpoint(macro_norm,True)>=1:
        raise ValueError('RECONSTRUCTION_CONTRACTION_NOT_CERTIFIED')
    print('full-column interval macrostep',flush=True)
    step=mean**4
    for head in reversed(matrices[:4]):
        step=head*step*head
    print('CAR reference interval macrostep',flush=True)
    ix=np.ix_(data['indices'],data['indices'])
    car_heads=[hermitian(x)[ix] for x in [data['one_body'],*data['fragments'][:3]]]
    rs=full_reference[ix]
    rnorm=b.real(Fraction(core['tail_norm_upper_exact']))
    rt,gamma=b.tail_log(b.matrix(rs),lam,theta,rnorm)
    reference_step=b.expm(delta*(-c.j*rt-gamma))
    for h in reversed(car_heads):
        half=b.expm(-c.j*delta*b.matrix(h)/2)
        reference_step=half*reference_step*half
    embedded=c.zeros(n,36)
    for i,row in enumerate(positions):
        for j in range(36):
            embedded[int(row),j]=reference_step[i,j]
    columns=c.matrix([[step[i,int(j)] for j in positions] for i in range(n)])
    connection=b.norm(columns-embedded)
    # Exact literal scalar used by the old SOURCE, and explicit raw target.
    scalar=float(data['ham'].constant)+old['identity']
    hmodel=sum([b.matrix(x) for x in car_heads],c.zeros(36))+b.matrix(rs)+b.real(scalar)*c.eye(36)
    h_reference=hermitian(data['physical'])[ix]
    outside_spin=np.setdiff1d(np.arange(256),data['indices'])
    if np.count_nonzero(data['physical'][np.ix_(outside_spin,data['indices'])]):
        raise ValueError('RAW_H_REFERENCE_SECTOR_CONNECTION_REQUIRED')
    model_error=b.norm(hmodel-b.matrix(h_reference))
    raw_error=b.norm(b.matrix(h_reference)-b.matrix(data['physical'][ix]))
    def serialize(v):
        return dict(lower_exact=str(endpoint(v)),upper_exact=str(endpoint(v,True)),
                    lower_float=float(endpoint(v)),upper_float=float(endpoint(v,True)))
    rows=[]
    for row in core['rounds']:
        elapsed=row['q']*delta
        addition=elapsed*model_error+row['q']*connection+c.expm1(elapsed*raw_error)
        d=b.real(Fraction(row['values']['core_distance_upper']['upper_exact']))+addition
        a=b.real(Fraction(row['values']['reference_radius_lower']['lower_exact']))
        good=endpoint(d,True)<=endpoint(a*c.sin(b.real('0.14')))
        phase=c.atan2(d,c.sqrt(a*a-d*d)) if endpoint(d,True)<endpoint(a) else None
        rows.append(dict(m=row['m'],model_term=serialize(elapsed*model_error),
                         connection_term=serialize(row['q']*connection),
                         raw_term=serialize(c.expm1(elapsed*raw_error)),
                         distance_upper=serialize(d),radius_lower=serialize(a-d),
                         phase_upper=None if phase is None else serialize(phase),
                         satisfies_014_reconstruction_model=good))
    return dict(status='RECONSTRUCTION_MODEL_ENCLOSURE_COMPLETE',
        input_identity=data['public'],parallel_frame_workers=workers,
        frame_worker_seconds=[x['elapsed'] for x in results],
        total_number_block_used_only_for_full_column_connection=True,
        tail_dictionary_minus_CAR_upper=serialize(tail_difference),
        macrostep_contraction_upper=serialize(macro_norm),
        macrostep_connection_upper=serialize(connection),model_difference_upper=serialize(model_error),
        raw_difference_upper=serialize(raw_error),rounds=rows,
        scope='Literal executable-coefficient reconstruction with binary64 basis/diagonal operands and exact matrix functions; not a native circuit or exact-involution dictionary certificate.',
        fixed_physical_law_guarantee='UNKNOWN_BASIS_AND_PRIMITIVE_SEMANTICS_NOT_CERTIFIED',
        implementation_inclusive_guarantee='UNKNOWN',new_raw_signals=0,new_damped_signals=0,
        RPE_cost_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['scripts/run_calpha_reconstruction_bridge.py','src/trotterlib/calpha_interval_bounds.py',
           'src/trotterlib/calpha_macrostep_bounds.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result=dict(started_UTC=datetime.now(timezone.utc).isoformat(),SOURCE_SHA256=hashes,status='STARTED')
    started=time.monotonic()
    try:
        result.update(evaluate(args.private_dir))
    except Exception as exc:
        result.update(status='TECHNICAL_EXCEPTION',reason=str(exc));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['SOURCE_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])
