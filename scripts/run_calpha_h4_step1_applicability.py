"""Approved C-alpha H4 Step 1 input/sector gate; no circuits or signals.

Reuses the fixed P1 input loader and its CAR convention. A failed sector gate
ends this Step 1; it does not authorize a different dictionary or sector.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time

BASE = '0d5ca357ccfd8fcc850bef800c5bcadfde2a153e'
MATCH = 1e-9  # Existing P1 independent-match tolerance; diagnostic, not enclosure.


def run(private_dir, output):
    import numpy as np
    from trotterlib import b2_p1_signal_validation as p1
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.df_rte_tail import exact_df_diagonal_coefficients

    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    sources = [Path(__file__).relative_to(root).as_posix(),
               'src/trotterlib/b2_p1_signal_validation.py',
               'src/trotterlib/df_trotter/decompose.py',
               'src/trotterlib/df_rte_tail.py',
               'scripts/run_o1_m1_h4_operator_correctness.py']
    result = dict(schema='calpha_h4_step1_applicability_v1', base_SOURCE=BASE,
        status='STARTED', SOURCE_SHA256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
        conditions=dict(target='H_phys', geometry='linear H4 1.00 Angstrom',basis='STO-3G',
            DF_rank=12,L_D=3,partition='generation_prefix',sector='Nalpha=Nbeta=2;dimension36',
            K=0,r=4,delta=.02,rounds=[0,17],beta_systematic_max=.14),
        signal_calls=0,ground_solve_calls=0,circuit_build_calls=0,compile_calls=0,
        sampling_calls=0,shot_calls=0,guarantee_status='not_certified')
    try:
        for name in sources[1:]:
            fixed = subprocess.check_output(['git','show',BASE+':'+name],cwd=root)
            if fixed != (root/name).read_bytes():
                raise ValueError('FIXED_REUSED_SOURCE_CHANGED:'+name)
        data = p1.load_private_inputs(private_dir)
        result['input_identity']=data['public']
        print('Fixed P1 input/state/DF identity PASS',flush=True)
        indices=data['indices']; outside=np.setdiff1d(np.arange(256),indices)
        operators=[('raw_H_phys',data['physical']),('H_solve',
            .5*(data['physical']+data['physical'].conj().T)),('DF',data['df']),
            ('one_body',data['one_body'])]
        operators.extend(('fragment_'+str(k),b) for k,b in enumerate(data['fragments']))
        leakage={name:float(np.max(np.abs(matrix[np.ix_(outside,indices)]),initial=0))
                 for name,matrix in operators}
        result['operator_sector_leakage_maxentry']=leakage
        result['Hermitian_reference']=dict(definition='H_solve=(raw_H_phys+raw_H_phys.H)/2; target retained',
            solve_minus_raw_spectral_diagnostic=float(np.linalg.norm(
                .5*(data['physical'].conj().T-data['physical']),2)),
            gap_source='89165cf70958e7e4676d0580e3acb7faf588772c; saved numerical gap only',
            gap_certified_lower_bound=None)
        frames=[]; nonidentity_mass=[]; identities=[]; failed=[]
        for k,(weight,g) in enumerate(zip(data['ham'].lambdas,data['ham'].g_matrices,strict=True)):
            u,eta=diag_hermitian(g,sort='descending_abs',assume_hermitian=True)
            coefficients=exact_df_diagonal_coefficients(eta,float(weight))
            if k<3:
                continue
            # Z_j=I-2 n_j.  In the original spin basis n_j has one-particle
            # projector u[:,j]u[:,j].H. A nonzero alpha/beta cross-block is
            # sufficient to refute [Z_j,Nalpha]=0 for the sampled Z_j atom.
            singles={support[0]:coefficient for support,coefficient in coefficients
                     if len(support)==1 and coefficient!=0}
            cross=[]
            for j in sorted(singles):
                block=np.outer(u[::2,j],u[1::2,j].conj())
                value=float(np.max(np.abs(block),initial=0))
                cross.append(dict(diagonal_mode=j,coefficient=singles[j],
                    alpha_weight=float(np.vdot(u[::2,j],u[::2,j]).real),
                    beta_weight=float(np.vdot(u[1::2,j],u[1::2,j]).real),
                    spin_cross_projector_maxentry=value))
                if value>MATCH:
                    failed.append(dict(fragment=k,mode=j,spin_cross_projector_maxentry=value))
            frames.append(dict(fragment=k,nonzero_single_Z_atoms=cross,
                unitary_defect_maxentry=float(np.max(abs(u.conj().T@u-np.eye(8)))),
                reconstruction_maxentry=float(np.max(abs((u*eta)@u.conj().T-g)))))
            for support,coefficient in coefficients:
                (identities if not support else nonidentity_mass).append(float(coefficient))
        import math
        result['tail_dictionary_diagnostic']=dict(frames=frames,
            lambda_R=math.fsum(abs(x) for x in nonidentity_mass),
            extracted_identity=math.fsum(identities),
            single_Z_spin_violations=failed,match_tolerance=MATCH,
            criterion='necessary spin commutation of each nonzero sampled single-Z atom',
            convention='rank-one orbital projector; independent of column relabeling/global phase')
        print('Tail single-Z spin violations:',len(failed),flush=True)
        if failed:
            result['status']='TECHNICAL_STOP_PRIMITIVE_SPIN_SECTOR_NOT_PRESERVED'
            result['stop_reason']='Existing generic DF eigenframes mix alpha and beta in nonzero sampled Z atoms; theory-review sector condition fails. No signal/moment evaluated.'
            result['mandatory_stop']=True
        elif any(v>MATCH for v in leakage.values()):
            result['status']='TECHNICAL_STOP_OPERATOR_SECTOR_NOT_PRESERVED'
            result['mandatory_stop']=True
        else:
            cache=output/'private'
            cache.mkdir(mode=0o700)
            (cache/'.gitignore').write_text('*\n')
            np.savez(cache/'inputs.npz',physical=data['physical'],df=data['df'],
                one_body=data['one_body'],fragments=np.asarray(data['fragments']),
                state=data['state'],indices=indices,energy=data['energy'],
                lambdas=data['ham'].lambdas,g_matrices=data['ham'].g_matrices,
                original_one_body=data['ham'].one_body,constant=data['ham'].constant)
            (cache/'inputs.npz').chmod(0o600)
            result['status']='APPLICABILITY_DIAGNOSTIC_PASS_NEXT_SCIENCE_PENDING'
            result['mandatory_stop']=False
            result['cache_SHA256']=hashlib.sha256((cache/'inputs.npz').read_bytes()).hexdigest()
    except Exception as exc:
        result.update(status='TECHNICAL_STOP_EXCEPTION',stop_reason=str(exc),exception_type=type(exc).__name__,mandatory_stop=True)
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start
        result['environment']=dict(python=platform.python_version(),cpu_affinity=len(os.sched_getaffinity(0)),
            thread_env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},gpu_used=False,
            policy='small 8/36/256 matrices use single-thread BLAS; no trajectory or q-fold loop')
        (output/'applicability.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(result['status'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--private-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.private_dir,args.output)
