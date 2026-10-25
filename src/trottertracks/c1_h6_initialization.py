"""Saved-H6 initialization diagnosis; no optimizer, rank search or circuits.

The old codec is measured unchanged. The column-sign proposal is artificial-test
only, with a round-trip guard because SO(n) alone does not settle log branches.
"""
from pathlib import Path
import importlib.metadata
import json
import resource
import subprocess
import time

from . import c1_stage1_execution as io
from . import c1_molecule_extension as molecule
from .c1_stage3_integrated import Calls, resource_plan

SCOPE = 'C1_H6_SAVED_INITIALIZATION_DIAGNOSIS_V1'
RANKS = (14, 18)
ATOL = 1e-10
FIT_LIMITS = dict(saved_fit_load=1, explicit_initializer=2, initialization_identity=2,
                  initial_codec=2, final_reencoding=2, tensor_assessment=6,
                  value_and_gradient=4, final_optimizer_consistency=2)
SECTOR_LIMITS = dict(input_and_snapshots_load=1, sector_context=1, physical_sector=1,
                     sector_assessment=6, saved_sector_consistency=3, diagnosis=2)
FAMILY = 'artifacts/research_c_c1_h6_initialization'


def reconstruct(J, U):
    import numpy as np
    # Same contraction as Stage1 and the locked ffsim objective.
    return np.einsum('kpi,kqi,kij,krj,ksj->pqrs', U, U, J, U, U, optimize=True)


def tensor_metrics(g, J, U):
    import numpy as np
    residual = g-reconstruct(J, U)
    molecule.stage1.finite_arrays(residual)
    frob = float(np.linalg.norm(residual.ravel()))
    return dict(tensor_maxabs=float(np.max(abs(residual))), tensor_frobenius=frob,
                objective_from_tensor=.5*frob*frob, actual_rank=len(J))


def codec(J, U):
    """Actual old real=True codec, with its existing per-matrix-logm bridge."""
    import numpy as np
    import scipy.linalg
    import ffsim.linalg.util as util
    from unittest.mock import patch
    from . import c1_ffsim_compat as bridge
    original = bridge.batch_logm
    seen = {}
    def log(mats):
        value = original(mats); seen['logs'] = value
        return value
    with patch.object(bridge, 'batch_logm', side_effect=log), bridge.per_matrix_logm():
        x = util.df_tensors_to_params(J, U, None, real=True)
    a, b = util.df_tensors_from_params(x, len(J), len(U[0]), None, real=True)
    logs = seen['logs']; n = U.shape[-1]
    generators = util.antihermitians_from_parameters(
        x[:len(J)*n*(n-1)//2], n, len(J), real=True)
    frames = []
    for i, (old, new, logmat, generator) in enumerate(zip(U, b, logs, generators, strict=True)):
        po = np.einsum('pi,qi->ipq', old, old)
        pn = np.einsum('pi,qi->ipq', new, new)
        frames.append(dict(frame=i, det_before=float(np.linalg.det(old)),
            det_after=float(np.linalg.det(new)),
            orthogonality_before=float(np.max(abs(old.T@old-np.eye(n)))),
            log_imaginary_frobenius=float(np.linalg.norm(logmat.imag)),
            log_real_skew_residual=float(np.linalg.norm(logmat.real+logmat.real.T)),
            generator_skew_residual=float(np.max(abs(generator+generator.T))),
            complex_log_exponential_residual=float(np.max(abs(scipy.linalg.expm(logmat)-old))),
            rotation_roundtrip_maxabs=float(np.max(abs(new-old))),
            density_projector_roundtrip_maxabs=float(np.max(abs(pn-po))),
            distance_eigenvalue_to_minus_one=float(np.min(abs(np.linalg.eigvals(old)+1)))))
    return a, b, x, dict(frames=frames, dtype=str(x.dtype), real=True,
        J_roundtrip_maxabs=float(np.max(abs(a-J))),
        tensor_roundtrip_frobenius=float(np.linalg.norm((reconstruct(a,b)-reconstruct(J,U)).ravel())),
        parameters_shape=list(x.shape))


def column_sign_proposal(J, U):
    """Artificial-only minimal gauge proposal; refuse unresolved SO branches."""
    import numpy as np
    fixed = np.array(U, copy=True)
    if np.max(abs(fixed.transpose(0,2,1)@fixed-np.eye(fixed.shape[-1]))) > ATOL:
        raise ValueError('ORTHOGONAL_INPUT_REQUIRED')
    for mat in fixed:
        if np.linalg.det(mat)<0: mat[:, -1] *= -1
    a,b,x,report = codec(J, fixed)
    baseline=reconstruct(J,U)
    if np.linalg.norm((reconstruct(a,b)-baseline).ravel()) > ATOL*max(1.,np.linalg.norm(baseline.ravel())):
        raise ValueError('REAL_LOG_BRANCH_UNRESOLVED')
    return a,b,x,report


def initial_decomposition(g, rank):
    """Same nonoptimized initializer as old CDF; no optimized entry is called."""
    import importlib
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    return d._double_factorized_explicit_cholesky(g, tol=1e-8, max_vecs=rank)


def initialization_identity(J,U,saved_J,saved_U):
    import numpy as np
    if J.shape!=saved_J.shape or U.shape!=saved_U.shape:
        raise ValueError('INITIAL_RANK_MISMATCH')
    record=dict(J_maxabs=float(np.max(abs(J-saved_J))),
                U_maxabs=float(np.max(abs(U-saved_U))), tolerance=ATOL,
                actual_rank=len(J), status='PASS')
    if max(record['J_maxabs'],record['U_maxabs'])>ATOL:
        raise ValueError('INITIAL_ROTATION_IDENTITY_NOT_ESTABLISHED: '+str(record))
    return record


def objective_gradient(g,x,rank):
    import importlib
    import numpy as np
    import jax, jax.numpy as jnp
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    if not jax.config.jax_enable_x64:raise ValueError('JAX_X64_REQUIRED')
    fun=d._make_compressed_value_and_grad(rank,len(g),None)
    value,grad=fun(jnp.asarray(x),jnp.asarray(g));grad=np.asarray(grad)
    molecule.stage1.finite_arrays(grad,value)
    return dict(objective=float(value),gradient_l2=float(np.linalg.norm(grad)),
                gradient_infinity=float(np.max(abs(grad)))),grad


def check_objective(value,expected):
    allowance=1e-12+1e-8*abs(expected)
    if abs(value-expected)>allowance:raise ValueError('OBJECTIVE_IDENTITY_FAIL')
    return dict(status='PASS',difference=abs(value-expected),allowance=allowance)


def _save_arrays(path,**arrays):
    import numpy as np
    with Path(path).open('xb') as f:np.savez(f,**arrays)


def _finish(output,calls,start):
    counts=dict(calls.counts)
    child=output/'codec'/'STOP.json'
    if child.exists():
        for k,v in io.read(child)['call_counts'].items():counts[k]=counts.get(k,0)+v
    io.save(output/'STOP.json',dict(mandatory_stop=True,failed=calls.stop.is_set(),
        retry=0,resume=0,scientific_units=sum(counts.values()),call_counts=counts))
    io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,
        process_lifetime_CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        child_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))


def fit_diagnosis(root,output,contract):
    """Fit-env analysis, optimizer entry forbidden even if a helper regresses."""
    import numpy as np
    import ffsim
    import scipy.optimize
    from unittest.mock import patch
    root,output=Path(root),Path(output);output.mkdir(exist_ok=False)
    calls=Calls(output,FIT_LIMITS,sum(FIT_LIMITS.values()));start=time.monotonic()
    source=root/contract['saved_run']
    try:
        def load():
            with np.load(source/'input.npz',allow_pickle=False) as a:g=a['g']
            arrays={}
            for key in ['XDF_full','XDF_selected','CDF_14','CDF_18']:
                with np.load(source/(key+'.npz'),allow_pickle=False) as a:arrays[key]=(a['J'],a['U'])
            molecule.stage1.finite_arrays(g,*[a for pair in arrays.values() for a in pair])
            return g,arrays
        g,saved=calls('saved_fit_load','fixed H6 arrays',load)
        results={}
        with patch.object(scipy.optimize,'minimize',side_effect=AssertionError('OPTIMIZATION_FORBIDDEN')):
            for rank in RANKS:
                label='R'+str(rank);J,U=calls('explicit_initializer',label,lambda:initial_decomposition(g,rank))
                # Verify exact column conventions too: a sign-only change can affect the old codec.
                def verify():
                    value=initialization_identity(J,U,saved['XDF_full'][0][:rank],saved['XDF_full'][1][:rank])
                    if rank==14:value['selected_XDF14']=initialization_identity(J,U,*saved['XDF_selected'])
                    return value
                identity=calls('initialization_identity',label,verify)
                io.save(output/(label+'_identity.json'),identity)
                afterJ,afterU,x0,diagnostic=calls('initial_codec',label,lambda:codec(J,U))
                io.save(output/(label+'_codec.json'),diagnostic)
                endJ,endU=saved['CDF_'+str(rank)]
                reJ,reU,endx,enddiag=calls('final_reencoding',label,lambda:codec(endJ,endU))
                io.save(output/(label+'_final_reencoding.json'),enddiag)
                metrics={}
                for state,A,B in [('before',J,U),('after',afterJ,afterU),('final',endJ,endU)]:
                    key=f'{label}_{state}';_save_arrays(output/(key+'.npz'),J=A,U=B)
                    metrics[state]=calls('tensor_assessment',key,lambda A=A,B=B:tensor_metrics(g,A,B))
                    io.save(output/(key+'_tensor.json'),metrics[state])
                initial_grad,gradient=calls('value_and_gradient',label+':x0',lambda:objective_gradient(g,x0,rank))
                check_objective(initial_grad['objective'],metrics['after']['objective_from_tensor'])
                _save_arrays(output/(label+'_initial_parameters.npz'),x=x0,gradient=gradient)
                # The old optimizer x/jac was NOT saved. This is a fresh chart diagnostic only.
                final_scale=max(1.,float(np.linalg.norm(reconstruct(endJ,endU).ravel())))
                final_grad=dict(status='UNRESOLVED_FINAL_REENCODING',original_optimizer_jac=None)
                if enddiag['tensor_roundtrip_frobenius']<=ATOL*final_scale:
                    value,gradient=calls('value_and_gradient',label+':final_chart',lambda:objective_gradient(g,endx,rank))
                    check_objective(value['objective'],metrics['final']['objective_from_tensor'])
                    _save_arrays(output/(label+'_final_reencoded_parameters.npz'),x=endx,gradient=gradient)
                    final_grad=dict(**value,status='REENCODED_CHART_ONLY',original_optimizer_jac=None)
                md=io.read(source/f'CDF_{rank}.json')
                consistency=calls('final_optimizer_consistency',label,lambda:check_objective(
                    metrics['final']['objective_from_tensor'],md['fun']))
                results[str(rank)]=dict(identity=identity,codec=diagnostic,tensor=metrics,
                    initial_gradient=initial_grad,final_chart_gradient=final_grad,
                    saved_optimizer=md,final_objective_consistency=consistency,
                    original_optimizer_x=None,original_optimizer_jac=None)
                io.save(output/(label+'_result.json'),results[str(rank)])
        io.save(output/'result.json',dict(ranks=results,new_fit=0,optimizer_iterations=0,
            repaired_H6_codec_evaluations=0))
    except BaseException as error:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(error),first_failure=calls.first_failure));raise
    finally:_finish(output,calls,start)


def preflight(root,contract_path,*,fit=False):
    root=Path(root);c=io.read(contract_path)
    if (c['scope']!=SCOPE or c['ranks']!=list(RANKS) or c['fit_calls']!=FIT_LIMITS
        or c['sector_calls']!=SECTOR_LIMITS or c['maximum_logical_units']!=35):
        raise ValueError('DIAGNOSIS_SCOPE')
    manifest=root/c['source_manifest']
    if io.sha(manifest)!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for f,w in io.read(manifest)['files'].items():
        if io.sha(root/f)!=w:raise ValueError('SOURCE: '+f)
    for f,w in c['saved_inputs'].items():
        if io.sha(root/f)!=w:raise ValueError('SAVED_INPUT: '+f)
    packages=c['fit_packages'] if fit else c['evaluation_packages']
    for k,w in packages.items():
        if importlib.metadata.version(k)!=w:raise ValueError('ENVIRONMENT: '+k)
    if fit:
        import ffsim
        for name,w in c['ffsim_source'].items():
            if io.sha(Path(ffsim.__file__).parent/name)!=w:raise ValueError('FFSIM_SOURCE: '+name)
    return c


def run(root,contract_path,output,fit_python):
    """One fresh two-environment diagnosis; fixed data, two ranks, zero fitting."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    root,output=Path(root),Path(output);c=preflight(root,contract_path)
    output.mkdir(exist_ok=False,parents=True);calls=Calls(output,SECTOR_LIMITS,sum(SECTOR_LIMITS.values()));start=time.monotonic()
    io.save(output/'contract.json',c)
    try:
        subprocess.run([str(Path(fit_python).absolute()),str(root/c['entry']),
            '--contract',str(Path(contract_path).absolute()),'--worker','--output',str(output/'codec')],check=True)
        def load():
            with np.load(root/c['saved_run']/'input.npz',allow_pickle=False) as a:packet={k:a[k] for k in a.files}
            snapshots={}
            for rank in RANKS:
                for state in ('before','after','final'):
                    with np.load(output/'codec'/f'R{rank}_{state}.npz',allow_pickle=False) as a:snapshots[rank,state]=(a['J'],a['U'])
            return packet,snapshots
        packet,snapshots=calls('input_and_snapshots_load','worker snapshots',load)
        context=calls('sector_context','sector400',lambda:molecule.SectorCAR(len(packet['h'])))
        physical=calls('physical_sector','same original h/g/c',lambda:context.physical(packet['h'],packet['g'],packet['constant']))
        plan=resource_plan(task_count=6,full_dimension=context.d);io.save(output/'resource_plan.json',plan)
        def assess(key):
            rank,state=key;label=f'R{rank}_{state}';J,U=snapshots[key]
            value,_=calls('sector_assessment',label,lambda:molecule.rank_assessment(context,packet,J,U,physical,rank))
            value.update(state=state,budget_Ha=c['DF_reserve_Ha'],
                budget_status='PASS' if value['sector_error']+value['numerical_guard']<=c['DF_reserve_Ha'] else 'NOT_MET')
            io.save(output/(label+'_sector.json'),value);return key,value
        with threadpool_limits(limits=1 if plan['workers']>1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:rows=dict(pool.map(assess,snapshots))
        for rank,state,file in [(14,'before','XDF_rank_14.json'),(14,'final','CDF_rank_14.json'),(18,'final','CDF_rank_18.json')]:
            def compare(rank=rank,state=state,file=file):
                old=io.read(root/c['saved_run']/file);now=rows[rank,state]
                if abs(now['sector_error']-old['sector_error'])>ATOL:raise ValueError('SAVED_SECTOR_REPLAY')
                return dict(status='PASS',difference=abs(now['sector_error']-old['sector_error']))
            value=calls('saved_sector_consistency',file,compare);io.save(output/(file.replace('.json','_replay.json')),value)
        diagnostic=io.read(output/'codec'/'result.json');conclusions={}
        for rank in RANKS:
            def diagnose(rank=rank):
                t=diagnostic['ranks'][str(rank)]['tensor'];before=t['before'];after=t['after']
                tensor_delta=after['tensor_frobenius']-before['tensor_frobenius']
                sector_delta=rows[rank,'after']['sector_error']-rows[rank,'before']['sector_error']
                changed=diagnostic['ranks'][str(rank)]['codec']['tensor_roundtrip_frobenius']>ATOL
                return dict(rank=rank,classification='INITIAL_CONVERSION_DEGRADATION_OBSERVED' if changed and tensor_delta>ATOL and sector_delta>ATOL else (
                    'INITIAL_CONVERSION_PRESERVED' if not changed else 'MIXED_OR_UNRESOLVED'),
                    tensor_error_increase=tensor_delta,sector_error_increase_Ha=sector_delta,
                    repaired_fit_performance=None,sole_cause_established=False)
            conclusions[str(rank)]=calls('diagnosis',str(rank),diagnose)
        io.save(output/'result.json',dict(status='LIMITED_REANALYSIS_COMPLETE',conclusions=conclusions,
            new_fit=0,optimizer_iterations=0,repaired_H6_codec_evaluations=0,
            original_optimizer_jac=None,retry=0,resume=0,mandatory_stop=True))
    except BaseException as error:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(error),first_failure=calls.first_failure));raise
    finally:_finish(output,calls,start)
