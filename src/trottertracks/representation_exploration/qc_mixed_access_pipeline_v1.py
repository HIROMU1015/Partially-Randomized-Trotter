"""One-shot registered pipeline; PySCF is imported only inside authorized input.

Importing this module does not generate molecular data. Synthetic helpers use
hand-written integrals, never molecular engines or existing scientific packs.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.linalg import expm

from . import qc_mixed_access_math_v1 as core
from .qc_mixed_access_execution_v1 import AuthorizedContext, LaunchError, THREAD_VARS
from .qc_mixed_access_checker_v1 import check_raw
from .qc_mixed_access_resource_v1 import evaluate_resources


def write_json(path,data,compact=False):
    def scalar(value):
        if isinstance(value,np.generic): return value.item()
        raise TypeError('UNSUPPORTED_JSON:'+type(value).__name__)
    Path(path).write_text(json.dumps(data,indent=None if compact else 2,separators=(',',':') if compact else None,allow_nan=False,default=scalar)+'\n')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_input(context,budget):
    if not isinstance(context,AuthorizedContext): raise LaunchError('AUTHORIZED_CONTEXT_REQUIRED')
    # Explicit runtime capability check before any engine import or construction.
    if context.receipt.get('type')!='USER_EXPLICIT_SCIENCE_APPROVAL': raise LaunchError('ENGINE_APPROVAL')
    from pyscf import gto, scf, ao2mo
    started=time.monotonic()
    target=context.contract['target']; budget.tick('molecular_inputs'); budget.tick('RHF_jobs')
    mol=gto.M(atom=[(a,(x,y,z)) for a,x,y,z in target['coordinates_Angstrom']],unit='Angstrom',
              basis=target['basis'],charge=target['charge'],spin=target['spin'],symmetry=False,verbose=0)
    job=scf.RHF(mol); job.conv_tol=target['SCF_conv_tol']; job.max_cycle=target['SCF_max_cycle']; job.init_guess=target['SCF_init_guess']
    job.max_memory=1024
    cycles=[]
    def callback(env):
        cycles.append(int(env['cycle'])+1)
    job.callback=callback; scf_started=time.monotonic(); job.kernel()
    scf_seconds=time.monotonic()-scf_started
    budget.tick('SCF_cycles',max(cycles,default=0))
    if not job.converged: raise core.ContractError('SCF_NOT_CONVERGED_NO_RETRY')
    mo=np.asarray(job.mo_coeff)
    if mo.shape[1]!=4 or not np.isrealobj(mo): raise core.ContractError('RHF_CANONICAL_PACK')
    integral_started=time.monotonic(); h=mo.T@job.get_hcore()@mo
    eri=ao2mo.restore(1,ao2mo.kernel(mol,mo),len(h))
    constant=float(mol.energy_nuc())
    integral_seconds=time.monotonic()-integral_started
    save_started=time.monotonic()
    np.savez(context.output/'input.npz',h=h,eri=eri,constant=constant,MO=mo)
    write_json(context.output/'input_seal.json',{'status':'SEALED_BEFORE_PF','input_SHA256':file_sha(context.output/'input.npz'),
        'recipe':target,'SCF_cycles':max(cycles,default=0),'FCI_jobs':0,'ground_gap_jobs':0,
        'SCF_extra_convergence_check':bool(job.conv_check),'PySCF_max_memory_MB':1024,
        'timings':{'SCF_seconds':scf_seconds,'integral_transform_seconds':integral_seconds,'pack_save_seconds':time.monotonic()-save_started,'total_input_seconds':time.monotonic()-started},
        'eigensolver_scope':'SCF one-body and later DF small-matrix diagonalizations only'})
    return h,eri,constant


def derived_df_matrix(model):
    # Validation reconstruction; optimized kernel basis, checked independently
    # against determinant frames in synthetic tests.
    result=np.eye(1<<model['n'],dtype=complex)*model['scalar']
    for actor in model['gaussian']:
        frame=core.gaussian_apply(np.eye(len(result)),actor['operations'])
        result+=(frame*actor['energy'][None,:])@frame.conj().T
    return result


def worker_probe(job):
    model,method,times,steps,references,particles,output,limits=job
    for name in THREAD_VARS: os.environ[name]='1'
    budget=core.Budget(limits); started=time.monotonic(); probes=[]; seeds={}; worst_unitarity=0.
    columns=np.array([i for i in range(1<<model['n']) if i.bit_count()==particles])
    unique=sorted({float(T/r) for T in times for r in steps})
    for h in unique:
        budget.tick('distinct_ideal_seed_matrices')
        seeds[h]=core.seed_matrix(method,h,model,budget)
        defect=float(np.linalg.norm(seeds[h].conj().T@seeds[h]-np.eye(len(seeds[h])),'fro'))
        worst_unitarity=max(worst_unitarity,defect)
        if defect>1e-8: raise core.ContractError('SEED_UNITARITY_GUARD')
    for T in times:
        for r in steps:
            budget.tick('PF_error_queries')
            unitary=core.power_doubling(seeds[float(T/r)],int(r),budget)
            error=unitary[:,columns]-references[T][:,columns]
            norm,sigma,vh=core.norm_certificate(error,budget)
            name=f'error_{method}_T{T}_r{r}.npz'
            np.savez(Path(output)/name,error=error,sigma=sigma,vh=vh,columns=columns)
            other=[i for i in range(len(unitary)) if i.bit_count()!=particles]
            probes.append({'method':method,'T':T,'r':r,'error':norm,'file':name,
                           'leakage_fro':float(np.linalg.norm(unitary[np.ix_(other,columns)],'fro'))})
    return {'method':method,'probes':probes,'counts':dict(budget.counts),
            'seed_unitarity_defect_max_fro':worst_unitarity,
            'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def reserved_counts(model,methods,times,steps):
    unique={float(T/r) for T in times for r in steps}; primitive=0
    for method in methods:
        for h in unique:
            for kind,index,t,adj in core.stages(method,h,model):
                primitive+=1+(2*len(model['gaussian'][index]['operations']) if kind=='D' else 0)
    return {'distinct_ideal_seed_matrices':len(methods)*len(unique),
            'PF_error_queries':len(methods)*len(times)*len(steps),
            'norm_SVD_including_checks':len(methods)*len(times)*len(steps),
            'dense_power_products':len(methods)*len(times)*sum(int(r).bit_length()-1 for r in steps),
            'primitive_applications':primitive}


def directory_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).iterdir() if p.is_file())


def resident_bytes(pid):
    try:
        return int(next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmRSS:')).split()[1])*1024
    except (FileNotFoundError,StopIteration): return 0


def monitor_wait(futures,pool,output,limit):
    peak=0
    while not all(f.done() for f in futures):
        # ProcessPoolExecutor has no public worker-PID accessor in Python 3.12;
        # guard this host-specific monitoring path explicitly.
        pids=[os.getpid()]+[p.pid for p in pool._processes.values()]
        rss=sum(resident_bytes(pid) for pid in pids); peak=max(peak,rss)
        if rss>limit['aggregate_RSS_bytes'] or directory_bytes(output)>limit['output_bytes']:
            for process in pool._processes.values(): process.terminate()
            raise core.ContractError('RESOURCE_STOP_AGGREGATE_RSS_OR_OUTPUT')
        time.sleep(.2)
    return peak


def run_registered(context):
    if not isinstance(context,AuthorizedContext): raise LaunchError('AUTHORIZED_CONTEXT_REQUIRED')
    started=time.monotonic(); budget=core.Budget(context.contract['limits']); out=context.output
    try:
        h,eri,constant=generate_input(context,budget)
        budget.tick('DF_decompositions'); model=core.build_model(h,eri,constant,budget)
        for key,count in (('M_eff',len(model['groups'])),('Pauli_keys',len(model['paulis'])),('DF_factors',len(model['factors']))): budget.tick(key,count)
        if model['n']!=8: raise core.ContractError('FULL_FOCK_SHAPE')
        df=derived_df_matrix(model); df_norm=float(np.linalg.norm(df-model['H'],'fro'))
        if df_norm>1e-8: raise core.ContractError('DF_MATRIX_RECONSTRUCTION')
        # Frobenius is a valid operator-norm upper bound for residual deduction.
        np.savez(out/'derived.npz',H=model['H'],eps=model['eps'],density=model['density'],
                 df_lambdas=np.array([x[0] for x in model['factors']]),df_factors=np.array([x[1] for x in model['factors']]))
        columns=[i for i in range(256) if i.bit_count()==4]
        B=model['H']-np.diag(model['diagonal']+model['scalar'])
        budget.tick('norm_SVD_including_checks'); coupling_norm=float(np.linalg.svd(B[:,columns],compute_uv=False)[0])
        structure={'M_eff':len(model['groups']),'M_raw_canonical_oriented_monomials':sum({i for i,a in ops if a}!={i for i,a in ops if not a} for ops in model['terms']),
                   'z_j':[sum(v!=0 for v in g['delta_weights'].values()) for g in model['groups']],
                   'group_descriptors':[{k:g[k] for k in ('create','annihilate','g0','g_weights','delta0','delta_weights','parity_mask')} for g in model['groups']],
                   'lambda_A':float(sum(abs(x) for x in model['eps'])+sum(abs(model['density'][i,j]) for i in range(8) for j in range(i+1,8))),
                   'lambda_B':float(sum(abs(g['g0'])+sum(abs(v) for v in g['g_weights'].values()) for g in model['groups'])),
                   'A_diagonal_range':[float(min(model['diagonal'])),float(max(model['diagonal']))],
                   'coupling_norm_B_Qnu':coupling_norm,'DF_factors_all':len(model['factors']),
                   'DF_reporting_rank_1e_12':sum(abs(x[0])>1e-12 for x in model['factors']),
                   'DF_internal_rank_reporting_1e_12':[int(sum(abs(v)>1e-12 for v in actor['eta'])) for actor in model['gaussian'][1:]],
                   'classical_preprocessing':model['preprocessing_timings'],'preprocessing_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   'DF_residual_fro_upper':df_norm,'DF_coefficient_residual':model['df_coefficient_residual'],
                   'no_rank_truncation':True,'classical_eigh_counts':dict(budget.counts)}
        write_json(out/'structure.json',structure)
        references={}; crosschecks=[]
        for T in context.contract['grid']['times_atomic']:
            budget.tick('reference_matrices'); pade=expm(-1j*T*model['H'])
            taylor,info=core.taylor_reference(model['H'],T,budget)
            defect=float(np.linalg.norm(pade-taylor,'fro'))
            unitarity=float(np.linalg.norm(pade.conj().T@pade-np.eye(len(pade)),'fro'))
            if defect>1e-10: raise core.ContractError('REFERENCE_CROSSCHECK')
            if unitarity>1e-8: raise core.ContractError('REFERENCE_UNITARITY')
            references[T]=pade; np.savez(out/f'reference_T{T}.npz',pade=pade,taylor=taylor)
            crosschecks.append({'T':T,'difference_fro':defect,'unitarity_fro':unitarity,**info})
        write_json(out/'reference_checks.json',crosschecks)
        grid=context.contract['grid']; methods=context.contract['methods']
        reservation=reserved_counts(model,methods,grid['times_atomic'],grid['steps'])
        # Uncompressed complex errors + right singular vectors fit the output
        # contract. Reserve another 4 MiB for resource/metadata/failure records.
        count=reservation['PF_error_queries']; cols=len(columns)
        projected=directory_bytes(out)+count*((256*cols+cols*cols)*16+cols*16+4096)+2**22
        if projected>budget.limits['output_bytes']: raise core.ContractError('PREDISPATCH_OUTPUT_CAP')
        # Reserve aggregate workloads before dispatch, rather than locking per gate.
        for key,count in reservation.items():
            if budget.counts[key]+count>budget.limits[key]: raise core.ContractError('PREDISPATCH_CAP:'+key)
        jobs=[(model,m,grid['times_atomic'],grid['steps'],references,4,str(out),budget.limits) for m in methods]
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(worker_probe,job) for job in jobs]
            peak=monitor_wait(futures,pool,out,budget.limits)
            worker_results=[f.result() for f in futures]
        for result in worker_results:
            for key,count in result['counts'].items(): budget.tick(key,count)
        actual={k:sum(w['counts'].get(k,0) for w in worker_results) for k in reservation}
        if actual!=reservation: raise core.ContractError('RESERVATION_ACCOUNTING')
        probes=[p for result in worker_results for p in result['probes']]; write_json(out/'probes.json',probes)
        write_json(out/'execution_metrics.json',{'reservation':reservation,'counts':dict(budget.counts),'worker_results':[{k:v for k,v in w.items() if k!='probes'} for w in worker_results],
                                               'aggregate_RSS_peak_sampled':peak,'sample_period_seconds':.2,'threads_per_worker':1,'workers':2,'wall_seconds_so_far':time.monotonic()-started})
        names=[p.name for p in out.iterdir() if p.is_file() and p.name!='launch_receipt.json']
        write_json(out/'raw_seal.json',{'status':'SEALED_BEFORE_SELECTION','files':{name:file_sha(out/name) for name in sorted(names)}})
        audit=check_raw(out,context.contract); write_json(out/'independent_audit.json',audit)
        evaluation=evaluate_resources(model,probes,context.contract,df_norm)
        budget.tick('resource_model_rows',len(evaluation['rows']))
        write_json(out/'resource_evaluation.json',evaluation,compact=True)
        result={'status':evaluation['status'],'mandatory_stop':True,'counts':dict(budget.counts),
                'wall_seconds':time.monotonic()-started,'constructor_independent_new_algorithm':'NOT_ADOPTED',
                'ground_energy_task_evaluated':False,'quantum_compile':False,'new_GPT_review_started':False}
        write_json(out/'result.json',result)
        if directory_bytes(out)>budget.limits['output_bytes']: raise core.ContractError('OUTPUT_CAP')
        return result
    except Exception as error:
        status='RESOURCE_STOP' if 'CAP' in str(error) or 'RESOURCE_STOP' in str(error) else 'TECHNICAL_STOP'
        write_json(out/'failure.json',{'status':status,'reason':str(error)[:4096],'type':type(error).__name__,
                  'counts':dict(budget.counts),'wall_seconds':time.monotonic()-started,'retry':False,'mandatory_stop':True})
        raise
