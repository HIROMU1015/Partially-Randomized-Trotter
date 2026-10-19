"""Lazy signal-only adapter; one saved snapshot/reference per system process."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace
import traceback
import h46_m5_signal_contract as c


def base_context(cfg,system):
    import concurrent.futures.thread,concurrent.futures._base,queue,_queue,threading
    import h46_m5_native as original
    import h46_m5_tol_native as tol
    original.install(cfg)
    import numpy as np,qiskit
    from h46_new2_native import OccupationOracle,prepare
    sealed={r['path'] for r in cfg['sealed_files']}
    c.require(all(str(Path(m.__file__).resolve()) in sealed for m in
        (concurrent.futures.thread,concurrent.futures._base,queue,_queue,threading)),'LOADED_SIGNAL_THREAD_RUNTIME')
    c.require(np.__version__=='1.26.4' and qiskit.__version__=='1.3.0' and
        all(str(Path(m.__file__).resolve()) in sealed for m in (np,qiskit)),'LOADED_SIGNAL_RUNTIME')
    info=cfg['systems'][system];local=dict(cfg,system=system,inputs=info['inputs'],binding=info['binding'])
    ham,sector,full,state,md=tol.restore(local)
    binding=info['binding']
    c.require(md['hamiltonian_hash']==binding['hamiltonian'] and md['state_hash']==binding['state'] and ham.n_blocks==binding['rank'],'SIGNAL_INPUT_IDENTITY')
    oracle=OccupationOracle(ham,full,prefix=ham.n_blocks)
    qstate=np.zeros(1<<ham.n_qubits,complex);qstate[oracle.qindices]=oracle.initial
    c.require(abs(np.linalg.norm(qstate)-1)<1e-12,'SAVED_SIGNAL_STATE_NORM')
    ref=Path(info['reference_root']);receipt=c.read(ref/'receipt.json')
    c.require(c.sha(ref/'reference.npz')==receipt['reference_sha256'],'SAVED_SIGNAL_REFERENCE_SHA')
    with np.load(ref/'reference.npz',allow_pickle=False) as z:
        c.require(set(z.files)=={'reference','qstate'} and np.array_equal(z['qstate'],qstate),'REFERENCE_SAME_SAVED_STATE')
        reference=z['reference'].copy()
    c.require(reference.shape==oracle.initial.shape and np.isfinite(reference).all(),'FINITE_SAVED_REFERENCE')
    fullprep=prepare(ham,ham.n_blocks)
    c.require(fullprep.preparation_hash==receipt['full_preparation'],'SAVED_FULL_PREPARATION_IDENTITY')
    matrices=tuple((s[1]*s[0])@s[2] for s in oracle.specs)
    return SimpleNamespace(np=np,ham=ham,oracle=oracle,qstate=qstate,fullprep=fullprep,
        reference=reference,scalar=ham.constant,matrices=matrices,prepare=prepare,system=system,
        context_key=(cfg['SOURCE'],system,c.digest(info['inputs']),binding['hamiltonian'],binding['state']))


def prefix_context(base,prefix):
    from h46_new2_core import spectrum,matrix_only_tail_coefficients
    np=base.np;ham=base.ham
    c.require(type(prefix) is int and 1<=prefix<=ham.n_blocks,'SIGNAL_PREFIX_RANGE')
    cut=base.fullprep if prefix==ham.n_blocks else base.prepare(ham,prefix)
    # The canonical helper accepts truncated prefixes only. A full PF has no
    # discarded fragments, hence exactly zero extracted tail scalar and lambda.
    scalar,lam=(0.,0.) if prefix==ham.n_blocks else matrix_only_tail_coefficients(ham,prefix)
    c.require(abs(cut.extracted_identity_coefficient-scalar)<1e-9 and abs(cut.exact_rte_lambda_r-lam)<1e-9,'PREFIX_SCALAR_LAMBDA')
    c.require(tuple(cut.deterministic_fragment_indices)==tuple(range(prefix)),'PREFIX_GENERATION_ORDER')
    tail=sum(base.matrices[prefix+1:],np.zeros_like(base.matrices[0]))-scalar*np.eye(len(base.oracle.initial))
    return SimpleNamespace(**{**base.__dict__,'cut':cut,'tail_identity':scalar,'lambda_R':lam,'tail_spec':spectrum(tail),
        'prefix':prefix,'context_key':base.context_key+(prefix,cut.preparation_hash)})


def evaluate(ctx,row):
    # The numerical function is the unchanged legacy signal code, K4 only.
    import h46_m5_native as original
    c.require(ctx.prefix==row['prefix'],'NO_ANCHOR_TAIL_REUSE_FOR_NEW_PREFIX')
    group=dict(row,policy=row['r_policy'],r_value=row['r_value'])
    return original.signal(ctx,group)


def one_candidate(cfg,row,ctx,stop):
    c.require(not stop.is_set(),'OWN_SYSTEM_STOP');c.reserve(cfg,row)
    reuse=cfg['systems'][row['system']]['reuse'].get(row['id'])
    if reuse:
        c.require(c.sha(reuse['path'])==reuse['sha256'],'SAVED_CANDIDATE_SIGNAL_SHA')
        record=c.read(reuse['path']);origin='SAVED_RAW_SIGNAL_FIELDS_REUSED_NOT_OLD_TASK_PASS'
    else:record=evaluate(ctx,row);origin='NEW_FINITE_MEAN_SIGNAL'
    c.require(not stop.is_set(),'OWN_SYSTEM_STOP_AFTER_SIGNAL')
    allowance=c.eligibility(record,row)
    result=dict(status='SIGNAL_DIAGNOSTIC_PASS',candidate=row,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        signal=record,signal_origin=origin,source_signal_sha256=None if reuse is None else reuse['sha256'],
        prefix_context_hash=c.digest(ctx.context_key),preparation_hash=ctx.cut.preparation_hash,
        empirical_eligibility=allowance,shot_allowance_formally_adopted=False,strict_bias_certificate=None,
        new_signal=0 if reuse else 1,new_compile=0,new_sampling=0,cost_accounting=False,ranking=False)
    c.save(Path(cfg['output'])/(row['id']+'.json'),result)
    return dict(id=row['id'],file=row['id']+'.json',sha256=c.sha(Path(cfg['output'])/(row['id']+'.json')),new_signal=result['new_signal'])


def system_worker(cfg,system,threads,stop):
    out=Path(cfg['output'])
    try:
        c.authorize(cfg);c.consumed(cfg)
        base=base_context(cfg,system)
        candidates=c.rows(system)
        prefixes=sorted({r['prefix'] for r in candidates})
        contexts={p:prefix_context(base,p) for p in prefixes}
        c.save(out/(system+'_context_receipt.json'),dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
          saved_snapshot_restores=1,saved_reference_reads=1,new_full_H_reference=0,
          full_preparation=base.fullprep.preparation_hash,prefixes={str(p):dict(preparation=contexts[p].cut.preparation_hash,
          context_hash=c.digest(contexts[p].context_key),lambda_R=contexts[p].lambda_R,tail_identity=contexts[p].tail_identity) for p in prefixes},
          threads=threads,new_compile=0,new_sampling=0))
        completed=[];errors=[]
        with ThreadPoolExecutor(max_workers=threads) as pool:
            pending={pool.submit(one_candidate,cfg,row,contexts[row['prefix']],stop):row for row in candidates}
            for future in as_completed(pending):
                try:completed.append(future.result())
                except Exception as exc:
                    stop.set();errors.append(dict(candidate=pending[future]['id'],reason=str(exc)))
        c.require(not errors and len(completed)==len(candidates),'OWN_SIGNAL_SYSTEM_INCOMPLETE:'+str(errors))
        c.save(out/(system+'_complete.json'),dict(status='SIGNAL_SYSTEM_COMPLETE_MANDATORY_STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
            system=system,results=sorted(completed,key=lambda x:x['id']),new_signal=sum(x['new_signal'] for x in completed),reused_signal=len(completed)-sum(x['new_signal'] for x in completed),
            new_compile=0,new_sampling=0,new_full_H_reference=0,mandatory_STOP=True,next_phase_started=False))
    except BaseException as exc:
        stop.set();c.save(out/(system+'_STOP.json'),dict(status='STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],system=system,reason=str(exc),traceback=traceback.format_exc(),retry=False,mandatory_STOP=True))
