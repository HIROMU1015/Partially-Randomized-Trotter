"""H2 saved-input adapter; reuse the fixed H4/H6 native and audit operations."""
import copy
import math
from pathlib import Path
from types import CodeType,FunctionType,SimpleNamespace
import h2_outer32_contract as c
from h46_pr_pilot_repair import BoundJSON

CTX={}
RUNTIME=None
ACTIVE_OUTPUT=None
PHASE_INDEX=0

def phase_transpile(circuit,**settings):
    from h2_m5_compiler_phase_guard import transpile
    result=transpile(circuit,**settings)
    receipt=result.metadata['H2_phase_guard']
    c.require(receipt['unchanged_gate_stream'] and not receipt['state_fit'],'H2_GATE_STREAM_GUARD')
    global PHASE_INDEX
    c.require(ACTIVE_OUTPUT is not None and PHASE_INDEX<2,'REGISTERED_H2_COMPILER_GUARD_CALL')
    c.save(ACTIVE_OUTPUT/('compiler_phase_guard_'+str(PHASE_INDEX)+'.json'),receipt);PHASE_INDEX+=1
    return result

def install(cfg):
    import sys
    root=Path(cfg['provider_root'])/'src'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    import numpy as np,qiskit
    from functools import partial
    import trotterlib.rte_compiled_cost as compiler
    c.require(np.__version__=='1.26.4' and qiskit.__version__=='1.3.0','H2_COMMON_PACKAGE_VERSIONS')
    indexed={r['path'] for r in cfg['sealed_files']}
    c.require(all(str(Path(mod.__file__).resolve()) in indexed for mod in (np,qiskit)),'H2_LOADED_RUNTIME_SEALED')
    compiler.transpile=partial(phase_transpile,num_processes=1)
    for name,mod in list(sys.modules.items()):
        if name.startswith(('trotterlib.','trottertracks.')) and getattr(mod,'__file__',None):
            c.require(Path(mod.__file__).resolve().is_relative_to(root),'H2_LOADED_PROVIDER_ORIGIN')

def context(cfg,system='H2',prefix=1):
    c.require(system=='H2' and prefix==1,'H2_FIXED_PREFIX1_ONLY')
    key=(cfg['SOURCE'],c.digest(cfg['inputs']))
    if key in CTX:return CTX[key]
    install(cfg)
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    from trotterlib.pr2_s0_s1_validation import _state_hash
    from h46_new2_native import OccupationOracle,prepare
    from h46_new2_core import spectrum,matrix_only_tail_coefficients
    record=c.read(cfg['inputs']['snapshot_receipt']['path']);path=cfg['inputs']['snapshot']['path']
    c.require(c.sha(path)==record['sha256']==cfg['binding']['snapshot'],'SAVED_H2_SNAPSHOT_SHA')
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    import hashlib
    for name,value in arrays.items():
        a=np.ascontiguousarray(value);saved=record['arrays'][name]
        actual=dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
        c.require(all(actual[k]==saved[k] for k in actual),'SAVED_H2_ARRAY:'+name)
    ham=DFHamiltonian(float(arrays['constant']),arrays['one_body'],arrays['lambdas'],tuple(arrays['g_matrices']),record['hamiltonian_metadata'])
    sector=PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
    full,state=arrays['state_vector'],arrays['sector_state_vector']
    c.require(np.array_equal(arrays['sector_basis_indices'],sector.basis_indices) and np.array_equal(full[sector.basis_indices],state),'SAVED_H2_SECTOR')
    c.require(df_hamiltonian_hash(ham)==cfg['binding']['hamiltonian'] and _state_hash(full,state)==cfg['binding']['state']
        and ham.n_blocks==3 and ham.n_qubits==4,'SAVED_H2_HAMILTONIAN_STATE')
    oracle=OccupationOracle(ham,full,prefix=3);qstate=np.zeros(16,complex);qstate[oracle.qindices]=oracle.initial
    c.require(abs(np.linalg.norm(qstate)-1)<1e-12,'SAVED_H2_STATE_NORM')
    fullprep=prepare(ham,3);cut=prepare(ham,1);matrices=tuple((s[1]*s[0])@s[2] for s in oracle.specs)
    scalar,lam=matrix_only_tail_coefficients(ham,1)
    c.require(abs(cut.extracted_identity_coefficient-scalar)<1e-9 and abs(cut.exact_rte_lambda_r-lam)<1e-9
        and tuple(cut.deterministic_fragment_indices)==(0,),'H2_FULL_FOCK_SCALAR_LAMBDA_PREFIX')
    tail=sum(matrices[2:],np.zeros_like(matrices[0]))-scalar*np.eye(len(oracle.initial))
    # Only SIGNAL may create the one registered full-H reference vector.
    reference=None
    if cfg['stage']=='H2_SIGNAL_OUTER32':
        from scipy.linalg import expm
        reference=expm(-.8j*(sum(matrices)+ham.constant*np.eye(len(oracle.initial))))@oracle.initial
        fp=c.read(cfg['inputs']['FP_result']['path'])['reference']['reference'];saved=complex(fp['real'],fp['imag'])
        c.require(abs(np.vdot(oracle.initial,reference)-saved)<1e-9,'H2_SAVED_REFERENCE_AGREEMENT')
    else:
        p=Path(cfg['reference_path']);c.require(str(p) in {r['path'] for r in cfg['sealed_files']},'H2_REFERENCE_SEALED')
        with np.load(p,allow_pickle=False) as z:
            c.require(np.array_equal(z['qstate'],qstate),'H2_REFERENCE_STATE_IDENTITY');reference=z['reference'].copy()
        c.require(cut.preparation_hash==cfg['preparation_hash'],'H2_PREPARATION_MATCHES_SIGNAL')
    ctx=SimpleNamespace(np=np,ham=ham,oracle=oracle,qstate=qstate,fullprep=fullprep,cut=cut,
        scalar=ham.constant,tail_identity=scalar,lambda_R=lam,tail_spec=spectrum(tail),matrices=matrices,reference=reference)
    CTX[key]=ctx;return ctx

def independent_signal(ctx,row):
    """Ascending matrix Taylor path; independent of the spectral tail path."""
    from scipy.linalg import expm
    import h46_m5_native as native
    np=ctx.np;v=np.repeat(ctx.oracle.initial[:,None],3,axis=1);logs=[]
    eye=np.eye(len(ctx.oracle.initial));tail=sum(ctx.matrices[2:],np.zeros_like(ctx.matrices[0]))-ctx.tail_identity*eye
    for step in native.c.core.schedule(row['formula'],row['q'],row['r_policy'],row['r_value']):
        t=step['signed_time'];r=step['rte_steps']
        for mat in ctx.matrices[:2]:v=expm(-.5j*t*mat)@v
        tau=ctx.lambda_R*t/r
        b=math.fsum(abs(tau)**k/math.factorial(k)*math.hypot(1.,tau/(k+1)) for k in (0,2,4))
        v[:,0]=expm(-1j*t*tail)@v[:,0]
        for _ in range(r):
            term=v[:,1:].copy();total=term.copy()
            for degree in range(1,6):term=(-1j*t/(r*degree))*(tail@term);total+=term
            v[:,1:]=total;v[:,2]/=b
        logs.append(r*math.log(b))
        for mat in reversed(ctx.matrices[:2]):v=expm(-.5j*t*mat)@v
    v*=np.exp(-.8j*(ctx.scalar+ctx.tail_identity))
    return [complex(np.vdot(ctx.oracle.initial,v[:,i])) for i in range(3)],math.fsum(logs)

def signal_stage(cfg):
    c.authorize(cfg);root=Path(cfg['run_root']);c.require(not root.exists(),'UNUSED_H2_SIGNAL_ROOT');root.mkdir()
    c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']));Path(cfg['ledger']).mkdir();Path(cfg['output']).mkdir()
    c.save(Path(cfg['ledger'])/'FULL_H_REFERENCE.json',dict(call=1,SOURCE=cfg['SOURCE'],state='RESERVED_NONREFUNDABLE'))
    ctx=context(cfg);np=ctx.np
    with (Path(cfg['output'])/'reference.npz').open('xb') as f:np.savez(f,reference=ctx.reference,qstate=ctx.qstate)
    import h46_m5_native as native
    from h46_m5_signal_contract import eligibility
    from concurrent.futures import ThreadPoolExecutor
    def one(row):
        c.save(Path(cfg['ledger'])/(row['id']+'.json'),dict(SOURCE=cfg['SOURCE'],signal_calls=1,state='RESERVED_NONREFUNDABLE'))
        record=native.signal(ctx,dict(row,policy=row['r_policy']));values,logB=independent_signal(ctx,row)
        errors=[abs(values[i]-complex(record[k]['real'],record[k]['imag'])) for i,k in enumerate(('exact_tail','corrected','raw'))]
        error=max(errors+[abs(logB-record['log_B'])]);c.require(error<1e-9,'H2_INDEPENDENT_SIGNAL_NORMALIZATION')
        return dict(candidate=row,signal=record,independent_forward_abs_errors=errors,
            normalization_error=abs(logB-record['log_B']),empirical_eligibility=eligibility(record,row))
    from run_h46_m5_signal import fresh_resources
    observation=fresh_resources(dict(cfg,systems={'H2':{}},resources=dict(system_RSS_proxy_bytes={'H2':cfg['resources']['worker_RSS_proxy_bytes']},disk_proxy_bytes=cfg['resources']['disk_proxy_bytes'])))
    workers=min(12,max(1,math.floor(observation['idle_CPU_equivalents'])))
    with ThreadPoolExecutor(max_workers=workers) as pool:records=list(pool.map(one,c.groups()))
    c.verify(cfg)
    result=dict(status='H2_OUTER32_SIGNAL_PASS',SOURCE=cfg['SOURCE'],records=records,preparation_hash=ctx.cut.preparation_hash,
        full_preparation_hash=ctx.fullprep.preparation_hash,reference_sha256=c.sha(Path(cfg['output'])/'reference.npz'),
        reference_accuracy='CONDITIONAL_ON_SAVED_REFERENCE_NOT_RIGOROUS',strict_bias_certificate=None,new_shots=0,new_compile=0,
        snapshot_restores=1,full_H_reference_jobs=1,group_threads=workers,formal_shot_policy_adopted=False)
    c.save(Path(cfg['output'])/'signals.json',result);c.save(root/'completion.json',dict(status=result['status'],result_sha256=c.sha(Path(cfg['output'])/'signals.json'),mandatory_STOP=True))
    return result

def replace_code(code,label):
    return code.replace(co_consts=tuple(replace_code(v,label) if isinstance(v,CodeType)
        else label if v=='DESIGN_PILOT' else v for v in code.co_consts))

def rebind(module,overrides,label):
    ns=dict(vars(module));ns.update(overrides)
    for name,fn in vars(module).items():
        if isinstance(fn,FunctionType) and fn.__module__==module.__name__:
            f=FunctionType(replace_code(fn.__code__,label),ns,name,fn.__defaults__,fn.__closure__);f.__kwdefaults__=fn.__kwdefaults__;ns[name]=f
    ns.update(overrides);return SimpleNamespace(**ns),ns

def enabled(cfg):
    c.require(cfg['stage'] in ('H2_DESIGN_PILOT','H2_PR_EXPLORATION') and cfg['resources']['process_start_method']=='fork','H2_FORK_OPTIMIZATIONS')
    return True

def prewarm(cfg,unused):
    context(cfg);return dict(contexts=1,keys=[['H2',1]],RNG_calls=0,compile_calls=0,signal_evaluations=0,readonly_shared=True)

def fresh(cfg):
    from h2_outer32_pipeline import admission
    return admission(cfg,len(cfg['tasks']))

def worker(task):
    global ACTIVE_OUTPUT,PHASE_INDEX
    cfg=RUNTIME.runner_namespace['WORKER_CFG'];stop=RUNTIME.runner_namespace['WORKER_STOP']
    ACTIVE_OUTPUT=Path(cfg['output'])/task['id'];PHASE_INDEX=0
    result=RUNTIME.native.run_task(cfg,task,stop)
    c.require(PHASE_INDEX==2,'H2_BOTH_COMPILER_GUARD_RECEIPTS');return result

def extra_audit(cfg):
    r=RUNTIME.auditor.audit(cfg)
    for t in cfg['tasks']:
        for i in range(2):
            p=Path(cfg['output'])/t['id']/('compiler_phase_guard_'+str(i)+'.json')
            a=c.read(p);c.require(a['unchanged_gate_stream'] and not a['state_fit'] and a['policy']==cfg['compiler_phase_policy'],'H2_COMPILER_GATE_STREAM_RECEIPT')
    c.save(Path(cfg['run_root'])/'compiler_policy_end_audit.json',dict(status='PASS',axes=2*len(cfg['tasks']),
        policy=cfg['compiler_phase_policy'],end_audit_sha256=c.sha(Path(cfg['run_root'])/'end_audit.json')))
    return r

def get_runtime(cfg):
    global RUNTIME
    if RUNTIME is not None:return RUNTIME
    import h46_pr_pilot_native as native
    import run_h46_pr_design_pilot as runner
    import audit_h46_pr_pilot_execution as audit
    from h46_pr_pilot_repair import BoundJSON
    label=c.cohort(cfg['stage']);c.require(label is not None,'H2_ACQUISITION_COHORT')
    view=SimpleNamespace(**{k:getattr(c,k) for k in ('ROOT','sha','read','save','seal','digest','require','authorize','verify',
        'consumed','reserve','task_limits','storage','COMPILE_TOLERANCE','OTHER_TOLERANCE')})
    view.design=SimpleNamespace(COHORT=label,event_stratum=c.event_stratum,COMPILER_HASH=c.previous.COMPILER_HASH,NATIVE_POLICY=c.previous.NATIVE_POLICY)
    view.SIGNAL_RECORDS=cfg['signal_path']
    nat,ns=rebind(native,dict(c=view,context=context,enabled=enabled,CACHE={},INPUT_CACHE=BoundJSON()),label)
    cli=str(ROOT_PATH/'scripts/run_h2_outer32_stage.py')
    run,rns=rebind(runner,dict(c=view,__file__=cli,repair=SimpleNamespace(enabled=enabled,prewarm=prewarm),fresh=fresh,worker=worker),label)
    run.execute.__kwdefaults__={'target':worker}
    run.supervise=FunctionType(runner.supervise.__code__,dict(runner.supervise.__globals__,c=view,__file__=cli),'supervise',runner.supervise.__defaults__,runner.supervise.__closure__)
    aud,ans=rebind(audit,dict(c=view),label)
    RUNTIME=SimpleNamespace(native=nat,native_namespace=ns,runner=run,runner_namespace=rns,auditor=aud,audit_namespace=ans)
    return RUNTIME

ROOT_PATH=c.ROOT
