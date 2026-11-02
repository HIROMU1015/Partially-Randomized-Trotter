"""Reuse frozen scientific operations; adapt ONLY the cohort label and bindings.

Recursive bytecode adaptation replaces the exact string DESIGN_PILOT only. No
operation, numerical constant, tolerance, RNG, schedule or compiler is changed.
Old modules and old runs are never patched. Real execution requires authorize().
"""
from pathlib import Path
from types import CodeType,FunctionType,SimpleNamespace
import h46_pr_exploration_contract as c
import h46_pr_pilot_repair as repair

def cohort_code(code):
    return code.replace(co_consts=tuple(cohort_code(v) if isinstance(v,CodeType)
        else 'EXPLORATION' if v=='DESIGN_PILOT' else v for v in code.co_consts))

def rebind(module,overrides):
    namespace=dict(vars(module));namespace.update(overrides)
    for name,fn in vars(module).items():
        if isinstance(fn,FunctionType) and fn.__module__==module.__name__:
            new=FunctionType(cohort_code(fn.__code__),namespace,name,fn.__defaults__,fn.__closure__)
            new.__kwdefaults__=fn.__kwdefaults__;namespace[name]=new
    namespace.update(overrides)
    return SimpleNamespace(**namespace),namespace

def enabled(cfg):
    r=cfg.get('resources',{})
    c.require(cfg['stage'] in c.STAGES and r.get('process_start_method')=='fork'
      and r.get('prewarm_readonly_contexts') is True and r.get('worker_input_cache') is True
      and r.get('worker_allocation_policy')=='FRESH_JOINT_CPU_RAM_V1','EXACT_EXPLORATION_OPTIMIZATIONS')
    return True

def prewarm(cfg,unused_original_context):
    enabled(cfg)
    keys=sorted({(t['candidate']['system'],t['candidate']['prefix']) for t in cfg['tasks']})
    for system,prefix in keys:RUNTIME.native.context(cfg,system,prefix)
    return dict(contexts=len(keys),keys=[list(k) for k in keys],RNG_calls=0,signal_evaluations=0,
      compile_calls=0,context_readonly_between_forked_workers=True)

def fresh(cfg):
    """Fresh physical admission, bounded by a jointly reserved resource share.

    The joint lease comes from fresh CPU/RAM evidence for both jobs; it is NOT
    a historical worker ceiling. Re-probe after shared context prewarming.
    """
    import os
    from run_h46_m5_signal import fresh_resources
    r=cfg['resources'];lease=c.read(r['joint_admission_path'])
    c.require(c.sha(r['joint_admission_path'])==r['joint_admission_sha256'],'JOINT_ADMISSION_SHA')
    c.require(lease['SOURCE']==cfg['SOURCE'] and lease['runtime_caps'] is False,'OWN_JOINT_LEASE')
    system=cfg['stage'][:2];allocation=lease['workers'][system]
    c.require(type(allocation) is int and allocation>=1,'FRESH_JOINT_WORKER_SHARE')
    probe=dict(cfg,systems={'H4':{},'H6':{}},resources=dict(
      system_RSS_proxy_bytes={'H4':1,'H6':1},disk_proxy_bytes=r['disk_proxy_bytes']))
    obs=fresh_resources(probe)
    count=repair.worker_count(len(cfg['tasks']),obs['idle_CPU_equivalents'],
       obs['effective_available_RAM_bytes'],r['worker_RSS_proxy_bytes'],historical_allocation=allocation)
    fs=os.statvfs(Path(cfg['run_root']).parent)
    c.require(fs.f_favail>=r['inode_proxy'],'FRESH_EXPLORATION_INODES')
    return dict(workers=count,observation=obs,worker_RSS_proxy_bytes=r['worker_RSS_proxy_bytes'],
        joint_workers=lease['workers'],storage_peak_proxy_bytes=r['disk_proxy_bytes'],
        inode_free=fs.f_favail,runtime_caps=False,proxy_is_upper_bound=False,
        IO_PSI_observation_only=Path('/proc/pressure/io').read_text())

def worker(task):
    return RUNTIME.native.run_task(RUNTIME.runner_namespace['WORKER_CFG'],task,
                                   RUNTIME.runner_namespace['WORKER_STOP'])

def runtime():
    """Lazy import; invalid drafts are refused before this or molecular IO."""
    import h46_pr_pilot_native as old_native
    import run_h46_pr_design_pilot as old_runner
    import audit_h46_pr_pilot_execution as old_audit
    native,ns=rebind(old_native,dict(c=c,enabled=enabled,CACHE={},INPUT_CACHE=repair.BoundJSON()))
    runner,rns=rebind(old_runner,dict(c=c,__file__=str(Path(__file__).with_name('run_h46_pr_exploration.py')),
       repair=SimpleNamespace(enabled=enabled,prewarm=prewarm),fresh=fresh,worker=worker))
    # execute's default must be a top-level importable function for the pool.
    runner.execute.__kwdefaults__={'target':worker}
    # The imported supervisor is a previously rebound function, not defined in
    # old_runner. Bind its observer globals explicitly to the new CLI/contract.
    runner.supervise=FunctionType(old_runner.supervise.__code__,dict(old_runner.supervise.__globals__,
       c=c,__file__=str(Path(__file__).with_name('run_h46_pr_exploration.py'))),
       'supervise',old_runner.supervise.__defaults__,old_runner.supervise.__closure__)
    auditor,ans=rebind(old_audit,dict(c=c))
    return SimpleNamespace(native=native,native_namespace=ns,runner=runner,runner_namespace=rns,
                           auditor=auditor,audit_namespace=ans)

RUNTIME=None

def get_runtime():
    global RUNTIME
    if RUNTIME is None:RUNTIME=runtime()
    return RUNTIME
