"""Frozen native operations; only CONFIRMATION labels and new pack bindings.

No numerical tolerance, PF coefficient, RNG, schedule or gate policy changes.
Context sharing, parse caching and long-task-first dispatch reuse proven paths.
"""
import math
import os
from pathlib import Path
from types import CodeType,FunctionType,SimpleNamespace
import h246_confirmation_contract as c
import h46_pr_pilot_repair as repair

RUNTIME=None

def cohort_code(code):
    return code.replace(co_consts=tuple(cohort_code(v) if isinstance(v,CodeType)
        else c.d.COHORT if v=='DESIGN_PILOT' else v for v in code.co_consts))

def rebind(module,overrides):
    ns=dict(vars(module));ns.update(overrides)
    for name,fn in vars(module).items():
        if isinstance(fn,FunctionType) and fn.__module__==module.__name__:
            f=FunctionType(cohort_code(fn.__code__),ns,name,fn.__defaults__,fn.__closure__)
            f.__kwdefaults__=fn.__kwdefaults__;ns[name]=f
    ns.update(overrides);return SimpleNamespace(**ns),ns

def enabled(cfg):
    r=cfg['resources']
    c.require(cfg['stage'] in c.STAGES and r['process_start_method']=='fork'
      and r['prewarm_readonly_contexts'] and r['worker_input_cache'],'FORK_READONLY_CACHE')
    return True

def prewarm(cfg,unused):
    enabled(cfg)
    for system,prefix in sorted({(t['candidate']['system'],t['candidate']['prefix']) for t in cfg['tasks']}):
        context(cfg,system,prefix)
    return dict(contexts=1,RNG_calls=0,signal_evaluations=0,compile_calls=0,
      context_readonly_between_forked_workers=True)

def context(cfg,system,prefix):
    if system=='H2':
        import h2_outer32_runtime as h2
        return h2.context(cfg,system,prefix)
    return RUNTIME.h46_context(cfg,system,prefix)

def fresh(cfg):
    from run_h46_m5_signal import fresh_resources
    path=os.environ.get('H246_CONFIRMATION_LEASE')
    c.require(path,'OWN_CONTROLLER_FRESH_LEASE_REQUIRED')
    lease=c.read(path);system=cfg['stage'][:2]
    c.require(lease['configs'].get(cfg['stage'])==dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'])
      and lease['runtime_caps'] is False,'FRESH_LEASE_IDENTITY')
    allocation=lease['workers'][system]
    c.require(type(allocation) is int and allocation>=1,'FRESH_WORKER_SHARE')
    obs=fresh_resources(dict(run_root=cfg['run_root'],systems={system:{}},resources=dict(
      system_RSS_proxy_bytes={system:1},disk_proxy_bytes=lease['peak_growth_proxy_bytes'])))
    workers=repair.worker_count(len(cfg['tasks']),obs['idle_CPU_equivalents'],obs['effective_available_RAM_bytes'],
      cfg['resources']['worker_RSS_proxy_bytes'],historical_allocation=allocation)
    fs=os.statvfs(Path(cfg['run_root']).parent)
    c.require(fs.f_favail>=cfg['resources']['inode_proxy'],'FRESH_INODE_GATE')
    return dict(workers=workers,observation=obs,worker_RSS_proxy_bytes=cfg['resources']['worker_RSS_proxy_bytes'],
      joint_workers=lease['workers'],inode_free=fs.f_favail,runtime_caps=False,proxy_is_upper_bound=False,
      storage_peak_proxy_bytes=lease['peak_growth_proxy_bytes'],IO_PSI_observation_only=Path('/proc/pressure/io').read_text())

def worker(task):
    cfg=RUNTIME.runner_namespace['WORKER_CFG'];stop=RUNTIME.runner_namespace['WORKER_STOP']
    if cfg['stage']=='H2_CONFIRMATION':
        import h2_outer32_runtime as h2
        h2.ACTIVE_OUTPUT=Path(cfg['output'])/task['id'];h2.PHASE_INDEX=0
    result=RUNTIME.native.run_task(cfg,task,stop)
    if cfg['stage']=='H2_CONFIRMATION':c.require(h2.PHASE_INDEX==2,'H2_BOTH_UNCHANGED_GATE_STREAM_RECEIPTS')
    return result

def runtime(cfg):
    import h46_pr_pilot_native as old_native
    import run_h46_pr_design_pilot as old_runner
    import audit_h46_pr_pilot_execution as old_audit
    view=c.native_contract(cfg)
    native,ns=rebind(old_native,dict(c=view,enabled=enabled,CACHE={},INPUT_CACHE=repair.BoundJSON()))
    h46_context=native.context;ns['context']=context;native.context=context
    cli=str(Path(__file__).with_name('run_h246_confirmation.py'))
    runner,rns=rebind(old_runner,dict(c=view,__file__=cli,
      repair=SimpleNamespace(enabled=enabled,prewarm=prewarm),fresh=fresh,worker=worker))
    runner.execute.__kwdefaults__={'target':worker}
    runner.supervise=FunctionType(old_runner.supervise.__code__,dict(old_runner.supervise.__globals__,
      c=view,__file__=cli),'supervise',old_runner.supervise.__defaults__,old_runner.supervise.__closure__)
    audit,ans=rebind(old_audit,dict(c=view))
    return SimpleNamespace(stage=cfg['stage'],native=native,native_namespace=ns,h46_context=h46_context,
      runner=runner,runner_namespace=rns,auditor=audit,audit_namespace=ans)

def get_runtime(cfg):
    global RUNTIME
    if RUNTIME is None:RUNTIME=runtime(cfg)
    c.require(RUNTIME.stage==cfg['stage'],'ONE_STAGE_PER_PROCESS')
    return RUNTIME

def audit(cfg):
    r=get_runtime(cfg).auditor.audit(cfg)
    if cfg['stage']=='H2_CONFIRMATION' and not r['errors']:
        for task in cfg['tasks']:
            for i in range(2):
                v=c.read(Path(cfg['output'])/task['id']/('compiler_phase_guard_'+str(i)+'.json'))
                c.require(v['policy']==cfg['compiler_phase_policy'] and v['unchanged_gate_stream']
                  and not v['state_fit'],'H2_GATE_STREAM_GUARD_END_AUDIT')
        c.save(Path(cfg['run_root'])/'compiler_policy_end_audit.json',dict(status='PASS',
          policy=cfg['compiler_phase_policy'],axes=cfg['call_scope']['compile_axes'],
          end_audit_sha256=c.sha(Path(cfg['run_root'])/'end_audit.json')))
    return r
