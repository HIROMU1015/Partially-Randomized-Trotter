"""Canonical cutoff arithmetic and opt-in, process-local pilot conveniences.

No molecular restore, RNG, native action or compiler runs on import. Old run
contracts remain at their original SOURCE; these options need a new contract.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

POLICY='H6_PR_PILOT_CANONICAL_CUTOFF_AND_SHARED_CONTEXT_V1'
CUTOFF_POLICY='SYMBOLIC_TAIL_LAMBDA_TIMES_PARENTHESIZED_SIGNED_STEP_V1'

def require(ok,reason):
    if not ok:raise ValueError(reason)

def canonical_tau(preparation,signed_time,rte_steps):
    """Exactly the same lambda and operation order as make_rte_config()."""
    require(type(rte_steps) is int and rte_steps>=1,'POSITIVE_REGISTERED_RTE_STEPS')
    lam=preparation.rte_preparation.symbolic_tail.lambda_r
    require(math.isfinite(float(lam)) and lam>0,'FINITE_POSITIVE_SYMBOLIC_LAMBDA')
    time=float(signed_time);require(math.isfinite(time),'FINITE_SIGNED_TIME')
    tau=lam*(time/rte_steps)
    require(math.isfinite(tau),'FINITE_CANONICAL_TAU')
    return tau

def enabled(cfg):
    resources=cfg.get('resources',{})
    value=resources.get('optimization_policy')
    if value is None:return False
    require(value==POLICY and cfg['stage']=='H6_DESIGN_PILOT','H6_ONLY_OPT_IN_REPAIR_POLICY')
    require(resources.get('process_start_method')=='fork'
      and resources.get('prewarm_readonly_contexts') is True
      and resources.get('worker_input_cache') is True
      and resources.get('worker_allocation_policy')=='FRESH_SINGLE_STAGE_CPU_RAM_V1',
      'EXACT_REPAIR_OPTIMIZATIONS')
    return True

def worker_count(tasks,idle_cpu,available_ram,proxy,*,historical_allocation=None):
    require(type(tasks) is int and tasks>=1 and type(proxy) is int and proxy>0,'TASKS_AND_RAM_PROXY')
    count=min(tasks,math.floor(idle_cpu),available_ram//proxy)
    if historical_allocation is not None:count=min(count,historical_allocation)
    require(count>=1,'CANNOT_ADMIT_ONE_PILOT_WORKER')
    return count

class BoundJSON:
    """Parse once, but check complete bytes/SHA on every lookup.

    No metadata-only acceptance or skipped cryptographic check. A selected
    entry is copied out, so callers cannot mutate the private cached value.
    """
    def __init__(self):self.cache={};self.parses=0
    def _load(self,cfg,path):
        path=Path(path);key=str(path.resolve())
        require(str(path)==key,'CANONICAL_CACHED_INPUT_PATH')
        records=[r for r in cfg['sealed_files'] if r['path']==key]
        require(len(records)==1,'UNIQUE_SEALED_CACHED_INPUT')
        seal=records[0];raw=path.read_bytes()
        require(len(raw)==seal['bytes'] and hashlib.sha256(raw).hexdigest()==seal['sha256'],
          'CACHED_INPUT_SHA_MISMATCH')
        identity=(cfg['SOURCE'],cfg['plan_sha256'],key,seal['sha256'])
        if identity not in self.cache:
            self.cache[identity]=json.loads(raw);self.parses+=1
        return self.cache[identity]
    def registry(self,cfg,path,task_id):
        return copy.deepcopy(self._load(cfg,path)[task_id])
    def signal(self,cfg,path,candidate_id):
        data=self._load(cfg,path)
        records=[r for r in data['records'] if r['candidate']['id']==candidate_id]
        require(len(records)==1,'ONE_CACHED_SIGNAL_CANDIDATE')
        return copy.deepcopy(records[0])

def prewarm(cfg,context):
    """Only a separately authorized runner calls this before forking."""
    require(enabled(cfg),'REPAIR_PREWARM_NOT_ADOPTED')
    keys=sorted({(t['candidate']['system'],t['candidate']['prefix']) for t in cfg['tasks']})
    require(all(s=='H6' for s,p in keys),'H6_CONTEXTS_ONLY')
    for system,prefix in keys:context(cfg,system,prefix)
    return dict(contexts=len(keys),keys=[list(k) for k in keys],
      policy=POLICY,RNG_calls=0,signal_evaluations=0,compile_calls=0,
      context_readonly_between_forked_workers=True)
