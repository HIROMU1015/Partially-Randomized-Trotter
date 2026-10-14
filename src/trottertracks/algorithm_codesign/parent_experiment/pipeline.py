"""Future authorized workflow. Registered inputs are read only by run_pipeline.

Old-price validation -> proxy/scale -> candidates+inventory freeze -> fresh
acquisition -> complete ledger. No registered computation at module import.
"""
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from .design import candidates,proposals,ledger,comparisons
from ..parent_allocation import RootAllocationGenerator,rotation_key,native_ir_g9,reweight_events
from ..parent_allocation_native import (tool_binding,canonical,legacy_spec,bridge_legacy_row,
    recheck_entry,checked_entry,native_accounting,operator_target,sha)
from ..return_aggregation import Interval
from ..rte_development.evidence import atomic_json,digest,dumps,safe_relative


def serial(value):
    if isinstance(value,Interval): return dict(lo=value.lo,hi=value.hi)
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def read_bound(root,entry):
    path=Path(root)/safe_relative(entry['path'])
    if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
        raise PermissionError('saved input raw identity: '+entry['path'])
    return json.loads(path.read_text())


def expected(ratio,tool,c):
    return rotation_key({'ratio':F(ratio)},precision=F(c['primitive_error']),tool_identity=tool)


def normalize_saved(saved,exp,*,old,c,raw_hashes):
    spec=legacy_spec(exp)
    if old:
        provenance=dict(source_commit=c['reuse']['old_source_commit'],result_commit=c['reuse']['old_result_commit'],
            cache_sha256=raw_hashes['cache'],saved_row_sha256=sha(canonical(saved)),
            contract_sha256=raw_hashes['old_contract'],inventory_sha256=raw_hashes['inventory'],
            guard_audit_sha256=raw_hashes['guard_audit'],strict_guard_source_sha256=raw_hashes['strict_guard_source'],
            tool_identity_sha256=tool_sha(exp),runtime_executable_sha256=exp['spec']['tool']['runtime_executable_sha256'],
            inventory_spec=spec,inventory_key=sha(canonical(spec)),guard_policy='STRICT_RTE_RZ_FROBENIUS_NO_PHASE_MINIMIZATION')
        return bridge_legacy_row(saved,exp,provenance=provenance)
    from ..rte_reallocation.numeric import validate_saved
    from ..rte_reallocation.native import Angle
    validate_saved(saved,Angle('atan',F(exp['spec']['ratio'])),F(c['primitive_error']))
    # The backend executes the fixed strict guard before returning this saved row.
    # Rebind that certificate without evaluating the primitive a second time.
    row=dict(key=exp['key'],spec=exp['spec'],sequence=saved['sequence'],sequence_sha256=saved['sequence_sha256'],
             up_to_phase=False,operator_error_upper=saved['strict_operator_error_upper'],complete=True,
             T_count=saved['T_count'],Tdagger_count=saved['Tdagger_count'],one_qubit_count=saved['one_qubit_count'],
             global_W_count=saved['global_W_count'])
    return recheck_entry(row,exp,guard=lambda *_args,**_kwargs:F(saved['strict_operator_error_upper']))


def tool_sha(exp):return exp['spec']['tool']['tool_identity_sha256']


def freeze_inventory(configs,old_cache,tool,c):
    needed=set();baseline=set();root_new=set();root_count=0
    for cell in configs:
        for item in cell['bindings']:
            for gate in item['native_ir']:
                if gate[0]=='R':
                    ratio=str(F(gate[2]));needed.add(ratio)
                    if cell['candidate']=='common':baseline.add(ratio)
                    elif item.get('new_root'):root_new.add(ratio)
            root_count += bool(item.get('new_root'))
    if not baseline<=set(old_cache):raise ArithmeticError('required old common/reference key missing; no extra acquisition')
    new=needed-set(old_cache)
    if not new<=root_new or len(new)>c['caps']['new_synthesis_keys']:
        raise ArithmeticError('fresh non-root acquisition or new root key cap')
    if root_count>c['caps']['new_root_events']:raise RuntimeError('root matrix scope cap')
    keys=[dict(**expected(ratio,tool,c),ratio=ratio,
               acquisition='REUSE_ONCE_STRICT_GUARD' if ratio in old_cache else 'NEW_ONCE')
          for ratio in sorted(needed,key=F)]
    old_count=len(needed&set(old_cache))
    if old_count>c['caps']['old_unique_guards'] or len(keys)>c['caps']['all_unique_primitive_guards']:
        raise RuntimeError('primitive guard inventory cap')
    return dict(keys=keys,used_old_keys=old_count,new_keys=len(new),new_root_events=root_count,
                planned_root_matrix_checks=2*root_count,complete_before_new_prices=True,
                positive_inverse_sequence_shared=True)


def acquire(root,c,identity,inventory,writer,guard):
    jobs=writer.directory/'jobs';jobs.mkdir();fresh=[k for k in inventory['keys'] if k['acquisition']=='NEW_ONCE']
    cache={};active={};receipts=[];next_index=0
    try:
        while next_index<len(fresh) or active:
            guard.check()
            while next_index<len(fresh) and len(active)<c['caps']['workers']:
                key=fresh[next_index];index=next_index;next_index+=1
                job=dict(**identity,key=key['key'],ratio=key['ratio'],parent_pid=os.getpid(),job_index=index)
                path=jobs/f'key_{index:03d}.json';atomic_json(path,job)
                out=path.with_suffix('.stdout.log').open('xb');err=path.with_suffix('.stderr.log').open('xb')
                env=dict(os.environ,PYTHONPATH=str(Path(root)/'src'),PYTHONDONTWRITEBYTECODE='1',
                         OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                child=subprocess.Popen([c['acquisition']['runtime_executable'],'-B',c['execution']['worker'],'--job',str(path)],
                                       cwd=root,env=env,stdout=out,stderr=err)
                active[child.pid]=(child,path,job,out,err);guard.children.add(child.pid)
            for pid,(child,path,job,out,err) in list(active.items()):
                waited,status,usage=os.wait4(pid,os.WNOHANG)
                if not waited:continue
                child.returncode=os.waitstatus_to_exitcode(status);guard.reaped_cpu+=usage.ru_utime+usage.ru_stime
                guard.children.discard(pid);out.close();err.close();del active[pid]
                result=path.with_suffix('.result.json')
                receipt=dict(key=job['key'],pid=pid,exit_code=child.returncode,CPU_seconds=usage.ru_utime+usage.ru_stime,
                             peak_RSS_KiB=usage.ru_maxrss,result_sha256=digest(result) if result.exists() else None)
                atomic_json(path.with_suffix('.process_receipt.json'),receipt);receipts.append(receipt)
                if child.returncode!=0 or not result.exists():raise RuntimeError('fresh acquisition failed; no retry')
                if usage.ru_maxrss>c['caps']['worker_RSS_MiB']*1024:raise RuntimeError('worker observed peak RSS cap; no retry')
                v=json.loads(result.read_text())
                if (v.get('status')!='KEY_COMPLETE' or v.get('synthesis_calls')!=1 or v.get('retries')!=0
                    or any(v.get(k)!=val for k,val in job.items())):raise PermissionError('worker completion binding')
                cache[job['ratio']]=v['saved']
                writer.write(f'raw_new_cache/key_{job["job_index"]:03d}.json',v)
                if len(dumps(cache).encode())>c['caps']['cache_bytes']:raise RuntimeError('cache byte cap')
            if active:time.sleep(.05)
    finally:
        for child,_,_,_,_ in active.values():
            if child.poll() is None:child.terminate()
        for pid,(child,path,job,out,err) in active.items():
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            out.close();err.close();guard.children.discard(pid)
            atomic_json(path.with_suffix('.process_receipt.json'),dict(key=job['key'],pid=pid,exit_code=child.returncode,
                terminated_due_to_peer_or_parent_failure=True,retries=0))
    if len(cache)!=len(fresh):raise ArithmeticError('incomplete fresh inventory')
    return cache,receipts


def root_validation(event,gates,price,ref,tolerance):
    """Two independent 16x16 diagnostics; analytic bias uses primitive bound."""
    import numpy as np
    from ..rte_development.numeric import controlled,strict_error
    operators=ref.operators;n=len(operators[0])
    u=operator_target(event,identity=lambda:np.eye(n,dtype=complex),multiply=lambda a,b:a@b,
        involution=lambda i:operators[i],scale=lambda a,b:a*b,
        rotation=lambda i,r,s:np.cos(np.arctan(float(r)))*np.eye(n)-1j*s*np.sin(np.arctan(float(r)))*operators[i])
    wanted=controlled(u)
    ideal=strict_error(ref.actual(gates,False),wanted);actual=strict_error(ref.actual(gates,True),wanted)
    if ideal>tolerance or actual>float(price['strict_native_error_upper'])+tolerance:
        raise ArithmeticError('new root controlled phase/order diagnostic failed')
    return dict(ideal_error_diagnostic=ideal,actual_error_diagnostic=actual,
                independent_target=True,up_to_phase=False,bias_certificate=False)


def validate_common(saved,new):
    if len(saved)!=len(new):raise ArithmeticError('old common event coverage mismatch')
    for item,e in zip(saved,new):
        old=item['event']
        for k in ('word','child','ratio','phase_i_power','coefficient','proposal','weight'):
            left,right=old[k],e[k]
            if k=='word':left,right=tuple(left),tuple(right)
            elif k in ('ratio','coefficient','proposal','weight'):left,right=F(left),F(right)
            if left!=right:raise ArithmeticError('old common event mismatch: '+k)


def run_pipeline(root,c,identity,writer,guard):
    root=Path(root);reuse=c['reuse']['inputs'];precision={k:(F(v) if k in ('eta','rho') else v) for k,v in c['precision'].items()}
    from ..synthesis_placement.wrapper_launch import verify_runtime
    verify_runtime(root,dict(tool_identity=c['acquisition']['tool_identity']))
    from ..rte_reallocation.numeric import configure
    configure(c['acquisition']['interval_dps'])
    maps=read_bound(root,reuse['resource_map']);old_cache=read_bound(root,reuse['cache']);old_inventory=read_bound(root,reuse['inventory'])
    old_contract=read_bound(root,reuse['old_contract']);read_bound(root,reuse['guard_audit'])
    if (old_contract['primitive_error']!=c['primitive_error'] or old_contract['synthesizer_options']!=c['acquisition']['options']
        or old_contract['interval_dps']!=c['acquisition']['interval_dps']):raise PermissionError('old cache acquisition policy mismatch')
    tool=tool_binding((root/c['acquisition']['tool_identity']['path']).read_bytes(),expected_tool_sha256=c['acquisition']['tool_identity']['sha256'],
        runtime_executable_sha256=c['acquisition']['runtime_executable_sha256'],options=c['acquisition']['options'],interval_dps=c['acquisition']['interval_dps'])
    hashes={k:v['sha256'] for k,v in reuse.items()};hashes['strict_guard_source']=c['reuse']['historical_guard_source']['sha256']
    sources={};baseline=[];cache={};guard_count=0;timing=[]
    for row in maps['rows']:
        if row['steps']!=1 or row['arm'] not in c['central_arms']+c['reference_arms']:continue
        saved=read_bound(root,next(v for v in reuse.values() if v['path'].endswith(row['events']['path'])))
        if len(saved)!=len({v['index'] for v in saved}):raise ArithmeticError('saved event index duplication')
        sources[row['arm']]=saved
        for item in saved:
            e=item['event']
            for k in ('coefficient','proposal','weight'):e[k]=F(e[k])
            if 'ratio' in e:e['ratio']=F(e['ratio'])
            for gate in item['native_ir']:
                if gate[0]!='R':continue
                ratio=str(F(gate[2]))
                if ratio in cache:continue
                if ratio not in old_cache:raise ArithmeticError('required old key missing')
                exp=expected(ratio,tool,c);record=next((k for k in old_inventory['keys'] if k['ratio']==ratio),None)
                if record is None or record['key']!=sha(canonical(legacy_spec(exp))):raise PermissionError('old inventory spec binding')
                entry=normalize_saved(old_cache[ratio],exp,old=True,c=c,raw_hashes=hashes)
                entry=recheck_entry(entry,exp);cache[ratio]=entry;guard_count+=1;guard.check()
                writer.write('old_guards/key_'+str(guard_count-1).zfill(3)+'.json',entry)
        prices=[native_accounting(item['native_ir'],cache) for item in saved]
        canonical_law=reweight_events([item['event'] for item in saved],**precision)
        baseline.append(dict(arm=row['arm'],canonical_expected_T=sum(e['proposal']*price['T'] for e,price in zip(canonical_law,prices))))
    if set(sources)!=set(c['central_arms']+c['reference_arms']):raise ArithmeticError('five baseline representations required')
    scale=max(v['canonical_expected_T'] for v in baseline)
    configs=[];freeze=[];p=tuple(map(F,c['input']['p']));x=F(c['input']['tau'])
    for arm in c['central_arms']:
        start=time.process_time()
        make=RootAllocationGenerator.ordinary if arm=='ordinary' else RootAllocationGenerator.closed_p5
        base=make(p,x,**precision);validate_common(sources[arm],list(base.reference_events()))
        proxy=[native_accounting(item['native_ir'],cache)['T'] for item in sources[arm] if not item['event']['raw_word']]
        if len(proxy)!=len(p):raise ArithmeticError('full common root proxy coverage')
        cand=candidates(base.root.a,base.root.b,proxy,c['candidates']['KKT']);freeze.append(dict(arm=arm,proxy_T=proxy,**cand))
        rest=[item for item in sources[arm] if item['event']['raw_word']]
        for v in cand['candidates']:
            generator=make(p,x,v['allocation'],**precision)
            # Reuse unchanged non-root binding, no new non-root circuit or matrix.
            root_bindings=[dict(index=i,event=generator.root_event(i),native_ir=native_ir_g9(generator.root_event(i)),new_root=True)
                           for i in range(len(generator.root.atoms))]
            saved_rest=[dict(item,new_root=False) for item in rest]
            error=(base.base.B.hi-base.base.B.lo-(base.root.mass.hi-base.root.mass.lo)
                   +generator.root.mass.hi-generator.root.mass.lo)/2
            if error<0:raise ArithmeticError('coefficient certificate subtraction')
            configs.append(dict(arm=arm,candidate=v['label'],bindings=root_bindings+saved_rest,coefficient_error_upper=error,
                                allocation=v['allocation'],normalizer_interval=generator.B))
        timing.append(dict(arm=arm,candidate_setup_CPU_seconds=time.process_time()-start))
    for arm in c['reference_arms']:
        configs.append(dict(arm=arm,candidate='common',bindings=[dict(item,new_root=False) for item in sources[arm]],
                            coefficient_error_upper=8*F(c['precision']['rho']),allocation=None,normalizer_interval=None))
    if any(len(v['bindings'])>c['caps']['events_per_configuration'] for v in configs):raise RuntimeError('per-configuration event cap')
    if len(configs)>c['caps']['configurations'] or sum(len(v['bindings']) for v in configs)>c['caps']['event_bindings']:
        raise RuntimeError('configuration/event scope cap')
    inventory=freeze_inventory(configs,old_cache,tool,c)
    cf=writer.write('candidate_freeze.json',serial(dict(candidates=freeze,scale=scale,baseline_T=baseline,
        configurations=[{k:v for k,v in cell.items() if k!='bindings'} for cell in configs],
        frozen_before_new_price=True,registration='known development fixture; not held-out')))
    iv=writer.write('inventory.json',inventory)
    identity=dict(identity,candidate_freeze_sha256=cf['sha256'],inventory_sha256=iv['sha256'])
    # Additional reused ratios from noncommon roots are guarded once, after freeze.
    for entry in inventory['keys']:
        ratio=entry['ratio']
        if ratio in cache or entry['acquisition']=='NEW_ONCE':continue
        cache[ratio]=recheck_entry(normalize_saved(old_cache[ratio],entry,old=True,c=c,raw_hashes=hashes),entry)
        guard_count+=1;writer.write('old_guards/key_'+str(guard_count-1).zfill(3)+'.json',cache[ratio]);guard.check()
    fresh,receipts=acquire(root,c,identity,inventory,writer,guard)
    for ratio,saved in fresh.items():cache[ratio]=normalize_saved(saved,expected(ratio,tool,c),old=False,c=c,raw_hashes=hashes)
    if set(cache)!=set(k['ratio'] for k in inventory['keys']):raise ArithmeticError('complete price/guard coverage')
    writer.write('guarded_cache.json',cache)
    # No comparison or final winner is emitted until all prices and root checks succeed.
    from ..rte_development.numeric import NativeReference
    ref=NativeReference(cache);root_checks=0
    for index,cell in enumerate(configs):
        for item in cell['bindings']:
            guard.check();price=native_accounting(item['native_ir'],cache);item['native_cost']=price
            if item['new_root']:
                item['new_validation']=root_validation(item['event'],item['native_ir'],price,ref,c['matrix_tolerance']);root_checks+=2
        if root_checks>c['caps']['new_root_matrix_checks']:raise RuntimeError('matrix check cap')
        cell['saved_binding']=writer.write(f'events/config_{index:02d}.json',serial(cell['bindings']))
    rows=[];dropped=[];law_count=0
    for index,cell in enumerate(configs):
        start=time.process_time();events=[v['event'] for v in cell['bindings']];prices=[v['native_cost'] for v in cell['bindings']]
        laws,omitted=proposals(events,[v['T'] for v in prices],scale,precision)
        dropped.append(dict(config=index,proposals=omitted))
        for law in laws:
            guard.check();law_count+=1
            if law_count>c['caps']['laws']:raise RuntimeError('law cap')
            info=ledger(law['events'],prices,cell['coefficient_error_upper'],c)
            row=dict(id=f'config_{index:02d}:{law["label"]}',arm=cell['arm'],candidate=cell['candidate'],proposal=law['label'],
                     kappa=law['kappa'],allocation=cell['allocation'],saved_events=cell['saved_binding'],
                     ledger=info,workspace_beyond_system=1,
                     law=writer.write(f'laws/law_{law_count-1:02d}.json',serial(law['events'])))
            writer.write(f'rows/row_{law_count-1:02d}.json',serial(row));rows.append(row)
        timing.append(dict(config=index,global_law_and_ledger_CPU_seconds=time.process_time()-start,online_sampling='NOT_MEASURED'))
    writer.write('resource_map.json',serial(dict(rows=rows,dropped=dropped,comparisons=comparisons(rows),C_star=scale,
        classification='PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW',classical_setup=timing,
        primitive_old_guards=guard_count,new_synthesis_calls=len(fresh),new_root_matrix_checks=root_checks,
        acquisition_process_receipts=receipts,quantum_shots_executed=0,retries=0,mandatory_STOP=True,
        shared_acquisition_is_not_selected_algorithm_setup_cost=True,scientific_adoption=False,future_science_authorized=False)))
    return dict(configurations=len(configs),laws=law_count,old_guards=guard_count,new_synthesis_calls=len(fresh),
                new_root_matrix_checks=root_checks,inventory_sha256=iv['sha256'],candidate_freeze_sha256=cf['sha256'])
