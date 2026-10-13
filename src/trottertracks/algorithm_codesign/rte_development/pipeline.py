"""Future gated acquisition and reference accounting; no calls at import time."""
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .accounting import step_bounds,whole_budget,validate_events,whole_cost,affine_relation
from .evidence import atomic_json,digest,dumps


def inventory_key(ratio,c):
    spec={'angle_kind':'atan','ratio':str(F(ratio)),'scale':'1',
          'epsilon':c['primitive_error'],'options':c['synthesizer_options'],
          'tool_identity_sha256':c['tool_identity']['sha256'],
          'relative_controlled_phase':'strict; signed inverse from actual adjoint'}
    return hashlib.sha256(dumps(spec).encode()).hexdigest()


def freeze_inventory(pending,old_cache,c):
    keys={str(e['ratio']) for row in pending for e in row['events'] if e['ratio']}
    # r=1 is a saved anchor, never a new acquisition opportunity.
    anchor={str(e['ratio']) for row in pending if row['steps']==1
            for e in row['events'] if e['ratio']}
    if not anchor<=set(old_cache):raise ArithmeticError('saved anchor key coverage incomplete')
    if len(old_cache)>c['caps']['saved_keys']:raise RuntimeError('saved key cap')
    keys |= set(old_cache)
    out=[]
    for ratio in sorted(keys,key=F):
        out.append({'key':inventory_key(ratio,c),'ratio':ratio,
                    'acquisition':'REUSE_FIXED_ANCHOR' if ratio in old_cache else 'NEW_ONCE',
                    'primitive_error':c['primitive_error']})
    if len(out)>c['caps']['cache_entries']:raise RuntimeError('inventory/cache cap')
    if sum(r['acquisition']=='NEW_ONCE' for r in out)>c['caps']['new_synthesis_keys']:
        raise RuntimeError('new key cap')
    return dict(keys=out,complete_before_acquisition=True,
                source_price_queried_before_freeze=False,registered_seed=c['synthesizer_options']['seed'])


def validate_worker_result(value,key,identity):
    if (value.get('status')!='KEY_COMPLETE' or value.get('key')!=key['key']
        or value.get('ratio')!=key['ratio'] or value.get('synthesis_calls')!=1
        or value.get('retries')!=0):raise ArithmeticError('worker did not complete its one key')
    for field,expected in identity.items():
        if value.get(field)!=expected:raise PermissionError('worker result provenance mismatch: '+field)
    return value['saved']


def collect_pending(c,guard):
    from ..g7_generator import make_generator
    from ..g7_reference import reference_events
    from ..g9_p5 import P5Closed
    from ..g9_native import cts_events
    p=tuple(map(F,c['input']['p'])); tau=F(c['input']['tau'])
    kwargs=dict(root_bits=c['root_bits'],probability_bits=c['probability_bits'],
                eta=F(c['eta']),rho=F(c['rho']))
    pending=[]; total=0; timing=[]
    for steps in c['steps']:
        x=tau/steps
        for arm in c['arms']:
            guard.check(); start=time.process_time()
            if arm=='matched_CTS':
                events,certificate=cts_events(p,x,H=c['probability_bits'],K=c['root_bits'],
                                              eta=F(c['eta']),rho=F(c['rho']))
                bounds=step_bounds(cts=events,c=c)
                construction='explicit I1 Pauli acquisition; same access offered to all arms'
            else:
                generator=(P5Closed(p,x,**kwargs) if arm=='closed_P5_full' else
                           make_generator(p,x,5,arm,**kwargs))
                bounds=step_bounds(generator=generator,c=c)
                # Plan before reference enumeration; never pass the table into it.
                budget=whole_budget(bounds,x,steps,c)
                events=list(generator.reference_events() if arm=='closed_P5_full'
                            else reference_events(generator))
                certificate=None; construction='production local/group constructor; reference-only enumeration'
            budget=whole_budget(bounds,x,steps,c)
            observed=validate_events(events,bounds)
            total+=len(events)
            if total>c['caps']['event_bindings']:raise RuntimeError('event binding cap')
            pending.append(dict(steps=steps,x=x,arm=arm,events=events,
                                bounds=bounds,budget=budget,reference_law=observed,
                                certificate=certificate))
            timing.append(dict(steps=steps,arm=arm,stage=construction,
                               CPU_seconds=time.process_time()-start,
                               production_reads_reference_table=False,
                               online_classical_sampler_timing_acquired=False))
    return pending,timing


def acquire(root,c,inventory,inventory_sha,old_cache,identity,guard):
    from ..rte_reallocation.native import Angle
    from ..rte_reallocation.numeric import validate_saved
    directory=Path(root)/c['result_directory']; jobs=directory/'jobs';jobs.mkdir()
    cache={}; fresh=[]
    for key in inventory['keys']:
        ratio=key['ratio']
        if key['acquisition']=='REUSE_FIXED_ANCHOR':
            validate_saved(old_cache[ratio],Angle('atan',F(ratio)),F(c['primitive_error']))
            cache[ratio]=old_cache[ratio]
        else:fresh.append(key)
    next_index=0; active={}; receipts=[]
    try:
        while next_index<len(fresh) or active:
            guard.check()
            while next_index<len(fresh) and len(active)<c['caps']['workers']:
                key=fresh[next_index]; index=next_index;next_index+=1
                job={'key':key['key'],'ratio':key['ratio'],'inventory_sha256':inventory_sha,**identity}
                path=jobs/f'key_{index:03d}.json'; atomic_json(path,job)
                out=(jobs/f'key_{index:03d}.stdout.log').open('xb')
                err=(jobs/f'key_{index:03d}.stderr.log').open('xb')
                env=dict(os.environ,PYTHONPATH=str(Path(root)/'src'),OPENBLAS_NUM_THREADS='1',
                         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                         PYTHONDONTWRITEBYTECODE='1')
                child=subprocess.Popen([c['runtime_executable'],'-B',
                    'scripts/tracks/algorithm_codesign/run_rte_development_synthesis_key.py',
                    '--job',str(path)],cwd=root,env=env,stdout=out,stderr=err)
                active[child.pid]=(child,path,key,out,err);guard.children.add(child.pid)
            for pid,(child,path,key,out,err) in list(active.items()):
                # wait4 preserves exact child CPU including startup and shutdown.
                waited,status,usage=os.wait4(pid,os.WNOHANG)
                if not waited:continue
                child.returncode=os.waitstatus_to_exitcode(status)
                guard.reaped_cpu+=usage.ru_utime+usage.ru_stime
                guard.children.remove(pid);out.close();err.close();del active[pid]
                if child.returncode!=0:raise RuntimeError('synthesis child failed; no retry')
                value=json.loads(path.with_suffix('.result.json').read_text())
                expected={'inventory_sha256':inventory_sha,**identity}
                saved=validate_worker_result(value,key,expected)
                validate_saved(saved,Angle('atan',F(key['ratio'])),F(c['primitive_error']))
                cache[key['ratio']]=saved
                receipts.append({'key':key['key'],'pid':pid,'exit_code':child.returncode,
                                 'CPU_seconds':usage.ru_utime+usage.ru_stime,
                                 'peak_RSS_KiB':usage.ru_maxrss,'result_sha256':digest(path.with_suffix('.result.json'))})
                if len(dumps(cache).encode())>c['caps']['cache_bytes']:raise RuntimeError('cache byte cap')
            if active:time.sleep(.05)
    finally:
        for pid,(child,path,key,out,err) in list(active.items()):
            if child.poll() is None:child.terminate()
        for pid,(child,path,key,out,err) in list(active.items()):
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            out.close();err.close();guard.children.discard(pid)
    return cache,receipts


def evaluate(pending,cache,c,writer,guard):
    import numpy as np
    from ..g9_native import native_ir,cost
    from .numeric import NativeReference,finite_target
    ref=NativeReference(cache);p=tuple(map(F,c['input']['p'])); rows=[]
    for index,cell in enumerate(pending):
        per={k:F(0) for k in ('T','CX','1Q')};worst={k:0 for k in per}
        mean=np.zeros((8,8),complex);bindings=[]
        for number,e in enumerate(cell['events']):
            guard.check();gates=native_ir(e);price=cost(gates,cache)
            u,diag=ref.evaluate(e,gates,price['strict_error_upper'],c['matrix_tolerance'])
            mean+=float(e['coefficient'])*u
            for k in per:per[k]+=e['proposal']*price[k];worst[k]=max(worst[k],price[k])
            bindings.append({'index':number,'event':e,'native_ir':gates,'native_cost':price,**diag})
        target=finite_target(ref.operators,p,cell['x'])
        difference=float(np.linalg.norm(mean-target,2))
        if difference>c['matrix_tolerance']:raise ArithmeticError('common finite mean diagnostic failed')
        r=cell['steps']; finite_product=np.linalg.matrix_power(target,r)
        actual_product=np.linalg.matrix_power(mean,r)
        product_residual=float(np.linalg.norm(actual_product-finite_product,2))
        if product_residual>c['matrix_tolerance']:raise ArithmeticError('finite product mean diagnostic failed')
        events_identity=writer.write(f'events/row_{index:02d}.json',bindings)
        totals=whole_cost(cell['reference_law']['acceptance'],per,r,cell['budget'],worst)
        row={'steps':r,'x':cell['x'],'degree':5,'arm':cell['arm'],'events':events_identity,
             'production_step_bounds':cell['bounds'],'budget':cell['budget'],
             'reference_law_diagnostic_only':cell['reference_law'],
             'per_step_trial_native':per,'worst_single_step_event_native':worst,**totals,
             'ideal_step_mean_residual_diagnostic':difference,
             'ideal_product_mean_residual_diagnostic':product_residual,
             'uniform_state_bias_bound':True,'state':'000','workspace_beyond_system':1,
             'all_path_enumeration':False,'whole_circuit_compile':False,
             'price_model':'common within-step literal lowering; no cross-step cancellation',
             'classical_online_per_trial_cost':'NOT_ACQUIRED; not treated as zero',
             'uses_true_signal_for_shots':False,'operator_reference_is_not_bias_certificate':True}
        writer.write(f'rows/row_{index:02d}.json',row);rows.append(row)
        del bindings,mean,target,actual_product,finite_product
    relations=[]
    for r in c['steps']:
        by={v['arm']:v for v in rows if v['steps']==r}
        for metric in ('T','CX','1Q'):
            left=by['closed_P5_full'];right=by['partial_return_tail']
            relation=affine_relation({'intercept':left['expected_native_intercept'][metric],
                                     'slope':left['prep_readout_slope']},
                                    {'intercept':right['expected_native_intercept'][metric],
                                     'slope':right['prep_readout_slope']})
            relations.append(dict(steps=r,metric=metric,left='closed_P5_full',right='partial_return_tail',**relation))
    return rows,relations


def run_pipeline(root,c,identity,writer,guard):
    from ..rte_reallocation.numeric import configure
    configure(c['interval_dps'])
    anchor=Path(root)/c['saved_anchor']['path']
    if digest(anchor)!=c['saved_anchor']['sha256']:raise PermissionError('saved anchor identity changed')
    with anchor.open() as stream:old=json.load(stream)
    if old.get('status')!='G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE':raise ArithmeticError('incomplete anchor')
    old_cache=old['synthesis_cache'];del old
    pending,timing=collect_pending(c,guard)
    inventory=freeze_inventory(pending,old_cache,c)
    inv=writer.write('inventory.json',inventory)
    cache,receipts=acquire(root,c,inventory,inv['sha256'],old_cache,identity,guard)
    del old_cache
    writer.write('synthesis_cache.json',cache)
    rows,relations=evaluate(pending,cache,c,writer,guard)
    writer.write('resource_map.json',{'rows':rows,'primary_relations':relations,
                 'classical_setup_timing':timing,'worker_receipts':receipts,
                 'classification':'D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW',
                 'quantum_shots_executed':0,'trajectory_sampling':0,
                 'new_synthesis_calls':len(receipts),'retries':0,'mandatory_STOP':True,
                 'method_adoption':False,'future_science_authorized':False})
    return dict(rows=len(rows),new_synthesis_calls=len(receipts),inventory_sha256=inv['sha256'])
