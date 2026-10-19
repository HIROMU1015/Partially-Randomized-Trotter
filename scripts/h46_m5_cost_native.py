"""Lazy cost adapter. Shared evolution/paired axes; no mean signal or shots."""
from pathlib import Path
import math
import resource
import time
import traceback
import h46_m5_cost_contract as c

CACHE={}

def context(cfg,system,prefix):
    from h46_m5_signal_adapter import base_context,prefix_context
    key=(cfg['SOURCE'],system,c.digest(cfg['systems'][system]['inputs']))
    if key not in CACHE:CACHE[key]=dict(base=base_context(cfg,system),prefixes={})
    entry=CACHE[key]
    if prefix not in entry['prefixes']:entry['prefixes'][prefix]=prefix_context(entry['base'],prefix)
    ctx=entry['prefixes'][prefix]
    expected=next(x for x in c.manifest()['candidate_table'] if x['candidate']['system']==system and x['candidate']['prefix']==prefix)['preparation_hash']
    c.require(ctx.cut.preparation_hash==expected,'COST_PREPARATION_MATCHES_SAVED_SIGNAL')
    return ctx

def gate(out,name,error,threshold):
    row=dict(check=name,error=float(error) if math.isfinite(error) else None,strictly_below=threshold,PASS=math.isfinite(error) and 0<=error<threshold,phase_alignment=False)
    c.save(Path(out)/(name+'.json'),row);c.require(row['PASS'],'NUMERIC_DISAGREEMENT:'+name);return row

def atomic_binary(path,writer):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    c.require(not path.exists() and not temporary.exists(),'UNUSED_QUARANTINE_ARTIFACT')
    with temporary.open('xb') as f:writer(f)
    temporary.rename(path);return dict(file=path.name,sha256=c.sha(path),bytes=path.stat().st_size)

def run_task(cfg,task,stop):
    c.authorize(cfg);c.consumed(cfg)
    c.require(task in cfg['tasks'] and not stop.is_set(),'REGISTERED_ACTIVE_OWN_COST_TASK')
    c.reserve(cfg,task,'TASK');out=Path(cfg['output'])/task['id'];out.mkdir(exist_ok=False)
    started=time.monotonic();row=task['candidate'];system=row['system']
    def checkpoint(phase):
        c.require(not stop.is_set(),'OWN_COST_STAGE_STOP')
        with (out/'progress.jsonl').open('a') as f:
            import json
            f.write(json.dumps(dict(phase=phase,elapsed=time.monotonic()-started))+'\n')
    try:
        checkpoint('restore_cached_context');ctx=context(cfg,system,row['prefix']);np=ctx.np
        import h46_m5_native as original
        from qiskit import QuantumCircuit,qpy
        from trottertracks.resource_applicability import ax2b_native_df_v5 as native
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        from trotterlib.df_partial_s2 import make_df_partial_s2_step_request
        from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
        from trotterlib.rte import CompilerSettings
        from trotterlib.rte_compiled_cost import transpile_and_measure_cost
        from trotterlib.df_trotter.circuit import simulate_statevector
        group=dict(row,policy=row['r_policy']);requests=();events=[];logB=0.
        if row['method']=='PR':
            c.reserve(cfg,task,'trajectories');seeds=c.read(cfg['registry_path'])[task['id']];counter=[0]
            def factory(*args,**kwargs):
                checkpoint('sample_registered_unconditional_piece');i=counter[0];c.reserve(cfg,task,'request_sampling_calls',i);counter[0]+=1
                return make_df_partial_s2_step_request(*args,**kwargs)
            requests=original.make_requests(group,ctx.cut,seeds,ctx.lambda_R,factory=factory,tolerance=_explicit_cutoff_tolerance,checkpoint=checkpoint)
            events=[[e.to_dict() for e in request.rte_occurrence.events] for request in requests]
            logB=math.fsum(len(r.rte_occurrence.events)*math.log(r.rte_distribution.exact_finite_distribution) for r in requests)
        stratum=c.design.event_stratum(events);event_digest=c.digest(events)
        c.save(out/'events_PRIVATE.json',dict(events=events,event_digest=event_digest,conditional=False,cohort='EXPLORATION' if row['method']=='PR' else 'DETERMINISTIC',**stratum))
        sig=next(x for x in c.read(c.ROOT/c.SIGNAL_RECORDS)['records'] if x['candidate']['id']==row['id'])
        checks=[gate(out,'saved_normalization_log_B',abs(logB-sig['signal']['log_B']),c.OTHER_TOLERANCE)]
        scalar=ctx.scalar+(ctx.tail_identity if row['method'] in ('PR','B0I') else 0.)
        prep=ctx.fullprep if row['method']=='FULL' else ctx.cut
        with use_fingerprint_reuse() as cache:
            for i in range(c.task_limits(task)['native_piece_compositions']):c.reserve(cfg,task,'native_piece_compositions',i)
            c.reserve(cfg,task,'native_evolution');checkpoint('native_build')
            evolution=original.compose(group,prep,requests,scalar,native,QuantumCircuit,cache.fingerprint)
            c.reserve(cfg,task,'independent_oracle_actions')
            if row['method']=='PR':expected=ctx.oracle.trajectory(requests)
            else:
                value=ctx.oracle.initial.copy()
                for step in original.c.core.schedule(row['formula'],row['q'],'PER_PIECE',1):
                    for spec in list(ctx.oracle.specs[:row['prefix']+1])+list(reversed(ctx.oracle.specs[:row['prefix']+1])):
                        value=spec[1]@(np.exp(-.5j*step['signed_time']*spec[0])*(spec[2]@value))
                value*=np.exp(-.8j*scalar);expected=np.zeros_like(ctx.qstate);expected[ctx.oracle.qindices]=value
            low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)];high=np.r_[np.zeros_like(ctx.qstate),ctx.qstate]
            c.reserve(cfg,task,'native_low_actions');checkpoint('native_low_control')
            checks.append(gate(out,'native_low',float(np.linalg.norm(simulate_statevector(evolution.circuit,low)-low)),c.OTHER_TOLERANCE))
            c.reserve(cfg,task,'native_high_actions');checkpoint('native_high_control');actual=simulate_statevector(evolution.circuit,high)
            checks.append(gate(out,'native_leakage',float(np.linalg.norm(actual[:len(ctx.qstate)])),c.OTHER_TOLERANCE))
            checks.append(gate(out,'independent_occupation_vector',float(np.linalg.norm(actual[len(ctx.qstate):]-expected)),c.OTHER_TOLERANCE))
            z=complex(np.vdot(ctx.qstate,expected));costs=[]
            for index,axis in enumerate(('cosine','sine')):
                checkpoint('wrapper_'+axis)
                wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=len(evolution.circuit.data)+4)
                fp=cache.fingerprint(wrapper);c.reserve(cfg,task,'uncompiled_wrapper_actions',index)
                before=simulate_statevector(wrapper.remove_final_measurements(inplace=False),low)
                observed=float(np.vdot(before[:len(ctx.qstate)],before[:len(ctx.qstate)]).real-np.vdot(before[len(ctx.qstate):],before[len(ctx.qstate):]).real)
                checks.append(gate(out,'hadamard_'+axis,abs(observed-(z.real if axis=='cosine' else z.imag)),c.OTHER_TOLERANCE))
                compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
                c.reserve(cfg,task,'compile_axes',index);checkpoint('compile_'+axis)
                cost=transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fp)
                c.require(cost.compiler_settings_hash==c.design.COMPILER_HASH and cost.actual_circuit_fingerprint==fp==cache.fingerprint(wrapper),'COST_COMPILER_WRAPPER_IDENTITY')
                metrics={k:getattr(cost,k) for k in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')}
                c.require(all(type(v) is int and v>=0 for v in metrics.values()) and metrics['rz_count']>0,'FINITE_INTEGER_COMPILED_METRICS')
                qdir=out/('quarantine_'+axis);qdir.mkdir()
                wrapper_file=atomic_binary(qdir/'wrapper.qpy',lambda f:qpy.dump(wrapper,f))
                compiled_file=atomic_binary(qdir/'compiled.qpy',lambda f:qpy.dump(cost.transpiled_circuit,f))
                c.save(qdir/'unvalidated_metrics.json',dict(status='UNVALIDATED_NOT_FOR_ACCOUNTING',metrics=metrics,wrapper_fingerprint=fp,compiler_settings_hash=cost.compiler_settings_hash,wrapper=wrapper_file,compiled=compiled_file))
                c.reserve(cfg,task,'compiled_wrapper_actions',index);checkpoint('compiled_action_'+axis)
                after=simulate_statevector(cost.transpiled_circuit.remove_final_measurements(inplace=False),low)
                atomic_binary(qdir/'vectors.npz',lambda f:np.savez(f,low=low,before=before,after=after))
                checks.append(gate(out,'compiled_vector_'+axis,float(np.linalg.norm(after-before)),c.COMPILE_TOLERANCE))
                receipt=dict(status='DIRECT_MEASURED_CT01_SINGLE_SAVED_INPUT_STATE',axis=axis,metrics=metrics,fingerprint=fp,compiler_settings_hash=cost.compiler_settings_hash,event_digest=event_digest,preparation_hash=prep.preparation_hash,native_policy=c.design.NATIVE_POLICY,compiled_SHA=compiled_file['sha256'])
                c.save(out/('cost_'+axis+'.json'),receipt);costs.append(receipt)
                del after,before,cost,wrapper
            checkpoint('final_identity');c.verify(cfg)
            result=dict(status='UNIT_COST_PASS_NOT_ACCOUNTING',task=task,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],stage=cfg['stage'],input_binding=cfg['systems'][system]['binding'],preparation_hash=prep.preparation_hash,
              scalar=scalar,log_B=logB,checks=checks,axis_costs=costs,events= stratum,event_digest=event_digest,seed_cohort='EXPLORATION' if row['method']=='PR' else None,
              conditional_sampling=False,paired_trajectory=True,cohort_pooling=False,shot_accounting=False,cost_ranking=False,strict_bias_certificate=None,
              wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mandatory_STOP=True)
            c.save(out/'result.json',result);c.save(out/'fingerprint_cache.json',cache.snapshot())
            return dict(id=task['id'],result_sha256=c.sha(out/'result.json'),peak_RSS_KiB=result['peak_RSS_KiB'])
    except BaseException as exc:
        stop.set();c.save(out/'STOP.json',dict(status='STOP',reason=str(exc),traceback=traceback.format_exc(),task=task['id'],SOURCE=cfg['SOURCE'],stage=cfg['stage'],retry=False,mandatory_STOP=True))
        raise
