"""Bounded byte/JSON-only posthoc projection; no scientific arrays or compiler imports."""
import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import time


def read(path):
    return json.loads(path.read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))


def stamp(path):
    raw=path.read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def put(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def table(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def counts(bias,B,epsilon):
    headroom=[epsilon/math.sqrt(2)-bias[axis] for axis in ('cosine','sine')]
    return None if min(headroom)<=0 else [math.ceil(2*B**2/h**2*math.log(80)) for h in headroom]


def exact_work(pairs,ns):
    return Fraction(sum(ns[0]*a+ns[1]*b for a,b in pairs),len(pairs))


def exact_se(pairs,ns):
    n=len(pairs)
    if n==1:return 0.
    sample=[ns[0]*a+ns[1]*b for a,b in pairs];total=sum(sample)
    var=Fraction(sum(v*v for v in sample),n*(n-1))-Fraction(total*total,n*n*(n-1))
    return math.sqrt(float(var))


def blocks(path):
    entries=[path,*path.rglob('*')] if path.is_dir() else [path]
    for item in entries:
        assert not item.is_symlink() and item.stat().st_uid==os.getuid()
    return dict(allocated_bytes=sum(item.stat().st_blocks*512 for item in entries),inodes=len(entries),logical_file_bytes=sum(item.stat().st_size for item in entries if item.is_file()))


def main():
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(60,60));signal.alarm(90)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--driver-native-exit-code',type=int,required=True);args=parser.parse_args()
    assert args.driver_native_exit_code==0
    bundle=Path(__file__).absolute().parent;repo=bundle.parents[3];plan=read(bundle/'plan.json')
    evidence=args.evidence;output=Path(plan['output']);control=Path(plan['control']);terminal=read(output/'terminal.json');launch=read(evidence/'launch_attempt.json');retire=read(output/'observer-retirement.json')
    assert terminal['status']=='PILOT6_COMPLETE_STOP' and terminal['cleanup_complete'] and terminal['records']==6
    assert terminal['actual_consumed_or_reserved']==6 and terminal['first_stop_reason'] is None
    assert terminal['native_worker_exit_code']==0 and terminal['native_observer_exit_code']==-15 and terminal['carry']==plan['carry']
    assert terminal['mandatory_stop'] and terminal['runtime_next_stage'] is False
    lock=read(Path(plan['one_shot_lock']));assert lock['pid']==launch['driver_pid'] and lock['plan_digest']==read(bundle/'binding.json')['plan_digest']
    assert read(output/'started.json')['source']==plan['source_commit']
    assert len(list(output.glob('reservation-*.json')))==len(list(output.glob('task-*.json')))==6
    native_files={str(path.relative_to(evidence)):stamp(path) for path in [*sorted(output.iterdir()),control/'observer.jsonl',Path(plan['one_shot_lock']),evidence/'driver.log',evidence/'launch_attempt.json'] if path.is_file()}
    obs=[json.loads(line) for line in (control/'observer.jsonl').read_text().splitlines()]
    assert obs and all(row['kind']=='observation' and row['first_failure'] is None for row in obs)
    driver=launch['driver_pid'];observer=retire['observer_identity']['pid'];worker_ids={p['pid'] for row in obs for p in row['processes'] if p['pid']!=driver};assert len(worker_ids)==1;worker=worker_ids.pop()
    identities={p['pid']:{k:p[k] for k in ('pid','start','parent','uid')} for row in obs for p in [*row['processes'],row['observer']]}
    assert identities[driver]['start']==launch['driver_start'] and identities[observer]==retire['observer_identity']
    assert identities[worker]['parent']==identities[observer]['parent']==driver
    residual=[]
    for pid,identity in identities.items():
        path=Path('/proc')/str(pid);same=False;state=None
        if path.exists():
            try:
                fields=(path/'stat').read_text().rsplit(')',1)[1].split();state=fields[0];same=fields[19]==identity['start']
            except FileNotFoundError:pass
        assert not same,'run identity remains after cleanup'
        residual.append(dict(identity=identity,same_run_identity_present=same,current_state=state))
    oom=obs[0]['memory']['oom_events'];assert all(row['memory']['oom_events']==oom for row in obs)
    assert all(0<=row[key]<=5 for row in obs for key in ('interval_seconds','observation_duration_seconds','staleness_seconds'))
    assert all(all(v==0 for v in row['memory']['psi_full_by_scope'].values()) for row in obs)
    peak={}
    for role,pid in [('driver',driver),('worker',worker),('observer',observer)]:
        samples=[p for row in obs for p in [*row['processes'],row['observer']] if p['pid']==pid]
        peak[role]=dict(pid=pid,max_sampled_RSS_bytes=max(v['rss'] for v in samples),max_sampled_AS_bytes=max(v['address_space'] for v in samples),samples=len(samples))
    start=obs[0]['monotonic']-obs[0]['wall_seconds'];end=start+terminal['stage_wall_seconds']
    phases={row['driver_phase']['name']:row['driver_phase'] for row in obs if row['driver_phase']['name'].startswith('compile_')}
    projection_path=repo/'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json'
    original={(v['geometry'],v['template_id']):v for v in read(projection_path)['candidates']}
    results=[];metrics_rows=[]
    for i,task in enumerate(plan['tasks']):
        result=read(output/('task-'+task['task_id']+'.json'));reservation=read(output/('reservation-%02d.json'%(i+1)))
        assert reservation['task_id']==task['task_id'] and reservation['call']==i+1
        assert result['task_id']==task['task_id'] and result['status']=='COMPLETE' and result['source_commit']==plan['source_commit']
        assert result['plan_digest']==lock['plan_digest'] and result['old_cache_reuse'] is False
        phase=phases['compile_'+task['geometry']+'_'+task['axis']]
        next_time=phases['compile_'+plan['tasks'][i+1]['geometry']+'_'+plan['tasks'][i+1]['axis']]['monotonic'] if i<5 else end
        samples=[p for row in obs if row['driver_phase']['name']==phase['name'] for p in row['processes'] if p['pid']==worker]
        reference=original[task['geometry'],task['original_B0_template']]['paired_metric_axis_means'][task['axis']]
        value={**result,'geometry_A':task['geometry'],'prefix':task['prefix'],'q':1,'axis':task['axis'],'original_B0_template':task['original_B0_template'],'tail_identity':task['tail_identity'],'phase_angle':task['phase_angle'],'input_identity':task['input_identity'],'input_file_byte_identity':{k:task['input_file'][k] for k in ('bytes','sha256')},'old_B0_reference_RZ':task['old_B0_reference_RZ'],'delta_RZ':result['metrics']['rz_count']-task['old_B0_reference_RZ'],'delta_metrics_vs_saved_B0':{k:result['metrics'][k]-reference[k] for k in result['metrics']},'phase_interval_seconds':next_time-phase['monotonic'],'phase_interval_kind':'begin-to-next-phase interval; includes preparation/hash/compile/metrics' if i<5 else 'begin-to-terminal upper bound; also includes shutdown/cleanup','transpile_only_wall_seconds':None,'phase_max_sampled_worker_RSS_bytes':max(p['rss'] for p in samples)}
        results.append(value);metrics_rows.append({k:value[k] for k in ('geometry_A','prefix','q','axis','old_B0_reference_RZ','delta_RZ','phase_interval_seconds','transpile_only_wall_seconds','phase_max_sampled_worker_RSS_bytes')}|result['metrics'])
    assert retire['complete_task_ids']==[t['task_id'] for t in plan['tasks']]
    put(bundle/'compiled_wrapper_metrics.json',{'schema_version':'h4-b0i-measured-six-wrappers-v1','source_commit':plan['source_commit'],'grant_commit':launch['grant_commit'],'plan_digest':lock['plan_digest'],'results':results})
    table(bundle/'compiled_wrapper_metrics.csv',metrics_rows)
    runtime_parts={name:blocks(path) for name,path in [('output',output),('control',control),('one_shot_lock',Path(plan['one_shot_lock'])),('driver_log',evidence/'driver.log'),('launch_receipt',evidence/'launch_attempt.json')]}
    allocated=sum(v['allocated_bytes'] for v in runtime_parts.values());inodes=sum(v['inodes'] for v in runtime_parts.values())
    carry=plan['carry'];budget=dict(historical_carry=carry,new_consumed_or_reserved=6,cumulative_actual_consumed_or_reserved=carry['actual_invocations']+6,actual_cap=carry['actual_invocation_cap'],remaining_actual=carry['actual_invocation_cap']-carry['actual_invocations']-6,new_runtime_allocated_bytes=allocated,charge_at_receipt_bytes=carry['charged_bytes']+allocated,charge_cap_bytes=carry['charge_cap_bytes'],conservative_charge_with_full_stage_cap=carry['charged_bytes']+plan['caps']['stage_output_bytes'],recorded_wall_seconds=carry['recorded_wall_seconds']+terminal['stage_wall_seconds'],wall_accounting='historical recorded wall plus measured stage; excludes pre-run binding and posthoc publication',failed_calls_refunded=False,publication_storage_separate=True,runtime_storage_parts=runtime_parts,runtime_storage_inodes=inodes)
    put(bundle/'native_resource_receipt.json',{'schema_version':'h4-b0i-native-resource-receipt-v1','terminal':terminal,'driver_native_exit_code':args.driver_native_exit_code,'driver_exit_evidence':'native return code 0 from execution session 1414; not inferred from terminal JSON','observer_retirement':retire,'run_residual_audit':residual,'observation_count':len(obs),'max_interval_seconds':max(v['interval_seconds'] for v in obs),'max_observation_duration_seconds':max(v['observation_duration_seconds'] for v in obs),'max_staleness_seconds':max(v['staleness_seconds'] for v in obs),'sampled_memory':peak,'RSS_scope':'sampled maxima; instantaneous peak RSS was not captured','min_effective_available_bytes':min(v['memory']['available'] for v in obs),'pressure_max_by_scope':{key:max(v['memory']['psi_full_by_scope'][key] for v in obs) for key in obs[0]['memory']['psi_full_by_scope']},'cgroup_OOM_counters_unchanged':oom,'host_OOM_baseline':read(output/'host-oom-baseline.json'),'first_failure':None,'native_evidence_files':native_files,'budget':budget})
    gridpath=repo/'artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/precision_grid_1812.csv'
    with gridpath.open(newline='') as stream:grid=list(csv.DictReader(stream))
    comparisons=[];selected=[];by_geometry={}
    for g in ('0.70','0.80','1.40'):
        tasks=[t for t in plan['tasks'] if t['geometry']==g];ms=[r for r in results if r['geometry_A']==g]
        assert [t['axis'] for t in tasks]==['cosine','sine']
        below=0;eligible=0;cross=[]
        for base in [row for row in grid if row['geometry']==g]:
            epsilon=float(base['epsilon']);ns=counts(tasks[0]['saved_bias_B0I'],1.,epsilon)
            b2=original[g,base['B2_template']];n2=counts(b2['saved']['bias'],b2['saved']['normalization'],epsilon);assert n2 is not None
            work2=exact_work(b2['saved']['paired_costs'],n2);se2=exact_se(b2['saved']['paired_costs'],n2)
            assert math.isclose(float(work2),float(base['B2_G']),rel_tol=1e-12) and math.isclose(se2,float(base['SE']),rel_tol=1e-10)
            work=sum(n*r['metrics']['rz_count'] for n,r in zip(ns,ms)) if ns else None
            ratio=work/float(work2) if ns else None
            row=dict(geometry_A=g,prefix=tasks[0]['prefix'],q=1,epsilon=epsilon,B0I_eligible=ns is not None,N_cos=None if ns is None else ns[0],N_sin=None if ns is None else ns[1],measured_RZ_cos=ms[0]['metrics']['rz_count'],measured_RZ_sin=ms[1]['metrics']['rz_count'],B0I_RZ_work=work,B2_point_min_template=base['B2_template'],B2_point_RZ_work=float(work2),B2_descriptive_SE=se2,B0I_to_B2_point_ratio=ratio,reduction_percent=None if ratio is None else 100*(1-ratio),B0I_below_B2_point=ns is not None and work<work2,B0I_below_B2_point_minus_2SE=ns is not None and work<float(work2)-2*se2)
            comparisons.append(row)
            if ns:eligible+=1
            if row['B0I_below_B2_point']:below+=1;cross.append(epsilon)
            if epsilon==.05:selected.append(row)
        by_geometry[g]=dict(fixed_prefix=tasks[0]['prefix'],q=1,grid_points=302,eligible_points=eligible,points_below_B2_point_min=below,epsilon_first_below_grid=min(cross) if cross else None,epsilon_last_below_grid=max(cross) if cross else None,scope='one measured configuration; neither full B0+I family minimum nor continuous-epsilon proof')
    assert len(comparisons)==906 and len(selected)==3
    table(bundle/'measured_fixed3_precision_906.csv',comparisons);table(bundle/'selected_epsilon_0p05.csv',selected)
    put(bundle/'result_summary.json',{'status':'H4_B0I_PILOT6_COMPLETE_AWAITING_SCIENTIFIC_RESULT_REVIEW','execution_status':terminal['status'],'client_date_JST':'2026-10-14','source_commit':plan['source_commit'],'grant_commit':launch['grant_commit'],'compile_calls':6,'all_six_delta_RZ_zero':all(r['delta_RZ']==0 for r in results),'stage_wall_seconds':terminal['stage_wall_seconds'],'selected_epsilon_0p05':selected,'fixed3_precision_projection':by_geometry,'budget':budget,'scientific_baseline_adopted_as_additional_target':True,'limits':['No unmeasured B0+I family optimum or 1812-point new winner map','B2 uses the same 32 paired trajectories for selection and cost estimation; descriptive SE is not population superiority certification','Bias/normalization/shot model retained; actual hardware shots were not executed','RZ count equality on six circuits does not establish other configurations or old-compiler output equivalence','No precise transpile-only wall or instantaneous peak RSS measured'],'next_stage_authorized':False,'mandatory_stop':True,'H8_or_shared_environment_changes':0,'new_SCF_DF_state_signal_trajectory':0})
    public_inputs=[projection_path,gridpath,repo/'artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13/saved_b0_phase_sensitivity_v1.json',repo/'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/verify_saved_result_projection.py']
    put(bundle/'result_provenance.json',{'source_commit':plan['source_commit'],'grant_commit':launch['grant_commit'],'native_runtime_root':str(evidence),'public_analysis_inputs':[dict(path=str(path.relative_to(repo)),**stamp(path)) for path in public_inputs],'native_evidence_files':native_files,'private_scientific_arrays_accessed_in_posthoc':False})
    print(json.dumps({'status':'COMPLETE_PROJECTED','rows':906,'selected':selected,'sampled_worker_RSS_MiB':peak['worker']['max_sampled_RSS_bytes']/2**20,'runtime_allocated_bytes':allocated,'budget':budget,'grid_summary':by_geometry},ensure_ascii=False))


if __name__=='__main__':main()
