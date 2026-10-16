"""V1 saved-only five-precision reaccounting; sealed one-shot execution only.

No molecular provider, circuit, scalar overlay, fit, sampler or compiler is
imported. Parsing is indexed once; all scopes share each shot calculation.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import gzip
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import track_a_v1_runner as r
import track_a_v1_costs as costs

c=r.contract
ROOT=r.ROOT
SCOPES=('old_grid_S2','fine_grid_S2','fine_grid_S2_plus_truncated_S4')
BIAS_RULES=('signed_signal','empirical_no_cancellation')
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'


def plan():
    return dict(schema='track_a_v1_reaccount_plan_v1',scope='saved_cost_reaccounting',
        phase_a_result_commit='bf865de1efae51ce1663c878ad1e9daa5bb915c2',
        phase_b_result_commit='66caef169940320eb263d1e6fd8e87d1068d73f2',
        frozen_model_sha256='406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211',
        target=c.plan()['target'],compiler_hash=COMPILER,primary=c.plan()['primary'],
        epsilons=list(c.EPSILONS),allowance_factors=list(c.FACTORS),scopes=list(SCOPES),
        bias_rules=list(BIAS_RULES),conservative_certificate=None,
        domain=dict(old_deterministic=32,fine_S2_deterministic=96,with_S4_deterministic=136,
                    retained_PR_exploration=72,primary_PR_confirmation=8,
                    full_S2_and_full_S4_included_in_every_scope=True),
        cost_policy='saved measured axis first, including four Phase B axes; frozen model for other eligible additional axes',
        selection='exact rational weighted RZ, then lexical cell_id; no tolerance ties',
        missing_eligible_cost='STOP',missing_signed_decomposition='UNRESOLVED; no full-domain diagnostic minimum',
        call_caps=dict(reaccounting_pass=1,accounting_evaluations=4160,model_axis_estimates=208,
            new_signal=0,new_scalar=0,new_primitive=0,new_compile=0,new_fit=0,new_sampling=0,V2=0),
        resources=dict(processes=2,max_workers=0,threads=1,AS_bytes=2*2**30,RSS_bytes=2**30,
            observer_AS_bytes=512*2**20,observer_RSS_bytes=512*2**20,
            aggregate_RSS_bytes=1536*2**20,CPU_seconds=300,aggregate_CPU_seconds=300,
            wall_seconds=600,output_bytes=64*2**20,stage_storage_bytes=256*2**20),
        parallelism='one indexed stdlib arithmetic process plus concurrent external observer; no duplicated per-scope accounting',
        refit=False,one_shot=True,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False)


def index_unique(rows,key):
    result={}
    for row in rows:
        k=key(row)
        if k in result:raise ValueError('DUPLICATE_INPUT_ROW:'+str(k))
        result[k]=row
    return result


def pairs_by_cell(rows,phase):
    ids={x['cell_id'] for x in rows}
    return {cid:r.accounting.paired_rows(rows,cell_id=cid,phase=phase,compiler_hash=COMPILER) for cid in ids}


def measured_axes(pairs,kind):
    return [dict(cell_id=cid,axis=axis,RZ=str(Fraction(sum(p[axis]['RZ'] for p in samples),len(samples))),
        status='MEASURED_COMPLETE',measured=True,source_kind=kind,sample_count=len(samples))
        for cid,samples in pairs.items() for axis in c.AXES]


def eligible_state(accounting):
    if accounting is None:return None
    if accounting['cost_acquisition_eligible']:return True
    if accounting['eligibility_under_declared_allowance']=='INELIGIBLE':return False
    return None


def value_row(record,accounting,axis_index,*,epsilon,factor,bias_rule):
    cid=record['cell']['id'];usable=eligible_state(accounting)
    row=dict(cell_id=cid,cell=record['cell'],method=record['cell']['method'],epsilon=epsilon,factor=factor,
        bias_rule=bias_rule,eligible=usable,accounting=accounting,G_exact=None,G_point=None,
        conservative_bias_certificate=None,population_winner_certified=False)
    if usable is None:
        row['cost_status']='UNRESOLVED_SIGNED_DECOMPOSITION' if accounting is None else 'NUMERICALLY_UNRESOLVED'
        return row
    if not usable:row['cost_status']='NOT_NEEDED_INELIGIBLE';return row
    selected=[axis_index.get((cid,axis)) for axis in c.AXES]
    if any(x is None for x in selected):raise ValueError('ELIGIBLE_COST_MISSING:'+cid)
    if any(x['status'] not in ('MEASURED_COMPLETE','VALIDATED_MODEL_ESTIMATE') for x in selected):
        raise ValueError('UNVALIDATED_COST:'+cid)
    total=sum(Fraction(x['RZ'])*accounting['axes'][channel]['shots']
              for x,channel in zip(selected,('real','imag')))
    if total<=0:raise ValueError('NONPOSITIVE_WEIGHTED_COST')
    row.update(G_exact=str(total),G_point=float(total),
        cost_status='VALIDATED_MODEL_ESTIMATE' if any(x['status']=='VALIDATED_MODEL_ESTIMATE' for x in selected) else 'MEASURED_COMPLETE',
        axis_costs=selected,N_total=accounting['N_total'],selection_is_point_estimate=True)
    return row


def choose(rows):
    """Missing eligibility prohibits a full-domain minimum, even with valid peers."""
    unresolved=sum(x['eligible'] is None for x in rows)
    eligible=[x for x in rows if x['eligible'] is True]
    if unresolved:return dict(status='UNRESOLVED_FULL_DOMAIN',unresolved_candidates=unresolved,best=None)
    if not eligible:return dict(status='NO_ELIGIBLE_CANDIDATE',unresolved_candidates=0,best=None)
    if any(x['G_exact'] is None for x in eligible):raise ValueError('ELIGIBLE_COST_MISSING')
    best=min(eligible,key=lambda x:(Fraction(x['G_exact']),x['cell_id']))
    return dict(status='POINT_MINIMUM',unresolved_candidates=0,best={k:best[k] for k in
        ('cell_id','G_exact','G_point','cost_status','N_total')},
        exact_ties=sorted(x['cell_id'] for x in eligible if Fraction(x['G_exact'])==Fraction(best['G_exact'])))


def scope_ids(scope):
    reg=c.registrations()
    old={x['id'] for x in reg['existing'] if x['method']!='B2'}
    extra={x['id'] for x in reg['additional'] if scope==SCOPES[2] or (scope==SCOPES[1] and x['formula']=='2nd')}
    return old|extra if scope!=SCOPES[0] else old


def comparison(det,pr):
    d=choose(det);p=choose(pr);ratio=None;reduction=None
    if d['best'] and p['best']:
        ratio=Fraction(p['best']['G_exact'])/Fraction(d['best']['G_exact']);reduction=100*(1-ratio)
    return dict(deterministic=d,confirmed_PR=p,PR_over_deterministic_exact=str(ratio) if ratio is not None else None,
        PR_over_deterministic=float(ratio) if ratio is not None else None,
        PR_reduction_percentage_points=float(reduction) if reduction is not None else None,
        PR_reduction_percentage_points_exact=str(reduction) if reduction is not None else None,
        point_only=True,population_winner_certified=False)


def evaluate(data,writer,guard=lambda:None):
    """Exactly one saved-data pass. Caller must validate authorization first."""
    frozen=data['frozen'];validation=data['validation']
    if (data['phase_b_terminal'].get('status')!='COMPLETE_MANDATORY_STOP' or
            data['phase_b_terminal'].get('exit_code')!=0 or
            data['phase_b_launcher'].get('status')!='COMPLETE_MANDATORY_STOP' or
            data['phase_b_launcher'].get('native_exit_code')!=0 or
            data['phase_b_launcher'].get('cleanup_errors')!=[] or
            data['phase_b_launcher'].get('all_reaped') is not True or
            data['phase_b_launcher'].get('first_stop') is not None or
            data['phase_a_terminal'].get('status')!='COMPLETE_MANDATORY_STOP' or
            data['phase_a_terminal'].get('exit_code')!=0):raise ValueError('COMPLETE_PREVIOUS_PHASES_REQUIRED')
    if frozen.get('fitted_after_validation') is not False:raise ValueError('REFITTED_MODEL_FORBIDDEN')
    reg=c.registrations();old_records=data['old_signals']['signals']+data['b0i_signals']
    additional=data['additional']['signals']
    records=index_unique(old_records+additional,lambda x:x['cell']['id'])
    if set(records)!={x['id'] for x in reg['existing']+reg['additional']}:raise ValueError('REGISTERED_SIGNAL_DOMAIN')
    if len(old_records)!=104 or len(additional)!=104:raise ValueError('SIGNAL_COUNTS')
    if data['additional'].get('status')!='COMPLETE_MANDATORY_STOP':raise ValueError('ADDITIONAL_SIGNAL_INCOMPLETE')
    estimate=costs.validated_additional_cost_rows(frozen,validation['rows'],additional,contract=c,
                                               frozen_model_sha256=plan()['frozen_model_sha256'])
    if len(estimate)>208:raise ValueError('MODEL_AXIS_ESTIMATE_CAP')
    exploration=pairs_by_cell(data['exploration_wrappers'],'exploration')
    confirmed=pairs_by_cell(data['confirmation_wrappers'],'confirmation')
    confirmed_ids={x['cell_id'] for x in data['old_accounting']['H6_CONFIRMATION_UNION8'] if x['method']=='B2'}
    if set(confirmed)!=confirmed_ids or len(confirmed)!=8 or any(len(x)!=32 for x in confirmed.values()):
        raise ValueError('CONFIRMATION_UNION8_N32_REQUIRED')
    if any(len(x)!=8 for cid,x in exploration.items() if records[cid]['cell']['method']=='B2'):
        raise ValueError('EXPLORATION_N8_REQUIRED')
    old_axes=measured_axes(exploration,'SAVED_EXPLORATION_COMPILED_WRAPPERS')
    for x in data['b0i_axes']:
        if x['compiler_hash']!=COMPILER or x['native_exit_code']!='0':raise ValueError('B0I_COMPILE_IDENTITY')
        cid=f"B0I_p{x['prefix']}_q{x['q']}_2nd"
        old_axes.append(dict(cell_id=cid,axis=x['axis'],RZ=x['RZ'],status='MEASURED_COMPLETE',measured=True,
            source_kind=x['acquisition'],origin_SOURCE=x['origin_SOURCE'],sample_count=1))
    axis_index=index_unique(old_axes+estimate,lambda x:(x['cell_id'],x['axis']))
    # Measured validation cells take precedence without changing the frozen fit.
    for x in validation['rows']:
        key=(x['cell_id'],x['axis'])
        if key not in axis_index:raise ValueError('VALIDATION_OUTSIDE_ADDITIONAL_DOMAIN')
        axis_index[key]=dict(cell_id=x['cell_id'],axis=x['axis'],RZ=str(x['measured']),status='MEASURED_COMPLETE',
            measured=True,source_kind='PHASE_B_REGISTERED_VALIDATION',sample_count=1,model_sha256=x['model_sha256'])
    confirmed_axis=index_unique(measured_axes(confirmed,'SAVED_CONFIRMATION_UNION8_N32'),lambda x:(x['cell_id'],x['axis']))
    additional_ids={x['cell']['id'] for x in additional}
    old_tables=index_unique(data['old_accounting']['H6_EXPLORATION92']+data['b0i_accounting'],
        lambda x:(x['cell_id'],x['epsilon'],float(x['factor'])))
    confirm_table=index_unique(data['old_accounting']['H6_CONFIRMATION_UNION8'],
        lambda x:(x['cell_id'],x['epsilon'],float(x['factor'])))
    accounting_rows=[];comparisons=[];explore_selections=[];checks=0;calls=0
    for epsilon in c.EPSILONS:
        for factor in c.FACTORS:
            for bias_rule in BIAS_RULES:
                current={}
                for cid,record in records.items():
                    guard();view=record
                    if bias_rule==BIAS_RULES[1]:
                        _,envelope=c.signal_envelopes(record)
                        if envelope is None:accounting=None
                        else:
                            view=deepcopy(record);view['error_decomposition']['total_signed']=envelope
                            accounting=r.accounting.accounting(view,epsilon,factor);calls+=1
                    else:accounting=r.accounting.accounting(record,epsilon,factor);calls+=1
                    row=value_row(record,accounting,axis_index,epsilon=epsilon,factor=factor,bias_rule=bias_rule)
                    if bias_rule==BIAS_RULES[0]:
                        previous=old_tables.get((cid,epsilon,factor))
                        if previous:
                            if accounting!=previous['accounting']:raise ValueError('OLD_ACCOUNTING_CHANGED:'+cid)
                            if row['eligible']:
                                saved=previous['metrics']['RZ'];g=Fraction(row['G_exact'])
                                exact=saved.get('G_exact')
                                expected=Fraction(int(exact['numerator']),int(exact['denominator'])) if exact else Fraction(saved['G_point'])
                                if g!=expected or float(g)!=saved['G_point']:raise ValueError('OLD_WEIGHTED_COST_CHANGED:'+cid)
                            checks+=1
                        if cid in additional_ids:
                            if accounting!=record['precision_rows'][str(epsilon)][str(factor)]:
                                raise ValueError('PHASE_A_ELIGIBILITY_CHANGED:'+cid)
                            checks+=1
                    current[cid]=row;accounting_rows.append(row)
                    if row['method']=='B2':
                        row.update(sample_scope='H6_EXPLORATION92_N8',paired_samples=8)
                        if row['eligible']:row['cost_status']='SAVED_PAIRED_SAMPLE_MEAN'
                pr=[]
                for cid in sorted(confirmed_ids):
                    source=current[cid];record=records[cid]
                    row=value_row(record,source['accounting'],confirmed_axis,epsilon=epsilon,factor=factor,bias_rule=bias_rule)
                    row.update(sample_scope='H6_CONFIRMATION_UNION8',paired_samples=32)
                    if row['eligible']:
                        row['cost_status']='SAVED_PAIRED_SAMPLE_MEAN'
                        shots={axis:row['accounting']['axes'][channel]['shots'] for axis,channel in r.accounting.AXIS_CHANNEL.items()}
                        metric=r.accounting.weighted_metric(confirmed[cid],shots,'RZ')
                        exact=metric['G_exact']
                        if Fraction(row['G_exact'])!=Fraction(int(exact['numerator']),int(exact['denominator'])):
                            raise ValueError('PR_PAIRED_WEIGHTED_COST')
                        row.update(G_standard_error=metric['G_standard_error'],uncertainty_scope='PAIRED_SAMPLE_SE_NOT_CI')
                    if bias_rule==BIAS_RULES[0]:
                        prev=confirm_table[(cid,epsilon,factor)]
                        if row['accounting']!=prev['accounting']:raise ValueError('CONFIRMED_ACCOUNTING_CHANGED')
                        if row['eligible'] and row['G_point']!=prev['metrics']['RZ']['G_point']:raise ValueError('CONFIRMED_COST_CHANGED')
                        checks+=1
                    pr.append(row);accounting_rows.append(row)
                for scope in SCOPES:
                    selected=[current[cid] for cid in sorted(scope_ids(scope))]
                    result=dict(epsilon=epsilon,factor=factor,bias_rule=bias_rule,scope=scope,**comparison(selected,pr))
                    families={'B0':lambda x:x['method']=='B0','B0I':lambda x:x['method']=='B0I',
                        'full_S2':lambda x:x['method']=='B1' and x['cell']['formula']=='2nd',
                        'full_S4':lambda x:x['method']=='B1' and x['cell']['formula']=='4th'}
                    result['family_minima']={name:choose([x for x in selected if predicate(x)]) for name,predicate in families.items()}
                    result['eligibility_counts']=dict(total=len(selected),eligible=sum(x['eligible'] is True for x in selected),
                        ineligible=sum(x['eligible'] is False for x in selected),unresolved=sum(x['eligible'] is None for x in selected))
                    if bias_rule==BIAS_RULES[0] and scope==SCOPES[0]:
                        prior=next(x for x in data['old_comparison']['comparisons'] if x['epsilon']==epsilon
                                   and x['factor']==factor and x['metric']=='RZ')
                        for name,prior_key in [('deterministic','best_registered_deterministic'),('confirmed_PR','confirmed_PR_union8')]:
                            best=result[name]['best'];expected=prior[prior_key]
                            if best['cell_id']!=expected['cell_id'] or best['G_point']!=expected['G_point']:
                                raise ValueError('PREVIOUS_BASELINE_NOT_REPRODUCED')
                        checks+=1
                    comparisons.append(result)
                if bias_rule==BIAS_RULES[0]:
                    explore_selections.append(dict(epsilon=epsilon,factor=factor,scope='EXPLORATION72_N8_SEPARATE',
                        selection=choose([x for x in current.values() if x['method']=='B2']),primary=False))
    if calls>plan()['call_caps']['accounting_evaluations']:raise ValueError('ACCOUNTING_CALL_CAP')
    triggers=[]
    for epsilon in c.EPSILONS:
        for factor in c.FACTORS:
            selected=[x for x in comparisons if x['epsilon']==epsilon and x['factor']==factor and x['bias_rule']==BIAS_RULES[0]]
            old,new=selected[0],selected[-1]
            change=new['PR_reduction_percentage_points']-old['PR_reduction_percentage_points']
            endpoint=epsilon in (c.EPSILONS[0],c.EPSILONS[-1])
            sign_flip=endpoint and (Fraction(old['PR_over_deterministic_exact'])<1)!=(Fraction(new['PR_over_deterministic_exact'])<1)
            triggers.append(dict(epsilon=epsilon,factor=factor,reduction_change_percentage_points=change,
                intermediate_change_ge_10pp=(not endpoint and abs(change)>=10),endpoint_advantage_sign_change=sign_flip))
    writer.write('candidate_accounting.json',dict(rows=accounting_rows))
    writer.write('comparisons.json',dict(rows=comparisons,report_triggers=triggers))
    writer.write('exploration_reference.json',dict(rows=explore_selections,primary=False,population_winner_certified=False))
    writer.write('axis_costs.json',dict(rows=list(axis_index.values())+list(confirmed_axis.values()),
        frozen_model_refit=False,measured_precedes_model=True))
    summary=dict(status='COMPLETE_MANDATORY_STOP',comparison_rows=len(comparisons),accounting_rows=len(accounting_rows),
        accounting_evaluations=calls,model_axis_estimates=len(estimate),saved_reproduction_checks=checks,
        new_signal=0,new_scalar=0,new_compile=0,new_fit=0,new_sampling=0,V2_executions=0,
        conservative_bias_certificate=None,next_stage_authorized=False,mandatory_stop=True)
    writer.write('result_summary.json',summary)
    return summary


def verify_grant(config,grant,fresh,source,config_sha,now=None):
    now=time.monotonic() if now is None else now
    if (grant.get('schema')!='track_a_v1_reaccount_grant_v1' or grant.get('approved_by_user') is not True
            or grant.get('runtime_authorization') is not True or grant.get('scope')!=plan()['scope']
            or grant.get('one_shot') is not True or grant.get('retry') is not False or grant.get('resume') is not False
            or grant.get('SOURCE')!=source or grant.get('config_sha256')!=config_sha
            or grant.get('plan_sha256')!=c.digest(plan()) or config.get('plan_sha256')!=c.digest(plan())
            or grant.get('resources')!=plan()['resources'] or grant.get('call_caps')!=plan()['call_caps']
            or grant.get('frozen_model_sha256')!=plan()['frozen_model_sha256']):raise ValueError('SEALED_REACCOUNT_GRANT_REQUIRED')
    cpus=[grant.get('driver_cpu'),grant.get('monitor_cpu')]
    if (grant.get('allowed_cpus')!=cpus or len(set(cpus))!=2 or any(type(x) is not int for x in cpus)
            or not set(cpus)<=os.sched_getaffinity(0)):raise ValueError('DISJOINT_IDLE_CPUS_REQUIRED')
    if (fresh.get('source')!=source or fresh.get('config_sha256')!=config_sha or fresh.get('allowed_cpus')!=cpus
            or type(fresh.get('monotonic_seconds')) not in (int,float)
            or not 0<=now-fresh['monotonic_seconds']<=30
            or any(fresh.get(k) is not True for k in ('profile_verified','resources_idle_verified',
                'protected_jobs_unchanged','external_RSS_monitor_ready'))):raise ValueError('FRESH_GATE_REQUIRED')


def verify_bytes(config,source,*,git_blobs=False):
    import hashlib
    for name,h in config['source_hashes'].items():
        if r.sha(ROOT/name)!=h or (git_blobs and hashlib.sha256(subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)).hexdigest()!=h):
            raise ValueError('SOURCE_CHANGED:'+name)
    for name,h in config['runtime_file_hashes'].items():
        if r.sha(name)!=h:raise ValueError('RUNTIME_CHANGED')
    for x in config['files']:
        p=Path(x['path'])
        if p.is_symlink() or p.stat().st_size!=x['bytes'] or r.sha(p)!=x['sha256']:raise ValueError('INPUT_CHANGED:'+x['logical_name'])


def load_data(config):
    data={}
    for key,path in config['input_mapping'].items():
        if key.endswith('_wrappers'):data[key]=r.accounting.parse_wrappers(Path(path).read_bytes())
        elif key=='b0i_axes':
            import csv
            data[key]=list(csv.DictReader(Path(path).open()))
        elif key=='additional':data[key]=r.accounting.strict_json(gzip.decompress(Path(path).read_bytes()))
        else:data[key]=r.read_json(path)
    return data


def execute(config,grant,source):
    verify_bytes(config,source)
    if Path(sys.executable).resolve()!=Path(config['python']).resolve():raise ValueError('PINNED_PYTHON_REQUIRED')
    os.sched_setaffinity(0,{grant['driver_cpu']})
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):
        if os.environ.get(k)!='1':raise ValueError('THREAD_ONE_ENV')
    output=Path(config['output']);lock=Path(config['lock'])
    for p in (output,lock):
        if not p.is_absolute() or p.resolve()!=p or not p.is_relative_to('/home/AbeHiromu/projects') or p.exists():raise ValueError('EXCLUSIVE_BOUND_PATH_REQUIRED')
    with lock.open('x') as f:f.write(source+'\n');f.flush();os.fsync(f.fileno())
    output.mkdir();limits=plan()['resources'];resource.setrlimit(resource.RLIMIT_AS,(limits['AS_bytes'],limits['AS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(limits['CPU_seconds'],limits['CPU_seconds']))
    start=time.monotonic();last=0
    def guard():
        nonlocal last
        now=time.monotonic()
        if now-start>limits['wall_seconds']:raise ValueError('WALL_CEILING')
        if now-last<.1:return
        last=now
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>limits['RSS_bytes']:raise ValueError('RSS_CEILING')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>limits['output_bytes']:raise ValueError('OUTPUT_CEILING')
    def stop(*args):raise ValueError('STOP_SIGNAL_OR_WALL')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signal.alarm(limits['wall_seconds']);writer=r.Writer(output,guard)
    try:
        writer.write('reservation.json',dict(reaccounting_pass=1,SOURCE=source,refund=False,retry=False))
        summary=evaluate(load_data(config),writer,guard)
        verify_bytes(config,source)
        writer.write('terminal.json',dict(status='COMPLETE_MANDATORY_STOP',exit_code=0,SOURCE=source,
            scope=plan()['scope'],wall_seconds=time.monotonic()-start,summary=summary,next_stage_authorized=False))
    except BaseException as exc:
        r.Writer(output).write('terminal.json',dict(status='STOP',SOURCE=source,error=str(exc),retry=False,next_stage_authorized=False))
        raise
    finally:signal.alarm(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',action='store_true');p.add_argument('--launch-approved',action='store_true');p.add_argument('--execute-approved',action='store_true')
    for name in ('config','grant','fresh-gate'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if not a.launch_approved and not a.execute_approved:
        if any((a.config,a.grant,a.fresh_gate)):p.error('sealed inputs need an explicit execution mode')
        print(json.dumps(plan(),indent=2));return
    if a.launch_approved==a.execute_approved or not all((a.config,a.grant,a.fresh_gate)):p.error('one mode and all sealed inputs required')
    config,grant,fresh=map(r.read_json,(a.config,a.grant,a.fresh_gate))
    # The launcher binds HEAD and Git blobs before supervision. The single
    # arithmetic driver checks those working bytes, without transient Git
    # children that would violate its one-process execution contract.
    source=grant['SOURCE'] if a.execute_approved else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    verify_grant(config,grant,fresh,source,r.sha(a.config))
    if a.execute_approved:return execute(config,grant,source)
    for name,h in config['source_hashes'].items():
        if r.sha(ROOT/name)!=h:raise ValueError('LAUNCHER_SOURCE_CHANGED')
    for path in (a.config,a.grant,a.fresh_gate,Path(config['control'])):
        if not path.is_absolute() or path.resolve()!=path or not path.is_relative_to('/home/AbeHiromu/projects'):raise ValueError('PRIVATE_ABSOLUTE_PATH')
    if any(Path(config[k]).exists() for k in ('output','lock','control')):raise ValueError('ONE_SHOT_ALREADY_SPENT')
    from launch_track_a_v1_phase_a import supervise
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):env[k]='1'
    command=[config['python'],'-B',str(Path(__file__).resolve()),'--execute-approved','--config',str(a.config),'--grant',str(a.grant),'--fresh-gate',str(a.fresh_gate)]
    old=os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0,{grant['monitor_cpu']})
        supervise(command,env=env,control=config['control'],output=config['output'],limits=plan()['resources'],allowed_cpus=grant['allowed_cpus'])
    finally:os.sched_setaffinity(0,old)


if __name__=='__main__':main()
