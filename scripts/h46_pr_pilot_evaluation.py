"""Saved pilot evaluation only; no molecular/sampling/compiler imports."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal,localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import h46_pr_design_pilot as design

ROOT=Path(__file__).resolve().parents[1]
STAGES=('H4_PILOT_EVALUATION','H6_PILOT_EVALUATION')
TARGETS=(.05,.02,.01)
INFLATIONS=(1,2,4)
SOURCE_PATHS=('scripts/h46_pr_pilot_evaluation.py','scripts/prepare_h46_pr_pilot_evaluation.py',
 'scripts/run_h46_pr_pilot_evaluation.py','scripts/h46_pr_design_pilot.py',
 'scripts/h46_m5_cost_design.py','references/h46_pr_design_pilot_proposal_v1.json',
 'references/h46_m5_signal_domain_v1.json','references/h46_m5_cost_readiness_v1.json',
 'tests/test_h46_pr_pilot_evaluation.py')
require=design.require
digest=design.digest
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seal(p,role):
    p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),role=role)
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def mixture(laws,n):
    """Ideal independent-step probabilities from saved binary64 law weights.

    Decimal recurrence avoids cancellation of unobserved tiny probabilities.
    It does not bound conditional costs or model the finite PRNG state space.
    """
    with localcontext() as context:
        context.prec=80
        s0,s1,sm,s4=map(Decimal,(1,0,0,0));largest_delta=Decimal(0);events=0
        for row in laws:
            require(row['orders']==[0,2,4] and row['K']==4,'FIXED_EVEN_K4_LAW')
            r=row['steps'];require(type(r) is int and r>=1,'INTEGER_STEPS')
            values=row['probabilities']
            require(len(values)==3 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values),
              'FINITE_NONNEGATIVE_SAVED_PROBABILITIES')
            p0,p2,p4=[Decimal.from_float(float(v)) for v in values];total=p0+p2+p4
            require(total>0 and abs(total-1)<=Decimal.from_float(64*math.ulp(1.)),
              'SAVED_PROBABILITY_NORMALIZATION')
            largest_delta=max(largest_delta,abs(total-1));p0,p2,p4=p0/total,p2/total,p4/total
            for _ in range(r):
                s0,s1,sm,s4=s0*p0,s1*p0+s0*p2,sm*(p0+p2)+s1*p2,s4+(s0+s1+sm)*p4
            events+=r
        probabilities=dict(zip(design.STRATA,(s0,s1,sm,s4)))
        require(abs(sum(probabilities.values())-1)<Decimal('1e-65'),'MIXTURE_CLOSURE')
        return dict(probabilities={k:float(v) for k,v in probabilities.items()},
          ideal_expected_counts={k:float(n*v) for k,v in probabilities.items()},
          ideal_probability_at_least_one_in_n={k:float(1-(1-v)**n) for k,v in probabilities.items()},
          events=events,model='IDEAL_INDEPENDENT_STEPS_FROM_SAVED_BINARY64_WEIGHTS_RENORMALIZED_WITHIN_ROUNDOFF',
          largest_input_probability_sum_deviation=float(largest_delta),
          finite_PRNG_independence_proved=False,population_cost_bound=None,guarantee=False)

def describe(rows,laws):
    value=design.describe_pilot(rows,33,relative_MCSE_targets=TARGETS,variance_inflations=INFLATIONS)
    model=mixture(laws,33)
    value.update(relative_axis_MCSE={a:value['axis_MCSE'][a]/value['axis_means'][a] for a in ('cosine','sine')},
      relative_pair_unit_MCSE=value['pair_unit_RZ_MCSE']/value['pair_unit_RZ_mean'],
      pair_unit_min=min(r['cosine_RZ']+r['sine_RZ'] for r in rows),
      pair_unit_max=max(r['cosine_RZ']+r['sine_RZ'] for r in rows),ideal_stratum_law=model,
      all_axis_variances_observed_zero=all(value['covariance'][i][i]==0 for i in (0,1)))
    value['ideal_unobserved_strata_probability_mass']=sum(model['probabilities'][s]
      for s in design.STRATA if value['strata'][s]['observed_count']==0)
    # Report direct pair variance and its covariance identity with a numerical
    # verification tolerance, without dropping the covariance term.
    xx,xy=value['covariance'][0];_,yy=value['covariance'][1]
    error=abs(value['pair_unit_RZ_variance']-(xx+2*xy+yy))
    require(error<=128*math.ulp(1.)*(abs(xx)+2*abs(xy)+abs(yy)+1),'PAIRED_VARIANCE_COVARIANCE_IDENTITY')
    value['covariance_identity_abs_roundoff']=error
    return value

def load_task(index,campaign):
    r=read(index['result_path']);task=index['task'];candidate=task['candidate']
    require(sha(index['result_path'])==index['result_sha256'] and r['task']==task
      and r['SOURCE']==index['SOURCE'] and r['plan_sha256']==index['plan_sha256'], 'ACCEPTED_RESULT_SHA_IDENTITY')
    require(r['status']=='UNIT_COST_PASS_NOT_ACCOUNTING' and r['native_scientific_PASS']
      and r['byte_restoration_PASS'] and r['paired_trajectory'] and r['seed_cohort']==design.COHORT,
      'ACCEPTED_PAIRED_PILOT_ONLY')
    require(not any(r[k] for k in ('conditional_sampling','cohort_pooling','shot_accounting','cost_ranking')),
      'NO_POOLING_OR_OTHER_ANALYSIS')
    require(r['input_binding']==index['input_binding'] and r['preparation_hash']==index['preparation_hash'],
      'SAME_SAVED_INPUT_PREPARATION')
    require(len(r['checks'])==8 and all(v['PASS'] and not v['phase_alignment']
      and v['strictly_below']==(1e-7 if v['check'].startswith('compiled_vector_') else 1e-9)
      and 0<=v['error']<v['strictly_below'] for v in r['checks']),'SAVED_NATIVE_GATES_PASS')
    costs=r['axis_costs'];require([v['axis'] for v in costs]==['cosine','sine'],'PAIRED_AXES')
    require(all(v['compiler_settings_hash']==design.COMPILER_HASH and v['native_policy']==design.NATIVE_POLICY
      and v['event_digest']==r['event_digest'] for v in costs),'FIXED_COMPILER_AND_EVENT_PAIR')
    events=r['events'];require(events['stratum'] in design.STRATA
      and all(type(events[k]) is int and events[k]>=0 for k in ('N2','N4')),'SAVED_EVENT_STRATUM')
    require(events['N2']+events['N4']<=32*candidate['q'],'OUTER32_EVENTS')
    expected='S4' if events['N4'] else 'S2_multi' if events['N2']>=2 else 'S2_single' if events['N2'] else 'S0'
    require(events['stratum']==expected,'SAVED_EVENT_COUNT_LABEL')
    laws=[]
    for i,p in enumerate(index['law_paths']):
        record=read(p);law=record['law']
        require(record['piece_request_index']==i and record['cohort']==design.COHORT
          and not record['conditioned'] and record['event_count']==record['rte_steps'],'SAVED_PIECE_RECEIPT')
        laws.append(dict(steps=record['rte_steps'],orders=law['orders'],probabilities=law['order_probabilities'],
          K=law['finite_taylor_order'],signed_time=record['signed_step_time'],tau=law['dimensionless_step_time'],
          normalization=law['exact_finite_distribution']))
    require(len(laws)==candidate['q']*{'2nd':1,'4th':3,'4th(m5_best)':11}[candidate['formula']]
      and sum(v['steps'] for v in laws)==32*candidate['q'],'FIXED_FORMULA_OUTER32_PIECES')
    row=dict(trajectory_id=task['id'],candidate_id=candidate['id'],campaign_id=campaign,cohort=design.COHORT,
      conditional=False,paired=True,cosine_RZ=costs[0]['metrics']['rz_count'],sine_RZ=costs[1]['metrics']['rz_count'],
      stratum=events['stratum'])
    return row,laws

def evaluate_group(group,campaign):
    loaded=[load_task(v,campaign) for v in group['tasks']]
    require(len(loaded)==33 and all(laws==loaded[0][1] for row,laws in loaded),'FIXED33_IDENTICAL_GENERATION_LAW')
    return dict(candidate=group['candidate'],statistics=describe([r for r,laws in loaded],loaded[0][1]),
      original_registered_samples=[v['task']['sample'] for v in group['tasks']],
      result_SOURCE_counts={s:sum(v['SOURCE']==s for v in group['tasks']) for s in sorted({v['SOURCE'] for v in group['tasks']})})

def family_sensitivity(groups):
    families=[]
    for formula in ('2nd','4th','4th(m5_best)'):
        selected=[v for v in groups if v['candidate']['formula']==formula]
        require({v['candidate']['q'] for v in selected}=={1,2,4,8},'ALL_FOUR_Q_RETAINED')
        scenarios=[]
        for rho in TARGETS:
            for factor in INFLATIONS:
                values=[next(s['n_E_nominal_only'] for s in v['statistics']['n_E_sensitivity']
                  if s['relative_MCSE_target']==rho and s['variance_inflation']==factor) for v in selected]
                scenarios.append(dict(relative_MCSE_target=rho,variance_inflation=factor,
                  common_n_E_nominal_only=None if any(v is None for v in values) else max(values),
                  reason='ZERO_OBSERVED_VARIANCE_GROUP_REMAINS_UNRESOLVED' if any(v is None for v in values) else 'MAXIMUM_OVER_FIXED_FOUR_Q_NOT_ADOPTED',
                  adopted=False,rare_tail_guarantee=False))
        families.append(dict(formula=formula,q=[1,2,4,8],scenarios=scenarios))
    return families

def execute_stage(cfg):
    authorize(cfg);root=Path(cfg['run_root'])
    require(not root.exists(),'ONE_SHOT_UNUSED_ANALYSIS_ROOT');root.mkdir()
    save(root/'CONSUMED.json',dict(grant=cfg['grant'],state='CONSUMED_NO_RETRY'))
    try:
        count=min(len(cfg['plan']['groups']),len(os.sched_getaffinity(0)))
        def one(group):
            identifier=group['candidate']['id']
            save(root/'ledger'/(identifier+'.json'),dict(SOURCE=cfg['SOURCE'],stage=cfg['stage'],
              plan_sha256=cfg['plan_sha256'],candidate_id=identifier,state='RESERVED_NONREFUNDABLE_EVALUATION',
              groups=1,trajectories=33,new_sampling=0,new_compile=0))
            value=evaluate_group(group,cfg['plan']['campaign_id'])
            save(root/'groups'/(identifier+'.json'),value)
            return value
        with ThreadPoolExecutor(max_workers=count) as pool:
            results=list(pool.map(one,cfg['plan']['groups']))
        require(len(results)==12 and sum(r['statistics']['n'] for r in results)==396,'ALL12_GROUPS_396_PAIRS')
        require({p.stem for p in (root/'ledger').glob('*.json')}==
          {g['candidate']['id'] for g in cfg['plan']['groups']},'EXACT12_GROUP_EVALUATION_LEDGER')
        require(read(root/'CONSUMED.json')==dict(grant=cfg['grant'],state='CONSUMED_NO_RETRY'),'CONSUMED_EVALUATION_MARKER')
        verify(cfg)
        output=dict(status='PILOT_DESCRIPTIVE_EVALUATION_COMPLETE_MANDATORY_STOP',SOURCE=cfg['SOURCE'],
          stage=cfg['stage'],plan_sha256=cfg['plan_sha256'],groups=results,family_n_E_sensitivity=family_sensitivity(results),
          numeric_group_evaluations=12,n_E_adopted=False,n_E=None,formal_shot_policy_adopted=False,AD03='DEFERRED',
          paired_unit_costs_not_shot_weighted=True,ranking=False,population_mean_guarantee=False,
          population_tail_cost_bound=None,new_sampling=0,new_compile=0,new_signal=0,new_shot_calculation=0,
          group_threads=count,process_parallelism='ONE_SEPARATE_PROCESS_PER_SYSTEM',next_phase_started=False,mandatory_STOP=True)
        save(root/'evaluation.json',output)
        save(root/'completion.json',dict(status=output['status'],SOURCE=cfg['SOURCE'],
          plan_sha256=cfg['plan_sha256'],result_sha256=sha(root/'evaluation.json'),retry=False,mandatory_STOP=True))
        return dict(stage=cfg['stage'],status=output['status'],result_sha256=sha(root/'evaluation.json'))
    except BaseException as exc:
        save(root/'STOP.json',dict(reason=str(exc),retry=False,next_phase_started=False,mandatory_STOP=True));raise

def verify(cfg):
    import subprocess
    import sys
    require(cfg['stage'] in STAGES,'H4_H6_PILOT_ANALYSIS_ONLY')
    plan=cfg['plan'];require(read(cfg['plan_path'])==plan and sha(cfg['plan_path'])==cfg['plan_sha256'],'ANALYSIS_PLAN_SHA')
    require(sys.executable==plan['python_executable'] and sys.version==plan['python_version'],'FIXED_EVALUATION_RUNTIME')
    require(plan['SOURCE']==cfg['SOURCE'] and plan['stage']==cfg['stage'],'SOURCE_STAGE_BINDING')
    require(plan['relative_MCSE_targets']==list(TARGETS) and plan['variance_inflations']==list(INFLATIONS)
      and plan['n_E_adopted'] is False and not plan['shot_accounting'] and not plan['ranking'],'PROPOSALS_NOT_ADOPTION')
    for row in plan['sealed_files']:
        p=Path(row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'SEALED_ANALYSIS_INPUT:'+row['role'])
    indexed={v['path'] for v in plan['sealed_files']}
    for rel in SOURCE_PATHS:
        p=ROOT/rel;require(str(p) in indexed and p.read_bytes()==subprocess.check_output(
          ['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT),'COMMITTED_ANALYSIS_SOURCE:'+rel)
    groups=plan['groups'];system=cfg['stage'][:2]
    expected=[v for v in design.groups() if v['system']==system]
    require([g['candidate'] for g in groups]==expected,'ALL12_FIXED_GROUPS_NO_SELECTION')
    tasks=[v['task'] for g in groups for v in g['tasks']]
    require(tasks==design.tasks_for(system+'_DESIGN_PILOT',{v['id']:33 for v in expected}),'EXACT396_ORIGINAL_TASKS')
    require(plan['complete_acquisition_accepted'] is True and plan['old_STOP_rewritten'] is False,'ACCEPTED_COMPLETION_LINEAGE')
    for g in groups:
        for row in g['tasks']:
            require(row['result_path'] in indexed and all(p in indexed for p in row['law_paths']),'ALL_EVALUATION_INPUTS_SEALED')

def authorize(cfg):
    require(cfg.get('authorized') is True and cfg.get('grant',{}).get('valid') is True,'EXPLICIT_SAVED_PILOT_EVALUATION_APPROVAL_REQUIRED')
    grant=cfg['grant']
    require(grant['SOURCE']==cfg['SOURCE'] and grant['stage']==cfg['stage']
      and grant['plan_sha256']==cfg['plan_sha256'] and grant['run_root']==cfg['run_root']
      and grant['scope']==dict(groups=12,trajectories=396,nominal_n_E_scenarios=108,new_sampling=0,
        family_n_E_aggregations=27,new_compile=0,new_signal=0,new_shot_calculation=0,ranking=False), 'EXACT_ANALYSIS_GRANT_SCOPE')
    require(grant['retry'] is False and grant['next_phase_authorized'] is False,'NO_ANALYSIS_RETRY_OR_AUTOMATIC_EXPLORATION')
    require(cfg['run_root']==cfg['plan']['run_root'] and not Path(cfg['run_root']).is_relative_to(ROOT),
      'PRIVATE_SEPARATE_ANALYSIS_OUTPUT')
    verify(cfg)
