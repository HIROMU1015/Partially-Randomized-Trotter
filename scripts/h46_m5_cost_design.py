"""Metadata-only candidate/call/seed design. No molecular libraries or shots."""
import hashlib
import json
from pathlib import Path

BASE='a960a41da5ebc564b0306e798572257091c8509c'
EPSILONS=(.05,.01,.005,.001,.0005)
FORMULAS=('2nd','4th','4th(m5_best)')
NATIVE_POLICY='M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1'
COMPILER_HASH='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
COMPILER=dict(qiskit='1.3.0',basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1,backend=None,target=None,coupling_map=None,layout=None)

def require(value,reason):
    if not value:raise ValueError(reason)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def selection(records):
    """Join stored labels at the same endpoint. Never recompute eligibility."""
    ids=set();out=[]
    for data in records:
        row=data['candidate'];require(row['id'] not in ids,'DUPLICATE_CANDIDATE');ids.add(row['id'])
        require(data['status']=='SIGNAL_DIAGNOSTIC_PASS','SIGNAL_NOT_COMPLETE')
        axes=data['empirical_eligibility']['rows']
        require(len(axes)==10 and {(x['epsilon'],x['axis']) for x in axes}=={(e,a) for e in EPSILONS for a in ('real','imag')},'EXACT_FIVE_EPS_TWO_AXES')
        require(all(x['status_under_proposed_empirical_policy'] in ('ELIGIBLE','INELIGIBLE','UNDETERMINED') for x in axes),'KNOWN_SAVED_LABELS')
        require(all(x['shots'] is None and x['cost'] is None for x in axes),'NO_SHOT_OR_COST_REANALYSIS')
        statuses={str(e):{x['axis']:x['status_under_proposed_empirical_policy'] for x in axes if x['epsilon']==e} for e in EPSILONS}
        eligible=[e for e in EPSILONS if all(x=='ELIGIBLE' for x in statuses[str(e)].values())]
        unresolved=[e for e in EPSILONS if 'UNDETERMINED' in statuses[str(e)].values() and 'INELIGIBLE' not in statuses[str(e)].values()]
        out.append(dict(candidate=row,eligible_epsilons=eligible,unresolved_epsilons=unresolved,saved_axis_status=statuses,
          preparation_hash=data['preparation_hash'],signal_origin=data['signal_origin'],
          proposal='COST_TARGET' if eligible else 'HOLD_UNDETERMINED' if unresolved else 'NO_COMMON_ELIGIBLE_ENDPOINT',
          policy='PROPOSED_REQUIRES_DOMAIN_AND_EMPIRICAL_ALLOWANCE_ACCEPTANCE',formal_shot_policy_adopted=False))
    return out

def tasks_for(stage,domain,reusable_axes,n_by_candidate=None):
    require(stage in ('H4_DET','H6_DET','H4_PR','H6_PR'),'FOUR_SEPARATE_STAGES')
    system,kind=stage.split('_');n_by_candidate=n_by_candidate or {};result=[]
    wanted=[r for r in domain if r['candidate']['system']==system and (r['candidate']['method']=='PR')==(kind=='PR') and r['proposal']=='COST_TARGET']
    if kind=='PR':require(set(n_by_candidate)=={r['candidate']['id'] for r in wanted},'EXACT_PR_N_DOMAIN')
    for item in wanted:
        row=item['candidate'];available=[a for a in ('cosine','sine') if (row['id'],a) in reusable_axes]
        if kind=='DET':
            require(len(available) in (0,2),'PARTIAL_PAIR_REQUIRES_SEPARATE_CONTRACT')
            if available:continue
            n=1
        else:
            n=n_by_candidate[row['id']];require(type(n) is int and n>=2,'N_E_NOT_ADOPTED_OR_LT_TWO')
        for sample in range(n):
            result.append(dict(id=row['id']+'__'+str(sample),candidate=row,sample=sample,axes=['cosine','sine'],stage=stage))
    return result

def call_scope(tasks,kind):
    require(kind in ('DET','PR'),'KNOWN_KIND')
    n=len(tasks)
    # Controls are checked for every acquired trajectory/candidate, not only s0.
    pieces=sum(t['candidate']['q']*({'2nd':1,'4th':3,'4th(m5_best)':11}[t['candidate']['formula']]) for t in tasks)
    return dict(trajectories=n if kind=='PR' else 0,request_sampling_calls=pieces if kind=='PR' else 0,
      rte_events=sum(32*t['candidate']['q'] for t in tasks) if kind=='PR' else 0,
      native_piece_compositions=pieces,native_evolution=n,compile_axes=2*n,
      native_low_actions=n,native_high_actions=n,independent_oracle_actions=n,
      uncompiled_wrapper_actions=2*n,compiled_wrapper_actions=2*n,
      new_signal=0,new_reference=0,SCF_DF_state_generation=0,shot_calculation=0,cost_accounting=0,fit=0,
      ranking=0,conditional_sampling=0,retry=False,next_phase_authorized=False,mandatory_STOP=True)

def seed(source,campaign,cohort,task_id,outer,piece):
    require(cohort in ('EXPLORATION','CONFIRMATION'),'EXPLICIT_COHORT')
    require(isinstance(campaign,str) and bool(campaign.strip()),'FRESH_CAMPAIGN_NAMESPACE')
    return int(digest(['H46_M5_OUTER32_COST_V1',source,campaign,cohort,task_id,outer,piece])[:16],16)

def registry(tasks,source,campaign,cohort,excluded,schedule):
    require(len(excluded)==len(set(excluded)) and all(type(x) is int for x in excluded),'EXCLUSION_INTS_UNIQUE')
    occupied=set(excluded);result={}
    for task in tasks:
        row=task['candidate']
        if row['method']!='PR':continue
        require(task['id'] not in result,'DUPLICATE_TASK')
        values={}
        for step in schedule(row['formula'],row['q'],'ABS_TIME_PER_OUTER',32):
            key=str(step['outer'])+':'+str(step['piece']);value=seed(source,campaign,cohort,task['id'],step['outer'],step['piece'])
            require(value not in occupied,'KNOWN_OR_NEW_SEED_COLLISION_NO_REHASH_RETRY')
            occupied.add(value);values[key]=value
        result[task['id']]=values
    return result

def sampling_proposal(domain):
    result=[]
    for item in domain:
        r=item['candidate']
        if r['method']!='PR':continue
        result.append(dict(candidate=r,n_E=None,eligible_epsilons=item['eligible_epsilons'],
          estimand='PAIRED_NATIVE_UNIT_COST_MEAN_EXPLORATION_NOT_A_POPULATION_RANK_GUARANTEE',
          same_law_cost_variance='MISSING',old_piece_r2_samples_usable_for_new_mean=False,
          design_options=['FIXED_EXPLORATORY_N_AFTER_USER_ACCEPTS_FINITE_SAMPLE_SCOPE',
            'SEPARATE_OUTER32_DESIGN_PILOT_THEN_FRESH_FIXED_N_E'],
          within_formula_all_q_retained=True,all_five_eps_share_same_trajectories=True,
          adaptive_top_up=False,selection_after_exploration=True,confirmation_n_C=None,
          confirmation_samples_pooled=False,conditional_rare_pilot_pooled=False,
          rare_event_policy='UNCONDITIONAL_LAW_NO_REJECTION_OR_FORCED_EVENT;LOG_S0_S2_SINGLE_S2_MULTI_S4',
          unseen_rare_cost=None,unseen_rare_variance=None,all_support_upper_bound=None))
    return result

def event_stratum(events):
    flat=[e for piece in events for e in piece]
    orders=[e['taylor_order'] for e in flat]
    require(all(type(k) is int and k in (0,2,4) for k in orders),'REGISTERED_K4_TAYLOR_ORDERS')
    n2=orders.count(2);n4=orders.count(4)
    return dict(N2=n2,N4=n4,stratum='s4' if n4 else 's2_multi' if n2>=2 else 's2_single' if n2 else 's0')
