"""Saved-scalar mixed-round adapter; passive import, no molecular generation.

Use only with a verified common Hamiltonian/ground/phase binding. Existing
accuracy, Hoeffding, serial-cost and energy checks remain unchanged.
"""
import hashlib
import json
import math

from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_connection as connection
from .c_pr_aware_df_m2 import TARGET

VERSION='common_energy_roundwise_D_PR_v1'
METHODS=('D','PR')
LANES=('evidence','ideal_exact_native','assumed_raw_signal_budget')
ROUNDS=tuple(range(13))
SCOPE='serial_native_kernel_composition_without_state_preparation'
STATE='own_DF_ground_state_oracle'
BINDING_KEYS=('input_sha256','JU_sha256','ground_sha256','reference_sha256',
    'physical_input_sha256','geometry','sector','frame_order','kappa_policy',
    'scalar_policy','spin_policy','state_model','phase_convention','delta')


def common_target(bindings):
    """Equality of scientific inputs/conventions, not old target-id spelling."""
    if set(bindings)!=set(METHODS):raise ValueError('TWO_METHOD_BINDINGS')
    first=bindings['D']
    if set(first)!=set(BINDING_KEYS) or any(bindings[a]!=first for a in METHODS):
        raise ValueError('COMMON_ENERGY_BINDING')
    if first['state_model']!=STATE or first['delta']!='1/10' or first['phase_convention']!='X+iY=exp(-iEt)':
        raise ValueError('COMMON_STATE_TIME_PHASE')
    identity=hashlib.sha256(json.dumps(first,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return 'same_saved_DF_energy:'+identity


def assess(target,identity,method,lane,row,*,evidence_id,alias_verified):
    if method not in METHODS or lane not in LANES or not evidence_id:raise ValueError('METHOD_LANE_EVIDENCE')
    m=row['m']
    if type(m)is not int or m not in ROUNDS or row['elapsed_time']!=target.base_time*(1<<m):
        raise ValueError('ROUND_TIME_GRID')
    if row['state_model']!=STATE or row['phase_convention']!='X+iY=exp(-iEt)':raise ValueError('STATE_PHASE_CONVENTION')
    for k in ('attenuation','pf_operator_error','rte_corrected_operator_error'):
        accuracy._finite(row[k],k)
    A=row['attenuation']
    if method=='D' and (A!=1. or row['rte_corrected_operator_error']!=0.):raise ValueError('DETERMINISTIC_NOT_R_ZERO')
    if lane=='evidence':
        syn=row['synthesis_signal_error'];law=0. if method=='D' else row['finite_law_signal_error']
        law_model='deterministic' if method=='D' else 'actual_implemented_law'
    elif lane=='ideal_exact_native':
        # Explicit counterfactual, never a qualification of unknown errors.
        syn=law=0.;law_model='deterministic' if method=='D' else 'ideal_finite_RTE_law'
    else:
        syn=.001*A;law=0. if method=='D' else .001*A
        law_model='deterministic' if method=='D' else 'actual_implemented_law'
    evidence=accuracy.RoundEvidence(m=m,attenuation=A,pf_operator_error=row['pf_operator_error'],
        rte_corrected_operator_error=row['rte_corrected_operator_error'],synthesis_signal_error=syn,
        finite_law_signal_error=law,target_id=identity,state_model=STATE,alias_branch_verified=alias_verified,
        iid_per_shot=row['iid_per_shot'],evidence_kind='numerically_qualified_pilot',law_model=law_model)
    try:value=accuracy.assess_round(target,evidence)
    except (OverflowError,ZeroDivisionError):value=dict(status='UNRESOLVED_SHOT_RANGE',m=m,shots=None)
    return dict(value,target_id=identity,method_id=method,evidence_id=evidence_id,lane=lane,
        source_target_id=row['source_target_id'],elapsed_time=row['elapsed_time'],
        assumptions_applied=lane!='evidence',actual_CA=None,guaranteed_shots=None,
        synthesis_error_used=syn,law_error_used=law)


def costs(identity,method,m,kernel_mean,shells):
    """Both axes; complete kernel repeats 2**m, outer shell exactly once."""
    if method not in METHODS or m not in ROUNDS:raise ValueError('COST_METHOD_ROUND')
    if kernel_mean is None or any(kernel_mean.get(k)is None for k in accuracy.METRICS):
        return [dict(m=m,axis=a,target_id=identity,method_id=method,mean=None) for a in ('X','Y')]
    kernel=dict(target_id=identity,policy='native_kernel_repeat_with_fences',
        mean={k:accuracy._finite(kernel_mean[k],k) for k in accuracy.METRICS})
    out=[]
    for axis in ('X','Y'):
        sh={k:math.fsum(accuracy._finite(shells[a][k],'shell') for a in ('head',axis)) for k in accuracy.METRICS}
        value=connection.compose_serial_cost(kernel,dict(axis=axis,policy='separate_native_readout',mean=sh),
            m=m,substeps=1,target_id=identity)
        out.append(dict(value,method_id=method))
    return out


def select(plans,cost_records):
    """O(13*2) point selection, RZ only; exact ties prefer D.

    Missing qualification/price is not evidence of infeasibility or dominance.
    Non-RZ metrics are reported for this same schedule, never reoptimized.
    """
    if set(plans)!=set(METHODS) or set(cost_records)!=set(METHODS):raise ValueError('METHOD_COVERAGE')
    identities={p['target_id'] for a in METHODS for p in plans[a]}
    if len(identities)!=1:raise ValueError('MIXED_DF_TARGET')
    if len({p['lane'] for a in METHODS for p in plans[a]})!=1:raise ValueError('MIXED_QUALIFICATION_LANES')
    lookup={}
    for a in METHODS:
        if [p['m'] for p in plans[a]]!=list(ROUNDS):raise ValueError('ROUND_COVERAGE')
        if any(p['method_id']!=a for p in plans[a]):raise ValueError('METHOD_PLAN_IDENTITY')
        rows=cost_records[a];d={(r['m'],r['axis']):r for r in rows}
        if len(d)!=26 or len(rows)!=26 or set(d)!={(m,x) for m in ROUNDS for x in ('X','Y')}:
            raise ValueError('COST_COVERAGE')
        if any(r['method_id']!=a or r['target_id'] not in identities for r in rows):raise ValueError('COST_IDENTITY')
        if any(r['mean']is not None and (r.get('scope')!=SCOPE or not r.get('long_round_coverage')
                or r.get('kernel_occurrences')!=1<<r['m']) for r in rows):raise ValueError('SERIAL_COST_SCOPE')
        lookup[a]=d
    choices=[];comparisons=[];dominated=True
    for m in ROUNDS:
        values={};unknown=False
        for a in METHODS:
            p=plans[a][m]
            if p['status'] in ('INFEASIBLE_PHASE','INFEASIBLE_RADIUS'):values[a]=None;continue
            if p['status']!='CONDITIONAL_ROUND_PASS':unknown=True;continue
            if any(lookup[a][m,x]['mean']is None for x in ('X','Y')):unknown=True;continue
            values[a]=accuracy._finite(math.fsum(p['shots']*accuracy._finite(lookup[a][m,x]['mean']['rz_count'],'RZ')
                for x in ('X','Y')),'round RZ')
        comparisons.append(dict(m=m,conditional_RZ=values,unqualified=unknown))
        if unknown:return dict(status='UNQUALIFIED',schedule=None,comparisons=comparisons)
        eligible=[a for a in METHODS if values.get(a)is not None]
        if not eligible:return dict(status='INFEASIBLE_REGISTERED_ROUND',schedule=None,comparisons=comparisons)
        chosen=min(eligible,key=lambda a:(values[a],METHODS.index(a)));choices.append(chosen)
        dominated &= 'D' in eligible and ('PR' not in eligible or values['D']<=values['PR'])
    return dict(status='MIXED_ROUND_DOMINATED' if dominated else 'MIXED_ROUND_POTENTIALLY_USEFUL',
        schedule=choices,strictly_mixed=len(set(choices))==2,comparisons=comparisons,
        qualification='conditional descriptive selection using saved calibration; not population-optimal',
        guaranteed_advantage=None)


def evaluate_schedule(target,df,plans,cost_records,schedule,*,alias_verified,call=None,label='synthetic'):
    call=call or (lambda kind,label,fn:fn())
    if schedule is not None and (len(schedule)!=13 or any(a not in METHODS for a in schedule)):
        raise ValueError('FULL13_SCHEDULE')
    selected=[] if schedule is None else [plans[a][m] for m,a in enumerate(schedule)]
    selected_costs=[] if schedule is None else [r for m,a in enumerate(schedule) for r in cost_records[a] if r['m']==m]
    energy=call('schedule_energy',label,lambda:accuracy.assess_energy(target,df,selected,
        state_model=STATE,alias_branch_verified=alias_verified))
    resources=call('schedule_total',label,lambda:accuracy.aggregate_totals(energy,selected,selected_costs,cost_scope=SCOPE))
    return dict(energy=energy,resources=resources,plans=selected)


def shared_uncertainty(evaluations,PR_kernel_SE):
    """Conditional on selected schedule, one shared PR calibration, not 13."""
    out={}
    for name,value in evaluations.items():
        if value['resources'].get('totals')is None:out[name]=dict(weight=None,weighted_SE=None);continue
        W=sum(2*p['shots']*(1<<p['m']) for p in value['plans'] if p['method_id']=='PR')
        out[name]=dict(weight=W,weighted_SE=None if PR_kernel_SE is None else
            {k:W*accuracy._finite(PR_kernel_SE[k],'calibration SE') for k in accuracy.METRICS})
    wd=out['all_D']['weight'];wm=out['mixed']['weight']
    delta=None if wd is None or wm is None or PR_kernel_SE is None else {
        k:abs(wm-wd)*accuracy._finite(PR_kernel_SE[k],'calibration SE') for k in accuracy.METRICS}
    return dict(schedules=out,mixed_minus_all_D_SE=delta,shared_calibration=True,
        selection_on_same_data=True,selection_bias_corrected=False,formal_CI=None,cost_support=None)
