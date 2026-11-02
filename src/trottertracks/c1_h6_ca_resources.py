"""Dormant saved-H6 scalar evidence -> existing RPE/serial-cost connection.

No molecule arrays, QPY, operator, fitter, simulator or compiler are decoded.
Only an explicitly permitted future run may join saved numerical evidence.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import math
import resource
import time
from . import c1_h6_synthesis_bound as synthesis
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_connection as connection

io=synthesis.io
integrated=synthesis.native.saved.molecule.integrated
det=synthesis.native.det
SCOPE='C1_H6_SAVED_CA_RPE_SERIAL_RESOURCES_V1'
FAMILY='artifacts/research_c_c1_h6_ca_resources'
CANDIDATES=synthesis.native.saved.CANDIDATES
TARGET=dict(energy_ha=.00159,df_reserve_fraction=1/8,base_time=.1,
    beta_total=math.pi/6,beta_stat=math.pi/12,alpha_axis=.01/26,alpha_total=.01)
COST_SCOPE='serial_native_kernel_composition_without_state_preparation'
RECORD_KEYS={'bound','bound_STOP','native','native_STOP','PF','PF_STOP','shells','identity'}
LIMITS=dict(saved_bundle_load=1,evidence_attachment=4,round_accuracy=52,
    energy_assessment=4,round_serial_cost=52,cost_aggregation=4,cost_comparison=2)
MAX_CALLS=sum(LIMITS.values())
FORBIDDEN=('new_fit','optimizer','rank_search','DF_recomputation','PF_recomputation',
    'synthesis_recomputation','state_action','build_compile','quantum_sampling','retry','resume')


def preflight(root,path):
    """Byte hashes, fixed policy and versions only; never join saved values."""
    root=Path(root);c=io.read(path)
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions']!=synthesis.native.saved.CONDITIONS or c['accuracy_target']!=TARGET
            or c['cost_scope']!=COST_SCOPE or c['call_limits']!=LIMITS
            or c['maximum_logical_units']!=MAX_CALLS
            or c['candidate_order']!=[f'{a}:s{s}' for a,s in CANDIDATES]
            or any(c[k] for k in FORBIDDEN)):
        raise ValueError('CA_RESOURCE_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in [io.read(root/c['source_manifest'])['files'],c['saved_inputs']]:
        for f,h in group.items():
            if io.sha(root/f)!=h:raise ValueError('IDENTITY: '+f)
    if set(c['records'])!=RECORD_KEYS or not set(c['records'].values())<=c['saved_inputs'].keys():raise ValueError('UNPINNED_RECORD')
    for p,v in c['evaluation_packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('ENVIRONMENT: '+p)
    return c


def load_bundle(root,c):
    """Future science only: reuse saved errors/metrics, no error regeneration."""
    b={k:io.read(Path(root)/v) for k,v in c['records'].items()}
    expected=dict(bound='H6_NUMERICAL_RESTRICTED_BOUND_COMPLETE',
        native='H6_NATIVE_FINITE_PROBE_COST_COMPLETE',PF='H6_SAVED_RANK14_SECTOR_PF_COMPLETE')
    for k,status in expected.items():
        if b[k]['status']!=status or b[k+'_STOP']['failed'] or not b[k+'_STOP']['mandatory_stop']:
            raise ValueError('SAVED_RUN_STATUS: '+k)
    if (b['bound']['evaluated_groups']!=4 or len(b['identity']['groups'])!=4
            or b['bound_STOP']['scientific_units']!=3335):raise ValueError('BOUND_GROUP_IDENTITY')
    for name in ('physical','CDF','XDF'):
        if b['PF']['ground'][name]['status']!='PASS':raise ValueError('SAVED_GROUND_ALIAS')
    target=accuracy.AccuracyTarget(**c['accuracy_target'])
    if target.horizon()['maximum_round_index_M']!=12:raise ValueError('REGISTERED_HORIZON')
    return b


def metric_record(value,*,measurement):
    for k in accuracy.METRICS:
        if type(value[k]) is not int or value[k]<0:raise ValueError('SAVED_INTEGER_NATIVE_METRIC')
    if type(value['measurement_count']) is not int or value['measurement_count']!=measurement:raise ValueError('SAVED_MEASUREMENT_SCOPE')
    return {k:value[k] for k in accuracy.METRICS}


def attach_evidence(bundle,arm,s):
    key=f'{arm}:s{s}';labels=[f'{arm}_s{s}_{r}' for r in ('literal','shared')]
    identity=bundle['identity'];routes=bundle['bound']['routes'];native=bundle['native']['candidates']
    if (identity['aliases'][labels[0]]!=identity['aliases'][labels[1]]
            or identity['identities'][labels[0]]['signature']!=identity['identities'][labels[1]]['signature']
            or routes[labels[0]]['rounds']!=routes[labels[1]]['rounds']
            or native[labels[0]]['metrics']!=native[labels[1]]['metrics']):
        raise ValueError('EXACT_IR_ROUTE_EVIDENCE_MISMATCH')
    row=routes[labels[0]];kernel=row['kernel']
    if (kernel['covered_columns']!=800 or kernel['input_dimension']!=800 or kernel['output_dimension']!=8192
            or kernel['evidence_level']!=synthesis.bounds.LEVEL):raise ValueError('COMPLETE_BASIS_LEVEL_B_REQUIRED')
    df=bundle['PF']['saved_DF'][arm]
    if (df['status']!='PASS' or df['actual_rank']!=14 or df['budget_Ha']!=.00159/8):
        raise ValueError('SAVED_DF_IDENTITY')
    for k in ('saved_sector_error_Ha','saved_guard_Ha','norm_plus_guard_Ha'):accuracy._finite(df[k],k)
    if df['saved_sector_error_Ha']+df['saved_guard_Ha']!=df['norm_plus_guard_Ha']:
        raise ValueError('DF_GUARD_COUNTING')
    if df['norm_plus_guard_Ha']>df['budget_Ha']:raise ValueError('SAVED_DF_BUDGET_IDENTITY')
    rows=row['rounds'];pf=bundle['PF']['candidates'][key]
    if pf['arm']!=arm or pf['s']!=s or [r['m'] for r in rows]!=list(range(13)) or [r['m'] for r in pf['rounds']]!=list(range(13)):
        raise ValueError('REGISTERED_ROUND_CANDIDATE_IDENTITY')
    for r,old in zip(rows,pf['rounds'],strict=True):
        if (r['s']!=s or r['occurrences']!=s*(1<<r['m'])
                or old['attenuation']!=1. or old['rte_corrected_operator_error']!=0.
                or r['saved_PF_operator_error']!=old['pf_operator_error']):
            raise ValueError('SAVED_PF_ROUND_IDENTITY')
    metrics=metric_record(native[labels[0]]['metrics'],measurement=0)
    shell=bundle['shells']['metrics']
    sm={k:metric_record(shell[k],measurement=0 if k=='head' else 1) for k in ('head','X','Y')}
    return dict(id=key,arm=arm,s=s,aliases=labels,df_error_with_guard=df['norm_plus_guard_Ha'],
        rounds=rows,kernel_metrics=metrics,shell_metrics=sm,
        evidence_level=synthesis.bounds.LEVEL,state_model='own_DF_ground_state_oracle',alias_verified=True)


def assess_saved_round(target,packet,row):
    """Reuse solver and fixed-N margin check; each stored error enters once."""
    m=row['m'];combined=row['combined_complex_error_pilot'];pf=row['saved_PF_operator_error'];syn=row['complex_signal_bound_pilot']
    base=dict(m=m,target_id=packet['id'],source_combined_complex_error=combined,
        source_PF_error=pf,source_synthesis_complex_error=syn,
        evidence_level=row.get('evidence_level'),CA_achieved=None)
    if (row.get('evidence_level')!=synthesis.bounds.LEVEL
            or any(x is None for x in (combined,pf,syn))
            or (combined<1. and row.get('phase_bias_bound_pilot') is None)):
        return dict(base,status='UNRESOLVED_SAVED_SIGNAL_BOUND',shots=None)
    for x in (combined,pf,syn):accuracy._finite(x,'saved scalar bound')
    if (math.fsum([pf,syn])!=combined or row['registered_phase_room']!=target.beta_total-target.beta_stat
            or (combined<1. and math.asin(combined)!=row['phase_bias_bound_pilot'])):
        raise ValueError('SAVED_COMBINED_SIGNAL_OR_PHASE_IDENTITY')
    evidence=accuracy.RoundEvidence(m=m,attenuation=1.,pf_operator_error=pf,
        rte_corrected_operator_error=0.,synthesis_signal_error=syn,finite_law_signal_error=0.,
        target_id=packet['id'],state_model=packet['state_model'],alias_branch_verified=packet['alias_verified'],
        iid_per_shot=True,law_model='deterministic')
    try:plan=accuracy.assess_round(target,evidence)
    except (OverflowError,ZeroDivisionError):return dict(base,status='UNRESOLVED_SHOT_NUMERIC_RANGE',shots=None)
    if plan['status']!='CONDITIONAL_ROUND_PASS':return dict(base,**plan)
    if plan['raw_signal_error']!=combined:raise ValueError('DOUBLE_COUNTED_SIGNAL_ERROR')
    n=plan['shots']
    if type(n) is not int or n<1:raise ValueError('POSITIVE_INTEGER_SHOTS_REQUIRED')
    if n>=1<<53:return dict(base,status='UNRESOLVED_SHOT_NUMERIC_RANGE',shots=None,unqualified_shots_decimal=str(n))
    # The existing fixed-shot check independently verifies the integer selected
    # by the same existing Hoeffding solver; never increase N after inspection.
    verified=det.assess_fixed_round(target,SimpleNamespace(id=packet['id'],s=packet['s']),
        dict(m=m,attenuation=1.,rte_corrected_operator_error=0.,pf_operator_error=pf),
        dict(m=m,axes={a:dict(optimistic_shots_decimal=str(n)) for a in ('X','Y')}),
        dict(occurrences=packet['s']*(1<<m),complex_signal_error_pilot=syn),alias_verified=packet['alias_verified'])
    if verified['status']!='CONDITIONAL_ROUND_PASS':
        return dict(base,status='UNRESOLVED_INTEGER_SHOT_MARGIN',shots=None,shot_check=verified)
    return dict(base,**plan,axes_shots_decimal={a:str(n) for a in ('X','Y')},
        statistical_margin=verified['fixed_statistical_margin'],shot_count_solved=True,
        candidate_failure_only=True)


def serial_round_cost(packet,m):
    metrics=packet['kernel_metrics']
    kernel=dict(target_id=packet['id'],policy='native_kernel_repeat_with_fences',mean=metrics,
        lower=metrics,upper=metrics,simultaneous_interval_qualified=True)
    records=[]
    for axis in ('X','Y'):
        sh={k:packet['shell_metrics']['head'][k]+packet['shell_metrics'][axis][k] for k in accuracy.METRICS}
        records.append(connection.compose_serial_cost(kernel,dict(axis=axis,policy='separate_native_readout',
            mean=sh,lower=sh,upper=sh,simultaneous_interval_qualified=True),m=m,substeps=packet['s'],target_id=packet['id']))
    return records


def aggregate_candidate(packet,plans,energy,costs):
    try:total=accuracy.aggregate_totals(energy,plans,costs,cost_scope=COST_SCOPE)
    except OverflowError:total=dict(status='UNRESOLVED_COST_NUMERIC_RANGE',totals=None)
    if total.get('totals') is not None and any(not math.isfinite(v) or v>=1<<53 or not v.is_integer() for v in total['totals'].values()):
        total=dict(status='UNRESOLVED_COST_NUMERIC_RANGE',totals=None)
    unresolved=(any(p['status'].startswith('UNRESOLVED') for p in plans)
        or total['status'].startswith('UNRESOLVED')
        or (energy.get('same_accuracy') and total.get('totals') is None))
    status='UNRESOLVED' if unresolved else 'NUMERICAL_PILOT_CA_PASS' if energy.get('same_accuracy') else 'INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION'
    if total.get('totals') is not None:total['measurement_count']=total['executions']
    return dict(id=packet['id'],arm=packet['arm'],s=packet['s'],aliases=packet['aliases'],primary=packet['s']==1,
        status=status,rounds=plans,per_shot_costs=costs,numerical_energy_assessment=energy,cost=total,
        evidence_level=synthesis.bounds.LEVEL,rigorous_enclosure=None,CA_achieved=None,
        guaranteed_actual_shots=None,guaranteed_total_resource_superiority=False,
        failure_scope='one predeclared candidate; no four-candidate or post-selection guarantee',
        state_preparation_cost=None)


def run(root,path,output):
    from concurrent.futures import ThreadPoolExecutor
    root,output=Path(root),Path(output);c=preflight(root,path)
    output.mkdir(parents=True,exist_ok=False);(output/'candidates').mkdir()
    calls=integrated.Calls(output,LIMITS,MAX_CALLS);start=time.monotonic();cpu=time.process_time()
    io.save(output/'contract.json',c);io.save(output/'SOURCE.json',io.read(root/c['source_manifest']))
    try:
        b=calls('saved_bundle_load','pinned JSON evidence/alias/horizon only',lambda:load_bundle(root,c))
        packets={}
        for a,s in CANDIDATES:
            key=f'{a}:s{s}';packet=calls('evidence_attachment',key,lambda a=a,s=s:attach_evidence(b,a,s))
            packets[key]=packet;io.save(output/'candidates'/(key.replace(':','_')+'.packet.json'),packet)
        target=accuracy.AccuracyTarget(**c['accuracy_target']);plan=integrated.resource_plan(task_count=4)
        io.save(output/'resource_plan.json',plan)
        def evaluate(key):
            packet=packets[key];plans=[]
            for row in packet['rounds']:
                value=calls('round_accuracy',f'{key}:m{row["m"]}',lambda row=row:assess_saved_round(target,packet,row))
                plans.append(value);integrated.progress(output,key,'round_accuracy',value)
            energy=calls('energy_assessment',key,lambda:accuracy.assess_energy(target,packet['df_error_with_guard'],plans,
                state_model=packet['state_model'],alias_branch_verified=packet['alias_verified']))
            integrated.progress(output,key,'energy_assessment',energy)
            costs=[]
            if energy.get('same_accuracy'):
                for row in plans:
                    pair=calls('round_serial_cost',f'{key}:m{row["m"]}:XY',lambda row=row:serial_round_cost(packet,row['m']))
                    costs.extend(pair);integrated.progress(output,key,'serial_cost',dict(m=row['m'],axes=pair))
            result=calls('cost_aggregation',key,lambda:aggregate_candidate(packet,plans,energy,costs))
            io.save(output/'candidates'/(key.replace(':','_')+'.json'),result);return key,result
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:results=dict(pool.map(evaluate,packets))
        comparisons={}
        for s in (1,4):
            def compare(s=s):
                a,b=(results[f'{arm}:s{s}'] for arm in ('CDF','XDF'))
                return dict(s=s,comparison=accuracy.compare_totals(a['cost'],b['cost']),
                    qualification='conditional numerical pilot only',rigorous_same_accuracy=None,
                    guaranteed_total_resource_superiority=False)
            comparisons[str(s)]=calls('cost_comparison',f's{s}',compare)
        io.save(output/'result.json',dict(status='H6_SAVED_CA_RPE_SERIAL_RESOURCES_COMPLETE',candidates=results,
            comparisons=comparisons,scientific_units=sum(calls.counts.values()),call_counts=dict(calls.counts),
            CA_achieved=None,rigorous_enclosure=None,new_fit=0,optimizer=0,rank_search=0,build_compile=0,
            DF_recomputation=0,PF_recomputation=0,synthesis_recomputation=0,state_action=0,quantum_shots_drawn=0,
            retry=0,resume=0,mandatory_stop=True))
    except BaseException as error:
        calls.stop.set();io.save(output/'failure.json',dict(error=str(error),first_failure=calls.first_failure));raise
    finally:
        io.save(output/'STOP.json',dict(failed=calls.stop.is_set(),scientific_units=sum(calls.counts.values()),
            call_counts=dict(calls.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
