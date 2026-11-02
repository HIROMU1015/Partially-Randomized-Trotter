"""Saved scalar evidence -> opt-in Stage C1 accuracy/serial-cost scenarios.

No molecular arrays, circuits, random sampling or numerical work on import or
preflight. Unknown bounds stay null in the evidence lane. Scenario assumptions
are explicit counterfactuals and never become scientific qualifications.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c1_stage1_execution as io
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_connection as connection
from . import c1_stage3_integrated as integrated
from . import c_pr_aware_df_stage_b_runner as b1
from . import c_pr_aware_df_stage_b2 as b2

VERSION='PR_aware_DF_C1_saved_scalar_scenarios_v1'
SCOPE='C_PR_AWARE_DF_STAGE_C1_CONDITIONAL_RESOURCES_V1'
FAMILY='artifacts/research_c_pr_aware_df_stage_c'
FRESH=FAMILY+'/2026-10-27/C1-run01'
ARMS=('baseline','proposed')
TARGET=dict(energy_ha=.00159,df_reserve_fraction=1/8,base_time=.1,
    beta_total=math.pi/6,beta_stat=math.pi/12,alpha_axis=.01/26,alpha_total=.01)
COST_SCOPE='serial_native_kernel_composition_without_state_preparation'
SCENARIOS=(dict(id='ideal_exact_native',synthesis_fraction_of_A=0.,law_fraction_of_A=0.,
    law_model='ideal_finite_RTE_law',assumptions=['exact native synthesis',
    'ideal finite-RTE IID law','finite-bit B2 sample mean is a proxy for ideal-law kernel cost']),
    dict(id='assumed_raw_signal_budget',synthesis_fraction_of_A=.001,law_fraction_of_A=.001,
    law_model='actual_implemented_law',assumptions=['all-law synthesis raw-signal error <= .001*A_m',
    'implemented-vs-ideal raw-signal law error <= .001*A_m',
    'B2 finite-bit sample mean is a descriptive cost proxy for fresh independent occurrences']))
LANES=('evidence',)+tuple(x['id'] for x in SCENARIOS)
LIMITS=dict(saved_bundle_load=1,identity_gate=1,target_validation=1,arm_attachment=2,
    cost_calibration=2,serial_cost=52,signal_qualification=6,round_accuracy=78,
    energy_assessment=6,total_aggregation=6,calibration_uncertainty=6,comparison=3)
MAX_CALLS=sum(LIMITS.values())
B1_SOURCE='7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9'
B2_SOURCE='a1cf0046967c206f414dbad568a75884ee076ebf'
SAMPLING_SOURCE='cdf79ed569127f5cb6c7ee017b76bce6422166a656bb3ae866de6e86917c45bf'
METRICS=accuracy.METRICS


def preflight(root,path):
    """Bytes, fixed contracts, metadata/schema and saved verdicts only."""
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['status']!='UNAPPROVED_PROPOSAL' or c['implementation']!=VERSION
            or c['target']!=TARGET or c['scenarios']!=list(SCENARIOS) or c['lanes']!=list(LANES)
            or c['call_limits']!=LIMITS or c['maximum_logical_units']!=MAX_CALLS
            or c['fresh_output']!=FRESH or c['cost_scope']!=COST_SCOPE
            or c['retry'] or c['resume'] or c['new_fit'] or c['new_sampling'] or c['build_compile']):
        raise ValueError('C1_SCOPE_OR_FIXED_CONDITIONS')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError('SOURCE_OR_INPUT: '+name)
    required={*c['records'].values(),*c['export_hashes'].keys()}
    for samples in c['samples'].values():
        for paths in samples:required.update(paths.values())
    if (set(c['samples'])!=set(ARMS) or any(len(x)!=32 for x in c['samples'].values())
            or not required<=c['saved_inputs'].keys()):raise ValueError('UNPINNED_EVIDENCE')
    for p,h in c['export_hashes'].items():
        if c['saved_inputs'][p]!=h:raise ValueError('FINAL_EXPORT_IDENTITY')
    for name,want in c['evaluation_packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT: '+name)
    identity_gate(load_bundle(root,c),c)
    return c


def load_bundle(root,c):
    root=Path(root)
    records={k:io.read(root/p) for k,p in c['records'].items()}
    samples={a:[{k:io.read(root/p) for k,p in paths.items()} for paths in c['samples'][a]] for a in ARMS}
    return dict(records=records,samples=samples)


def identity_gate(bundle,c):
    """Never infer new performance from saved scalars during this gate."""
    r=bundle['records']
    expected={'A':('STAGE_A_PROXY_AND_DF_PASS',30,'e1085c95f5c01b68566e17d543ace5832cb94061'),
        'B1':('B1_MECHANISM_EVALUATION_COMPLETE',141,B1_SOURCE),
        'B2':('B2_FINITE_BIT_NATIVE_CALIBRATION_COMPLETE',652,B2_SOURCE)}
    for stage,(status,count,source) in expected.items():
        stop=r[stage+'_STOP']
        if (r[stage+'_result']['status']!=status or stop['failed'] or not stop['mandatory_stop']
                or stop['scientific_units']!=count or r[stage+'_SOURCE']['execution_commit']!=source):
            raise ValueError('SAVED_SOURCE_OR_STOP: '+stage)
    if (r['B1_STOP']['call_counts']!=b1.LIMITS or r['B2_STOP']['call_counts']!=b2.LIMITS
            or r['B2_STOP']['technical_relaunch']!=1 or r['B2_STOP']['resume']
            or r['B2_STOP']['retry_within_run'] or r['B2_STOP']['cumulative_scientific_units']!=1081
            or r['B1_contract']['policy']!=b1.POLICY or r['B2_original_contract']['policy']!=b2.POLICY
            or r['B2_original_contract']['compiler']!=b2.profile_dict()
            or r['B2_SOURCE']['sampling_SOURCE_sha256']!=SAMPLING_SOURCE
            or r['B2_result']['cost_support'] is not None
            or r['B2_result']['whole_law_synthesis_bound'] is not None
            or r['B2_result']['implemented_law_error'] is not None):
        raise ValueError('SAVED_POLICY_OR_FALSE_QUALIFICATION')
    for stage in ('B1','B2'):
        source=r[stage+'_SOURCE'];manifest=r[stage+'_manifest']
        if source['files']!=manifest['files']:raise ValueError('SAVED_SOURCE_FILES')
    if (r['B1_contract']['final_exports']!=c['final_exports']
            or r['B2_original_contract']['final_exports']!=c['final_exports']
            or set(r['B2_shells'])!={'head','X','Y'} or r['B2_result']['shells']!=r['B2_shells']):
        raise ValueError('EXPORT_OR_SHELL_IDENTITY')
    for arm in ARMS:
        final=c['final_exports'][arm];final_hash=c['export_hashes'][final]
        if any(r[k]['saved_inputs'][final]!=final_hash for k in ('B1_contract','B2_original_contract')):
            raise ValueError('ARM_FINAL_EXPORT_HASH')
        br=r['B1_result']['arms'][arm]
        if (br['arm']!=arm or br['classification']['status']!='POTENTIALLY_FEASIBLE'
                or [x['m'] for x in br['rounds']]!=list(range(13))
                or r[arm+'_ground']['status']!='PASS'
                or r['B2_result']['arms'][arm]['samples']!=32):raise ValueError('ARM_ROUND_OR_GROUND')
        for row in br['rounds']:
            if (set(row['reference'])!={'pf_operator_error','rte_corrected_operator_error','attenuation','raw_signal_parts','evidence_kind'}
                    or row['synthesis_error'] is not None or row['implemented_law_error'] is not None):
                raise ValueError('B1_ERROR_SCHEMA_OR_UNQUALIFIED_PROMOTION')
        seeds=set()
        for i,sample in enumerate(bundle['samples'][arm]):
            t=sample['trajectory']
            if (t['arm']!=arm or t['sample']!=i or t['SOURCE_sha256']!=SAMPLING_SOURCE
                    or t['final_JU_sha256']!=final_hash or t['microsteps']!=16
                    or t['quantum_shots'] or t['reused_as_coherent_repetitions']
                    or t['seed'] in seeds):raise ValueError('SAMPLE_ARM_SEED_OR_COHORT')
            seeds.add(t['seed'])
            if sample['XY']['status']!='PASS' or sample['metrics']['measurement_count']!=0:
                raise ValueError('NATIVE_SAMPLE_VERDICT')
            reports=[sample['raw']['reference'],sample['raw']['unitarity'],
                sample['compiled']['reference'],sample['compiled']['raw'],sample['compiled']['unitarity']]
            reports.extend(sample['adjoint'][kind][key] for kind in ('raw','compiled') for key in ('reference','returned'))
            if any(v['rtol']!=0 or v['tolerance']!=1e-10 or v['max_entry']>v['tolerance'] for v in reports):
                raise ValueError('RAW_COMPILED_REFERENCE_GATE')
    return dict(status='STATIC_IDENTITY_SCHEMA_AND_SAVED_VERDICTS_PASS',arms=list(ARMS),
        B2_calibration_samples=64,old_run01_samples_pooled=False,numerical_reanalysis=False)


def attach(bundle,c,arm):
    r=bundle['records'];saved=r['B1_result']['arms'][arm]
    accuracy._finite(saved['saved_DF_bound_Ha'],'saved DF bound')
    for x in saved['rounds']:
        for key in ('pf_operator_error','rte_corrected_operator_error','attenuation'):
            accuracy._finite(x['reference'][key],key)
    return dict(arm=arm,target_id=arm+':'+c['export_hashes'][c['final_exports'][arm]],
        rounds=saved['rounds'],df_bound_Ha=saved['saved_DF_bound_Ha'],
        kernel_samples=[s['metrics'] for s in bundle['samples'][arm]],
        saved_calibration=r['B2_result']['arms'][arm],shells=r['B2_shells'],
        B1_SOURCE=B1_SOURCE,B2_SOURCE=B2_SOURCE,sampling_SOURCE=SAMPLING_SOURCE,
        error_evidence=None,DF_and_reference_qualification='binary64 numerical pilot with engineering guard')


def cost_calibration(packet,*,support=None,alpha_per_metric=None):
    """Unknown support -> existing point-only path. Sample extrema are rejected."""
    if support is not None and (support.get('kind') not in ('analytic_all_law','exhaustive_all_law')
            or support.get('law')!='actual_implemented_law' or not support.get('provenance')):
        raise ValueError('OBSERVED_EXTREMA_ARE_NOT_COST_SUPPORT')
    out=accuracy.bounded_cost_intervals(packet['kernel_samples'],
        support_bounds=None if support is None else support['bounds'],alpha_per_metric=alpha_per_metric,
        support_provenance=None if support is None else support['provenance'])
    if out['mean']!=packet['saved_calibration']['sample_mean'] or out['classical_cost_samples']!=32:
        raise ValueError('SAVED_KERNEL_CALIBRATION_IDENTITY')
    return dict(out,target_id=packet['target_id'],policy='native_kernel_repeat_with_fences',
        estimand='finite-bit implementation kernel cost',cost_support=None if support is None else support)


def serial_cost(packet,kernel,m,axis):
    shell={k:math.fsum(packet['shells'][x]['metrics'][k] for x in ('head',axis)) for k in METRICS}
    record=dict(axis=axis,mean=shell,lower=shell,upper=shell,policy='separate_native_readout',
        simultaneous_interval_qualified=True)
    return connection.compose_serial_cost(kernel,record,m=m,substeps=4,target_id=packet['target_id'])


def qualification(packet,lane):
    if lane!='evidence':
        scenario=next(x for x in SCENARIOS if x['id']==lane)
        values=[dict(m=x['m'],synthesis=x['reference']['attenuation']*scenario['synthesis_fraction_of_A'],
            law=x['reference']['attenuation']*scenario['law_fraction_of_A']) for x in packet['rounds']]
        return dict(kind='ASSUMED_SCENARIO',law_model=scenario['law_model'],round_errors=values,
            assumptions=scenario['assumptions'],evidence_promoted=False,all_law_bound_qualified=False)
    missing=packet['error_evidence']
    if missing is None:
        return dict(kind='UNQUALIFIED',law_model='actual_implemented_law',
            round_errors=[dict(m=x['m'],synthesis=None,law=None) for x in packet['rounds']],
            reason='No all-law synthesis or implemented-law raw signal bound;B2 sample residuals are not bounds',
            evidence_promoted=False,all_law_bound_qualified=False)
    # Extension hook for independently obtained C2 evidence; C1's saved loader
    # always sets None and cannot silently substitute sampled diagnostics.
    if (missing['kind'] not in ('all_law_numerical_bound','rigorous_all_law_bound')
            or missing['norm']!='raw_complex_signal' or missing['law_model']!='actual_implemented_law'
            or missing['coverage']!='all_trajectories_all_occurrences_and_shells'
            or not missing.get('provenance') or [x['m'] for x in missing['round_errors']]!=list(range(13))):
        raise ValueError('UNQUALIFIED_ERROR_EVIDENCE_SCOPE')
    return dict(missing,all_law_bound_qualified=True,evidence_promoted=False)


def assess_round(target,packet,q,row):
    ref=row['reference'];error=q['round_errors'][row['m']]
    evidence=accuracy.RoundEvidence(m=row['m'],attenuation=ref['attenuation'],
        pf_operator_error=ref['pf_operator_error'],rte_corrected_operator_error=ref['rte_corrected_operator_error'],
        synthesis_signal_error=error['synthesis'],finite_law_signal_error=error['law'],
        target_id=packet['target_id'],state_model='own_DF_ground_state_oracle',alias_branch_verified=True,
        iid_per_shot=True,evidence_kind='numerically_qualified_pilot',law_model=q['law_model'])
    try:value=accuracy.assess_round(target,evidence)
    except (OverflowError,ZeroDivisionError):
        return dict(status='UNRESOLVED_NUMERICAL_SHOT_RANGE',m=row['m'],shots=None,
            reason='existing binary64 shot formula underflow/overflow;no clipping',qualification=q['kind'])
    return dict(value,qualification=q['kind'],assumption_applied=q['kind']=='ASSUMED_SCENARIO',
        actual_CA_achieved=None,rigorous_shots=None)


def energy(target,packet,plans):
    return accuracy.assess_energy(target,packet['df_bound_Ha'],plans,
        state_model='own_DF_ground_state_oracle',alias_branch_verified=True)


def calibration_uncertainty(packet,plans,totals):
    if totals.get('totals') is None:return dict(status='UNRESOLVED_ACCURACY',weighted_SE=None)
    # The SAME 32 samples price every round. Their errors are perfectly shared
    # across reuses: coefficient W=sum N*n multiplies a single sample mean.
    W=sum(2*p['shots']*4*(1<<p['m']) for p in plans)
    return dict(status='DESCRIPTIVE_SHARED_CALIBRATION_ONLY',kernel_reuse_weight=W,
        weighted_SE={k:W*packet['saved_calibration']['sample_mean_standard_error'][k] for k in METRICS},
        samples_per_arm=32,independent_round_calibrations=1,
        guaranteed_CI=None,ideal_law_cost_transfer_qualified=False,
        note='No sqrt(13) reduction;no 32x shot multiplier;SE is not an enclosure')


def compare(baseline,proposed):
    a,b=(x['resources'] for x in (baseline,proposed))
    if a.get('totals') is None or b.get('totals') is None:
        return dict(status='UNRESOLVED_ACCURACY_OR_ERRORS',mean_difference=None,mean_ratio=None,
            evidence_qualified_advantage=None,guaranteed_advantage=None)
    if a['cost_scope']!=b['cost_scope']:raise ValueError('COMPARISON_SCOPE')
    assumed=any(x['qualification']['kind']=='ASSUMED_SCENARIO' for x in (baseline,proposed))
    return dict(status='SCENARIO_POINT_ESTIMATES_ONLY' if assumed else 'EVIDENCE_CONDITIONAL_POINT_ESTIMATES_ONLY',
        mean_difference={k:b['totals'][k]-a['totals'][k] for k in METRICS},
        mean_ratio={k:b['totals'][k]/a['totals'][k] if a['totals'][k] else None for k in METRICS},
        evidence_qualified_advantage=None,guaranteed_advantage=None,
        ideal_law_mean=None,actual_CA_achieved=None,
        reason='Assumed errors and/or cost-transfer and finite calibration;not certified same-CA superiority')


def execute(root,c,output,call):
    output=Path(output)
    bundle=call('saved_bundle_load','saved JSON only;no NPZ/QPY decode',lambda:load_bundle(root,c))
    io.save(output/'identity.json',call('identity_gate','A/B1/B2 SOURCE and arm/seed/correctness identity',lambda:identity_gate(bundle,c)))
    def target_gate():
        target=accuracy.AccuracyTarget(**TARGET)
        if target.horizon()['maximum_round_index_M']!=12:raise ValueError('REGISTERED_13_ROUND_HORIZON')
        return target
    target=call('target_validation','fixed CA/phase/failure/horizon',target_gate)
    packets={};costs={}
    for arm in ARMS:
        packet=call('arm_attachment',arm,lambda arm=arm:attach(bundle,c,arm));packets[arm]=packet
        kernel=call('cost_calibration',arm,lambda:cost_calibration(packet))
        io.save(output/(arm+'_kernel_cost.json'),kernel)
        costs[arm]=[]
        for m in range(13):
            for axis in ('X','Y'):
                costs[arm].append(call('serial_cost',f'{arm}:m{m}:{axis}',lambda m=m,axis=axis:serial_cost(packet,kernel,m,axis)))
        io.save(output/(arm+'_serial_costs26.json'),costs[arm])
    plan=integrated.resource_plan(task_count=6,full_dimension=1)
    plan.update(policy='Independent scalar arm/scenario tasks only;no BLAS/Qiskit;no shared settings changed',BLAS_threads_per_worker=None)
    io.save(output/'resource_plan.json',plan)
    def task(arm,lane):
        packet=packets[arm];label=arm+':'+lane
        q=call('signal_qualification',label,lambda:qualification(packet,lane))
        plans=[]
        for row in packet['rounds']:
            value=call('round_accuracy',label+':m'+str(row['m']),lambda row=row:assess_round(target,packet,q,row))
            plans.append(value);io.save(output/(arm+'_'+lane+f'_m{row["m"]:02d}.json'),value)
        assessment=call('energy_assessment',label,lambda:energy(target,packet,plans))
        totals=call('total_aggregation',label,lambda:accuracy.aggregate_totals(assessment,plans,costs[arm],cost_scope=COST_SCOPE))
        uncertainty=call('calibration_uncertainty',label,lambda:calibration_uncertainty(packet,plans,totals))
        value=dict(arm=arm,lane=lane,target_id=packet['target_id'],qualification=q,rounds=plans,
            conditional_energy_assessment=assessment,resources=totals,calibration_uncertainty=uncertainty,
            evidence_qualified_shots=None if q['kind'] in ('UNQUALIFIED','ASSUMED_SCENARIO') else [p['shots'] for p in plans],
            actual_CA_achieved=None,rigorous_same_CA=None,guaranteed_total_resources=None,
            B1_SOURCE=B1_SOURCE,B2_SOURCE=B2_SOURCE,sampling_SOURCE=SAMPLING_SOURCE)
        io.save(output/(arm+'_'+lane+'.json'),value);return (arm,lane),value
    results={};failures=[]
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures=[pool.submit(task,a,lane) for a in ARMS for lane in LANES]
        for f in as_completed(futures):
            try:key,value=f.result();results[key]=value
            except BaseException as error:call.stop.set();failures.append(error)
    if failures:raise failures[0]
    comparisons={lane:call('comparison',lane,lambda lane=lane:compare(results['baseline',lane],results['proposed',lane])) for lane in LANES}
    value=dict(status='C1_SAVED_SCALAR_SCENARIOS_COMPLETE',scope=SCOPE,
        evidence_lane={a:results[a,'evidence'] for a in ARMS},
        assumed_scenarios={lane:{a:results[a,lane] for a in ARMS} for lane in LANES[1:]},
        comparisons=comparisons,scientific_units=sum(call.counts.values()),
        CA_achieved=None,shots_guaranteed=None,total_resources_guaranteed=None,
        all_law_synthesis=None,implemented_law_error=None,cost_support=None,
        new_sampling=0,quantum_shots=0,new_fit=0,build_compile=0,mandatory_stop=True)
    io.save(output/'result.json',value);return value


def run(root,path,output):
    root=Path(root).resolve();c=preflight(root,path);output=Path(output).resolve()
    if output!=root/FRESH or output.exists():raise ValueError('FRESH_C1_NO_RETRY_RESUME')
    output.mkdir(parents=True);call=integrated.Calls(output,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        io.save(output/'contract.json',c);io.save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,output,call)
    except BaseException as error:
        call.stop.set();io.save(output/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(output/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
