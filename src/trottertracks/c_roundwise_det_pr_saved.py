"""Opt-in saved M2/M4 scalar reanalysis. Preflight is hash/metadata only.

No molecular array decode, operator generation, PF/RTE, fitter, sampling,
QPY/build/compile or quantum measurements. Execution still requires separate
user permission; no approval certificate/record/grant is used.
"""
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c_roundwise_det_pr as mixed
from . import c1_stage1_execution as io
from . import c1_stage3_integrated as integrated

SCOPE='C_ROUNDWISE_SAME_CDF8_DET_PR_SAVED_V1'
BASE='0cb2e5ea986320283465ea51e5b273d0b3313b96'
FRESH='artifacts/research_c_roundwise_det_pr/2026-10-30/run01'
LIMITS=dict(saved_bundle=1,common_energy_binding=1,qualification=3,
    round_accuracy=78,round_cost=78,allocation=3,schedule_energy=9,
    schedule_total=9,shared_uncertainty=3,comparison=3,summary=1)
MAX_CALLS=sum(LIMITS.values())
EXECUTIONS=dict(D='a6ffc23d94f6488a4769596f089a91c895d49214',
    PR='7dbabc7866cc62a5460bf1579ce84c59493d8e1e',N='dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad')
STATUSES=dict(D='M2_SAME_JU_DETERMINISTIC_COMPLETE',PR='M4_R128_SHOTS_BREAKEVEN_COMPLETE',N='NATIVE128_FINITE_COHORT_COMPLETE')


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    for key,want in dict(scope=SCOPE,base_commit=BASE,implementation=mixed.VERSION,
            target=mixed.TARGET,lanes=list(mixed.LANES),call_limits=LIMITS,
            maximum_logical_units=MAX_CALLS,fresh_output=FRESH,retry=0,resume=0).items():
        if c[key]!=want:raise ValueError('FIXED_MIXED_CONTRACT:'+key)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    source=io.read(root/c['source_manifest'])
    for files in (source['files'],c['saved_inputs']):
        for p,h in files.items():
            file=(root/p).resolve()
            if not file.is_relative_to(root) or io.sha(file)!=h:raise ValueError('SOURCE_OR_INPUT:'+p)
    if not set(c['records'].values())<=set(c['saved_inputs']):raise ValueError('UNPINNED_RECORD')
    for p,h in source['reuse_files'].items():
        if io.sha(root/p)!=h:raise ValueError('REUSED_SOURCE:'+p)
    for package,want in c['packages'].items():
        if importlib.metadata.version(package)!=want:raise ValueError('ENVIRONMENT:'+package)
    if (root/FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def load_bundle(root,c):
    return {k:io.read(Path(root)/p) for k,p in c['records'].items()}


def bridge(bundle,c):
    """Bind saved SOURCE/contract/H/ground before creating common target IDs."""
    d,p,n=(bundle[a+'_contract'] for a in ('D','PR','N'))
    for a,old in [('D',d),('PR',p),('N',n)]:
        stop=bundle[a+'_STOP'];source=bundle[a+'_SOURCE']
        if (source['execution_commit']!=EXECUTIONS[a] or bundle[a]['status']!=STATUSES[a]
                or stop['failed'] or not stop['mandatory_stop'] or stop['retry'] or stop['resume']
                or stop['scientific_units']!=old['maximum_logical_units']):raise ValueError('OLD_SOURCE_STATUS_STOP:'+a)
        if old['target']!=mixed.TARGET:raise ValueError('ACCURACY_POLICY')
        for name in ('compiler','gaussian','compiler_phase','shells'):
            if old[name]!=d[name]:raise ValueError('NATIVE_POLICY:'+name)
    if (d['conditions']['s']!=1 or d['conditions']['PF']!='all-frame ordered symmetric S2'
            or p['policy']['s']!=1 or p['policy']['r']!=128 or p['policy']['K']!=0
            or p['policy']['ld']!=2 or p['policy']['rounds']!=list(mixed.ROUNDS)):
        raise ValueError('REGISTERED_TWO_METHODS')
    # The saved native contract extends precisely the same M4 scientific input.
    for key in ('input','exports','grounds','references','original_input_sha256'):
        if n[key]!=p[key]:raise ValueError('M4_NATIVE_BINDING:'+key)
    for path in (p['input'],p['exports']['proposed'],p['grounds']['proposed'],p['references']['proposed']):
        if n['saved_inputs'][path]!=p['saved_inputs'][path]:raise ValueError('M4_NATIVE_INPUT_HASH')
    for key,value in p['policy'].items():
        if key not in ('law','PF_reuse') and n['policy'].get(key)!=value:
            raise ValueError('M4_NATIVE_SCIENTIFIC_CONDITION:'+key)
    fields={}
    for method,old,condition in [('D',d,d['conditions']),('PR',p,p['policy'])]:
        paths=dict(input_sha256=old['input'],JU_sha256=old['exports']['proposed'],
            ground_sha256=old['grounds']['proposed'],reference_sha256=old['references']['proposed'])
        hashes={key:old['saved_inputs'][path] for key,path in paths.items()}
        if any(c['saved_inputs'].get(path)!=hashes[key] for key,path in paths.items()):raise ValueError('H_GROUND_INPUT_HASH')
        fields[method]=dict(**hashes,physical_input_sha256=old['original_input_sha256'],
            geometry=[condition['atoms'],condition['distance_A'],condition['basis']],
            sector=[condition['Nalpha'],condition['Nbeta'],36],frame_order=condition['frame_order'],
            kappa_policy=condition['original_g_kappa'],scalar_policy=condition['scalar'],
            spin_policy='existing exact-spin-lift kappa, original full-frame U',state_model=condition['oracle'],
            phase_convention='X+iY=exp(-iEt)',delta=condition['delta'])
    if fields['D']['geometry']!=[4,1.,'STO-3G'] or fields['D']['sector']!=[2,2,36]:raise ValueError('H4_SECTOR')
    identity=mixed.common_target(fields);D=bundle['D']['candidates']['proposed'];PR=bundle['PR']
    if D['id']!='D_proposed' or D['export_sha256']!=fields['D']['JU_sha256']:raise ValueError('D_EXPORT')
    pr_id='proposed:'+fields['PR']['JU_sha256']
    if any(x['target_id']!='D_proposed' for x in D['rounds']) or any(
            x['target_id']!=pr_id for lane in mixed.LANES[1:] for x in PR['B'][lane]['rounds']):
        raise ValueError('OLD_METHOD_TARGET_ID')
    # Old unknown-error rows have no target_id. Their packet binding comes
    # from the pinned M4 SOURCE/contracts, not a new string inserted into them.
    if any(x.get('target_id') not in (None,pr_id) or x['status']!='UNRESOLVED_SIGNAL_ERROR'
            for x in PR['B']['evidence']['rounds']):raise ValueError('OLD_UNKNOWN_LANE_SCHEMA')
    df=PR['A']['saved_DF_bound_Ha'];mixed.accuracy._finite(df,'saved DF')
    if df!=bundle['B1']['arms']['proposed']['saved_DF_bound_Ha']:raise ValueError('SAME_DF_ERROR_SOURCE')
    if d['records']['B1']!=p['records']['B1']:raise ValueError('DF_PROVENANCE')
    E=bundle['ground']['energy_Ha']
    if (not math.isfinite(E) or not -10.<=E<=10. or bundle['ground']['status']!='PASS' or bundle['ground']['alias_margin']<=0
            or bundle['physical_ground']['status']!='PASS' or bundle['physical_ground']['alias_margin']<=0
            or d['allowances']['alias_window_Ha']!=[-10.,10.]):raise ValueError('GROUND_ALIAS')
    if any([r['m'] for r in rows]!=list(mixed.ROUNDS) for rows in
            [D['PF_rounds'],D['synthesis_rounds'],D['rounds'],PR['A']['rounds']]):raise ValueError('ROUND_COVERAGE')
    if bundle['N']['calibration']['samples']!=32:raise ValueError('SAVED_CALIBRATION32')
    # Reference-defined deterministic RTE/law are non-applicable, not r=0 PR.
    rows={a:[] for a in mixed.METHODS}
    for m in mixed.ROUNDS:
        common=dict(m=m,elapsed_time=.1*(1<<m),state_model=mixed.STATE,
            phase_convention='X+iY=exp(-iEt)',iid_per_shot=True)
        rows['D'].append(dict(common,attenuation=1.,pf_operator_error=D['PF_rounds'][m]['pf_operator_error'],
            rte_corrected_operator_error=0.,synthesis_signal_error=D['synthesis_rounds'][m]['complex_signal_bound_pilot'],
            finite_law_signal_error=0.,source_target_id=D['id']))
        ref=PR['A']['rounds'][m]['reference']
        rows['PR'].append(dict(common,**{k:ref[k] for k in ('attenuation','pf_operator_error','rte_corrected_operator_error')},
            synthesis_signal_error=None,finite_law_signal_error=None,source_target_id=pr_id))
    shells={name:bundle['shells'][name]['metrics'] for name in ('head','X','Y')}
    return dict(target_id=identity,bindings=fields,rows=rows,df=df,alias_verified=True,
        kernel_means=dict(D=D['metrics'],PR=bundle['N']['calibration']['sample_mean']),shells=shells,
        PR_SE=bundle['N']['calibration']['sample_mean_standard_error'],saved_PR_rounds={lane:PR['B'][lane]['rounds'] for lane in mixed.LANES},
        evidence_sources={a:c['records'][a] for a in mixed.METHODS},
        scope='same DF ground/energy; methods and evidence remain distinct; no target string replacement proof')


def comparison(evaluations,allocation):
    totals={k:v['resources'].get('totals') for k,v in evaluations.items()}
    delta=None if totals['mixed']is None or totals['all_D']is None else {
        k:totals['mixed'][k]-totals['all_D'][k] for k in mixed.accuracy.METRICS}
    return dict(status=allocation['status'],conditional_totals=totals,mixed_minus_all_D=delta,
        schedule=allocation['schedule'],selection_statistic='RZ point minimum; D on exact tie',
        actual_CA=None,population_advantage=None,guaranteed_advantage=None)


def execute(root,c,out,call):
    data=call('saved_bundle','M2/M4/native/contracts/ground metadata only',lambda:load_bundle(root,c))
    packet=call('common_energy_binding','H/JU/ground/reference/phase/source',lambda:bridge(data,c))
    io.save(out/'common_energy.json',{k:packet[k] for k in ('target_id','bindings','scope')})
    target=mixed.accuracy.AccuracyTarget(**mixed.TARGET);results={}
    for lane in mixed.LANES:
        q=call('qualification',lane,lambda:dict(lane=lane,assumed=lane!='evidence',
            D_synthesis='saved numerical pilot' if lane=='evidence' else '0 counterfactual' if lane=='ideal_exact_native' else '.001*A counterfactual',
            PR_synthesis_law='unknown' if lane=='evidence' else '0 counterfactual' if lane=='ideal_exact_native' else '.001*A each counterfactual',
            evidence_promoted=False))
        plans={a:[] for a in mixed.METHODS};costs={a:[] for a in mixed.METHODS}
        for a in mixed.METHODS:
            for row in packet['rows'][a]:
                m=row['m'];label=f'{lane}/{a}/m{m:02d}'
                def registered_assessment():
                    value=mixed.assess(target,packet['target_id'],a,lane,row,
                        evidence_id=packet['evidence_sources'][a]+f':m{m}',alias_verified=packet['alias_verified'])
                    if a=='PR':
                        old=packet['saved_PR_rounds'][lane][m]
                        if value['status']!=old['status'] or value['shots']!=old['shots']:
                            raise ValueError('SAVED_PR_SHOTS_OR_STATUS_CHANGED')
                    return value
                plan=call('round_accuracy',label,registered_assessment)
                price=call('round_cost',label,lambda a=a,m=m:mixed.costs(packet['target_id'],a,m,packet['kernel_means'][a],packet['shells']))
                plans[a].append(plan);costs[a].extend(price)
                io.save(out/(lane+'_'+a+f'_m{m:02d}.json'),dict(accuracy=plan,cost=price))
        allocation=call('allocation',lane,lambda:mixed.select(plans,costs))
        io.save(out/(lane+'_allocation.json'),allocation)
        schedules=dict(all_D=['D']*13,all_PR=['PR']*13,mixed=allocation['schedule']);evaluated={}
        for name,schedule in schedules.items():
            evaluated[name]=mixed.evaluate_schedule(target,packet['df'],plans,costs,schedule,
                alias_verified=packet['alias_verified'],call=call,label=lane+'/'+name)
        uncertainty=call('shared_uncertainty',lane,lambda:mixed.shared_uncertainty(evaluated,packet['PR_SE']))
        compared=call('comparison',lane,lambda:comparison(evaluated,allocation))
        results[lane]=dict(qualification=q,allocation=allocation,evaluations=evaluated,uncertainty=uncertainty,comparison=compared)
        io.save(out/(lane+'_result.json'),results[lane])
    result=call('summary','all three lanes; no population/actual CA claim',lambda:dict(
        status='SAVED_ROUNDWISE_CONDITIONAL_REANALYSIS_COMPLETE',lanes=results,
        new_fit=0,PF_RTE_operator_evaluation=0,sampling=0,build_compile=0,quantum_shots=0,
        actual_CA=None,all_law_synthesis=None,cost_support=None,guaranteed_advantage=None,mandatory_stop=True))
    io.save(out/'result.json',result);return result


def run(root,path):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH;out.mkdir(parents=True,exist_ok=False)
    call=integrated.Calls(out,LIMITS,MAX_CALLS);tick=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),first_failure=call.first_failure,retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
