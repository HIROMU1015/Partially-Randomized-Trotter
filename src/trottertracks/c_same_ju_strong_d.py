"""Opt-in single saved J/U -> existing M2 strong full-frame S2 pipeline.

Import/preflight are byte/metadata-only. No new optimizer, Gaussian emitter,
shot formula, or approval-proof machinery. Science requires separate permission.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import time
from . import c_pr_aware_df_m2 as m2

io,accuracy,integrated=m2.io,m2.accuracy,m2.integrated
BASE='aced1647b81e8d28091058cdc7b6e3e1ee54896a'
SCOPE='C_SAME_SELECTED_JU_STRONG_DETERMINISTIC_S2_V1'
VERSION='saved_selected_JU_M2_single_arm_v1'
FAMILY='artifacts/research_c_same_ju_strong_d/2026-11-02'
FRESH=FAMILY+'/run01'
CONDITIONS=dict(m2.CONDITIONS,saved_PR='same selected JU;partial S2 LD2 s1 r128 K0;13 rounds',
    candidate='update005',DF_budget_Ha=.00019875,deterministic_r=None,deterministic_K=None,
    DF_error='reuse saved sector norm+engineering guard;no recomputation')
COMPILER=dict(qiskit='1.3.0',optimization_level=1,seed_transpiler=17,
    basis_gates=['rz','sx','x','cx'],approximation_degree=1.,qubits_initially_zero=False)
RECORDS={'new_result','new_STOP','new_SOURCE','new_contract','physical','ground_report',
    'physical_ground_report','shell_reports','old_D','old_D_STOP','old_D_SOURCE',
    'old_PR','old_PR_STOP','old_PR_SOURCE','old_D_contract','old_PR_contract'}
LIMITS=dict(evidence_load=1,input_load=1,fit_load=1,reference_load=1,candidate_binding=1,
    car_context=1,raw_factor=9,sector_gate=1,ground_load=1,ground_gate=1,
    sector_eigenpair=1,native_model=1,PF_kernel=1,PF_round=13,full_reference=1,
    reference_sector_gate=1,gaussian_build=9,relative_gaussian=16,shell_load=6,
    shell_decode=6,shell_structure=3,shell_segment=3,kernel_build=1,fusion_gate=1,
    compile=1,operator_decode=2,segment_assessment=1,adjoint_build=2,adjoint_decode=2,
    adjoint_gate=2,XY_gate=1,metrics=1,round_synthesis=13,round_accuracy=13,energy=1,
    round_serial_cost=13,aggregate=1,ideal_round_accuracy=13,ideal_energy=1,
    ideal_aggregate=1,comparison=6,result_summary=1)
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    for k,w in dict(base_commit=BASE,scope=SCOPE,implementation=VERSION,conditions=CONDITIONS,
        compiler=COMPILER,target=m2.TARGET,allowances=m2.ALLOWANCES,call_limits=LIMITS,
        maximum_logical_units=MAX_CALLS,fresh_output=FRESH,retry=0,resume=0).items():
        if c[k]!=w:raise ValueError('FIXED_SAME_JU:'+k)
    if set(c['records'])!=RECORDS:raise ValueError('RECORD_SCHEMA')
    paths={c[k] for k in ('input','export','reference','basis','ground')}
    paths.update(c['records'].values());paths.update(p for x in c['shells'].values() for p in x.values())
    if not paths<=c['saved_inputs'].keys():raise ValueError('UNPINNED_INPUT')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,h in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=h:raise ValueError('SOURCE_INPUT:'+name)
    for package,w in c['packages'].items():
        if importlib.metadata.version(package)!=w:raise ValueError('ENVIRONMENT:'+package)
    from .c1_h4_native_fair import GAUSSIAN,PHASE
    if c['gaussian']!=GAUSSIAN or c['compiler_phase']!=PHASE:raise ValueError('NATIVE_POLICY')
    if (root/FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def evidence(root,c):
    r={k:io.read(Path(root)/v) for k,v in c['records'].items()};new=r['new_result'];selected=new['selected']
    if (r['new_STOP']['failed'] or r['new_STOP']['scientific_units']!=543
            or new['performance']['status']!='PERFORMANCE_NATIVE_COMPLETE'
            or selected['id']!='update005' or not selected['qualification']['status']=='NUMERICAL_PILOT_ADMITTED'
            or selected['physical']!=r['physical']):raise ValueError('NEW_SELECTED_PROVENANCE')
    orig=r['new_contract']
    if (orig['scope']!='C_PR_INTEGRATED_H4_REDUCED_PERFORMANCE_V1'
            or orig['target']!=m2.TARGET or orig['compiler']!=COMPILER
            or orig['input_paths']['input']!=c['input']
            or orig['saved_inputs'][c['input']]!=c['saved_inputs'][c['input']]
            or io.sha(Path(root)/c['records']['new_SOURCE'])!=orig['source_manifest_sha256']):
        raise ValueError('ORIGINAL_SOURCE_INPUT_POLICY')
    df=r['physical']
    if (df['candidate_sha256']!=c['saved_inputs'][c['export']] or df['actual_rank']!=8
            or df['kappa']!='original_g_corrected' or df['scalar']!='nuclear_once'
            or df['budget_Ha']!=.00019875 or not df['budget_PASS']
            or df['sector_error']+df['numerical_guard']>df['budget_Ha']):raise ValueError('SAVED_DF_BUDGET_IDENTITY')
    if any(r[k]['status']!='PASS' or r[k]['alias_margin']<=0 for k in ('ground_report','physical_ground_report')):
        raise ValueError('SAVED_GROUND_ALIAS')
    for lane,x in new['performance']['lanes'].items():
        if [v['m'] for v in x['rounds']]!=list(range(13)):raise ValueError('SAVED_PR_ROUNDS')
        if lane!='evidence' and any(v['target_id']!='integrated:'+df['candidate_sha256'] for v in x['rounds']):
            raise ValueError('SAVED_PR_JU_IDENTITY')
        if lane!='evidence' and not x['energy']['same_accuracy']:raise ValueError('SAVED_PR_SCENARIO')
    if (r['old_D']['status']!='M2_SAME_JU_DETERMINISTIC_COMPLETE' or r['old_D_STOP']['failed']
            or r['old_D_STOP']['scientific_units']!=253
            or r['old_D_SOURCE']['execution_commit']!='a6ffc23d94f6488a4769596f089a91c895d49214'
            or r['old_PR']['status']!='NATIVE128_FINITE_COHORT_COMPLETE' or r['old_PR_STOP']['failed']
            or r['old_PR_STOP']['scientific_units']!=332
            or r['old_PR_SOURCE']['execution_commit']!='dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad'):
        raise ValueError('OLD_REFERENCE_PROVENANCE')
    oldD,oldPR=r['old_D_contract'],r['old_PR_contract'];old_export=oldD['exports']['proposed']
    if (oldPR['exports']['proposed']!=old_export
            or oldD['saved_inputs'][old_export]!=oldPR['saved_inputs'][old_export]
            or r['old_D']['candidates']['proposed']['export_sha256']!=oldD['saved_inputs'][old_export]
            or any(x['target']!=m2.TARGET or x['compiler']!=COMPILER
                or x['input']!=c['input'] or x['saved_inputs'][c['input']]!=c['saved_inputs'][c['input']]
                for x in (oldD,oldPR))):
        raise ValueError('OLD_SAME_JU_COMPARISON')
    return r


def ideal_plans(pf,packet,call):
    """Matched PR ideal scenario only; not an assertion about actual synthesis."""
    target=accuracy.AccuracyTarget(**m2.TARGET);plans=[]
    for row in pf:
        def assess(row=row):
            e=accuracy.RoundEvidence(m=row['m'],attenuation=1.,pf_operator_error=row['pf_operator_error'],
                rte_corrected_operator_error=0.,synthesis_signal_error=0.,finite_law_signal_error=0.,
                target_id=packet['id'],state_model=packet['state_model'],alias_branch_verified=packet['alias_verified'],
                iid_per_shot=True,law_model='deterministic')
            return accuracy.assess_round(target,e)
        plans.append(call('ideal_round_accuracy','m'+str(row['m']),assess))
    energy=call('ideal_energy','explicit counterfactual exact native',lambda:accuracy.assess_energy(
        target,packet['df_error_with_guard'],plans,state_model=packet['state_model'],alias_branch_verified=packet['alias_verified']))
    return dict(qualification='COUNTERFACTUAL_EXACT_NATIVE;not measured or actual CA',rounds=plans,energy=energy)


def comparisons(saved,numerical,ideal,call):
    pr=saved['new_result']['performance']['lanes'];old=saved['old_PR']['lanes'];oldD=saved['old_D']['candidates']['proposed']['cost']
    output={}
    entries=[('new_PR_ideal/new_D_numerical',pr['ideal_exact_native']['conditional_totals'],numerical,
        'same selected JU/CA/serial policy;PR counterfactual versus D level-B numerical synthesis'),
        ('new_PR_assumed/new_D_numerical',pr['assumed_raw_signal_budget']['conditional_totals'],numerical,
        'same selected JU;PR unverified all-law budgets versus D numerical synthesis'),
        ('new_PR_ideal/new_D_ideal',pr['ideal_exact_native']['conditional_totals'],ideal,
        'same selected JU and matched exact-native counterfactual;PR IID law/finite price mean unqualified'),
        ('old_PR_ideal/old_D',dict(totals=old['ideal_exact_native']['conditional_totals'],cost_scope=oldD['cost_scope']),oldD,
        'same old JU;quote old finite-cohort PR and numerical D scopes'),
        ('old_PR_assumed/old_D',dict(totals=old['assumed_raw_signal_budget']['conditional_totals'],cost_scope=oldD['cost_scope']),oldD,
        'same old JU;different synthesis qualifications retained'),
        ('new_D_numerical/old_D',numerical,oldD,'DF-update descriptive comparison;not objective causal identification')]
    for label,left,right,qualification in entries:
        output[label]=call('comparison',label,lambda left=left,right=right,label=label,qualification=qualification:
            m2.numerical_compare(left,right,label,qualification=qualification))
    return output


def execute(root,c,out,call):
    import numpy as np
    from scipy.linalg import eigh
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b_runner as loader,c_pr_aware_df_stage_b2 as b2
    from . import c1_h4_native_fair as fair,c1_h4_full_synthesis as synthesis,c1_molecule_extension as molecule
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h6_ca_resources import serial_round_cost
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    root,out=Path(root),Path(out)
    for leaf in ('circuits','operators'):(out/leaf).mkdir()
    saved=call('evidence_load','fixed selected and old reference identities',lambda:evidence(root,c))
    p=call('input_load','original g corrected kappa/scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    d=call('fit_load','saved selected JU;no fit',lambda:loader.load(root/c['export'],('J','U')))
    def references():
        ref=loader.load(root/c['reference'],('J','U','generators','H'))
        basis=loader.load(root/c['basis'],('sector_basis_indices',))['sector_basis_indices']
        ref['full_input_indices']=b.stage1.bit_reverse_indices(8)[basis];return ref
    sector=call('reference_load','saved independent sector generators and basis',references)
    def bind():
        m2.validate_arrays(p,d,'selected')
        if not all(np.array_equal(d[k],sector[k]) for k in ('J','U')):raise ValueError('SAME_SELECTED_JU_ARRAYS')
        return dict(status='PASS',selected_export=c['saved_inputs'][c['export']],rank=8,DF_error_recomputed=False)
    io.save(out/'identity.json',call('candidate_binding','selected/physical/PR identity',bind))
    car=call('car_context','full256 CAR',lambda:b.stage1.car_bilinears(8))
    G=[call('raw_factor',str(i),lambda i=i:b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)
        if i==0 else b.stage1.raw_frame(d['J'][i-1],d['U'][i-1],car)[0]) for i in range(9)]
    ix,checks=call('sector_gate','9 full CAR vs saved SectorCAR factors',lambda:m2.sector_match(G,sector));io.save(out/'sector.json',checks)
    ground=call('ground_load','same saved own-DF oracle',lambda:loader.load(root/c['ground'],('energy','sector_state')))
    def ground_gate():
        residual=float(np.linalg.norm(sector['H']@ground['sector_state']-ground['energy']*ground['sector_state']))
        norm=float(abs(np.vdot(ground['sector_state'],ground['sector_state'])-1))
        if (residual>1e-10 or norm>1e-12 or not np.isfinite(residual+norm)
                or abs(float(ground['energy'])-saved['ground_report']['energy_Ha'])>1e-12):raise ValueError('SAVED_GROUND_CORRECTNESS')
        return dict(status='PASS',residual=residual,normalization=norm,alias_margin=saved['ground_report']['alias_margin'])
    io.save(out/'ground.json',call('ground_gate','residual/normalization/alias',ground_gate))
    def eigenpair():
        fair.component_check(sector['H'],sector['H'].conj().T,1e-10,label='saved sector Hermitian')
        return eigh(sector['H'])
    pair=call('sector_eigenpair','PF exact reference;reuse across13 rounds',eigenpair)
    if abs(float(pair[0][0])-float(ground['energy']))>1e-10:raise ValueError('GROUND_NOT_LOWEST')
    arm=call('native_model','same JU full-frame D;no PR split',lambda:b.prepare_arm(**p,**d,role='proposed',
        source_identity=c['saved_inputs'][c['export']],input_identity=c['saved_inputs'][c['input']],ld=8))
    sk={}
    def pf_call(kind,label,fn):
        value=call('PF_kernel' if kind=='sector_kernel_reference' else 'PF_round',label,fn)
        if kind=='sector_kernel_reference':sk['pf']=value[0]['pf']
        return value
    pf=molecule.PF_rounds(sector['generators'],1,pf_call,'selected',eigenpair=pair);io.save(out/'PF.json',pf)
    df=saved['physical'];packet=dict(id='D_selected:'+c['saved_inputs'][c['export']],arm='selected',s=1,aliases=['D_selected'],
        df_error_with_guard=df['sector_error']+df['numerical_guard'],state_model='own_DF_ground_state_oracle',alias_verified=True)
    ideal=ideal_plans(pf,packet,call)
    numerical=None;metrics=None;costs=[]
    if ideal['energy'].get('same_accuracy'):
        ref=call('full_reference','independent full-frame CAR S2',lambda:b.reference.reference_kernels(G,Fraction(1,10),ld=8)['pf'])
        io.save(out/'full_sector_kernel.json',call('reference_sector_gate','full/sector S2',lambda:fair.component_check(
            ref[np.ix_(ix,ix)],sk['pf'],1e-10,label='full vs sector S2')))
        shells={};shell_reports={};shell_metrics={}
        for label in ('head','X','Y'):
            qs={};matrices={}
            for kind in ('raw','compiled'):
                def load(label=label,kind=kind):
                    with (root/c['shells'][label][kind]).open('rb') as f:items=qpy.load(f)
                    if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                    return items[0]
                qs[kind]=call('shell_load',label+kind,load)
                matrices[kind]=call('shell_decode',label+kind,lambda kind=kind:fair.decode(qs[kind]))
            call('shell_structure',label,lambda label=label:b2.shell_assessment(qs['raw'],qs['compiled'],label))
            H=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
            target=np.asarray(np.kron(H if label!='Y' else H@np.diag([1.,-1j]),np.eye(256)),complex)
            report=call('shell_segment',label,lambda label=label:synthesis.segment(matrices['raw'],matrices['compiled'],target,'shell:'+label))
            report['metrics']=saved['shell_reports'][label]['metrics'];shells[label]=matrices;shell_reports[label]=report
            shell_metrics[label]={k:report['metrics'][k] for k in accuracy.METRICS}
        io.save(out/'shells.json',shell_reports)
        builder=StrongBuilder(relative_fusion=True,call=call)
        for block in arm.model.blocks:call('gaussian_build',block.block_id,lambda block=block:fair.basis_request(builder,arm.model,block))
        raw=call('kernel_build','one deterministic h=.1 kernel',lambda:builder.ordered(arm.model,Fraction(1,10),optimized=True))
        io.save(out/'fusion.json',call('fusion_gate','M2 order/direction/shared rules',lambda:m2.fusion_gate(builder,arm.model)))
        profile=CompilerProfile();profile.validate()
        compiled=call('compile','fixed P0 once',lambda:fair.compile_checked(raw,profile));qs=dict(raw=raw,compiled=compiled)
        for kind,q in list(qs.items()):qs[kind+'_adjoint']=call('adjoint_build',kind,lambda q=q:b.native.actual_adjoint(q))
        for kind,q in qs.items():
            with (out/'circuits'/(kind+'.qpy')).open('xb') as f:qpy.dump(q,f)
        plan=integrated.resource_plan(task_count=4,full_dimension=256)
        plan['policy']='four independent raw/compiled/actual-adjoint full512 decodes;build/compile/inverse creation sequential'
        io.save(out/'resource_plan.json',plan)
        def decode(kind):return kind,call('adjoint_decode' if kind.endswith('adjoint') else 'operator_decode',kind,lambda:fair.decode(qs[kind]))
        with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:mats=dict(pool.map(decode,qs))
        for kind,matrix in mats.items():loader.save_arrays(out/'operators'/(kind+'.npz'),matrix=matrix)
        target=fair.controlled(ref);segment=call('segment_assessment','full512 raw/compiled/reference',lambda:synthesis.segment(mats['raw'],mats['compiled'],target,'selected'))
        io.save(out/'segment.json',segment)
        adj={}
        for kind in ('raw','compiled'):
            def gate(kind=kind):
                x=mats[kind+'_adjoint'];return dict(dagger=fair.component_check(x,mats[kind].conj().T,1e-10,label='actual dagger'),
                    reference=fair.component_check(x,target.conj().T,1e-10,label='adjoint/reference'),
                    recovery=fair.component_check(x@mats[kind],np.eye(512),1e-10,label='adjoint return'))
            adj[kind]=call('adjoint_gate',kind,gate)
        io.save(out/'adjoint.json',adj);psi=np.zeros(256,complex);psi[ix]=ground['sector_state']
        io.save(out/'XY.json',call('XY_gate','raw/compiled complex X/Y',lambda:b2.XY_gate(mats['raw'],mats['compiled'],ref,psi,shells)))
        metrics=call('metrics','one P0 kernel',lambda:circuit_metrics(compiled));io.save(out/'metrics.json',metrics)
        packet.update(kernel_metrics={k:metrics[k] for k in accuracy.METRICS},shell_metrics=shell_metrics)
        numerical=m2.evaluate_accuracy(packet,pf,shell_reports,segment,out,call)
        costs=numerical['per_shot_costs']
        if not costs and ideal['energy'].get('same_accuracy'):
            for row in ideal['rounds']:costs.extend(call('round_serial_cost',str(row['m']),lambda row=row:serial_round_cost(packet,row['m'])))
    absent=dict(status='NO_TOTAL_UNQUALIFIED_ACCURACY',totals=None,cost_scope='serial_native_kernel_composition_without_state_preparation')
    ideal['cost']=call('ideal_aggregate','matched counterfactual totals',lambda:accuracy.aggregate_totals(
        ideal['energy'],ideal['rounds'],costs,cost_scope=absent['cost_scope']))
    numcost=absent if numerical is None else numerical['cost']
    compared=comparisons(saved,numcost,ideal['cost'],call)
    result=call('result_summary','one candidate/two D lanes/six scoped comparisons',lambda:dict(
        status='SAME_JU_STRONG_D_COMPLETE' if numerical is not None else 'INFEASIBLE_D_PF_NO_NATIVE',
        selected_export=c['saved_inputs'][c['export']],saved_DF=df,numerical=numerical,ideal_exact_native=ideal,
        metrics=metrics,comparisons=compared,PR_update_ratios_quoted={k:v.get('new_over_old_PR') for k,v in saved['new_result']['performance']['lanes'].items()},
        strict_DF_enclosure=None,actual_CA_achieved=None,PR_all_law_synthesis=None,implemented_PR_law_error=None,
        guaranteed_resource_superiority=None,causal_objective_effect=None,new_fit=0,optimizer=0,PR_recalculation=0,
        quantum_sampling=0,deterministic_kernel_samples=1 if metrics is not None else 0,mandatory_stop=True))
    io.save(out/'result.json',result);return result


def run(root,path):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH;out.mkdir(parents=True)
    call=integrated.Calls(out,LIMITS,MAX_CALLS);start=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):result=execute(root,c,out,call)
        return result
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=repr(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),first_failure=call.first_failure,
            stop_reason=result['status'] if 'result' in locals() else 'CORRECTNESS_OR_EXECUTION_FAILURE',retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
