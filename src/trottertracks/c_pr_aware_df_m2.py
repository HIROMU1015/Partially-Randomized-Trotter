"""Two saved CDF arms -> full-frame strong S2 / existing accuracy and costs.

No science on import/preflight. No fitter, new sampler, Gaussian compiler,
shot solver or cost aggregator. Old runners and their fixed scopes stay intact.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c1_stage1_execution as io
from . import c1_stage3_accuracy as accuracy
from . import c1_stage3_integrated as integrated

VERSION='same_saved_CDF_two_arm_strong_S2_s1_v1'
SCOPE='C_PR_AWARE_DF_M2_SAME_JU_DETERMINISTIC_V1'
BASE='23499d0d84678a53af3c9bccb819901063e9d16a'
FAMILY='artifacts/research_c_pr_aware_df_M2/2026-10-28'
FRESH=FAMILY+'/run01'
ARMS=('baseline','proposed')
TARGET=dict(energy_ha=.00159,df_reserve_fraction=1/8,base_time=.1,
    beta_total=math.pi/6,beta_stat=math.pi/12,alpha_axis=.01/26,alpha_total=.01)
CONDITIONS=dict(atoms=4,distance_A=1.,basis='STO-3G',Nalpha=2,Nbeta=2,sector=36,
    modes=8,controlled_dimension=512,rank=8,frame_order=list(range(8)),
    original_g_kappa=True,scalar='nuclear once',PF='all-frame ordered symmetric S2',
    s=1,delta='1/10',h='1/10',rounds=list(range(13)),PR_split=None,
    oracle='own_DF_ground_state_oracle',state_preparation='EXCLUDED',
    native='StrongBuilder relative_fusion=True,exact-spin-lift kappa,original full-frame U',
    serial='separate head/kernel/readout,all-wire fences,scheduled depth sum',
    saved_PR='partial S2 LD2 s4 h1/40 r16 K0;two saved C1 assumed scenarios')
ALLOWANCES=dict(component_abs=1e-10,unitarity_abs=1e-10,signal_abs=2e-10,
    normalization=1e-12,ground_residual=1e-10,rtol=0,phase_fit=False,
    alias_window_Ha=[-10.,10.],accounting_rel=1e-12,accounting_abs=1e-6)
LIMITS=dict(evidence_load=1,input_load=1,fit_load=2,native_model=2,car_context=1,
    raw_factor=18,reference_load=2,sector_gate=2,ground_load=2,ground_gate=2,
    sector_eigenpair=2,PF_kernel=2,PF_round=26,full_reference=2,reference_sector_gate=2,
    gaussian_build=18,relative_gaussian=32,shell_load=6,shell_decode=6,shell_segment=3,
    shell_structure=3,kernel_build=2,fusion_gate=2,compile=2,operator_decode=4,
    segment_assessment=2,adjoint_build=4,adjoint_decode=4,adjoint_gate=4,
    XY_gate=2,metrics=2,round_synthesis=26,round_accuracy=26,energy=2,
    round_serial_cost=26,aggregate=2,comparison=7,result_summary=1)
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['implementation']!=VERSION or c['base_commit']!=BASE
            or c['conditions']!=CONDITIONS or c['target']!=TARGET or c['allowances']!=ALLOWANCES
            or c['call_limits']!=LIMITS or c['maximum_logical_units']!=MAX_CALLS
            or c['fresh_output']!=FRESH or c['retry'] or c['resume']):
        raise ValueError('M2_FIXED_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for n,h in group.items():
            p=(root/n).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=h:raise ValueError('SOURCE_OR_INPUT:'+n)
    for package,version in c['packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('ENVIRONMENT:'+package)
    if (root/FRESH).exists():raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY')
    return c


def evidence(root,c):
    b={k:io.read(Path(root)/p) for k,p in c['records'].items()}
    if (b['A']['status']!='STAGE_A_PROXY_AND_DF_PASS'
            or b['B1']['status']!='B1_MECHANISM_EVALUATION_COMPLETE'
            or b['B2']['status']!='B2_FINITE_BIT_NATIVE_CALIBRATION_COMPLETE'
            or b['C1']['status']!='C1_SAVED_SCALAR_SCENARIOS_COMPLETE'
            or b['M1']['status']!='M1_DESCRIPTIVE_SAVED_ANALYSIS_COMPLETE'):
        raise ValueError('SAVED_STAGE_STATUS')
    for k,units,commit in [('B1',141,'7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9'),
        ('B2',652,'a1cf0046967c206f414dbad568a75884ee076ebf'),
        ('C1',164,'b1c735da394d4b0a444cbc618ead9781ef4d9134')]:
        stop=b[k+'_STOP']
        if (stop['failed'] or not stop['mandatory_stop'] or stop['scientific_units']!=units
                or b[k+'_SOURCE']['execution_commit']!=commit):raise ValueError('SAVED_SOURCE_STOP:'+k)
    old=b['B2_contract']
    if (old['final_exports']!=c['exports'] or old['stage_A_input']!=c['input']
            or old['B1_references']!=c['references'] or old['B1_grounds']!=c['grounds']
            or old['original_input']['sha256']!=c['original_input_sha256']):raise ValueError('ARM_INPUT_BINDING')
    if old['compiler']!=c['compiler'] or old['gaussian']!=c['gaussian'] or old['compiler_phase']!=c['compiler_phase']:
        raise ValueError('SAVED_NATIVE_POLICY')
    if b['physical_ground']['status']!='PASS' or b['physical_ground']['alias_margin']<=0:
        raise ValueError('SAVED_PHYSICAL_GROUND_ALIAS')
    for arm in ARMS:
        export=c['exports'][arm]
        if old['saved_inputs'][export]!=c['saved_inputs'][export]:raise ValueError('EXPORT_HASH_BINDING')
        df=b['B1']['arms'][arm]['saved_DF_bound_Ha'];accuracy._finite(df,'saved DF')
        if df>TARGET['energy_ha']/8:raise ValueError('SAVED_DF_BUDGET')
        if b[arm+'_ground']['status']!='PASS':raise ValueError('SAVED_GROUND_STATUS')
        for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
            x=b['C1']['assumed_scenarios'][lane][arm]
            if x['target_id']!=arm+':'+c['saved_inputs'][export] or x['qualification']['kind']!='ASSUMED_SCENARIO':
                raise ValueError('PR_SCENARIO_IDENTITY')
    if (b['external']['status']!='H4_STRONG_BASELINE_S1_CA_RESOURCES_COMPLETE'
            or b['external_STOP']['failed'] or b['external_STOP']['scientific_units']!=86
            or b['external_SOURCE']['execution_commit']!='1aeedd96178a3f50a15bc2701c3c332f4f391b64'):
        raise ValueError('EXTERNAL_XDF_PROVENANCE')
    return b


def validate_arrays(packet,values,arm):
    import numpy as np
    h,g,scalar=packet['h'],packet['g'],packet['constant'];J,U=values['J'],values['U']
    if h.shape!=(4,4) or g.shape!=(4,)*4 or scalar.shape!=() or J.shape!=(8,4,4) or U.shape!=J.shape:
        raise ValueError('FIXED_H4_RANK8_SCHEMA:'+arm)
    if any(x.dtype!=np.float64 or not np.isfinite(x).all() for x in (h,g,scalar,J,U)):
        raise ValueError('FLOAT64_FINITE_ARRAYS')
    if not np.array_equal(J,J.transpose(0,2,1)) or np.max(abs(U.transpose(0,2,1)@U-np.eye(4)))>1e-10:
        raise ValueError('J_SYMMETRY_OR_U_ORTHOGONALITY')
    return True


def sector_match(G,saved):
    import numpy as np
    from .c1_stage1 import sector_indices
    from .c1_stage3_molecular import component_check
    ix=saved['full_input_indices']
    if not np.array_equal(np.sort(ix),sector_indices(4,2,2)) or saved['generators'].shape!=(9,36,36):
        raise ValueError('SAVED_SECTOR_MAPPING')
    checks=[component_check(x[np.ix_(ix,ix)],y,1e-10,label='same JU full/sector factor'+str(i))
        for i,(x,y) in enumerate(zip(G,saved['generators'],strict=True))]
    checks.append(component_check(sum(saved['generators']),saved['H'],1e-10,label='saved sector sum'))
    return ix,checks


def fusion_gate(builder,model):
    from . import c1_stage2_native as native
    if [b.block_id for b in model.blocks]!=['one_body']+[f'F{i}' for i in range(8)]:raise ValueError('ORIGINAL_FRAME_ORDER')
    groups=native.runs(model,native.ordered_schedule(9,Fraction(1,10)),optimized=True)
    pairs=[(model.blocks[a[0][0]].block_id,model.blocks[b[0][0]].block_id) for a,b in zip(groups,groups[1:])]
    if ([(r['previous'],r['following']) for r in builder.fusion_records]!=pairs
            or any(r['multiplication']!='V_following.conj().T @ V_previous' for r in builder.fusion_records)):
        raise ValueError('RELATIVE_DIRECTION_OR_SCHEDULE')
    return dict(status='PASS',frame_order=list(range(8)),s=1,h='1/10',
        schedule=[(i,str(t)) for i,t in native.ordered_schedule(9,Fraction(1,10))],
        relative_records=builder.fusion_records,relative_boundaries=len(pairs))


def numerical_compare(left,right,label,*,qualification):
    if left.get('totals') is None or right.get('totals') is None:
        return dict(label=label,status='UNRESOLVED_OR_INELIGIBLE',difference=None,ratio=None)
    if left['cost_scope']!=right['cost_scope']:raise ValueError('COST_SCOPE')
    a,b=left['totals'],right['totals']
    return dict(label=label,status='CONDITIONAL_NUMERICAL_POINT_COMPARISON',
        difference={k:a[k]-b[k] for k in accuracy.METRICS},
        ratio={k:a[k]/b[k] if b[k] else None for k in accuracy.METRICS},
        lower_RZ='left' if a['rz_count']<b['rz_count'] else 'right' if a['rz_count']>b['rz_count'] else 'equal',
        qualification=qualification,actual_CA_achieved=None,guaranteed_superiority=None)


def evaluate_accuracy(packet,pf,shell_reports,segment,out,call):
    from .c1_h4_full_synthesis import propagate
    from . import c1_h6_ca_resources as reused
    target=accuracy.AccuracyTarget(**TARGET);rows=[];plans=[];costs=[];key=packet['id']
    for p in pf:
        row=call('round_synthesis',key+':m'+str(p['m']),lambda p=p:propagate(segment,shell_reports,s=1,m=p['m'],saved_pf_error=p['pf_operator_error']))
        rows.append(row)
        plan=call('round_accuracy',key+':m'+str(p['m']),lambda row=row:reused.assess_saved_round(target,packet,row))
        plans.append(plan);io.save(out/(key+f'_m{p["m"]:02d}.json'),dict(PF=p,synthesis=row,accuracy=plan))
    energy=call('energy',key,lambda:accuracy.assess_energy(target,packet['df_error_with_guard'],plans,
        state_model=packet['state_model'],alias_branch_verified=packet['alias_verified']))
    if energy.get('same_accuracy'):
        for p in plans:
            costs.extend(call('round_serial_cost',key+':m'+str(p['m']),lambda p=p:reused.serial_round_cost(packet,p['m'])))
    value=call('aggregate',key,lambda:reused.aggregate_candidate(packet,plans,energy,costs))
    value.update(PF_rounds=pf,synthesis_rounds=rows)
    return value


def execute(root,c,out,call):
    import numpy as np
    from scipy.linalg import eigh
    from threadpoolctl import threadpool_limits
    from qiskit import qpy
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_pr_aware_df_stage_b2 as b2
    from . import c1_h4_native_fair as fair
    from .c1_h4_strong_baseline import StrongBuilder
    from . import c1_h4_full_synthesis as synthesis
    from . import c1_molecule_extension as molecule
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    root,out=Path(root),Path(out)
    for leaf in ('circuits','operators','candidates'): (out/leaf).mkdir()
    def arrays(name,**x):loader.save_arrays(out/'operators'/(name+'.npz'),**x)
    def qsave(name,qc):
        with (out/'circuits'/(name+'.qpy')).open('xb') as f:qpy.dump(qc,f)
    saved=call('evidence_load','saved A/B1/B2/C1/M1 and external XDF',lambda:evidence(root,c))
    p=call('input_load','original h/g/scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    car=call('car_context','full256 CAR',lambda:b.stage1.car_bilinears(8))
    profile=CompilerProfile();profile.validate()
    shells={};shell_reports={};shell_metrics={}
    for label in ('head','X','Y'):
        circuits={};matrices={}
        for kind in ('raw','compiled'):
            def load_qpy():
                with (root/c['shells'][label][kind]).open('rb') as f:qs=qpy.load(f)
                if len(qs)!=1 or qs[0].num_qubits!=9:raise ValueError('SAVED_SHELL_QPY_SCHEMA')
                return qs[0]
            qc=call('shell_load',label+':'+kind,load_qpy);circuits[kind]=qc
            matrices[kind]=call('shell_decode',label+':'+kind,lambda qc=qc:fair.decode(qc))
        call('shell_structure',label,lambda label=label:b2.shell_assessment(circuits['raw'],circuits['compiled'],label))
        # Independent exact ancilla shell matrix, not an ideal inferred from raw.
        H=np.array([[1.,1.],[1.,-1.]])/math.sqrt(2)
        ideal=np.asarray(np.kron(H if label!='Y' else H@np.diag([1.,-1j]),np.eye(256)),dtype=np.complex128)
        report=call('shell_segment',label,lambda:synthesis.segment(matrices['raw'],matrices['compiled'],ideal,'shell:'+label))
        report['metrics']=saved['shell_reports'][label]['metrics']
        shells[label]=matrices;shell_reports[label]=report
        shell_metrics[label]={k:report['metrics'][k] for k in accuracy.METRICS}
    io.save(out/'shells.json',shell_reports)
    prepared={}
    # Build/compile/inverse creation are sequential Qiskit operations. Heavy
    # independent full-Operator work below is parallel across the two arms.
    for arm in ARMS:
        data=call('fit_load',arm,lambda arm=arm:loader.load(root/c['exports'][arm],('J','U')))
        validate_arrays(p,data,arm)
        a=call('native_model',arm,lambda arm=arm:b.prepare_arm(**p,**data,role=arm,
            source_identity=c['saved_inputs'][c['exports'][arm]],input_identity=c['original_input_sha256'],ld=8))
        G=[]
        for i in range(9):
            G.append(call('raw_factor',arm+':'+str(i),lambda i=i:b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)
                if i==0 else b.stage1.raw_frame(data['J'][i-1],data['U'][i-1],car)[0]))
        sector=call('reference_load',arm,lambda arm=arm:loader.load(root/c['references'][arm],('generators','H','full_input_indices')))
        indices,checks=call('sector_gate',arm,lambda:sector_match(G,sector));io.save(out/(arm+'_sector.json'),checks)
        ground=call('ground_load',arm,lambda arm=arm:loader.load(root/c['grounds'][arm],('energy','sector_state')))
        def ground_check():
            H=sector['H'];psi=ground['sector_state'];energy=float(ground['energy'])
            r=float(np.linalg.norm(H@psi-energy*psi));norm=float(abs(np.vdot(psi,psi)-1))
            old=saved[arm+'_ground']
            if r>1e-10 or norm>1e-12 or abs(energy-old['energy_Ha'])>1e-12 or old['alias_margin']<=0:
                raise ValueError('SAME_JU_GROUND_ALIAS')
            return dict(status='PASS',residual=r,normalization=norm,energy_Ha=energy,
                alias_margin=old['alias_margin'],saved_own_ground=True)
        io.save(out/(arm+'_ground.json'),call('ground_gate',arm,ground_check))
        pair=call('sector_eigenpair',arm,lambda:eigh(sector['H']))
        if abs(float(pair[0][0])-float(ground['energy']))>1e-10:raise ValueError('SAVED_GROUND_IS_NOT_LOWEST')
        # Reuse the sector PF engine, including repeated squaring, with M2's
        # explicitly renamed logical units only; the old runner is never run.
        sector_kernel={}
        def pf_call(kind,label,fn):
            value=call('PF_kernel' if kind=='sector_kernel_reference' else 'PF_round',label,fn)
            if kind=='sector_kernel_reference':sector_kernel['pf']=value[0]['pf']
            return value
        pf=molecule.PF_rounds(sector['generators'],1,pf_call,arm,eigenpair=pair)
        ref=call('full_reference',arm,lambda:b.reference.reference_kernels(G,Fraction(1,10),ld=8)['pf'])
        gate=call('reference_sector_gate',arm,lambda:fair.component_check(ref[np.ix_(indices,indices)],
            sector_kernel['pf'],1e-10,label='full vs sector S2'))
        io.save(out/(arm+'_PF.json'),dict(arm=arm,s=1,h='1/10',rounds=pf,sector_gate=gate))
        arrays(arm+'_reference',matrix=ref)
        builder=StrongBuilder(relative_fusion=True,call=call)
        for block in a.model.blocks:
            call('gaussian_build',arm+':'+block.block_id,lambda block=block:fair.basis_request(builder,a.model,block))
        raw=call('kernel_build',arm,lambda:builder.ordered(a.model,Fraction(1,10),optimized=True))
        io.save(out/(arm+'_fusion.json'),call('fusion_gate',arm,lambda:fusion_gate(builder,a.model)))
        compiled=call('compile',arm,lambda:fair.compile_checked(raw,profile))
        qsave(arm+'_raw',raw);qsave(arm+'_compiled',compiled)
        adjoints={}
        for kind,qc in [('raw',raw),('compiled',compiled)]:
            adjoints[kind]=call('adjoint_build',arm+':'+kind,lambda qc=qc:b.native.actual_adjoint(qc))
            qsave(arm+'_'+kind+'_adjoint',adjoints[kind])
        psi=np.zeros(256,complex);psi[indices]=ground['sector_state']
        prepared[arm]=dict(a=a,raw=raw,compiled=compiled,adjoints=adjoints,ref=ref,psi=psi,pf=pf)
    plan=integrated.resource_plan(task_count=2,full_dimension=256)
    plan.update(policy='two independent arm Operator/matrix chains;Qiskit build/compile/inverse sequential',
        BLAS_threads_per_worker=1,CPU_cap=None,RAM_cap=None,wall_cap=None)
    io.save(out/'resource_plan.json',plan)
    def task(arm):
        x=prepared[arm];target=fair.controlled(x['ref']);mats={}
        for kind in ('raw','compiled'):
            mats[kind]=call('operator_decode',arm+':'+kind,lambda kind=kind:fair.decode(x[kind]))
            arrays(arm+'_'+kind,matrix=mats[kind])
        segment=call('segment_assessment',arm,lambda:synthesis.segment(mats['raw'],mats['compiled'],target,arm))
        io.save(out/(arm+'_segment.json'),segment)
        adj={}
        for kind in ('raw','compiled'):
            matrix=call('adjoint_decode',arm+':'+kind,lambda kind=kind:fair.decode(x['adjoints'][kind]))
            def check(kind=kind,matrix=matrix):
                return dict(dagger=fair.component_check(matrix,mats[kind].conj().T,1e-10,label='actual dagger'),
                    reference=fair.component_check(matrix,target.conj().T,1e-10,label='adjoint reference'),
                    recovery=fair.component_check(matrix@mats[kind],np.eye(512),1e-10,label='actual return'))
            adj[kind]=call('adjoint_gate',arm+':'+kind,check)
        io.save(out/(arm+'_adjoint.json'),adj)
        XY=call('XY_gate',arm,lambda:b2.XY_gate(mats['raw'],mats['compiled'],x['ref'],x['psi'],shells))
        metrics=call('metrics',arm,lambda:circuit_metrics(x['compiled']))
        packet=dict(id='D_'+arm,arm=arm,s=1,aliases=['D_'+arm],
            df_error_with_guard=saved['B1']['arms'][arm]['saved_DF_bound_Ha'],
            state_model='own_DF_ground_state_oracle',alias_verified=True,
            kernel_metrics={k:metrics[k] for k in accuracy.METRICS},shell_metrics=shell_metrics)
        value=evaluate_accuracy(packet,x['pf'],shell_reports,segment,out,call)
        value.update(segment=segment,adjoint=adj,XY=XY,metrics=metrics,
            export_sha256=c['saved_inputs'][c['exports'][arm]])
        io.save(out/'candidates'/(arm+'.json'),value)
        return arm,value
    results={};failures=[]
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for future in as_completed([pool.submit(task,a) for a in ARMS]):
                try:k,v=future.result();results[k]=v
                except BaseException as error:call.stop.set();failures.append(error)
    if failures:raise failures[0]
    comparisons={}
    for arm in ARMS:
        for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
            label=arm+':PR_'+lane+'/D_s1'
            comparisons[label]=call('comparison',label,lambda arm=arm,lane=lane,label=label:numerical_compare(
                saved['C1']['assumed_scenarios'][lane][arm]['resources'],results[arm]['cost'],label,
                qualification='saved PR assumed scenario s4 versus deterministic numerical synthesis pilot s1;same JU,not pure PF-only ablation'))
    label='D_proposed/D_baseline'
    comparisons[label]=call('comparison',label,lambda:numerical_compare(results['proposed']['cost'],results['baseline']['cost'],label,
        qualification='two numerical deterministic s1 pilots;different saved J/U,not optimizer causal identification'))
    ext=saved['external']['candidates']['B3_XDF_spin_relative']
    for arm in ARMS:
        label='D_'+arm+'/external_XDF8_B3'
        comparisons[label]=call('comparison',label,lambda arm=arm,label=label:numerical_compare(results[arm]['cost'],ext['cost'],label,
            qualification='external saved XDF8 spin+relative numerical pilot;same CA/model/serial policy,different DF representation and errors'))
    def summary():return dict(status='M2_SAME_JU_DETERMINISTIC_COMPLETE',scope=SCOPE,candidates=results,
        comparisons=comparisons,saved_PR_scenarios=saved['C1']['assumed_scenarios'],
        external_XDF=dict(source='1aeedd96178a3f50a15bc2701c3c332f4f391b64',candidate=ext),
        CA_achieved=None,rigorous_enclosure=None,PR_all_law_bound=None,population_mean_advantage=None,
        guaranteed_resource_superiority=False,causal_effect=None,
        new_fit=0,optimizer=0,rank_search=0,sampling=0,deterministic_s4=0,additional_guarantees=0,
        mandatory_stop=True)
    r=call('result_summary','two arms seven fixed comparisons',summary);io.save(out/'result.json',r);return r


def run(root,path):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH
    out.mkdir(parents=True,exist_ok=False);call=integrated.Calls(out,LIMITS,MAX_CALLS);tick=time.monotonic()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        # Matrices are at most512; two independent Operator chains are the
        # dominant parallel workload. Coordinate small BLAS calls explicitly.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),first_failure=call.first_failure,retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
