"""Current PR r32/s1 reference bridge under an explicit IID 53-bit model.

Passive imports/preflight. Stage A reuses atom/CAR/table algebra and saved
references; no circuit, trajectory sampling, PF generation or shots solver.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import subprocess
import time

from . import c1_stage1_execution as io
from . import c_all_law_signal_ca as previous

BASE='3a784888a188db946681a6143c5e18479ddf284a'
CANDIDATE=previous.CANDIDATE
FAMILY='artifacts/research_c_iid_grid_all_law/2026-11-03'
FRESH=FAMILY+'/stageA-run01'
ENTRY='scripts/run_research_c_iid_grid_bridge.py'
CONTRACT='docs/research/research_c_iid_grid_all_law_contract_unapproved_20261103.json'
MODEL=dict(id='IID_UNIFORM_53_BIT_WORDS_COUNTERFACTUAL',bits=53,
    mapping='NumPy1.26.4 binary64 normalize/cumsum/normalize;searchsorted_right',
    independent_events=True,independent_coherent_occurrences=True,
    independent_measurement_shots=True,independent_XY_measurements=True,
    inverse='inverse of the same realized circuit where used;not an independently redrawn inverse',
    PCG64_joint_stream_qualified=False)
LIMITS=dict(saved_metadata_load=1,identity_gate=1,input_load=1,sector_context=1,
    candidate_arrays_load=1,model_replay=1,factor_binding=9,law_prepare=1,
    atom_context=1,choice_grid_analysis=1,atom_audit=1,event_mean=2,
    head_reference=1,kernel_bridge=1,round_powers=1,round_bridge=13,summary=1)
MAX_CALLS=sum(LIMITS.values())


def fixed():
    return dict(scope='C_CURRENT_PR_R32_IID_GRID_REFERENCE_BRIDGE_V1',base_commit=BASE,
        status='UNAPPROVED_EXECUTABLE_STAGE_A_ONLY',candidate_sha256=CANDIDATE,
        PR_scientific_SOURCE=previous.PR_SOURCE,model=MODEL,conditions=previous.fixed()['conditions'],
        entry=ENTRY,call_limits=LIMITS,maximum_logical_units=MAX_CALLS,
        fresh_output=FRESH,retry=0,resume=0,mandatory_stop=True,
        new_native_build_compile=0,new_trajectory_sampling=0,new_PF_RTE_generation=0,
        ground_eigensolve=0,new_shots_evaluation=0,Stage_B='P0_UNIFORM_NOT_ESTABLISHED',
        Stage_C='NOT_READY_BOUND_INPUTS',Stage_D='OUT_OF_SCOPE')


def identity_gate(d,c):
    pr=d['PR_result']['arms']['PR'];old=d['PR_contract']
    if (pr['candidate']!=c['candidate_sha256'] or old['candidate_sha256']['PR']!=c['candidate_sha256']
            or old['compiler']!=c['compiler'] or old['target']!=c['target']
            or old['policy']['s']!=1 or old['policy']['r']!=32 or old['policy']['K']!=0
            or old['policy']['h']!='1/10' or old['policy']['ld']!=2
            or [r['m'] for r in pr['PR']['rounds']]!=list(range(13))
            or pr['PR']['all_law_synthesis'] is not None or pr['PR']['actual_CA_achieved'] is not None):
        raise ValueError('CURRENT_CANDIDATE_REFERENCE_SCOPE')
    for key,n in [('PR_STOP',1231),('statistics_STOP',7)]:
        s=d[key]
        if s['failed'] or s['scientific_units']!=n or s['retry'] or s['resume'] or not s['mandatory_stop']:
            raise ValueError('SAVED_STOP')
    return dict(status='PASS',candidate=c['candidate_sha256'],model=MODEL,
        PCG64_joint_stream_error=None,all_law_synthesis=None,actual_CA_achieved=None)


def prepare_arrays(root,c,ctx):
    import numpy as np
    from .c_pr_aware_df_stage_b_runner import load
    roles=c['input_roles']
    data=load(Path(root)/roles['candidate'],('J','U'))
    saved=load(Path(root)/roles['factors'],('J','U','generators'))
    saved.update(load(Path(root)/roles['PF_RTE_reference'],('corrected',)))
    psi=load(Path(root)/roles['own_ground'],('sector_state',))['sector_state']
    if (data['J'].shape!=(8,4,4) or data['U'].shape!=(8,4,4)
            or saved['generators'].shape!=(9,36,36) or saved['corrected'].shape!=(36,36)
            or psi.shape!=(36,) or len(ctx.sector.basis_indices)!=36):raise ValueError('ARRAY_SCHEMA')
    from .c1_stage1 import finite_arrays
    finite_arrays(*data.values(),*saved.values(),psi)
    if (not np.array_equal(data['J'],saved['J']) or not np.array_equal(data['U'],saved['U'])
            or abs(np.vdot(psi,psi)-1)>1e-12):raise ValueError('CANDIDATE_OR_GROUND_BINDING')
    return data,saved,psi


def heads(generators):
    # Old C2 heads hardcodes h/2=1/80; only parameter routing differs here.
    from .c1_stage3_reference import _ordered_exponentials
    schedule=tuple((i,Fraction(1,20)) for i in range(3));cache={}
    return dict(left=_ordered_exponentials(generators,schedule,cache),
        right=_ordered_exponentials(generators,tuple(reversed(schedule)),cache))


def prepare_law(arm,saved,normalization):
    """Existing law construction plus current saved lambda/tau/identity binding."""
    from . import c_pr_aware_df_stage_b as b
    tail,config,dist=b.configure(arm,Fraction(1,10),32,0)
    if (len(arm.plan.atoms)!=saved['atoms'] or saved['atoms']!=216
            or arm.plan.extracted_identity!=saved['identity'] or normalization['r']!=32
            or not math.isclose(tail.lambda_r,saved['lambda_R'],rel_tol=1e-12,abs_tol=1e-14)
            or not math.isclose(config.dimensionless_step_time,normalization['tau'],rel_tol=1e-12,abs_tol=1e-14)
            or list(dist.orders)!=[0] or list(dist.order_probabilities)!=[1.]):
        raise ValueError('SAVED_CURRENT_LAW_NORMALIZATION_BINDING')
    return tail,config,dist


def kernel_bridge(arm,config,means,head,saved):
    """New p/q endpoint; reuse saved CAR kernel instead of regenerating RTE.

    Old C2 kernel_bridge regenerates a different h=1/40 reference and only p.
    This connector adds q and h=.1 routing, retaining the saved current kernel.
    """
    import numpy as np
    from .c1_stage3_accuracy import _legacy
    from .c_pr_aware_df_c2_bridge import compare
    if (config.rte_steps!=32 or config.finite_taylor_order!=0 or config.evolution_time!=.1):
        raise ValueError('R32_S1_H_ROUTING')
    phase=np.exp(-1j*.1*arm.plan.extracted_identity)
    value={k:phase*head['right']@np.linalg.matrix_power(means[k]['matrix'],32)@head['left'] for k in ('ideal','grid')}
    A=_legacy('rte').finite_rte_attenuation(config,tail_evolutions=1)
    value.update(raw=A*saved['corrected'],corrected=saved['corrected'],config=config)
    gate=compare(value['ideal'],value['raw'],'r32 ideal-atom/saved CAR raw kernel')
    value['report']=dict(status='CURRENT_KERNEL_REFERENCE_NUMERICAL_PASS',ideal_vs_raw=gate,
        model=MODEL,h='1/10',r=32,s=1,attenuation=A,identity_phase_parts=[phase.real,phase.imag],
        scalar='nuclear once in saved G0;only extracted tail identity applied separately',
        numerical_reference_uncertainty=None,rigorous_bridge_bound=None,all_law_synthesis=None)
    return value


def power_cache(context):
    """Four dependent repeated-squaring chains, shared by 13 round checks."""
    import numpy as np
    result={}
    for key in ('ideal','grid','raw','corrected'):
        values=[context[key]]
        for _ in range(12):values.append(np.linalg.matrix_power(values[-1],2))
        for a in values:a.setflags(write=False)
        result[key]=values
    return result


def round_bridge(context,powers,m,saved_row,psi,table):
    import numpy as np
    from .c1_stage3_accuracy import _legacy
    previous._m(m)
    A=_legacy('rte').finite_rte_attenuation(context['config'],tail_evolutions=1<<m)
    if abs(A-saved_row['reference']['attenuation'])>1e-12:raise ValueError('ATTENUATION_BINDING')
    ref=A*powers['corrected'][m]
    z=complex(np.vdot(psi,ref@psi));old=complex(*saved_row['reference']['raw_signal_parts'])
    replay=abs(z-old)
    if replay>2e-10:raise ValueError('SAVED_RAW_SIGNAL_REPLAY')
    def distance(a,b):
        difference=a-b
        values=dict(operator2_observed=float(np.linalg.norm(difference,2)),
            frobenius_numerical_upper_pilot=float(np.linalg.norm(difference,'fro')),
            raw_complex_signal_observed=float(abs(np.vdot(psi,difference@psi))))
        if not all(math.isfinite(x) for x in values.values()):raise ValueError('NONFINITE_BRIDGE')
        return values
    signals={k:complex(np.vdot(psi,powers[k][m]@psi)) for k in ('ideal','grid')}
    return dict(m=m,kernel_occurrences=1<<m,events=32*(1<<m),model=MODEL,
        ideal_atom_vs_saved_CAR=distance(powers['ideal'][m],ref),
        grid_vs_ideal_atom=distance(powers['grid'][m],powers['ideal'][m]),
        grid_vs_saved_CAR=distance(powers['grid'][m],ref),
        normalization_power_path=distance(powers['raw'][m],ref),
        raw_reference_signal_parts=[z.real,z.imag],
        raw_signal_parts={k:[v.real,v.imag] for k,v in signals.items()},
        saved_signal_replay_error=replay,conditional_categorical=previous.categorical_preview(table,m),
        saved_PF_operator_error=saved_row['reference']['pf_operator_error'],
        saved_corrected_RTE_operator_error=saved_row['reference']['rte_corrected_operator_error'],
        reference_numerical_uncertainty=None,rigorous_bridge_bound=None,
        error_evidence=None,all_law_synthesis=None,PCG64_joint_stream_qualified=False,
        actual_CA_achieved=None,shots=None,
        qualification='sector36 numerical correspondence and conditional IID categorical bound;not final compiled all-law evidence')


def execute(root,c,out,call):
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_c2_bridge as bridge
    from . import c_pr_aware_df_stage_c2 as categorical
    from .c_pr_aware_df_stage_b_runner import load,save_arrays
    from .c1_molecule_extension import SectorCAR
    from .c1_stage3_integrated import resource_plan
    from threadpoolctl import threadpool_limits
    root,out=Path(root),Path(out)
    d=call('saved_metadata_load','current PR and fixed old records',lambda:{k:io.read(root/p) for k,p in previous.records().items()})
    io.save(out/'identity.json',call('identity_gate','current r32/s1 only',lambda:identity_gate(d,c)))
    p=call('input_load','original h/g/scalar;no physical/ground solve',lambda:load(root/c['input_roles']['input'],('h','g','constant')))
    ctx=call('sector_context','saved spin/bit ordering,sector36',lambda:SectorCAR(4))
    data,saved,psi=call('candidate_arrays_load','candidate/factors/corrected/ground binding',lambda:prepare_arrays(root,c,ctx))
    arm=call('model_replay','existing general-K and one-body basis;no circuit',lambda:b.prepare_arm(
        **p,**data,role='proposed',source_identity=c['candidate_sha256'],input_identity=c['input_sha256'],ld=2))
    for i in range(9):call('factor_binding',str(i),lambda i=i:b.component_check(
        b.candidate_factor(ctx,arm,i),saved['generators'][i],1e-10,label='saved/current factor '+str(i)))
    tail,config,dist=call('law_prepare','r32/K0/h=.1 saved lambda/tau/identity',lambda:prepare_law(arm,d['PR_law'],d['PR_normalization']))
    operators=call('atom_context','same signed atoms and full sector',lambda:b.atom_operators(ctx,arm))
    table=call('choice_grid_analysis','exact p versus IID-grid q,no random draws',lambda:categorical.analyze_table(
        [abs(a.coefficient) for a in arm.plan.atoms],[x.probability for x in tail.components]))
    io.save(out/'categorical_table.json',table)
    io.save(out/'atoms.json',call('atom_audit','all216 signed atoms/CAR',lambda:bridge.atom_audit(arm,tail,operators,saved['generators'],table)))
    means={}
    for key,probabilities in [('ideal',table['ideal_probabilities']),('grid',table['grid']['probabilities_exact'])]:
        # event_mean is generic weighted K0 algebra despite its legacy input
        # field name. A fresh local view supplies q; the real p table is intact.
        means[key]=call('event_mean',key,lambda probabilities=probabilities:bridge.event_mean(
            arm,tail,dist,operators,dict(ideal_probabilities=probabilities)))
        io.save(out/(key+'_event_mean.json'),dict(model='ideal_saved_atom' if key=='ideal' else MODEL['id'],report=means[key]['report']))
    head=call('head_reference','two h/2=.05 head halves',lambda:heads(saved['generators']))
    context=call('kernel_bridge','ideal and grid versus saved raw CAR',lambda:kernel_bridge(arm,config,means,head,saved))
    io.save(out/'kernel.json',context['report'])
    save_arrays(out/'mean_kernels.npz',ideal_event_mean=means['ideal']['matrix'],grid_event_mean=means['grid']['matrix'],
        ideal_kernel=context['ideal'],grid_kernel=context['grid'],raw_CAR_kernel=context['raw'])
    powers=call('round_powers','four shared binary-power chains,n=1..4096',lambda:power_cache(context))
    plan=resource_plan(task_count=13,full_dimension=36);io.save(out/'resource_plan.json',plan)
    def task(m):
        row=call('round_bridge',str(m),lambda:round_bridge(context,powers,m,d['PR_result']['arms']['PR']['PR']['rounds'][m],psi,table))
        io.save(out/f'm{m:02d}.json',row);return row
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            rows=[f.result() for f in as_completed([pool.submit(task,m) for m in range(13)])]
    rows.sort(key=lambda x:x['m'])
    result=call('summary','Stage A only;no synthesis/accuracy promotion',lambda:dict(
        status='IID_GRID_REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE',model=MODEL,rounds=rows,
        candidate=c['candidate_sha256'],PR_scientific_SOURCE=previous.PR_SOURCE,
        Stage_B='P0_UNIFORM_NOT_ESTABLISHED',Stage_C='NOT_READY_BOUND_INPUTS',
        PCG64_joint_stream_qualified=False,PCG64_joint_stream_error=None,
        all_law_synthesis=None,reference_numerical_uncertainty=None,
        error_evidence=None,actual_CA_achieved=None,rigorous_CA=None,
        new_sampling=0,new_native_build_compile=0,new_PF_RTE_generation=0,ground_eigensolve=0,
        shots_evaluation=0,mandatory_stop=True))
    io.save(out/'result.json',result);return result


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    for k,v in fixed().items():
        if c[k]!=v:raise ValueError('FIXED_IID_BRIDGE:'+k)
    # Reuse the unchanged prior candidate/source/input/STOP gate, without its
    # NOT_READY whole-program status blocking this separate reference stage.
    old,_=previous.preflight(root,root/previous.CONTRACT)
    if (c['input_roles']!=old['scientific_input_roles'] or c['compiler']!=old['compiler']
            or c['target']!=old['target'] or c['input_sha256']!=old['saved_inputs'][c['input_roles']['input']]):
        raise ValueError('CURRENT_INPUT_COMPILER_TARGET')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    source=io.read(root/c['source_manifest'])
    if ENTRY not in source['files'] or not set(c['input_roles'].values())<=c['saved_inputs'].keys():raise ValueError('UNPINNED_INPUT')
    for group in (source['files'],c['saved_inputs']):
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError('SOURCE_INPUT:'+name)
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT:'+name)
    identity_gate({k:io.read(root/p) for k,p in previous.records().items()},c)
    if (root/FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return c,dict(status='PASSIVE_STAGE_A_READY',scientific_calls=0,maximum_logical_units=MAX_CALLS,
        model=MODEL,Stage_B='P0_UNIFORM_NOT_ESTABLISHED',Stage_C='NOT_READY_BOUND_INPUTS',actual_CA_achieved=None)


def run(root,path,*,execute_science=False):
    if not execute_science:raise PermissionError('SEPARATE_FIXED_STAGE_A_SCIENTIFIC_PERMISSION_REQUIRED')
    from .c1_stage3_integrated import Calls
    root=Path(root).resolve();c,_=preflight(root,path);out=root/FRESH
    out.mkdir(parents=True,exist_ok=False);call=Calls(out,LIMITS,MAX_CALLS);start=time.monotonic();complete=False
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        result=execute(root,c,out,call);complete=True;return result
    except BaseException as exc:
        call.stop.set();io.save(out/'failure.json',dict(error=repr(exc),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=not complete or call.stop.is_set(),scientific_units=sum(call.counts.values()),
            maximum_logical_units=MAX_CALLS,call_counts=dict(call.counts),first_failure=call.first_failure,
            wall_seconds=time.monotonic()-start,retry=0,resume=0,mandatory_stop=True))
