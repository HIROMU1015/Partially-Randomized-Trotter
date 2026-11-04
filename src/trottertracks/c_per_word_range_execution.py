"""Passive entry and dormant Stage R only; no per-word/calibration auto-launch."""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.metadata
import json
import resource
import time

from . import c_per_word_population_range as method
from . import c_fixed_word_pilot as prior

io=prior.io
SCOPE='C_PR_R32_LITERAL_POPULATION_NORM_STAGE_R_V1'
LIMITS=dict(evidence_load=1,input_load=1,candidate_load=1,native_model=1,
    car_context=1,raw_factor=3,candidate_binding=1,event_catalog=1,grid_table_load=1,
    saved_sector_heads_load=1,head_sector_binding=1,
    support_gate=1,head_norm=3,frame_norm=8,event_norm=216,population_norm=1,
    growth_profile=13,range_summary=1)
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    if (c['base_commit']!=method.BASE or c['target']!=method.TARGET
            or c['precision_bits']!=192 or c['retry'] or c['resume'] or not c['mandatory_stop']
            or c['stages']['R']['scope']!=SCOPE or c['stages']['R']['call_limits']!=LIMITS
            or c['stages']['R']['maximum_logical_units']!=MAX_CALLS
            or c['stages']['R']['executable'] is not True or c['stages']['P']['executable'] is not False
            or c['stages']['C']['executable'] or c['stages']['A']['executable']):
        raise ValueError('FIXED_STAGE_R_TECHNICAL_CONTRACT')
    source=io.read(root/c['source_manifest'])
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (source['files'],c['saved_inputs']):
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError('SOURCE_INPUT:'+name)
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT:'+name)
    # Same target/model/P0/RPE identity; no numerical import in passive mode.
    if c['identity']!=prior.integration.previous.identity():raise ValueError('CANDIDATE_SOURCE_MODEL_COMPILER')
    for output in (c['stages']['R']['fresh_output'],c['stages']['P']['fresh_output']):
        if (root/output).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_STAGE_R_READY_FOR_SEPARATE_PERMISSION',
        scientific_calls=0,arrays_decoded=0,execution_started=False,
        Stage_R_maximum_logical_units=MAX_CALLS,Stage_P_executable=False,
        calibration_method_adopted=False,Stage_C_ready=False,mandatory_stop=True)


def execute_range(root,c,out,call):
    """Scientific path: not invoked by technical preparation or passive tests."""
    import numpy as np
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_m2 as m2
    from .c_pr_aware_df_stage_b_runner import load,save_arrays
    parent=c['parent_roles']
    call('evidence_load','preserved endpoint/DF/source/STOP',lambda:prior.evidence(root,parent))
    p=call('input_load','same h/g/scalar',lambda:load(root/parent['input'],('h','g','constant')))
    data=call('candidate_load','saved J/U only',lambda:load(root/parent['candidate'],('J','U')))
    def bind():
        m2.validate_arrays(p,data,'literal population range')
        if io.sha(root/parent['candidate'])!=c['identity']['candidate']:raise ValueError('FIXED_CANDIDATE_HASH')
        return dict(status='PASS',new_fit=0)
    call('candidate_binding','same H4 CDF8',bind)
    arm=call('native_model','reuse type/general-K split;no circuit',lambda:b.prepare_arm(**p,**data,role='proposed',
        source_identity=c['identity']['candidate'],input_identity=c['saved_inputs'][parent['input']],ld=2))
    full=call('car_context','independent full256 CAR',lambda:b.stage1.car_bilinears(8))
    from concurrent.futures import ThreadPoolExecutor
    from threadpoolctl import threadpool_limits
    from .c1_stage3_integrated import resource_plan
    plan=resource_plan(3,full_dimension=256)
    def factor(i):
        return call('raw_factor',str(i),lambda:b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],full)
            if i==0 else b.stage1.raw_frame(data['J'][i-1],data['U'][i-1],full)[0])
    # Dominant independent CAR factors; small scalar Arb norm bounds below
    # remain serial to preserve process-local precision without spawn overhead.
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        generators=list(pool.map(factor,range(3)))
    heads=generators[:3]
    def metadata():
        atoms=io.read(root/c['atoms'])['atoms'];mean=io.read(root/c['event_mean'])
        if mean['model']!=c['identity']['model']:raise ValueError('SAVED_ANGLE_MODEL')
        angle=mean['report']['angle'];actual={a.atom_id:a for a in arm.plan.atoms}
        if len(atoms)!=216 or set(actual)!={x['atom_id'] for x in atoms}:raise ValueError('COMPLETE_ATOM_SOURCE')
        for row in atoms:
            a=actual[row['atom_id']]
            if (a.frame_id!=row['frame_id'] or list(a.support)!=row['support']
                    or F(a.coefficient)!=F(row['coefficient'])):raise ValueError('SAVED_SIGNED_ATOM_BINDING')
        return dict(ids=[x['atom_id'] for x in atoms],catalog={x['atom_id']:NS(
            rotation_component_id=x['atom_id'],rotation_angle=angle,taylor_order=0,phase=1,
            product_component_ids=()) for x in atoms})
    events=call('event_catalog','saved K0 angle and full signed catalog;no RTE resampling',metadata)
    table=call('grid_table_load','saved exact53-bit counts;no q reevaluation',lambda:io.read(root/c['grid_table'])['grid'])
    def saved_heads():
        a=load(root/parent['sector'],('generators',))
        a.update(load(root/parent['basis'],('sector_basis_indices',)))
        return a
    saved=call('saved_sector_heads_load','saved B2/A reference identity only',saved_heads)
    def head_binding():
        ix=b.stage1.bit_reverse_indices(8)[saved['sector_basis_indices']]
        # Exact equality is provenance, not a norm/precision claim. A mismatch
        # blocks automatic B2 transport; it does not change either target.
        equal=[bool(np.array_equal(G[np.ix_(ix,ix)],saved['generators'][i])) for i,G in enumerate(heads)]
        return dict(exact_head_bits_equal=equal,B2_head_identity_qualified=all(equal),
            additional_head_bridge_required=not all(equal),norm_or_PF_recomputed=False)
    binding=call('head_sector_binding','full-head vs saved sector-head bytes',head_binding)
    io.save(out/'reference_head_binding.json',binding)
    save_arrays(out/'reference_heads.npz',heads=np.asarray(heads))
    value=method.population_norm(arm,heads,events['catalog'],table,expected_ids=events['ids'],call=call)
    io.save(out/'population.json',value)
    return call('range_summary','range only;no per-word C/P0 or calibration',lambda:dict(
        status='LITERAL_POPULATION_NORM_STAGE_R_COMPLETE',population=value,head_binding=binding,
        scientific_input=c['identity'],native_build_compile=0,new_sampling=0,new_fit=0,
        resource_plan=plan,
        new_PF_RTE_errors=0,confidence_computed=False,Stage_P_executed=False,
        Stage_C_ready=False,actual_CA_achieved=None,mandatory_stop=True))


def run(root,path,*,execute_science=False,executor=None):
    if not execute_science:raise PermissionError('SEPARATE_FIXED_STAGE_R_SCIENTIFIC_PERMISSION')
    root=Path(root).resolve();preflight(root,path);c=io.read(path)
    from .c_pr_aware_df_zsxx_enclosure import runtime_model
    from .c1_stage3_integrated import Calls
    runtime_model()
    out=root/c['stages']['R']['fresh_output'];out.mkdir(parents=True,exist_ok=False)
    calls=Calls(out,LIMITS,MAX_CALLS);tick=time.monotonic()
    io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
    try:
        result=(executor or execute_range)(root,c,out,calls)
        io.save(out/'result.json',result);return result
    except BaseException as error:
        calls.stop.set()
        if calls.first_failure is None:
            calls.first_failure=dict(kind='runner',error=type(error).__name__,message=str(error))
        io.save(out/'partial_failure.json',dict(error=type(error).__name__,message=str(error)))
        raise
    finally:
        io.save(out/'STOP.json',dict(scientific_units=sum(calls.counts.values()),call_counts=calls.counts,
            failed=calls.first_failure,retry=0,resume=0,mandatory_stop=True,Stage_P_executed=False,
            Stage_C_executed=False,actual_CA_achieved=None))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
