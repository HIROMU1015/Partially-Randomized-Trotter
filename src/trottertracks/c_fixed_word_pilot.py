"""Four predeclared word diagnostics; no sampling or accuracy calculation.

Imports/preflight use bytes and JSON only. Molecular evaluation is confined
to the separately permitted runner. Reuses the complete native verifier.
"""
from fractions import Fraction
import importlib.metadata
from pathlib import Path
import resource
import time
from . import c_pr_integrated_performance as perf
from . import c_b1_b2_reference_bound_integration as integration

io=integration.io
BASE='891aa78f47e66184eaf8a6d527ca8773ba6bbdd0'
FAMILY='artifacts/research_c_fixed_word_pilot/2026-11-04'
CONTRACT='docs/research/research_c_fixed_word_pilot_contract_unapproved_20261104.json'
ENTRY='scripts/run_research_c_fixed_word_pilot.py'
SCOPE='C_PR_R32_FIXED_FOUR_WORD_NUMERICAL_NATIVE_PILOT_V1'
LABELS=['same_atom','same_basis_Z_ZZ','alternating_frames','six_frame_cycle']
POLICY=dict(samples=4,events_per_sample=32,mode='fixed finite words;no PRNG or IID calibration')
SAMPLE_KINDS=('fixed_word_events','kernel_build','trajectory_reference','raw_operator_gate','compile',
              'compiled_operator_gate','adjoint_pair_gate','XY_gate','kernel_metrics')
COMMON={k:v for k,v in perf.m3.B_LIMITS.items() if k not in ('trajectory_sample','calibration_summary','delta_cost_comparison',*SAMPLE_KINDS)}
LIMITS=dict(evidence_load=1,input_load=1,candidate_load=1,reference_load=1,basis_load=1,ground_load=1,
    candidate_binding=1,native_model=1,ground_gate=1,law_prepare=1,law_gate=1,
    event_catalog=1,word_gate=1,**COMMON,**{k:4 for k in SAMPLE_KINDS},finite_summary=1,result_summary=1)
MAX_CALLS=sum(LIMITS.values())


def words(atoms):
    """Lexical IDs/support/frame metadata only; no coefficient optimization."""
    def pick(frame,size):
        found=sorted(a['atom_id'] for a in atoms if a['frame_id']=='CDF_proposed:F'+str(frame) and len(a['support'])==size)
        if not found:raise ValueError('MISSING_REGISTERED_WORD_ATOM')
        return found[0]
    z,zz=pick(2,1),pick(2,2);other=pick(3,2);cycle=[pick(i,2) for i in range(2,8)]
    seq=[[z]*32,[z,zz]*16,[z,other]*16,[cycle[i%6] for i in range(32)]]
    return [dict(id=label,atom_ids=ids) for label,ids in zip(LABELS,seq)]


def fixed():
    return dict(scope=SCOPE,base_commit=BASE,entry=ENTRY,status='UNAPPROVED_EXECUTABLE_FINITE_NUMERICAL_PILOT',
        conditions=integration.previous.identity(),finite_policy=POLICY,word_labels=LABELS,
        precision='binary64;existing full512 component gates;no interval/rigorous inference',
        call_limits=LIMITS,maximum_logical_units=MAX_CALLS,fresh_output=FAMILY+'/run01',retry=0,resume=0,
        new_fit=0,new_sampling=0,new_ground_eigensolve=0,new_PF_RTE_error=0,new_shots_energy=0,
        Stage_C_executed=False,whole_law_synthesis_bound=None,mandatory_stop=True)


def evidence(root,c):
    old=io.read(root/c['origin_contract']);result=io.read(root/c['origin_result']);stop=io.read(root/c['origin_STOP'])
    a=result['arms']['PR'];h=integration.previous.identity()['candidate']
    if (a['candidate']!=h or not a['PR']['native_admitted'] or old['candidate_sha256']['PR']!=h
        or stop['failed'] or stop['scientific_units']!=1231 or stop['retry'] or stop['resume']
        or not stop['mandatory_stop'] or old['source_manifest_sha256']!=c['origin_SOURCE_sha256']):
        raise ValueError('PRESERVED_CANDIDATE_ORIGIN')
    if (not a['D']['saved_DF']['budget_PASS'] or a['D']['saved_DF']['candidate_sha256']!=h):
        raise ValueError('SAVED_DF_QUALIFICATION')
    return dict(source=old,physical=a['D']['saved_DF'])


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    if any(c.get(k)!=v for k,v in fixed().items()):raise ValueError('FIXED_FINITE_PILOT')
    source=io.read(root/c['source_manifest'])
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (source['files'],c['saved_inputs']):
        for p,h in group.items():
            q=(root/p).resolve()
            if not q.is_relative_to(root) or io.sha(q)!=h:raise ValueError('SOURCE_INPUT:'+p)
    for k,v in c['packages'].items():
        if importlib.metadata.version(k)!=v:raise ValueError('ENVIRONMENT:'+k)
    if c['saved_inputs'][c['candidate']]!=integration.previous.identity()['candidate']:raise ValueError('CANDIDATE_HASH')
    if io.sha(root/c['origin_SOURCE'])!=c['origin_SOURCE_sha256']:raise ValueError('ORIGIN_SOURCE')
    evidence(root,c)
    atoms=io.read(root/c['atoms'])['atoms']
    if c['words']!=words(atoms):raise ValueError('REGISTERED_WORD_FAMILY')
    from .c1_h4_native_fair import GAUSSIAN,PHASE
    if c['gaussian']!=GAUSSIAN or c['compiler_phase']!=PHASE:raise ValueError('P0_PRECISION_POLICY')
    if (root/c['fresh_output']).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def prepare(root,c,out,call):
    import numpy as np
    from . import c_pr_aware_df_m2 as m2
    from .c_rc2_pr_stage_b import distribution_gate
    b,loader=perf.b,perf.loader
    saved=call('evidence_load','same fixed PR endpoint and DF verdict',lambda:evidence(root,c))
    p=call('input_load','original h/g/scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    d=call('candidate_load','fixed CDF8',lambda:loader.load(root/c['candidate'],('J','U')))
    sector=call('reference_load','same candidate saved independent CAR',lambda:loader.load(root/c['sector'],('J','U','generators','H')))
    basis=call('basis_load','saved physical sector indices only',lambda:loader.load(root/c['basis'],('sector_basis_indices',)))
    g=call('ground_load','saved numerical own-DF state;no new solve',lambda:loader.load(root/c['ground'],('energy','sector_state')))
    def bind():
        m2.validate_arrays(p,d,'finite current PR')
        if not all(np.array_equal(d[k],sector[k]) for k in ('J','U')):raise ValueError('CANDIDATE_REFERENCE')
        sector['full_input_indices']=b.stage1.bit_reverse_indices(8)[basis['sector_basis_indices']]
        return dict(status='PASS',saved_DF=saved['physical'],new_DF_error=0)
    io.save(out/'identity.json',call('candidate_binding','exact same J/U and input',bind))
    arm=call('native_model','existing original kappa/scalar/frame/Gaussian policy',lambda:b.prepare_arm(**p,**d,role='proposed',
        source_identity=integration.previous.identity()['candidate'],input_identity=c['saved_inputs'][c['input']],ld=2))
    pair=(np.asarray([float(g['energy'])]),g['sector_state'][:,None])
    io.save(out/'ground.json',call('ground_gate','saved-state residual/norm/alias;not certified eigenstate distance',lambda:loader.ground_gate(sector['H'],pair)))
    tail,config,dist=call('law_prepare','same r32 K0,not sampling',lambda:b.configure(arm,Fraction(1,10),32,0))
    def law():
        distribution_gate(tail,config,dist,expected_steps=32)
        original=io.read(root/c['atoms'])['atoms']
        actual={a.atom_id:a for a in arm.plan.atoms}
        if set(actual)!={a['atom_id'] for a in original}:raise ValueError('ATOM_IDENTITY')
        for row in original:
            atom=actual[row['atom_id']]
            if (atom.frame_id!=row['frame_id'] or list(atom.support)!=row['support']
                or Fraction(atom.coefficient)!=Fraction(row['coefficient'])):raise ValueError('SIGNED_ATOM_TARGET')
        return dict(status='PASS',atoms=len(actual),random_draws=0,unitary_trajectory_premise_qualified=False)
    io.save(out/'law.json',call('law_gate','bind Stage A signed atom definitions',law))
    def catalog():
        from trotterlib.rte import enumerate_rte_events
        return {e.rotation_component_id:e for e in enumerate_rte_events(tail.components,dist)}
    events=call('event_catalog','existing K0 event constructor,no RNG',catalog)
    def assemble():
        value=words(io.read(root/c['atoms'])['atoms'])
        if value!=c['words'] or len({tuple(w['atom_ids']) for w in value})!=4:raise ValueError('WORD_ORDER_OR_COUNT')
        return tuple(tuple(events[x] for x in w['atom_ids']) for w in value)
    fixed_words=call('word_gate','four prescribed 32-event words',assemble)
    return dict(arm=arm,p=p,d=d,sector=sector,ground=g['sector_state'],tail=tail,config=config,dist=dist),fixed_words


def execute(root,c,out,call,*,preparer=prepare,native=perf.native):
    a,fixed_words=preparer(root,c,out,call)
    rows,summary=native(root,c,out,call,a,sampling=POLICY,fixed_words=fixed_words)
    return call('result_summary','finite correctness only;no RPE or law claim',lambda:dict(
        status='FIXED_FOUR_WORD_NUMERICAL_PILOT_COMPLETE',words=c['words'],metrics=rows,native=summary,
        precision='numerical observations only',whole_law_synthesis_bound=None,categorical_unitary_premise_qualified=False,
        B2_bounds_recomputed=0,Stage_C_ready=False,actual_CA_achieved=None,
        population_mean=None,guaranteed_resource_superiority=None,mandatory_stop=True))


def run(root,path,*,execute_science=False):
    if not execute_science:raise PermissionError('EXPLICIT_FIXED_FINITE_SCIENTIFIC_PERMISSION_REQUIRED')
    root=Path(root).resolve();c=preflight(root,path);out=root/c['fresh_output'];out.mkdir(parents=True,exist_ok=False)
    call=perf.Calls(out,LIMITS,MAX_CALLS);tick,cpu=time.monotonic(),time.process_time();complete=False
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
        r=execute(root,c,out,call);io.save(out/'result.json',r);complete=True;return r
    except BaseException as exc:
        call.stop.set();io.save(out/'failure.json',dict(error=repr(exc),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=not complete or call.stop.is_set(),scientific_units=sum(call.counts.values()),
            maximum_logical_units=MAX_CALLS,call_counts=dict(call.counts),first_failure=call.first_failure,
            retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
