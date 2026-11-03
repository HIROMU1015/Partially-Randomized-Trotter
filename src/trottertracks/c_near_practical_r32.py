"""Fixed two-endpoint r32 downstream adapter; no molecular work on import.

Reuse ground/CAR, PF/RTE/RPE, complete native32, P0 and same-J/U strong D.
Old update005/064 quantities never enter the new candidate's numerical packet.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import time
from . import c_update064_r32_downstream as downstream
from . import c_update064_strong_d as deterministic
from . import c_near_r32_statistics as statistics

io,perf=downstream.io,downstream.perf
BASE='2ceb0a8825785ebd4f21241d8ef0f20cb321bf69'
SCOPE='C_NEAR_PRACTICAL_TWO_ENDPOINT_COMMON_R32_DOWNSTREAM_V1'
VERSION='candidate_bound_shared_ground_existing_PF_RTE_native32_strongD_v1'
ENTRY='scripts/run_research_c_near_practical_r32.py'
FAMILY='artifacts/research_c_near_practical_r32/2026-11-03'
FRESH=FAMILY+'/run01'
INPUT=downstream.INPUT
CANDIDATES=dict(PR='a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a',
 ENTRY='345dd11a14a281bf716f42009565b2d1875fdcf9845b928a15c102164208cb3f')
POLICY={k:v for k,v in downstream.POLICY.items() if k!='candidate'}
COMMON={k:1 for k in ('evidence_load','input_load','physical_reference_load','sector_context',
    'physical_ground','target_validation','seed_manifest','main_comparison','result_summary')}
A=dict(selected_load=1,reference_load=1,candidate_binding=1,native_model=1,
    general_factor_gate=9,candidate_ground=1,law_prepare=1,law_gate=1,normalization_gate=1)
B=downstream.B_LIMITS
C=downstream.C_LIMITS
D=downstream.D_LIMITS
E=dict(accounting_binding=1,native_rows_validation=1,price_statistics=2,secondary_summary=1,arm_summary=1)
STAGES=dict(A=A,B=B,C=C,D=D,E=E)
LIMITS={**{'common.'+k:v for k,v in COMMON.items()},
    **{f'{arm}.{stage}.{k}':v for arm in CANDIDATES for stage,ks in STAGES.items() for k,v in ks.items()}}
MAX_CALLS=sum(LIMITS.values())


def sampling(arm):
    if arm not in CANDIDATES:raise ValueError('REGISTERED_ENDPOINT_ONLY')
    return dict(master=2026110304,cohort='near_practical_'+arm+'_r32_fixed32_v1',
        axis='NEAR_PRACTICAL_'+arm+'_PRICE_ONLY',samples=32,events_per_sample=32,
        PRNG='numpy.random.PCG64',paired=False,adaptive=False)


def seed_manifest(digest):
    rows=[]
    for arm,candidate in CANDIDATES.items():
        policy=sampling(arm);master=(policy['master'],policy['cohort'],digest)
        for i in range(32):
            seed=perf.b.connection.trajectory_seed(master,'proposed',candidate,-1,policy['axis'],i,0)
            rows.append(dict(arm=arm,sample=i,seed=seed))
    if len({x['seed'] for x in rows})!=64:raise ValueError('SEED_COLLISION')
    return dict(status='PASS',rows=rows,master=2026110304,PRNG_draws=0,
        disjoint_new_cohort_names=True,old_trajectories_reused=0,old_samples_pooled=False,
        qualification='hash-separated finite-bit seed streams;not proof of ideal IID law or cost transfer')


def fixed():
    return dict(base_commit=BASE,scope=SCOPE,implementation=VERSION,entry=ENTRY,
        candidate_sha256=CANDIDATES,policy=POLICY,target=downstream.m4.TARGET,
        compiler=downstream.retuning.COMPILER,allowances=downstream.m3.ALLOWANCES,
        scenarios=list(perf.accounting.SCENARIOS),sampling={a:sampling(a) for a in CANDIDATES},
        statistics=statistics.POLICY,call_limits=LIMITS,maximum_logical_units=MAX_CALLS,
        fresh_output=FRESH,retry=0,resume=0,new_fit=0,optimizer=0,new_DF_error=0,mandatory_stop=True)


def evidence(root,c):
    root=Path(root);s={k:io.read(root/p) for k,p in c['origin_records'].items()}
    result,stop,contract=s['result'],s['STOP'],s['contract']
    if (result['status']!='PAIRED_FINITE_RECIPE_ENDPOINTS_RECORDED' or stop['failed']
        or stop['scientific_units']!=645 or stop['retry'] or stop['resume'] or not stop['mandatory_stop']):
        raise ValueError('FIXED_ENDPOINT_ORIGIN_STOP')
    if (contract['scope']!='C_NEAR_OBJECTIVE_PRACTICAL_COMPARISON_II_O_V1' or contract['saved_inputs'][c['input']]!=INPUT
        or io.sha(root/c['origin_records']['SOURCE'])!=contract['source_manifest_sha256']):
        raise ValueError('ORIGINAL_HAMILTONIAN_BINDING')
    physical={}
    for arm,digest in CANDIDATES.items():
        a=result['arms'][arm];df=io.read(root/c['endpoints'][arm]['physical'])
        if (a['endpoint_sha256']!=digest or a['physical']!=df or not a['endpoint_eligible']
            or a['trajectory']['accepted_U_updates']!=16 or a['status']!='MAXIMUM_ACCEPTED_UPDATES'
            or a['candidate_identity']['source_manifest_sha256']!=contract['source_manifest_sha256']):
            raise ValueError('ENDPOINT_PROVENANCE:'+arm)
        if (df['candidate_sha256']!=digest or df['actual_rank']!=8 or not df['budget_PASS']
            or df['guard_policy']!='sector_operation_scale_guard8_proposal_v1'
            or df['budget_Ha']!=.00019875 or df['kappa']!='original_g_corrected' or df['scalar']!='nuclear_once'
            or not math.isfinite(df['sector_error']+df['numerical_guard'])
            or min(df['sector_error'],df['numerical_guard'])<0 or df['sector_error']+df['numerical_guard']>.00019875):
            raise ValueError('SAVED_PHYSICAL_QUALIFICATION:'+arm)
        physical[arm]=df
    return dict(physical=physical,shells=io.read(root/c['shell_reports']),source=s['SOURCE'])


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    for k,w in fixed().items():
        if c[k]!=w:raise ValueError('FIXED_NEAR_R32:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    src=io.read(root/c['source_manifest'])
    if ENTRY not in src['files']:raise ValueError('UNPINNED_ENTRY')
    for group in (src['files'],c['saved_inputs']):
        for name,digest in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=digest:raise ValueError('SOURCE_INPUT:'+name)
    needed={c['input'],c['physical_reference'],c['shell_reports'],*c['origin_records'].values()}
    needed.update(p for x in c['endpoints'].values() for p in x.values())
    needed.update(p for x in c['shells'].values() for p in x.values())
    if not needed<=c['saved_inputs'].keys() or c['saved_inputs'][c['input']]!=INPUT:
        raise ValueError('UNPINNED_INPUT_ROLES')
    for a,h in CANDIDATES.items():
        if c['saved_inputs'][c['endpoints'][a]['candidate']]!=h:raise ValueError('WRONG_CANDIDATE:'+a)
    for name,v in c['packages'].items():
        if importlib.metadata.version(name)!=v:raise ValueError('ENVIRONMENT:'+name)
    from .c1_h4_native_fair import GAUSSIAN,PHASE
    if c['gaussian']!=GAUSSIAN or c['compiler_phase']!=PHASE:raise ValueError('P0_PRECISION_POLICY')
    evidence(root,c)
    if (root/FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def dispatch(call,arm,stage):
    def f(k,l,fn):return call(f'{arm}.{stage}.{k}',l,fn)
    f.stop=call.stop
    return f


def ground(H,out,label):
    """Existing eigensolve/residual gate; alias ineligibility is not correctness."""
    pair=perf.loader.ground(H,out,label)
    try:row=perf.loader.ground_gate(H,pair)
    except ValueError as exc:
        if str(exc)!='UNRESOLVED_GROUND_ALIAS':raise
        row=dict(status='UNRESOLVED_GROUND_ALIAS',eligible=False)
    io.save(out/(label+'_ground.json'),row)
    return pair,row


def prepare_arm(root,c,out,arm,common,call):
    import numpy as np
    b,loader=perf.b,perf.loader
    digest=CANDIDATES[arm];paths=c['endpoints'][arm]
    d=call('selected_load',arm,lambda:loader.load(root/paths['candidate'],('J','U')))
    sector=call('reference_load',arm,lambda:loader.load(root/paths['reference'],('J','U','generators','H')))
    def bind():
        downstream.strong.m2.validate_arrays(common['p'],d,arm)
        if not all(np.array_equal(d[k],sector[k]) for k in ('J','U')):raise ValueError('MIXED_CANDIDATE_REFERENCE')
        from .c1_stage1 import sector_indices
        ix=common['basis']
        if sector['generators'].shape!=(9,36,36) or sector['H'].shape!=(36,36) or not np.array_equal(np.sort(ix),sector_indices(4,2,2)):
            raise ValueError('SECTOR_REFERENCE_SCHEMA')
        b.component_check(sum(sector['generators']),sector['H'],1e-10,label='same candidate saved sector sum')
        sector['full_input_indices']=ix
        return dict(status='PASS',candidate=digest,saved_DF=common['saved']['physical'][arm],new_DF_error=0,old_005_064_values_used=False)
    io.save(out/'identity.json',call('candidate_binding',arm,bind))
    model=call('native_model',arm,lambda:b.prepare_arm(**common['p'],**d,role='proposed',source_identity=digest,input_identity=INPUT,ld=2))
    for i in range(9):
        call('general_factor_gate',str(i),lambda i=i:b.component_check(b.candidate_factor(common['ctx'],model,i),sector['generators'][i],1e-10,label='candidate GeneralK/saved independent CAR'))
    pair,row=call('candidate_ground',arm,lambda:ground(sector['H'],out,'candidate'))
    if row['status']!='PASS':return None,dict(status='UNRESOLVED_GROUND_ALIAS',candidate=digest)
    tail,config,dist=call('law_prepare',arm,lambda:b.configure(model,Fraction(1,10),32,0))
    def law():
        from .c_rc2_pr_stage_b import distribution_gate
        value=b.tail_gate(common['ctx'],model,sector['generators'],tail)
        distribution_gate(tail,config,dist,expected_steps=32)
        if not math.isclose(value['lambda_R'],common['saved']['physical'][arm]['exact_tail_lambda_Ha'],rel_tol=1e-12,abs_tol=1e-14):raise ValueError('LAMBDA_CONVENTION')
        return value
    io.save(out/'law.json',call('law_gate',arm,law))
    io.save(out/'normalization.json',call('normalization_gate',arm,lambda:downstream.retuning.normalization(tail,config,32)))
    return dict(candidate_identity=digest,candidate_label=arm,p=common['p'],d=d,sector=sector,
        arm=model,ctx=common['ctx'],pair=pair,ground=pair[1][:,0],target=common['target'],tail=tail,config=config,dist=dist,
        saved=dict(physical=common['saved']['physical'][arm],shells=common['saved']['shells'])),dict(status='PASS',candidate=digest)


def execute(root,c,out,call,*,preparer=prepare_arm,reference=downstream.references,native=downstream.calibrate,strong=deterministic.evaluate):
    from . import c1_molecule_extension as molecule
    b,loader=perf.b,perf.loader
    for name in ('shared',*CANDIDATES):(out/name).mkdir()
    def cc(k,l,f):return call('common.'+k,l,f)
    saved=cc('evidence_load','fixed pair and saved DF norms',lambda:evidence(root,c))
    p=cc('input_load','original H4 h/g/nuclear scalar',lambda:loader.load(root/c['input'],('h','g','constant')))
    physical=cc('physical_reference_load','original H and common sector basis only',lambda:loader.load(root/c['physical_reference'],('H','sector_basis_indices')))
    ctx=cc('sector_context','independent SectorCAR4',lambda:molecule.SectorCAR(4))
    _,physical_ground=cc('physical_ground','shared physical Hamiltonian,not candidate ground',lambda:ground(physical['H'],out/'shared','physical'))
    target=cc('target_validation','existing common CA/RPE13',lambda:downstream.m4.accuracy.AccuracyTarget(**downstream.m4.TARGET))
    if target.horizon()['maximum_round_index_M']!=12:raise ValueError('REGISTERED_HORIZON')
    io.save(out/'seed_manifest.json',cc('seed_manifest','64 fresh candidate/cohort-specific keys',lambda:seed_manifest(c['source_manifest_sha256'])))
    if physical_ground['status']!='PASS':return dict(status='SHARED_PHYSICAL_ALIAS_UNQUALIFIED',arms={},mandatory_stop=True)
    common=dict(p=p,saved=saved,ctx=ctx,basis=b.stage1.bit_reverse_indices(8)[physical['sector_basis_indices']],target=target)
    prepared={};results={}
    # Tiny sector setup is ordered; the dominant 13-round and native32 kernels
    # are parallel. Qiskit global-equivalence mutation contexts must not overlap.
    for name in CANDIDATES:
        for leaf in ('A','B','C','D','E'):(out/name/leaf).mkdir()
        a,status=preparer(root,c,out/name/'A',name,common,dispatch(call,name,'A'))
        prepared[name]=a;results[name]=dict(candidate=CANDIDATES[name],A=status,PR=None,D=None,statistics=None)
    def refs(name):
        if prepared[name] is None:return name,None,None
        result,packet=reference(prepared[name],out/name/'B',dispatch(call,name,'B'))
        return name,result,packet
    with ThreadPoolExecutor(max_workers=2) as pool:
        refs_done=list(pool.map(refs,CANDIDATES))
    packets={}
    for name,pr,packet in refs_done:results[name]['PR']=pr;packets[name]=packet
    for name in CANDIDATES:
        a=prepared[name]
        if a is None:continue
        # D has its own PF admission, independent of PR's r32 phase room.
        d=strong(root,c,out/name/'D',dispatch(call,name,'D'),a);results[name]['D']=d
        pr=results[name]['PR']
        if pr['native_admitted']:
            native(root,c,out/name/'C',dispatch(call,name,'C'),a,pr,packets[name],sampling=sampling(name))
            ec=dispatch(call,name,'E');rows=packets[name]['kernel_samples']
            ec('native_rows_validation','all32;no dropping',lambda:statistics.shared.validate_rows(rows,32))
            binding=ec('accounting_binding','W kernel occurrences + shells',lambda:statistics.affine(packets[name],pr))
            s=ec('secondary_summary','descriptive six metrics;secondary individual PR/D intervals',lambda:statistics.arm_summary(rows,binding,d,calibration=pr['calibration'],call=ec))
            io.save(out/name/'E'/'accounting.json',binding);io.save(out/name/'E'/'statistics.json',s)
            results[name]['statistics']=s;results[name]['bindings']=binding
        io.save(out/name/'result.json',dispatch(call,name,'E')('arm_summary','candidate results;unknown bounds null',lambda:results[name]))
    def compare():
        eligible=all(results[a]['statistics'] is not None for a in CANDIDATES)
        if not eligible:return dict(status='UNRESOLVED_ONE_OR_BOTH_PR_INELIGIBLE',primary=None,point_winner=None)
        primary=statistics.primary(packets['PR']['kernel_samples'],packets['ENTRY']['kernel_samples'],
            results['PR']['bindings']['ideal_exact_native'],results['ENTRY']['bindings']['ideal_exact_native'],
            calibrations=[results[a]['PR']['calibration'] for a in CANDIDATES])
        return dict(status='CONDITIONAL_FIXED_R32_PRICE_COMPARISON',primary=primary,
            qualification='finite-bit price proxy + ideal-law/exact-native assumptions,not actual CA or validated coverage')
    comparison=cc('main_comparison','one prespecified ideal total-RZ difference',compare)
    return cc('result_summary','fixed two-endpoint scope then STOP',lambda:dict(status='NEAR_PRACTICAL_R32_DOWNSTREAM_COMPLETE',
        arms=results,comparison=comparison,actual_CA_achieved=None,guaranteed_resource_superiority=None,
        all_law_synthesis=None,implemented_law_error=None,independent_novelty=None,retry=0,resume=0,mandatory_stop=True))


def run(root,path,*,execute_science=False):
    if execute_science is not True:raise PermissionError('EXPLICIT_SCIENTIFIC_EXECUTION_REQUIRED')
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH;out.mkdir(parents=True)
    call=downstream.m4.integrated.Calls(out,LIMITS,MAX_CALLS);tick,cpu=time.monotonic(),time.process_time();result=None
    try:
        io.save(out/'SOURCE.json',io.read(root/c['source_manifest']));io.save(out/'contract.json',c)
        result=execute(root,c,out,call);io.save(out/'result.json',result);return result
    except BaseException as exc:
        call.stop.set();io.save(out/'failure.json',dict(error=repr(exc),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),first_failure=call.first_failure,maximum_logical_units=MAX_CALLS,
            reason=result['status'] if result else 'CORRECTNESS_OR_EXECUTION_FAILURE',retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
