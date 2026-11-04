"""Separate dormant head/P entries; shared byte/JSON preflight and STOP handling."""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.metadata
import resource
import time
from . import c_per_word_range_schema as schema
from . import c_head_reference_bridge as head
from . import c_stage_p_literal as perword

io = schema.io
BASE = '1c0e1d1b4967efecb08dd3d8e7a747eba6c83039'
FAMILY = 'artifacts/research_c_head_bridge_stage_p/2026-11-04'
ENTRY = 'scripts/prepare_research_c_head_bridge_stage_p.py'
CONTRACTS = {k:'docs/research/research_c_'+name+'_contract_unapproved_20261104.json'
             for k,name in [('H','head_bridge'),('P','stage_p_literal')]}
SCOPES = dict(H='C_PR_R32_SAVED_HEAD_BRIDGE_V1',P='C_PR_R32_ONE_WORD_LITERAL_STAGE_P_V1')
LIMITS = dict(
 H=dict(stage_R_metadata=1,B2_metadata=1,arrays_load=1,identity_gate=1,native_model=1,
        sector_invariance=1,half_exponential_pair=3,head_products=1,kernel_bridge=1,
        round_transport=13,B2_connection=1,summary=1),
 P=dict(evidence_metadata=1,input_load=1,candidate_load=1,Stage_R_heads_load=1,native_model=1,
        event_catalog=1,word_binding=1,gaussian_build=5,relative_gaussian=4,
        kernel_build=1,compile=1,records=2,literal_reference=1,compiled_raw_certificate=1,
        raw_literal_certificate=1,perword_triangle=1,adjoint_build=2,adjoint_record=2,
        adjoint_certificate=2,shell_load=6,shell_certificate=3,finite_signal_profile=1,summary=1))
MAXIMUM = {k:sum(v.values()) for k,v in LIMITS.items()}


def stage_r(root,c):
    r=io.read(root/c['R_result']);p=io.read(root/c['population']);s=io.read(root/c['R_STOP'])
    if (r['status']!='LITERAL_POPULATION_NORM_STAGE_R_COMPLETE' or r['population']!=p
        or s['scientific_units']!=256 or s['failed'] is not None or s['retry'] or s['resume']
        or s['mandatory_stop'] is not True or not schema.same_json(r['scientific_input'],c['identity'])
        or p['target']!=schema.previous.method.TARGET or p['status']!='VALIDATED_LITERAL_POPULATION_NORM'
        or p['target_id']!=c['population_target_id'] or p['D_norm_upper_exact']!=c['population_a_exact']
        or p['d_range_upper_exact']!=c['population_M_exact']
        or F(p['d_range_upper_exact'])!=1+F(p['D_norm_upper_exact'])
        or p['all_q_words_compile_success_qualified'] is not False):
        raise ValueError('FIXED_STAGE_R_IDENTITY_AND_EXACT_RANGE')
    return p


def b2_metadata(root,c):
    from . import c_b1_b2_reference_bound_integration as b2
    r=io.read(root/c['B2_result']);s=io.read(root/c['B2_STOP'])
    b2.b2_records(r,pins=b2.publications())
    if s['scientific_units']!=27 or s['failed'] or s['retry'] or s['resume'] or not s['mandatory_stop']:
        raise ValueError('PRESERVED_B2_STOP')
    return r


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path);stage=c['stage']
    if (stage not in SCOPES or c['base_commit']!=BASE or c['scope']!=SCOPES[stage]
        or c['call_limits']!=LIMITS[stage] or c['maximum_logical_units']!=MAXIMUM[stage]
        or c['fresh_output']!=FAMILY+'/'+stage+'-run01' or c['retry'] or c['resume']
        or not c['mandatory_stop'] or c['precision_bits']!=(384 if stage=='H' else 192)
        or c['basis_order']!='bit_reverse(saved_interleaved_sector_indices)'
        or c['head_time']!='1/20' or c['h']!='1/10'
        or not schema.same_json(c['identity'],schema.previous.prior.integration.previous.identity())):
        raise ValueError('FIXED_HEAD_OR_P_CONTRACT')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,want in group.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or io.sha(f)!=want:raise ValueError('SOURCE_INPUT:'+p)
    for role in ('R_result','population','R_STOP','full_heads','B2_result','B2_STOP','event_mean','grid_event_mean','atoms','stage_A_result','grid_table','normalization','law'):
        if c[role] not in c['saved_inputs']:raise ValueError('UNPINNED_ROLE:'+role)
    if (c['parent_roles']!=io.read(root/schema.CONTRACT)['parent_roles']
            or c['saved_inputs'][c['parent_roles']['candidate']]!=c['identity']['candidate']):
        raise ValueError('FIXED_PARENT_INPUT_ROLES')
    if stage=='P':
        shells=io.read(root/'docs/research/research_c_fixed_word_pilot_contract_unapproved_20261104.json')['shells']
        if c['shells']!=shells:raise ValueError('FIXED_SHELL_ROLES')
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT:'+name)
    stage_r(root,c);b2_metadata(root,c);meta=schema.saved_metadata(root,c)
    law=io.read(root/c['law']);norm=io.read(root/c['normalization'])
    if (float(law['identity']).hex()!=c['tail_phase_identity_hex'] or norm['r']!=32
        or norm['identity_phase_separate'] is not True):raise ValueError('SCALAR_OR_RTE_METADATA')
    if stage=='P':
        old=io.read(root/schema.CONTRACT)['stages']['P']['proposed_word']
        if c['word']!=old or len(c['word'])!=32 or any(x not in {a['atom_id'] for a in meta['atoms']} for x in c['word']):
            raise ValueError('ONE_PREVIOUSLY_PROPOSED_WORD_ONLY')
        probs=io.read(root/c['grid_table'])['ideal_probabilities']
        normalization=io.read(root/c['event_mean'])['report']['normalization']
        if (type(probs) is not list or len(probs)!=216 or any(type(x) is not str for x in probs)
                or type(normalization) is not float or not 0<normalization<float('inf')
                or type(norm['tau']) is not float or not 0<norm['tau']<float('inf')):
            raise ValueError('SAVED_TYPED_EVENT_METADATA_SCHEMA')
    if (root/c['fresh_output']).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_READY_FOR_SEPARATE_PERMISSION',stage=stage,
        maximum_logical_units=MAXIMUM[stage],scientific_calls=0,arrays_decoded=0,
        head_bridge_upper_exact=None,per_word_upper_exact=None,Stage_C_ready=False,
        all_q_compiler_totality_qualified=False,actual_CA_achieved=None,mandatory_stop=True)


def execute_h(root,c,out,call):
    import numpy as np
    from .c_pr_aware_df_stage_b_runner import load
    from . import c_pr_aware_df_stage_b as b
    p=call('stage_R_metadata','saved exact a/M;no R recomputation',lambda:stage_r(root,c))
    B2=call('B2_metadata','old two disjoint bounds',lambda:b2_metadata(root,c))
    def arrays():
        d=load(root/c['parent_roles']['sector'],('J','U','generators'))
        d.update(load(root/c['parent_roles']['input'],('h','g','constant')))
        d['full']=load(root/c['full_heads'],('heads',))['heads']
        d['indices']=load(root/c['parent_roles']['basis'],('sector_basis_indices',))['sector_basis_indices']
        d['candidate']=load(root/c['parent_roles']['candidate'],('J','U'))
        return d
    d=call('arrays_load','five saved inputs;no generator regeneration',arrays)
    def identity():
        if any(not np.array_equal(d[k],d['candidate'][k]) for k in ('J','U')):
            raise ValueError('SAVED_JU_IDENTITY')
        if d['full'].shape!=(3,256,256) or d['generators'].shape!=(9,36,36) or d['indices'].shape!=(36,):
            raise ValueError('SAVED_HEAD_DIMENSIONS')
        return True
    call('identity_gate','same candidate/scalar/frame and basis provenance',identity)
    def model():
        arm=b.prepare_arm(d['h'],d['g'],d['constant'],d['J'],d['U'],role='proposed',
            source_identity=c['identity']['candidate'],input_identity=c['saved_inputs'][c['parent_roles']['input']],ld=2)
        if float(arm.plan.extracted_identity).hex()!=c['tail_phase_identity_hex']:
            raise ValueError('COMMON_TAIL_PHASE_IDENTITY')
        return arm
    call('native_model','same tail identity;no circuit',model)
    ix=b.stage1.bit_reverse_indices(8)[d['indices']]
    A=call('sector_invariance','exact reducing sector;both cross blocks zero',lambda:head.identity(
        d['full'],d['generators'][:3],ix,candidate=c['identity']['candidate'],expected_candidate=c['identity']['candidate']))
    tasks=[(a,z,F(1,10)) for a,z in zip(A,d['generators'][:3])]
    from .c1_stage3_integrated import resource_plan
    from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
    import multiprocessing
    plan=resource_plan(3,full_dimension=36)
    with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn')) as processes:
        def task(pair):
            i,t=pair
            def evaluate():
                value=processes.submit(head.half_pair,t).result();io.save(out/('head'+str(i)+'.json'),value);return value
            return call('half_exponential_pair',str(i),evaluate)
        with ThreadPoolExecutor(max_workers=plan['workers']) as threads:pairs=list(threads.map(task,enumerate(tasks)))
    h=call('head_products','both orders and phase-sensitive differences',lambda:head.products(pairs));io.save(out/'head_products.json',h)
    kernel=call('kernel_bridge','shared literal tail;head difference once',lambda:head.kernel_bridge(h,p));io.save(out/'kernel_bridge.json',kernel)
    rows=[]
    for m in range(13):
        row=call('round_transport',str(m),lambda m=m:head.round_bridge(kernel,m));rows.append(row);io.save(out/f'm{m:02d}.json',row)
    connected=call('B2_connection','head+B2 bridge+B2 uncertainty once',lambda:head.connect_b2(rows,B2,target_id=p['target_id']))
    io.save(out/'B2_connection.json',connected)
    return call('summary','reference only;no calibration/CA',lambda:dict(status='HEAD_BRIDGE_COMPLETE',
        kernel=kernel,rounds=rows,connection=connected,resource_plan=plan,Stage_C_ready=False,actual_CA_achieved=None))


def execute_p(root,c,out,call):
    from .c_pr_aware_df_stage_b_runner import load
    from . import c_pr_aware_df_stage_b as b
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h4_native_fair import basis_request,compile_checked
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from qiskit import qpy
    pop=call('evidence_metadata','same Stage R target and exact range',lambda:stage_r(root,c))
    p=call('input_load','original h/g/scalar',lambda:load(root/c['parent_roles']['input'],('h','g','constant')))
    d=call('candidate_load','saved J/U',lambda:load(root/c['parent_roles']['candidate'],('J','U')))
    heads=call('Stage_R_heads_load','saved full heads only;no R replay',lambda:load(root/c['full_heads'],('heads',))['heads'])
    arm=call('native_model','unchanged general-K adapter',lambda:b.prepare_arm(**p,**d,role='proposed',
        source_identity=c['identity']['candidate'],input_identity=c['saved_inputs'][c['parent_roles']['input']],ld=2))
    def catalog():
        bound=schema.event_catalog(root,c,arm);meta=schema.saved_metadata(root,c)
        if float(arm.plan.extracted_identity).hex()!=c['tail_phase_identity_hex']:raise ValueError('TAIL_PHASE_IDENTITY')
        probs=io.read(root/c['grid_table'])['ideal_probabilities']
        normalization=io.read(root/c['event_mean'])['report']['normalization']
        return perword.events_from_saved(arm,c['word'],meta,probs,normalization)
    events=call('event_catalog','one fixed word with saved angle;no RNG',catalog)
    binding=call('word_binding','exact Stage R head/atom/angle digest',lambda:schema.previous.method.validate(arm,heads,events,h=F(1,10),r=32))
    perword.bind_population(dict(binding=binding),pop)
    builder=StrongBuilder(relative_fusion=True,call=call);shared=SharedQiskitCalls(call)
    for i in (0,1,2,7,8):call('gaussian_build',str(i),lambda i=i:basis_request(builder,arm.model,arm.model.blocks[i]))
    for order in ((0,1,2),(2,1,0)):
        for i,j in zip(order,order[1:]):builder.transition(arm.model,arm.model.blocks[i],arm.model.blocks[j])
    norm=io.read(root/c['normalization'])
    config=NS(rte_steps=32,evolution_time=.1,dimensionless_step_time=norm['tau'])
    raw=shared('kernel_build','one finite word',lambda:b.build_kernel(arm,F(1,10),events,config,builder=builder))
    with (out/'raw.qpy').open('xb') as f:qpy.dump(raw,f)
    compiled=shared('compile','same P0',lambda:compile_checked(raw,CompilerProfile()))
    with (out/'compiled.qpy').open('xb') as f:qpy.dump(compiled,f)
    rec={}
    for name,q in [('raw',raw),('compiled',compiled)]:
        rec[name]=call('records',name,lambda q=q:perword.fixed.record(q));io.save(out/(name+'_record.json'),rec[name])
    reference=call('literal_reference','all256 spin blocks;no projection of native gates',lambda:schema.previous.method.build_literal_reference(arm,heads,events,parallel=True))
    io.save(out/'literal_reference.json',reference);perword.bind_population(reference,pop)
    cr=call('compiled_raw_certificate','full512 columns',lambda:perword.compiled_raw(rec['compiled'],rec['raw'],parallel=True));io.save(out/'compiled_raw.json',cr)
    rd=call('raw_literal_certificate','full512 columns;no cap2',lambda:schema.previous.method.compare_literal(rec['raw'],reference,parallel=True));io.save(out/'raw_literal.json',rd)
    bound=call('perword_triangle','two legs once;Head-to-B2 not included',lambda:perword.triangle(rec['raw'],rec['compiled'],reference,cr,rd));io.save(out/'per_word.json',bound)
    inverse={}
    with shared.adjoint_builds():
        for name,q in [('raw',raw),('compiled',compiled)]:
            inv=call('adjoint_build',name,lambda q=q:b.native.actual_adjoint(q))
            with (out/(name+'_adjoint.qpy')).open('xb') as f:qpy.dump(inv,f)
            record=call('adjoint_record',name,lambda:perword.fixed.record(inv));io.save(out/(name+'_adjoint_record.json'),record)
            def inverse_certificate(name=name):
                value=perword.fixed.compare(record,perword.fixed.adjoint(rec[name]),parallel=True)
                forward=bound['upper_exact'] if name=='compiled' else rd['upper_exact']
                return dict(value,to_literal_adjoint_upper_exact=str(F(value['upper_exact'])+F(forward)),
                    same_realized_forward_record=schema.previous.method._digest(rec[name]),
                    literal_target='D dagger;not inverse of nonunitary D',literal_inverse_claimed=False)
            inverse[name]=call('adjoint_certificate',name,inverse_certificate)
            io.save(out/(name+'_actual_inverse.json'),inverse[name])
    shells={}
    for label in ('head','X','Y'):
        qs={}
        for kind in ('raw','compiled'):
            def shell_load():
                with (root/c['shells'][label][kind]).open('rb') as f:items=qpy.load(f)
                if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                return items[0]
            qs[kind]=call('shell_load',label+kind,shell_load)
        shells[label]=call('shell_certificate',label,lambda:perword.shell_certificate(qs['raw'],qs['compiled'],label));io.save(out/(label+'_shell.json'),shells[label])
    signal=call('finite_signal_profile','one occurrence,both norm factors;not expected bias',lambda:perword.finite_signal(bound,pop,
        {k:v['compiled_ideal']['upper_exact'] for k,v in shells.items()}));io.save(out/'finite_signal.json',signal)
    return call('summary','finite word only;all-q totality remains false',lambda:dict(status='STAGE_P_ONE_WORD_COMPLETE',
        per_word=bound,actual_inverse=inverse,shells=shells,finite_signal=signal,word=c['word'],
        finite_word_compile_success=True,all_q_compile_success_qualified=False,calibration_ready=False,
        population_binding=perword.bind_population(reference,pop),Stage_C_ready=False,actual_CA_achieved=None))


def run(root,path,*,execute_science=False,executor=None):
    if not execute_science:raise PermissionError('SEPARATE_FIXED_HEAD_OR_P_SCIENTIFIC_PERMISSION')
    root=Path(root).resolve();preflight(root,path);c=io.read(path)
    from .c_pr_aware_df_zsxx_enclosure import runtime_model
    from .c1_stage3_integrated import Calls
    runtime_model();out=root/c['fresh_output'];out.mkdir(parents=True,exist_ok=False)
    call=Calls(out,LIMITS[c['stage']],MAXIMUM[c['stage']]);tick=time.monotonic()
    io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
    try:
        value=(executor or (execute_h if c['stage']=='H' else execute_p))(root,c,out,call)
        io.save(out/'result.json',value);return value
    except BaseException as exc:
        call.stop.set()
        if call.first_failure is None:call.first_failure=dict(error_type=type(exc).__name__,error=str(exc))
        io.save(out/'partial_failure.json',dict(error_type=type(exc).__name__,error=str(exc)));raise
    finally:
        io.save(out/'STOP.json',dict(call_counts=call.counts,scientific_units=sum(call.counts.values()),failed=call.first_failure,
            retry=0,resume=0,mandatory_stop=True,Stage_C_executed=False,actual_CA_achieved=None))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
