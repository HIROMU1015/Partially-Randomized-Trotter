"""Artificial candidates only. No molecular I/O, scientific sampling or fit."""
from copy import deepcopy, copy
from types import SimpleNamespace, ModuleType
from pathlib import Path
import json
import ast
import subprocess
import sys

import pytest
from compiled_price_artificial import ROOT, np, P, E, dfq, costs
from trottertracks import candidate_price_adapter as A
from trottertracks import pr_rpe_rmse_controller as B
from check_pr_rpe_rmse_controller import spec
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.rte import make_rte_config, _make_event
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_partial_s2 import DFPartialS2StepRequest

SOURCE = {'fixture': 'ARTIFICIAL_EXECUTION_ONLY'}
INPUT = {'fixture': 'INVENTED_MATRICES_NOT_H4_INPUT'}


def mock_compile(qc, settings, **kwargs):
    ops=qc.count_ops()
    return SimpleNamespace(compiler_settings_hash=B.p.PROFILE['compiler'],rz_count=int(ops.get('rz',0)),
        cx_count=int(ops.get('cx',0)),rz_depth=int(qc.depth()),cx_depth=int(qc.depth()),
        total_depth=int(qc.depth()),circuit_size=len(qc.data))


@pytest.fixture
def env(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No input/normal sampler in these tests')
    monkeypatch.setattr(np,'load',forbidden)
    from trotterlib import rte
    from trotterlib.df_rte_circuit import DFRTEEventPreparation
    monkeypatch.setattr(rte,'sample_rte_events',forbidden)
    monkeypatch.setattr(rte,'iter_sample_rte_events',forbidden)
    monkeypatch.setattr(DFRTEEventPreparation,'sample_occurrence_request',forbidden)
    # Arbitrary eight-wire matrices for unchanged wire-8 Accounting. Not H4.
    one=np.diag([.31,-.17,.09,-.23,.11,.07,-.13,.19]).astype(complex)
    gs=[np.diag(np.arange(1,9)*a-.41) for a in (.07,-.13,.11)]
    gs[1][0,1]=gs[1][1,0]=.043
    ham=DFHamiltonian(constant=.173,one_body=one,lambdas=np.array([.23,-.37,.19]),
        g_matrices=tuple(gs),metadata={'synthetic':True})
    # Use real pinned prepare on invented input to bind the actual input hash/lambda.
    runtime=A.NativeRuntime(ROOT,ham,binding={},input_source=INPUT,execution_source=SOURCE)
    t=runtime._tools();t.compile=mock_compile
    preparations={k:t.h['prepare'](ham,k) for k in (1,2)}
    binding=dict(system='H4',rank=3,hamiltonian=preparations[1].hamiltonian_hash,
        state='MOCK_STATE',snapshot='MOCK_SNAPSHOT',decomposition='MOCK_DF',source=INPUT)
    runtime.binding=deepcopy(binding)
    return runtime,preparations,binding


def request(env,*,prefix=1,K=2,formula='2nd',q=2,R=4,delta=.125,axes=('X','Y'),source=SOURCE):
    runtime,preps,binding=env
    s=spec(L_D=prefix,M=0,kappa=R)
    s.update(binding=deepcopy(binding),h=delta*q,delta_pf=delta,finite_K=K,formula=formula,
        profile=deepcopy(B.p.PROFILE))
    s['split']['lambda_R']=preps[prefix].rte_preparation.symbolic_tail.lambda_r
    s['event_mapping']['kind']='ARTIFICIAL_EXPLICIT_MAPPING'
    def mapper(spec,row):
        return dict(kind=spec['event_mapping']['kind'],source=deepcopy(spec['event_mapping']['source']),
            kappa=deepcopy(spec['kappa']),R_outer=R,event_count=row['q_PF']*R)
    config=B.generate(s,event_mapper=mapper)
    c=dict(system='H4',method='PR',formula=formula,prefix=prefix,K=K,q=q,R_outer=R,T=delta*q,delta=delta)
    c['id']=B.p.candidate_id(c)
    return config,A.from_b(config,0,c,input_source=INPUT,execution_source=source,axes=axes)


def path(runtime,req):
    ctx=runtime.context(req);c=req['physical_context']['physical_candidate'];out=[]
    for step in ctx.h['core'].schedule(c['formula'],c['q'],'ABS_TIME_PER_OUTER',c['R_outer'],T=c['T']):
        config,dist=make_rte_config(ctx.prep.rte_preparation.symbolic_tail,
            evolution_time=step['signed_time'],rte_steps=step['rte_steps'],truncation_tolerance=1.,finite_taylor_order=c['K'])
        comps=ctx.prep.rte_preparation.symbolic_tail.components
        choices=[i for i,v in enumerate(comps) if len(v.diagonal_pauli_support)==2]
        events=[]
        for j in range(step['rte_steps']):
            rot=choices[j%len(choices)]
            indices=tuple([rot]+[choices[k%len(choices)] for k in range(c['K'])]) if c['K']>=2 and j==1 else (rot,)
            events.append(_make_event(indices,comps,dist,(len(indices)-1)//2))
        occ=DFRTEEventSequenceCircuitRequest(events=tuple(events),component_specs=ctx.prep.rte_preparation.component_specs,
            controlled=True,ancilla_qubit=8,tail_id=config.tail_id,tail_hash=config.tail_hash,occurrence_rte_steps=step['rte_steps'])
        out.append(DFPartialS2StepRequest(ctx.prep,step['signed_time'],config,dist,occ,controlled=True,ancilla_qubit=8,seed=j))
    return out


def test_preparation_head_tail_independence_and_context_reuse(env):
    runtime,_,_=env
    _,r1=request(env,prefix=1);_,r2=request(env,prefix=2)
    a=runtime.context(r1);b=runtime.context(r2)
    assert a is runtime.context(r1)
    assert a.prep is not b.prep and a.prep.preparation_hash!=b.prep.preparation_hash
    assert len(a.blocks[0])==2 and len(b.blocks[0])==3
    assert a.prep.randomized_block_indices==(1,2) and b.prep.randomized_block_indices==(2,)
    assert a.prep.rte_preparation.symbolic_tail.tail_hash!=b.prep.rte_preparation.symbolic_tail.tail_hash
    assert len(runtime.preparations)==2
    assert a.scalar==runtime.ham.constant+a.prep.extracted_identity_coefficient


@pytest.mark.parametrize('K',[0,2,4,6])
def test_formula_K_q_R_context_and_signed_word(env,K):
    runtime,_,_=env
    _,r=request(env,K=K,formula='4th(m5_best)',q=1,R=13)
    ctx=runtime.context(r);events=path(runtime,r)
    seeds,laws=A._law_records(ctx,r['physical_context']['physical_candidate'],events)
    assert len(events)==11 and any(x.step_time<0 for x in events)
    assert all(l['orders']==list(range(0,K+1,2)) for l in laws)
    assert sum(len(x.rte_occurrence.events) for x in events)==13
    _,other=request(env,K=K,q=1,R=13)
    assert r['context_key']!=other['context_key']


def test_same_event_count_different_schedule_and_sources_do_not_alias(env):
    runtime,_,_=env
    _,a=request(env,q=1,R=8);_,b=request(env,q=2,R=4)
    assert a['physical_context']['structure']['event_count']==b['physical_context']['structure']['event_count']
    assert runtime.context(a) is not runtime.context(b)
    _,c=request(env,source={'fixture':'OTHER_SOURCE'})
    with pytest.raises(ValueError,match='SOURCE'):runtime.context(c)
    d=deepcopy(a);d['physical_context']['physical_candidate']['K']=6
    with pytest.raises(ValueError,match='mutated'):runtime.context(d)


@pytest.mark.parametrize('axes',[('X','Y'),('Y',),('X',),('Y','X')])
def test_axis_selection_quantities_lowering_wrapper_and_scalar(env,tmp_path,axes):
    runtime,_,_=env;_,r=request(env,axes=axes,q=1,R=2)
    seen=[];ctx=runtime.context(r)
    original=ctx.compile
    def compile(qc,*args,**kw):
        assert qc.count_ops().get('measure')==1
        assert dfq.QiskitDFRTEEventCircuitBuilder.lowering_id==E.LOWERING_ID
        seen.append(qc)
        return original(qc,*args,**kw)
    ctx.compile=compile
    result=A.Adapter(ROOT,runtime,cache_root=tmp_path/'cache').acquire(r,path(runtime,r),
        cohort='ARTIFICIAL_SUPPLIED_PATH',index=0,directory=tmp_path/'out')
    assert len(seen)==len(axes) and result['new_compiles']==len(axes)
    assert [x['axis'] for x in result['records']]==list(axes)
    for rec in result['records']:
        assert rec['D']==rec['C']-rec['S']
        assert rec['compiled_quantity']!=rec['structural_quantity']
    assert result['mean_compiled_price'] is None and not result['qualified_expectation']
    assert getattr(dfq.QiskitDFRTEEventCircuitBuilder,'lowering_id',None)!=E.LOWERING_ID
    # The scalar appears once on the ancilla immediately before the suffix.
    for qc,axis in zip(seen,axes):
        offset=4 if axis=='Y' else 3
        scalar_op=qc.data[-offset]
        assert scalar_op.operation.name=='p'
        assert float(scalar_op.operation.params[0])==-ctx.scalar*r['physical_context']['physical_candidate']['T']
        assert qc.find_bit(scalar_op.qubits[0]).index==8
        matching=[op for op in qc.data if op.operation.name=='p' and len(op.qubits)==1
            and qc.find_bit(op.qubits[0]).index==8 and float(op.operation.params[0])==float(scalar_op.operation.params[0])]
        assert len(matching)==1
    pending=A.pending_price_records(r,path_source=result['source'])
    assert all(v['mean']['rz_count'] is None and v['law_context']['ideal_sampler_distance_qualified'] is None for v in pending)


def test_fixed_prefix_pipeline_matches_C1_S_fingerprints_and_six_metrics(env,tmp_path):
    runtime,_,_=env;_,r=request(env,q=2,R=4,delta=.1);ctx=runtime.context(r);events=path(runtime,r)
    new=A.Adapter(ROOT,runtime,cache_root=tmp_path/'none').acquire(r,events,
        cohort='SYNTHETIC',index=0,directory=tmp_path/'new')
    c=r['physical_context']['physical_candidate'];st=r['physical_context']['structure']
    plan=dict(source=SOURCE,menus=dict(H4=dict(candidates=[dict(candidate=c,structure=st)])),runtime=dict(old_cache=str(tmp_path/'none')))
    # Reuse the literal original C1 fixed-context head constructor separately.
    tree=ast.parse(P._blob(ROOT,P.C1_SCIENCE,'src/trottertracks/r_price_phase1_science.py'))
    method=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='Backend'
                for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='context')
    loop=next(n for n in method.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='w')
    scope=dict(blocks=[],delta=.1,prep=ctx.prep,system='H4',np=np,core={'weights':ctx.h['core'].weights},
        one_body_diagonal_primitives=runtime._tools().one,df_squared_diagonal_primitives=runtime._tools().squared)
    exec(compile(ast.Module(body=[loop],type_ignores=[]),'original_C1_head','exec'),scope)
    assert ctx.blocks==scope['blocks']
    oldctx=copy(ctx);oldctx.blocks=scope['blocks']
    old=runtime.science.Backend(ROOT,plan);old.contexts['H4']=oldctx
    cell=dict(id=r['request_id'],candidate_id=c['id'],system='H4',cohort='SYNTHETIC')
    with runtime.lowering():
        prior=old.price_requests(cell,0,tmp_path/'old',events,[],[],conditional=False)
    assert [x['C'] for x in new['records']]==prior['C']
    assert [x['S'] for x in new['records']]==prior['S']
    assert [x['fingerprint'] for x in new['records']]==prior['fingerprints']
    for axis,rec in zip(('cosine','sine'),new['records']):
        assert rec['metrics']==json.loads((tmp_path/'old'/(axis+'.json')).read_text())['metrics']


def test_wrong_path_K_preparation_and_order_fail_before_compile(env,tmp_path):
    runtime,_,_=env;_,r=request(env,q=1,R=4);events=path(runtime,r)
    _,foreign=request(env,prefix=2,q=1,R=4)
    a=A.Adapter(ROOT,runtime,cache_root=tmp_path/'cache')
    for req,batch in [(foreign,events),(r,events+events)]:
        with pytest.raises(ValueError):a.acquire(req,batch,cohort='TEST',index=0,directory=tmp_path/'bad')
    _,differentK=request(env,K=4,q=1,R=4)
    with pytest.raises(ValueError,match='Foreign'):a.acquire(differentK,events,cohort='TEST',index=0,directory=tmp_path/'bad')
    _,signed=request(env,formula='4th(m5_best)',q=1,R=13)
    batch=path(runtime,signed)
    batch[0],batch[5]=batch[5],batch[0]
    with pytest.raises(ValueError):a.acquire(signed,batch,cohort='TEST',index=0,directory=tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_cache_hit_and_foreign_compiler_refusal(env,tmp_path):
    runtime,_,_=env;_,r=request(env,q=1,R=2);cache=tmp_path/'cache';cache.mkdir()
    adapter=A.Adapter(ROOT,runtime,cache_root=cache);events=path(runtime,r)
    first=adapter.acquire(r,events,cohort='TEST',index=0,directory=tmp_path/'one')
    for axis in ('cosine','sine'):
        raw=json.loads((tmp_path/'one'/(axis+'.json')).read_text())
        (cache/(raw['fingerprint']+'.json')).write_text(json.dumps(raw))
    runtime.context(r).compile=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('cache must hit'))
    hit=adapter.acquire(r,events,cohort='TEST',index=1,directory=tmp_path/'two')
    assert hit['cache_hits']==2 and hit['new_compiles']==0
    assert all(x['cache_reference'] for x in hit['records'])
    _,foreign=request(env,prefix=2,q=1,R=2)
    different=adapter.acquire(foreign,path(runtime,foreign),cohort='TEST',index=0,directory=tmp_path/'foreign_split')
    assert different['cache_hits']==0 and different['new_compiles']==2
    assert different['resolved_context_key']!=first['resolved_context_key']
    assert {v['fingerprint'] for v in different['records']}.isdisjoint(v['fingerprint'] for v in first['records'])
    file=cache/(first['records'][0]['fingerprint']+'.json');bad=json.loads(file.read_text());bad['compiler']='FOREIGN';file.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='CACHE_BINDING'):adapter.acquire(r,events,cohort='TEST',index=2,directory=tmp_path/'three')


def test_A_job_bridge_matches_B_and_does_not_consume_signal_or_N(env):
    config,r=request(env);row=config['rounds'][0]
    source='821c5c1ffd7c95e1690d45d2b6e8ff4ccb15bd1c'
    module=ModuleType('trottertracks._artificial_A_price_job');module.__package__='trottertracks'
    exec(compile(P._blob(ROOT,source,'src/trottertracks/pr_rpe_signal_plan.py'),source+':A_plan','exec'),module.__dict__)
    recipe=dict(source=SOURCE,deterministic_PF='DIRECT_EXPM',finite_RTE='ARTIFICIAL_ONLY',sampling=False)
    jobs=module.plan(config,{0:r['physical_context']['physical_candidate']},b_validate=B._validate,
        b_exact_scalar=B.a.exact_scalar,numerical_recipe=recipe)['jobs']
    job=jobs[0]
    other=A.from_a_job(config,job,input_source=INPUT,execution_source=SOURCE)
    assert other==r
    job['B_configuration_id']='FOREIGN'
    with pytest.raises(ValueError,match='Foreign A'):A.from_a_job(config,job,input_source=INPUT,execution_source=SOURCE)


def test_metadata_profile_physical_ID_and_axes_rejected(env):
    config,r=request(env);c=r['physical_context']['physical_candidate']
    for axes in ((),('X','X'),('cosine',)):
        with pytest.raises(ValueError):A.from_b(config,0,c,input_source=INPUT,execution_source=SOURCE,axes=axes)
    bad=deepcopy(c);bad['q']=True;bad['id']=B.p.candidate_id(bad)
    with pytest.raises(ValueError,match='integers'):A.from_b(config,0,bad,input_source=INPUT,execution_source=SOURCE)
    bad=deepcopy(c);bad['prefix']=2
    with pytest.raises(ValueError):A.from_b(config,0,bad,input_source=INPUT,execution_source=SOURCE)
    req=deepcopy(r);req['axes']=['Y']
    with pytest.raises(ValueError):A.check_request(req)


def test_native_heads_use_original_signed_piece_time_even_off_dyadic_q(env):
    runtime,_,_=env;_,r=request(env,q=3,R=13,formula='4th(m5_best)')
    ctx=runtime.context(r)
    for bs,step in zip(ctx.blocks,ctx.h['core'].schedule('4th(m5_best)',3,'ABS_TIME_PER_OUTER',13,T=.375)):
        b=ctx.prep.deterministic_blocks[0]
        expected=runtime._tools().one(np.array(b.diagonal_eigenvalues),step['signed_time']/2)
        assert bs[0]['diagonal']==expected


def test_profile_and_preparation_expectations_never_silently_fallback(env):
    runtime,_,_=env;config,r=request(env)
    for field,value in [('compiler','FOREIGN'),('lowering_id','LEGACY')]:
        s=deepcopy(config['spec']);s['profile'][field]=value
        mapping=lambda s,row:deepcopy(config['rounds'][0]['mapping'])
        foreign=B.generate(s,event_mapper=mapping)
        with pytest.raises(ValueError):A.from_b(foreign,0,r['physical_context']['physical_candidate'],
            input_source=INPUT,execution_source=SOURCE)
    wrong=A.from_b(config,0,r['physical_context']['physical_candidate'],input_source=INPUT,
        execution_source=SOURCE,preparation_hash='WRONG')
    with pytest.raises(ValueError,match='Preparation'):runtime.context(wrong)


def test_adapter_import_is_passive_and_has_no_scientific_orchestration():
    code="import sys; import trottertracks.candidate_price_adapter; assert not any(x in sys.modules for x in ('numpy','qiskit','pyscf'))"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_pending_records_cannot_satisfy_B_compiled_mean_qualification(env):
    from check_pr_rpe_rmse_controller import data
    config,r=request(env);packet,_,_=data(config)
    pending=A.pending_price_records(r)
    out=B.assess(config,packet,pending,cohorts={(config['rounds'][0]['candidate']['id'],a):'PENDING_CANDIDATE_MEAN' for a in r['axes']})
    assert not out['eligible'] and out['cost'] is None
    assert all(v['quantity']=='FULL_WRAPPER_MEAN_COMPILED_PRICE' and v['mean']['rz_count'] is None for v in pending)


def test_saved_LD1_B_configuration_is_accepted_as_metadata_only():
    # Identity/recipe compatibility only; never restore its H, events or prices.
    config=json.loads(P._blob(ROOT,P.B_CONFIG[0],P.B_CONFIG[1]))
    row=config['rounds'][0]
    result=A.from_b(config,0,row['physical_candidate'],input_source=INPUT,execution_source=SOURCE)
    assert result['identity']==row['identity'] and result['law_binding']==row['law_binding']
    assert result['physical_context']['one_body_policy']=='DETERMINISTIC'
    assert result['physical_context']['preparation_hash'] is None
