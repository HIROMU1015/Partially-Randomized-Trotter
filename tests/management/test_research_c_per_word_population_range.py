"""Invented CAR/gates/intervals only. Never decode real molecular artifacts."""
from copy import deepcopy
from fractions import Fraction as F
from types import SimpleNamespace as NS
from unittest.mock import patch
import math
import numpy as np
import pytest

from trottertracks import c_per_word_population_range as m
from trottertracks import c_pr_aware_df_fixed_certificate as fixed
from trottertracks import c_pr_aware_df_stage_b as b
from trottertracks import c_pr_aware_df_fixed_car as car
from trottertracks import c_iid_grid_expected_bias as old


def packet(nonorthogonal=False, r=2):
    from trotterlib.rte import _make_event
    h=np.array([[.125,.03125],[.03125,-.25]])
    J=np.array([[[.25,.0625],[.0625,-.125]],[[.125,-.03125],[-.03125,.0625]],
        [[-.25,.0625],[.0625,.125]],[[.0625,.03125],[.03125,-.125]]])
    # Exactly orthogonal dyadic matrices to expose exact norm1 reasoning.
    U=np.array([np.eye(2),[[0.,-1.],[1.,0.]],np.eye(2),[[0.,1.],[-1.,0.]]])
    if nonorthogonal:
        U[2]=np.array([[1.+2e-12,1e-12],[0.,1.]])
        t=.37;U[3]=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    arm=b.prepare_arm(h,np.zeros((2,)*4),.125,J,U,role='proposed',
        source_identity='invented_CDF',input_identity='invented_input',ld=2)
    tail,config,dist=b.configure(arm,F(1,10),r,0)
    catalog={a.atom_id:_make_event((i,),tail.components,dist,0) for i,a in enumerate(arm.plan.atoms)}
    a=next(k for k,v in zip(catalog,arm.plan.atoms) if len(v.support)==1)
    z=next(k for k,v in zip(catalog,arm.plan.atoms) if len(v.support)==2 and not k.startswith(a.split(':F')[0]+':F2'))
    events=tuple(catalog[x] for x in (a,z)[:r]) if r<=2 else tuple(catalog[a] for i in range(r))
    full=b.stage1.car_bilinears(4)
    G=[b.stage1.raw_base(h,.125,full)]+[b.stage1.raw_frame(j,u,full)[0] for j,u in zip(J,U)]
    ids=list(catalog);D=1<<53;n=len(ids);counts=[D//n]*n;counts[-1]+=D-sum(counts)
    table=dict(counts=counts,denominator=D,bits=53,mapping='searchsorted_right')
    return arm,G[:3],events,catalog,table,config


def raw(arm,events,config):
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    lock=SharedQiskitCalls(lambda k,l,f:f())
    return lock('kernel_build','invented',lambda:b.build_kernel(arm,F(1,10),events,config))


def policies(n):
    from test_research_c_iid_grid_expected_bias import policy
    return policy(n)


def support(M='2'):
    return dict(qualification='deterministic_population_bound',upper_exact=M,provenance='invented valid range theorem')


def interval(lo,hi):
    return dict(kind='rigorous_enclosure',lower_exact=str(lo),upper_exact=str(hi),provenance='invented true enclosure')


@pytest.fixture(scope='module')
def example():
    arm,heads,events,catalog,table,config=packet(True)
    ref=m.build_literal_reference(arm,heads,events,r=2,parallel=False)
    qc=raw(arm,events,config)
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    compiled=compile_checked(qc,CompilerProfile())
    rec=fixed.record(qc);comp=fixed.record(compiled)
    value=m.evaluate_word(rec,comp,ref,parallel=False)
    return arm,heads,events,catalog,table,config,ref,qc,rec,value


def test_two_legs_nonunitary_literal_difference_and_independent_interval_enclosure(example):
    *_,ref,qc,rec,v=example
    from qiskit.quantum_info import Operator
    from flint import ctx
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    D=np.eye(32,dtype=complex)
    with ctx.workprec(192):
        for block in ref['blocks']:
            ix=block['indices'];A=car.unpack(block['entries'],len(ix))
            for i,x in enumerate(ix):
                for j,y in enumerate(ix):D[16+x,16+y]=complex(A[i,j].mid())
    observed=float(np.linalg.norm(Operator(qc).data-D,2))
    assert F(v['upper_exact'])>0 and observed <= float(F(v['upper_exact']))+1e-13
    assert len(v['raw_literal']['columns'])==32
    assert v['raw_literal']['upper_method'].endswith('no cap2')
    assert not v['all_word_uniform'] and not v['Stage_C_ready']
    assert v['raw_literal']['reference_uncertainty'].startswith('all CAR/gate')
    assert ref['input_U_replaced'] is ref['kappa_projected'] is ref['polar_bridge_added'] is False


def test_no_polar_call_and_exact_orthogonal_reference_agrees_with_raw():
    arm,heads,events,_,_,config=packet(False)
    with patch.object(car,'polar_bridge',side_effect=AssertionError('NO_POLAR_TARGET')):
        ref=m.build_literal_reference(arm,heads,events,r=2,parallel=False)
        v=m.compare_literal(fixed.record(raw(arm,events,config)),ref,parallel=False)
    assert F(v['upper_exact'])<F(1,10**12)
    assert v['phase_fit'] is False


@pytest.mark.parametrize('fault',['phase','event_order','head_order','control','target','dimension'])
def test_wrong_target_control_phase_and_order_detected(example,fault):
    arm,heads,events,_,_,config,ref,qc,rec,_=example
    if fault=='phase':
        rec=deepcopy(rec);rec['gates'].append(dict(name='p',qubits=[4],params_hex=[.125.hex()]))
    if fault=='event_order':
        rec=fixed.record(raw(arm,tuple(reversed(events)),config))
    if fault=='head_order':
        ref=m.build_literal_reference(arm,list(reversed(heads)),events,r=2,parallel=False)
    if fault=='control':rec=fixed.identity(5)
    if fault=='target':
        ref=deepcopy(ref);ref['target']='POLAR_TARGET'
        with pytest.raises(ValueError):m.compare_literal(rec,ref,parallel=False)
        return
    if fault=='dimension':
        with pytest.raises(ValueError):m.compare_literal(fixed.identity(4),ref,parallel=False)
        return
    v=m.compare_literal(rec,ref,parallel=False)
    assert F(v['lower_exact'])>F(1,10**8)


def test_actual_inverse_same_realized_circuit_no_new_word_and_shell(example):
    from trottertracks.c1_stage2_native import actual_adjoint
    from qiskit.quantum_info import Operator
    *_,qc,rec,value=example
    inv=actual_adjoint(qc)
    v=fixed.compare(fixed.record(inv),fixed.adjoint(rec),parallel=False)
    assert F(v['upper_exact'])<F(1,10**12)
    assert np.linalg.norm(Operator(inv).data@Operator(qc).data-np.eye(32))<1e-12
    from trottertracks.c1_stage3_connection import readout_segments
    p,t=readout_segments(4,'Y');t.remove_final_measurements()
    shell=fixed.compare(fixed.record(p),fixed.record(p),parallel=False)
    assert shell['dimension']==32 and F(shell['upper_exact'])<F(1,10**12)


def test_full512_artificial_identity_all_columns():
    # Invented zero heads and angle0, identity U, one signed CAR atom. Not H4.
    from trotterlib.rte import RTEEvent
    atoms=(NS(atom_id='toy:F2:z',frame_id='toy:F2',support=(0,),coefficient=F(-1)),)
    arm=NS(J=np.zeros((3,4,4)),U=np.array([np.eye(4)]*3),
        model=NS(modes=8,label='toy',source_identity='invented_full512'),
        plan=NS(ld=2,atoms=atoms,frames={'toy:F2':np.eye(8)},extracted_identity=.125))
    e=NS(rotation_component_id='toy:F2:z',taylor_order=0,phase=1,product_component_ids=(),rotation_angle=0.)
    heads=[np.zeros((256,256)) for _ in range(3)]
    ref=m.build_literal_reference(arm,heads,(e,)*32,parallel=False)
    # Include the exact controlled scalar phase from h*identity, no phase fit.
    rec=fixed.identity(9);rec['gates']=[dict(name='p',qubits=[8],params_hex=[(-.0125).hex()])]
    v=m.compare_literal(rec,ref,parallel=False)
    assert v['dimension']==v['columns_evaluated']==512
    assert len({i for block in ref['blocks'] for i in block['indices']})==256
    assert F(v['upper_exact'])<F(1,10**14)


def test_population_exact_unitary_range2_without_sample_max_and_nonunitary_range_greater():
    arm,heads,_,catalog,table,_=packet(False)
    p=m.population_norm(arm,heads,catalog,table,expected_ids=list(catalog),r=2)
    assert F(p['D_norm_upper_exact'])==1 and F(p['d_range_upper_exact'])==2
    assert not p['all_q_words_compile_success_qualified'] and not p['calibration_ready']
    arm,heads,_,catalog,table,_=packet(True)
    p=m.population_norm(arm,heads,catalog,table,expected_ids=list(catalog),r=2)
    assert F(p['D_norm_upper_exact'])>1 and F(p['d_range_upper_exact'])>2
    assert all(F(row['D_product_norm']['upper_exact'])>=1 for row in p['round_growth'])
    assert p['all_word_synthesis_error_bound'] is None


def test_exact_density_CAR_event_norm_covers_all_artificial_catalog():
    arm,heads,_,catalog,table,_=packet(True)
    full=b.stage1.car_bilinears(4)
    ctx=NS(d=16,bilinears={(p,q):NS(toarray=lambda p=p,q=q:full[p,q]) for p,q in []})
    from scipy.sparse import csr_matrix
    ctx.bilinears={(p,q):csr_matrix(full[p,q]) for p in range(4) for q in range(4) if p%2==q%2}
    ops=b.atom_operators(ctx,arm)
    from trotterlib.rte import event_unitary
    p=m.population_norm(arm,heads,catalog,table,expected_ids=list(catalog),r=2)
    for key,e in catalog.items():
        assert np.linalg.norm(event_unitary(e,ops),2)<=float(F(p['events'][key]['event_norm_upper_exact']))+1e-14


@pytest.mark.parametrize('fault',['missing','negative','float_counts','wrong_bits','wrong_event_key','duplicate_id'])
def test_population_support_missing_or_invalid_refused(fault):
    arm,heads,_,catalog,table,_=packet()
    ids=list(catalog)
    if fault=='missing':catalog.pop(ids[-1])
    if fault=='negative':table['counts'][0]=-1
    if fault=='float_counts':table['counts'][0]=1.
    if fault=='wrong_bits':table['bits']=52
    if fault=='wrong_event_key':catalog[ids[0]]=catalog[ids[1]]
    if fault=='duplicate_id':ids[-1]=ids[0]
    with pytest.raises(ValueError):m.population_norm(arm,heads,catalog,table,expected_ids=ids,r=2)


def test_exact_zero_support_only_no_probability_threshold():
    arm,_,_,catalog,table,_=packet()
    ids=list(catalog);table['counts']=[0]*(len(ids)-2)+[1,(1<<53)-1]
    s=m.grid_support(catalog,table,expected_ids=ids)
    assert s['positive']==ids[-2:] and s['counts_by_id'][ids[-2]]==1


def test_nonhermitian_heads_are_retained_bounded_and_noninvariant_fails():
    arm,heads,events,catalog,table,_=packet()
    # Non-Hermitian entry within one spin sector (basis columns1 and4).
    bad=[G.copy() for G in heads];bad[0][1,4]+=.125
    p=m.population_norm(arm,bad,catalog,table,expected_ids=list(catalog),r=2)
    assert not p['heads'][0]['exact_Hermitian'] and F(p['D_norm_upper_exact'])>1
    bad[0][0,1]=.125
    with pytest.raises(ValueError,match='SPIN_BLOCK'):m.validate(arm,bad,events,h=F(1,10),r=2)


def test_large_growth_serialization_and_no_implicit_cap():
    x=m.power_upper('1000000',4096)
    assert x['upper_exact'] is None and not x['clipped']
    assert x['upper_dyadic']['exponent2']>10000


def test_calibration_requires_common_target_and_total_domain_premise(example):
    arm,heads,events,catalog,table,_,_,_,_,v=example
    p=m.population_norm(arm,heads,catalog,table,expected_ids=list(catalog),r=2)
    assert m.calibration_inputs([v],p)['support'] is None
    theorem=dict(kind='structural_totality_argument',scope='all_legal_q_words',gates='exact_ideal_P0_unitaries',provenance='invented total finite compiler')
    inputs=m.calibration_inputs([v],p,compiler_domain=theorem)
    assert inputs['status']=='CONDITIONAL_CALIBRATION_INPUTS'
    v=deepcopy(v);v['binding']['target_id']='wrong'
    with pytest.raises(ValueError):m.calibration_inputs([v],p,compiler_domain=theorem)


def test_tail_zero_bad_straddle_bad_unknown_and_fresh():
    p=policies(20)
    good=[interval('0','1/10000')]*20
    r=m.threshold_tail(good,support=support(),threshold='1/100',delta='1/20',policy=p)
    assert r['bad']==r['straddle']==0 and F(r['tail_probability_upper'])>0
    rows=list(good);rows[0]=interval('0','1/50');rows[1]=interval('1/50','1/40')
    r=m.threshold_tail(rows,support=support(),threshold='1/100',delta='1/20',policy=p)
    assert r['bad']==r['straddle']==1
    rows[0]=dict(rows[0],upper_exact=None)
    assert m.threshold_tail(rows,support=support(),threshold='1/100',delta='1/20',policy=p)['bound'] is None
    p['origin']='old_native32'
    with pytest.raises(ValueError):m.threshold_tail(good,support=support(),threshold='1/100',delta='1/20',policy=p)


def test_empirical_Bernstein_includes_nonzero_enclosure_radius_and_no_small_variance_assumption():
    p=policies(20)
    exact=[interval('1/10','1/10')]*20
    uncertain=[interval('0','1/5')]*20
    a=m.variance_sensitive(exact,support=support(),delta='1/20',policy=p)
    b=m.variance_sensitive(uncertain,support=support(),delta='1/20',policy=p)
    assert F(a['sample_variance_upper_exact'])==0
    assert F(b['sample_variance_upper_exact'])>0
    with pytest.raises(ValueError):m.variance_sensitive(exact[:1],support=support(),delta='1/20',policy=policies(1))
    assert m.variance_sensitive([dict(x,kind='observed_residual') for x in exact],support=support(),delta='1/20',policy=p)['bound'] is None


def test_large_nonunitary_literal_has_no_cap2():
    arm,heads,events,_,_,_=packet()
    # Only the reference changes in this adversarial artificial problem.
    U=arm.U.copy();U[2]=2*np.eye(2)
    altered=b.prepare_arm(np.zeros((2,2)),np.zeros((2,)*4),0.,arm.J,U,role='proposed',
        source_identity='invented_large_nonunitary',input_identity='toy',ld=2)
    atom=next(a for a in altered.plan.atoms if a.frame_id.endswith('F2') and len(a.support)==1)
    e=NS(rotation_component_id=atom.atom_id,taylor_order=0,phase=1,product_component_ids=(),rotation_angle=math.pi/2)
    ref=m.build_literal_reference(altered,[np.zeros((16,16)) for _ in range(3)],(e,),r=1,parallel=False)
    v=m.compare_literal(fixed.identity(5),ref,parallel=False)
    assert F(v['lower_exact'])>2 and F(v['upper_exact'])>2


def test_invalid_M2_and_sample_max_population_rejected(example):
    arm,heads,_,catalog,table,*_=example
    p=m.population_norm(arm,heads,catalog,table,expected_ids=list(catalog),r=2)
    assert F(p['D_norm_upper_exact'])>1
    p['d_range_upper_exact']='2'
    with pytest.raises(ValueError,match='RANGE_DERIVATION'):m.calibration_inputs([],p)
    p['d_range_upper_exact']=str(1+F(p['D_norm_upper_exact']));p['no_sample_maximum_used']=False
    with pytest.raises(ValueError,match='RANGE_DERIVATION'):m.calibration_inputs([],p)


def test_variance_interval_projection_upper_covers_all_endpoint_variances():
    from itertools import product
    rows=[interval('0','1/5'),interval('2/5','3/5'),interval('3/10','7/10')]
    v=m.variance_sensitive(rows,support=support('1'),delta='1/20',policy=policies(3))
    for x in product(*[(F(row['lower_exact']),F(row['upper_exact'])) for row in rows]):
        avg=sum(x,F())/3;var=sum(((z-avg)**2 for z in x),F())/2
        assert var<=F(v['sample_variance_upper_exact'])


def test_arb_resource_strategy_no_artificial_caps_and_passive_import():
    import subprocess,sys
    from trottertracks.c_pr_aware_df_fixed_certificate import resource_plan
    p=resource_plan(512)
    assert p['workers']>=1 and p['CPU_cap'] is p['RAM_cap'] is p['wall_cap'] is None
    code="import sys;from trottertracks import c_per_word_population_range,c_per_word_range_execution;assert not any(x in sys.modules for x in ('numpy','scipy','flint','qiskit','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],check=True)


@pytest.mark.parametrize('failed',[False,True])
def test_stage_R_mock_calls_partial_failure_and_mandatory_STOP(tmp_path,monkeypatch,failed):
    from trottertracks import c_per_word_range_execution as execution
    import json
    c=dict(source_manifest='SOURCE.json',stages=dict(R=dict(fresh_output='run01')))
    (tmp_path/'contract.json').write_text(json.dumps(c));(tmp_path/'SOURCE.json').write_text('{}')
    monkeypatch.setattr(execution,'preflight',lambda *a:dict(scientific_calls=0))
    def run(root,c,out,call):
        if failed:
            return call('head_norm','invented correctness failure',lambda:(_ for _ in ()).throw(ValueError('TEST_CORRECTNESS')))
        return call('range_summary','invented conditional range only',lambda:dict(status='ARTIFICIAL_RANGE_ONLY'))
    with pytest.raises(PermissionError):execution.run(tmp_path,tmp_path/'contract.json')
    if failed:
        with pytest.raises(ValueError,match='TEST_CORRECTNESS'):
            execution.run(tmp_path,tmp_path/'contract.json',execute_science=True,executor=run)
    else:
        r=execution.run(tmp_path,tmp_path/'contract.json',execute_science=True,executor=run)
        assert r['status']=='ARTIFICIAL_RANGE_ONLY'
    stop=json.loads((tmp_path/'run01/STOP.json').read_text())
    assert bool(stop['failed'])==failed and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert not stop['Stage_P_executed'] and not stop['Stage_C_executed']
    with pytest.raises(FileExistsError):execution.run(tmp_path,tmp_path/'contract.json',execute_science=True,executor=run)


def test_compact_Arb_expected_propagation_agrees_with_old_formula_and_missing_shells():
    shells={k:'1/1000000' for k in ('head','X','Y')}
    rows=m.propagate_calibrated_mean('1/100000',dict(D_norm_upper_exact='100001/100000'),shells=shells)
    for row in rows[:4]:
        direct=old.propagate_expected('1/100000',norm_upper='100001/100000',shells=shells,m=row['m'])
        assert float(F(row['bound']))==pytest.approx(float(F(direct['bound'])),rel=1e-14)
    assert len(rows)==13 and rows[-1]['bound'] is not None
    assert all(x['bound'] is None for x in m.propagate_calibrated_mean(None,{},shells=shells))
    shells['Y']=None
    assert all(x['bound'] is None for x in m.propagate_calibrated_mean('1/10',dict(D_norm_upper_exact='1'),shells=shells))


def test_missing_per_word_numeric_interval_not_promoted_by_confidence(example):
    arm,heads,_,catalog,table,_,_,_,_,v=example
    p=m.population_norm(arm,heads,catalog,table,expected_ids=list(catalog),r=2)
    theorem=dict(kind='structural_totality_argument',scope='all_legal_q_words',gates='exact_ideal_P0_unitaries',provenance='invented')
    v=deepcopy(v);v['kind']='observed_residual'
    assert m.calibration_inputs([v],p,compiler_domain=theorem)['samples'] is None


@pytest.mark.parametrize('fault',['none','source','input','fresh','retry','resume','model'])
def test_stage_R_passive_binding_fresh_retry_scope(tmp_path,monkeypatch,fault):
    from trottertracks import c_per_word_range_execution as e
    import json,hashlib
    (tmp_path/'code').write_text('invented');(tmp_path/'input').write_bytes(b'invented metadata')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'SOURCE.json').write_text(json.dumps(dict(files={'code':sha(tmp_path/'code')})))
    c=dict(base_commit=m.BASE,target=m.TARGET,precision_bits=192,retry=0,resume=0,mandatory_stop=True,
        source_manifest='SOURCE.json',source_manifest_sha256=sha(tmp_path/'SOURCE.json'),
        saved_inputs={'input':sha(tmp_path/'input')},packages={},identity=e.prior.integration.previous.identity(),
        stages=dict(R=dict(scope=e.SCOPE,executable=True,call_limits=e.LIMITS,maximum_logical_units=e.MAX_CALLS,fresh_output='R'),
            P=dict(executable=False,fresh_output='P'),C=dict(executable=False),A=dict(executable=False)))
    if fault in ('source','input'):(tmp_path/('code' if fault=='source' else 'input')).write_text('changed')
    if fault=='fresh':(tmp_path/'R').mkdir()
    if fault in ('retry','resume'):c[fault]=1
    if fault=='model':c['identity']['model']['id']='ACTUAL_PCG64'
    (tmp_path/'contract').write_text(json.dumps(c))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('NO_REAL_ARRAY_DECODE'))
    if fault=='none':assert e.preflight(tmp_path,tmp_path/'contract')['scientific_calls']==0
    else:
        with pytest.raises(ValueError):e.preflight(tmp_path,tmp_path/'contract')


def test_real_contract_passive_cold_without_numerical_imports():
    from pathlib import Path
    import subprocess,sys
    # Source/contract will be frozen before this test enters the release suite.
    code="""
import sys
from pathlib import Path
class Block:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','flint','ffsim','pyscf','openfermion'):
   raise AssertionError('NO_SCIENTIFIC_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
from trottertracks import c_per_word_range_execution as m
r=m.preflight(Path.cwd(),m.method.CONTRACT)
assert r['scientific_calls']==r['arrays_decoded']==0 and r['Stage_C_ready'] is False
assert r['Stage_R_maximum_logical_units']==256
"""
    subprocess.run([sys.executable,'-B','-c',code],check=True)


def test_artificial_full_Stage_R_pipeline_exact_256_calls_no_native_or_sampling(tmp_path,monkeypatch):
    from trottertracks import c_per_word_range_execution as execution
    from trottertracks.c1_stage3_integrated import Calls
    from trottertracks.c_iid_grid_bridge import MODEL
    import json
    J=np.full((8,4,4),.0009765625);U=np.array([np.eye(4)]*8)
    p=dict(h=np.diag([.125,.25,.375,.5]),g=np.zeros((4,)*4),constant=np.asarray(.125))
    np.savez(tmp_path/'input.npz',**p);np.savez(tmp_path/'candidate.npz',J=J,U=U)
    digest=execution.io.sha(tmp_path/'candidate.npz')
    arm=b.prepare_arm(**p,J=J,U=U,role='proposed',source_identity=digest,input_identity='invented',ld=2)
    atoms=[dict(atom_id=a.atom_id,frame_id=a.frame_id,support=list(a.support),coefficient=str(F(a.coefficient))) for a in arm.plan.atoms]
    assert len(atoms)==216
    counts=[(1<<53)//216]*216;counts[-1]+=(1<<53)-sum(counts)
    for name,data in [('atoms',dict(atoms=atoms)),('mean',dict(model=MODEL,report=dict(angle=.015625))),
            ('grid',dict(grid=dict(counts=counts,denominator=1<<53,bits=53,mapping='searchsorted_right')))]:
        (tmp_path/(name+'.json')).write_text(json.dumps(data))
    np.savez(tmp_path/'sector.npz',generators=np.zeros((9,36,36)))
    np.savez(tmp_path/'basis.npz',sector_basis_indices=np.arange(36))
    c=dict(parent_roles=dict(input='input.npz',candidate='candidate.npz',sector='sector.npz',basis='basis.npz'),
        saved_inputs={'input.npz':execution.io.sha(tmp_path/'input.npz')},identity=dict(candidate=digest,model=MODEL),
        atoms='atoms.json',event_mean='mean.json',grid_table='grid.json')
    monkeypatch.setattr(execution.prior,'evidence',lambda *args:dict(status='INVENTED_SAVED_EVIDENCE'))
    monkeypatch.setattr(b,'build_kernel',lambda *a,**k:pytest.fail('NO_NATIVE_BUILD'))
    out=tmp_path/'output';out.mkdir();calls=Calls(out,execution.LIMITS,execution.MAX_CALLS)
    result=execution.execute_range(tmp_path,c,out,calls)
    assert sum(calls.counts.values())==256 and calls.counts==execution.LIMITS
    assert result['status']=='LITERAL_POPULATION_NORM_STAGE_R_COMPLETE'
    assert result['population']['d_range_upper_exact']=='2'
    assert result['head_binding']['additional_head_bridge_required'] is True
    assert result['new_sampling']==result['native_build_compile']==0 and not result['Stage_C_ready']
