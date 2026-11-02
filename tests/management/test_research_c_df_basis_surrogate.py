"""Invented J/U, events and metrics only; no saved-H4 features/statistics."""
from pathlib import Path
from fractions import Fraction
from itertools import product
from copy import deepcopy
import sys,json,subprocess,math
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_basis_surrogate as m
from trottertracks import c_df_basis_reanalysis as run


def fixture():
    from trottertracks.pr_dictionary_validation.prototype import MixedAtom,make_tail
    from trotterlib.rte import finite_rte_distribution,_make_event
    atoms=(MixedAtom('a','FRAME_Z',Fraction(1),('F2',),('Z','I'),'A','F2'),
        MixedAtom('b','FRAME_Z',Fraction(-1),('F3',),('Z','Z'),'B','F3'),
        MixedAtom('c','FRAME_Z',Fraction(2),('F4',),('I','Z'),'A','F4'))
    tail=make_tail(atoms,'ARTIFICIAL');dist=finite_rte_distribution(.01,0)
    def payload(indices):
        events=[_make_event((i,),tail.components,dist,0).to_dict() for i in indices]
        return json.loads(json.dumps(dict(events=events)))
    distances={(a.frame_id,b.frame_id):float(a.basis_key!=b.basis_key) for a in atoms for b in atoms}
    return atoms,payload,distances


def test_passive():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_df_basis_reanalysis as r;assert r.MAX_CALLS==148;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_IID_exact_enumeration_grouped_basis_not_frame():
    atoms,make,dist=fixture();f=m.law_features(atoms,r=3)
    assert f['q_basis']=={'A':.75,'B':.25};assert f['P_same']==.625
    totals=[0.,0.]
    probabilities=(.25,.25,.5)
    for indices in product(range(3),repeat=3):
        v=m.trajectory_features(make(indices),atoms,dist,event_count=3);p=math.prod(probabilities[i] for i in indices)
        totals[0]+=p*v['basis_switches'];totals[1]+=p*v['run_count']
    assert totals==pytest.approx([f['expected_switches'],f['expected_runs']])
    assert f['raw_central_RZ']==6 and f['raw_central_CX']==7.5
    assert f['compiled_price'] is f['head_price'] is None


def test_K0_128_runs_Z_ZZ_signs_identity_and_order():
    atoms,make,dist=fixture();v=m.trajectory_features(make([0,2]*32+[1]*64),atoms,dist)
    assert v['basis_switches']==1 and v['frame_switches']==64
    assert v['run_lengths']==[64,64] and v['Z_count']==v['ZZ_count']==64
    assert v['negative_count']==64 and v['raw_central_RZ']==256 and v['raw_central_CX']==384
    assert v['signed_angle_sum']==pytest.approx(0.)
    with pytest.raises(ValueError,match='128'):m.trajectory_features(make([0]*127),atoms,dist)
    bad=make([0]*128);bad['events'][0]['application_sequence'][0]['coefficient_sign']=-1
    with pytest.raises(ValueError):m.trajectory_features(bad,atoms,dist)


def test_zero_tail_no_fictional_random_events():
    f=m.law_features(());assert f['status']=='NO_RANDOM_TAIL' and f['P_same'] is None
    assert f['expected_runs']==f['raw_central_RZ']==0
    from trottertracks.pr_dictionary_validation.prototype import MixedAtom
    a=MixedAtom('I','IDENTITY',Fraction(1),('phase',),('I','I'),None)
    with pytest.raises(ValueError,match='IDENTITY_EXTRACTED'):m.law_features((a,))


def test_overlap_range_gauge_and_general_J_tensor():
    from trottertracks.c1_h6_initialization import reconstruct
    a=np.array([[1.,0.],[0.,1.]]);b=np.array([[1.,1.],[-1.,1.]])/math.sqrt(2)
    assert m.overlap(a,b)==pytest.approx(1.) and m.overlap(a,a)==0
    assert m.overlap(a,b*np.array([-1,1]))==pytest.approx(1.)
    assert m.overlap(a[:,::-1],b)==pytest.approx(1.)
    J=np.array([[[-2.,.3],[.3,1.]]]);U=np.array([b]);perm=[1,0]
    np.testing.assert_allclose(reconstruct(J,U),reconstruct(J[:,perm][:,:,perm],U[:,:,perm]),atol=1e-14)
    np.testing.assert_allclose(reconstruct(J,U),reconstruct(J,U*np.array([-1,1])),atol=1e-14)
    import hashlib
    assert hashlib.sha256(repr(tuple(map(tuple,b))).encode()).digest()!=hashlib.sha256(repr(tuple(map(tuple,b*np.array([-1,1])))).encode()).digest()
    with pytest.raises(ValueError,match='NONORTHOGONAL'):m.overlap(a,2*a)
    with pytest.raises(ValueError,match='SHAPE'):m.overlap(a,np.eye(3))


def test_geometry_saved_key_roundtrip_and_frame_order():
    from trottertracks import c1_stage3_connection as conn
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks.pr_dictionary_validation.prototype import atom_record
    J=np.repeat(np.diag([-1.,2.,3.,4.])[None],8,axis=0);U=np.repeat(np.eye(4)[None],8,axis=0)
    model=b.prepare_arm(np.eye(4),np.zeros((4,)*4),.31,J,U,role='proposed',source_identity='INVENTED',input_identity='INVENTED',ld=2)
    atoms=m.atoms_from_records(json.loads(json.dumps([atom_record(a) for a in model.plan.atoms])))
    distances=m.geometry(atoms,U);assert len(distances)==36 and max(distances.values())==0
    assert m.law_features(atoms)['P_same']==pytest.approx(1.)
    U[2,:,0]*=-1
    with pytest.raises(ValueError,match='SAVED_BASIS_KEY'):m.geometry(atoms,U)


def rows32():
    rng=np.random.default_rng(31129);rows=[]
    for i in range(32):
        s,z,p=map(int,rng.integers(1,20,3));y=100+2*s+3*z-p
        rows.append(dict(sample=i,features=dict(basis_switches=s,ZZ_count=z,same_basis_support_pairs=p,overlap_path=float(s)+float(rng.normal())),metrics={k:y for k in run.METRICS}))
    return rows


def test_fixed_folds_composite_is_falsifiable_not_price_engine():
    x=m.describe_models(rows32(),'rz_count');assert x['C_composite']['MAE']<1e-10
    assert x['A_switch']['MAE']>0 and len(x['C_composite']['folds'])==4
    assert m.qualification({a:{'rz_count':x} for a in ('M4','P4')})['status']=='RETAIN_FOR_INDEPENDENT_DF_VALIDATION'
    assert not m.qualification({a:{'rz_count':x} for a in ('M4','P4')})['new_DF_fit_GO']


def test_constant_rank_deficient_missing_not_zero_filled():
    rows=rows32()
    for row in rows:row['features']['ZZ_count']=0
    x=m.describe_models(rows,'rz_count');assert x['C_composite']['MAE'] is None
    assert x['C_composite']['correlations']['ZZ_count']['Pearson'] is None
    assert m.qualification({a:{'rz_count':x} for a in ('M4','P4')})['status']=='SURROGATE_NOT_QUALIFIED'
    rows[0]['metrics']['rz_count']=None
    with pytest.raises(ValueError):m.describe_models(rows,'rz_count')


def test_existing_literal_DP_not_whole_P0():
    from trottertracks.pr_dictionary_validation.pricing import AtomPrice,expected_literal_price,direct_literal_price
    prices={a:AtomPrice(a,a,False,False,10,10,4,2) for a in ('A','B')}
    exact=expected_literal_price(prices,{'A':.5,'B':.5},{0:1.},q=1,r=2,fixed_price=0)
    enumerated=sum(direct_literal_price([(a,'rotation'),(b,'rotation')],prices,event_count=2,scalar_reserves=1)/4 for a,b in product(prices,repeat=2))
    assert exact[0]==pytest.approx(1.) and exact[1]==pytest.approx(enumerated)


def test_raw_census_no_compile_and_schema(tmp_path):
    from qiskit import QuantumCircuit,qpy
    q=QuantumCircuit(9);q.rz(.3,0);q.cx(8,0);q.metadata=dict(arm='artificial',source='JU')
    with (tmp_path/'raw.qpy').open('wb') as f:qpy.dump(q,f)
    r=run.census(tmp_path,dict(role='artificial',export_sha256='JU'),dict(raw_qpy='raw.qpy'))
    assert r['opcode_counts']==dict(rz=1,cx=1) and r['isolated_head_price'] is None


def test_mock_full148_and_partial_STOP(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'load_bundle',lambda *x:{})
    monkeypatch.setattr(run,'identity',lambda *x:dict(status='MOCK'))
    monkeypatch.setattr(run,'static',lambda *x:((),{},dict(static_sample_count=1)))
    rows=rows32()
    monkeypatch.setattr(run,'trajectory',lambda root,spec,row,*x:dict(rows[row['sample']],seed=row['sample']))
    monkeypatch.setattr(run,'census',lambda *x:dict(mock=True))
    c=dict(cohorts={k:dict(samples=[dict(sample=i) for i in range(32)]) for k in ('M4','P4')})
    out=tmp_path/'complete';out.mkdir();calls=run.Calls(out,run.LIMITS,run.MAX_CALLS)
    r=run.execute(tmp_path,c,out,calls);assert dict(calls.counts)==run.LIMITS and sum(calls.counts.values())==148
    assert r['DF_clusters']==2 and not r['qualification']['new_DF_fit_GO']
    run.save(tmp_path/'source.json',dict(files={}));c['source_manifest']='source.json'
    monkeypatch.setattr(run,'preflight',lambda *x:c);monkeypatch.setattr(run.subprocess,'check_output',lambda *x,**k:'MOCK\n')
    def fail(*x):raise ValueError('MOCK_INPUT_IDENTITY')
    monkeypatch.setattr(run,'identity',fail)
    with pytest.raises(ValueError,match='MOCK_INPUT_IDENTITY'):run.run(tmp_path,tmp_path/'contract.json')
    s=run.read(tmp_path/run.FRESH/'STOP.json');assert s['failed'] and s['mandatory_stop'] and s['retry']==s['resume']==0


def test_passive_hash_scope_and_fresh(tmp_path):
    run.save(tmp_path/'source.json',dict(files={}))
    (tmp_path/'opaque.npz').write_bytes(b'ARTIFICIAL_OPAQUE_NEVER_DECODED')
    c=dict(scope=run.SCOPE,fresh_output=run.FRESH,call_limits=run.LIMITS,maximum_logical_units=148,
        models={k:list(v) for k,v in m.MODELS.items()},folds='sample_index_mod4_within_each_cohort',DF_clusters=2,
        samples_per_cohort=32,event_count=128,retry=0,resume=0,new_DF_fit=0,sampling=0,new_build_compile=0,
        cohorts={name:dict(role=role,samples=[dict(sample=i) for i in range(32)]) for name,role in [('M4_Proposed','proposed'),('tail_P4','tail_p4')]},
        source_manifest='source.json',source_manifest_sha256=run.sha(tmp_path/'source.json'),saved_inputs={'opaque.npz':run.sha(tmp_path/'opaque.npz')},packages={})
    run.save(tmp_path/'contract.json',c);assert run.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/run.FRESH).mkdir(parents=True)
    with pytest.raises(ValueError,match='FRESH'):run.preflight(tmp_path,tmp_path/'contract.json')
    (tmp_path/'opaque.npz').write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='INPUT'):run.preflight(tmp_path,tmp_path/'contract.json')
