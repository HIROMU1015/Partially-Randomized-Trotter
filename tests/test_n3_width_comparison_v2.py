"""Synthetic only. Never reads the registered science pack or historical ledger."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import pytest
import numpy as np

from trottertracks.representation_exploration.n3_width_contract_v2 import (
    METHODS,BLOCKS,SOS_LEVEL,CAPS,active_data,decode_case,enc,digest,
)
from trottertracks.representation_exploration import n3_width_algebra_v2 as car
from trottertracks.representation_exploration.n3_width_generator_v2 import (
    Engine,acquire_bounds,acquire_mode,spectral_certificate,transport_witness,
)
from trottertracks.representation_exploration.n3_width_check_v2 import (
    Checks,action,spectral,sos_lower,verify_witness,verify_sdp_refutation,one_body_sector,sector_states,
)
from trottertracks.representation_exploration.n3_width_pipeline_v2 import select_modes,summarize_block,full_h_case


@pytest.fixture(autouse=True)
def reject_registered_science_input_reads(monkeypatch):
    from pathlib import Path
    original=Path.open
    def guarded(path,*args,**kwargs):
        if 'artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_pack.json' in path.as_posix():
            raise AssertionError('Synthetic tests must not read the registered science inputs')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)


def fixture(nu=2):
    return enc({'id':'synthetic_only','family':'test','seed':0,'modes':4,'nu':nu,
        'h':[[F(-3,4),F(1,32),F(0),F(1,10)],
             [F(1,32),F(-1,2),F(1,32),F(1,12)],
             [F(0),F(1,32),F(1,4),F(1,16)],
             [F(1,10),F(1,12),F(1,16),F(3)]],
        'density':[[0,1,F(1,5)],[0,2,F(1,7)],[0,3,F(1,8)],[1,2,F(1,9)],[1,3,F(1,10)]],
        'candidate_modes':[3],'epsilon_proj':F(1,20)})


def operator(word,states):
    ix={b:i for i,b in enumerate(states)};M=np.zeros((len(states),len(states)))
    for j,b in enumerate(states):
        w=tuple((x>0,abs(x)-1) for x in word);ans=action(w,b)
        if ans is not None and ans[0] in ix:M[ix[ans[0]],j]=ans[1]
    return M


def test_all_short_CAR_words_against_independent_ladder_actions():
    labels=(1,2,3,-1,-2,-3);states=list(range(8))
    for degree in range(5):
        for w in product(labels,repeat=degree):
            rebuilt=sum((c*operator(z,states) for z,c in car.normal(w)),np.zeros((8,8)))
            assert np.array_equal(operator(w,states),rebuilt)


def test_sos_contains_OCC_exactly_including_number_term():
    m,nu=3,2;k=F(1,20);tau=F(-1,10);A=[[F(-2),F(0),F(0)],[F(0),F(0),F(0)],[F(0),F(0),F(10)]]
    v=[F(2),F(0),F(1)];J={(0,1):F(2),(0,2):F(1,3),(1,2):F(1,7)}
    Z=[[x*y for y in v] for x in v];R=[[k*A[i][j]-(tau if i==j else 0) for j in range(m)] for i in range(m)]
    blocks,mult,keys=car.structure(m,nu);grams={name:[[F(0) for _ in range(b['size'])] for _ in range(b['size'])] for name,b in blocks.items()}
    grams['particle']=R;grams['hole']=Z
    for i,pair in enumerate(sorted(J)):grams['D'][i][i]=k*J[pair]
    s=nu*tau-sum(Z[i][i] for i in range(m));eta=[tau]+[F(0)]*(len(mult)-1)
    target=car.add(car.scale(car.add(car.one_body(A),car.density(J)),k),
                   car.scale(car.one_body(Z),-1),{():-s})
    rhs=car.add(*(car.assemble_gram(blocks[name],grams[name]) for name in BLOCKS),
                *(car.scale(p,x) for p,x in zip(mult,eta)))
    assert car.add(target,car.scale(rhs,-1))=={}
    params=enc({'level':SOS_LEVEL,'grams':grams,'multipliers':eta,'ell':s,'residual_bound':F(0),'s':s})
    assert sos_lower(A,v,J,nu,k,params,Checks())==s
    bad=deepcopy(params);bad['grams']['D'][0][0]=[-1,1]
    with pytest.raises(ValueError):sos_lower(A,v,J,nu,k,bad,Checks())


@pytest.mark.parametrize('nu',[1,2,3])
def test_spectral_certificate_nu_one_degeneracy_and_full_occupation(nu):
    A=[[F(1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(2)]]
    w=spectral_certificate(A,nu,Engine());s=spectral(A,nu,w,Checks())
    assert s<=sum([F(1),F(1),F(2)][:nu])


@pytest.mark.parametrize('method',METHODS)
def test_real_solver_synthetic_certificates_sound(method):
    case=fixture();r=acquire_mode(case,3,method)
    assert 'STOP' not in r['status'],r['status']
    assert r['candidates'],r
    if method!='FULL_H_LOWER':
        for w in r['candidates']:verify_witness(case,3,r['bounds'],w)
    # Synthetic dense-sector reference checks the physical bound; never used in selection.
    h,J,nu,modes,eps=decode_case(case);states=sector_states(4,nu);H=one_body_sector(h,states,Checks())
    for i,b in enumerate(states):H[i][i]+=sum((x for (p,q),x in J.items() if b&(1<<p) and b&(1<<q)),F(0))
    active,A,v,den,E,hdd,nu,eps=active_data(case,3);sA=sector_states(3,nu);small=one_body_sector(A,sA,Checks())
    for i,b in enumerate(sA):small[i][i]+=sum((x for (p,q),x in den.items() if b&(1<<p) and b&(1<<q)),F(0))
    truth=float(np.linalg.eigvalsh(np.array(small,float))[0]-np.linalg.eigvalsh(np.array(H,float))[0])
    assert truth>=-1e-12
    for w in r['candidates']:assert truth<=float(F(*w['width']['positive_proof_delta']))+1e-12


def test_exact_SDP_refutation_strictness_and_particle_count():
    A=[[F(-1),F(0)],[F(0),F(1)]];v=[F(1),F(0)];G=[[F(1),F(0)],[F(0),F(0)]]
    w=enc({'gamma':G,'t':F(0),'objective':F(1)})
    assert verify_sdp_refutation(A,v,1,F(0),F(1),F(1,10),w)['physical_impossibility_claimed'] is False
    for key,value in [('gamma',enc([[F(2),F(0)],[F(0),F(-1)]])),('t',[-1,1]),('objective',[11,100])]:
        bad=deepcopy(w);bad[key]=value
        with pytest.raises(ValueError):verify_sdp_refutation(A,v,1,F(0),F(1),F(1,10),bad)
    full=enc({'gamma':[[F(1),F(0)],[F(0),F(1)]],'t':F(0),'objective':F(1)})
    verify_sdp_refutation(A,v,2,F(0),F(1),F(1,10),full)


@pytest.mark.parametrize('key,value',[('nu',True),('truth_gap',[1,1]),('candidate_modes',[1,1]),('epsilon_proj',[0,1]),('density',[[0,1,[-1,1]]])])
def test_invalid_case_rejected(key,value):
    case=fixture();case[key]=value
    with pytest.raises(ValueError):decode_case(case)


def test_tampered_certificate_bounds_and_binding_rejected():
    case=fixture();r=acquire_mode(case,3,'NORM');w=r['candidates'][0]
    for key,value in [('case_sha256','0'*64),('external',0),('s',[0,1]),('kappa',[1,1])]:
        bad=deepcopy(w);bad[key]=value
        with pytest.raises(ValueError):verify_witness(case,3,r['bounds'],bad)
    bad=deepcopy(r['bounds']);bad['c']=[100,1]
    with pytest.raises(ValueError):verify_witness(case,3,bad,w)
    bad=deepcopy(w);bad['width']['high']=[0,1]
    with pytest.raises(ValueError):verify_witness(case,3,r['bounds'],bad)


def test_H_endpoint_not_number_of_mode_candidates_and_fallback():
    case=fixture();case['candidate_modes']=[2,3]
    def row(d,width):return {'external':d,'status':'synthetic','candidates':[{'width':{'low':enc(width),'high':enc(width),'positive_proof_delta':enc(width)}}]}
    outcome=select_modes(case,[row(2,F(1,100)),row(3,F(1,200))])
    assert outcome['modes_removed']==1 and outcome['selected_external']==3
    assert outcome['reduced_H']['density']  # preserve interaction, not just the one-body matrix
    fallback=select_modes(case,[{'external':d,'status':'UNDETERMINED','candidates':[]} for d in (2,3)])
    assert fallback['modes_removed']==0 and not fallback['H_any_mode_reduction']


def test_full_lower_shared_once_and_no_c_required():
    case=fixture();case['candidate_modes']=[2,3];r=full_h_case(case)
    assert r['resources']['calls']['eigh']==1 and r['full_lower_acquired_once_per_H']


def test_cache_exact_identity_and_fresh_engine():
    A=[[F(1),F(0)],[F(0),F(2)]];e=Engine();e.eig(A);e.eig(A)
    assert e.counts['eigh']==1 and e.counts['eigh_cache_hits']==1
    changed=deepcopy(A);changed[0][0]+=F(1,100);e.eig(changed);assert e.counts['eigh']==2
    other=Engine();other.eig(A);assert other.counts['eigh_cache_hits']==0


def test_failed_finite_search_never_becomes_mathematical_refutation(monkeypatch):
    import trottertracks.representation_exploration.n3_width_generator_v2 as g
    monkeypatch.setattr(g,'sdp_candidates',lambda *args:([],[]));r=g.acquire_mode(fixture(),3,'SDP_EQUIVALENT')
    assert r['status']=='UNDETERMINED_FINITE_PROPOSALS' and not r['physical_impossibility_claimed']


def test_bit_and_operation_caps(monkeypatch):
    from trottertracks.representation_exploration.n3_width_contract_v2 import rat
    with pytest.raises(ArithmeticError):rat([1<<5000,1])
    monkeypatch.setitem(CAPS,'checker_operations',1);a=Checks();a.q(1)
    with pytest.raises(ArithmeticError):a.q(2)


def test_saved_witness_transport_keeps_parameters_at_stronger_c():
    case=fixture();e=Engine();strong=acquire_bounds(case,3,e);weak=deepcopy(strong)
    m=3;r=F(1,100);Z=weak['external']['Z']
    for i in range(m):Z[i][i]=enc(F(*Z[i][i])+r)
    weak['external']['s']=enc(F(*weak['external']['s'])-m*r);weak['c']=enc(F(*weak['c'])-m*r)
    rec=acquire_mode(case,3,'OCC',weak);w=rec['candidates'][rec['best_index']]
    new=transport_witness(case,3,weak,strong,w)
    assert new['parameters']==w['parameters'] and new['kappa']==w['kappa']
    assert F(*new['width']['high'])<=F(*w['width']['high'])
    with pytest.raises(ValueError):transport_witness(case,3,strong,weak,new)


def test_maximum_dimension_density_SOS_against_independent_checker():
    case=fixture(nu=3);n=6
    h=[[F(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):h[i][i]=F(i-4,5)
    h[5][5]=F(4);h[0][5]=h[5][0]=F(1,40);h[3][5]=h[5][3]=F(1,50)
    case.update(modes=n,h=enc(h),density=enc([[i,j,F(1,100)] for i in range(5) for j in range(i+1,5)]),candidate_modes=[5])
    r=acquire_mode(case,5,'DENSITY_SOS')
    assert r['status']=='CERTIFICATES_FOUND',r['status']
    for w in r['candidates']:verify_witness(case,5,r['bounds'],w)


def test_authorization_scope_identity_one_shot_and_false_template():
    from trottertracks.representation_exploration.n3_width_execution_v2 import validate_authorization,SCOPE,EXECUTION_BRANCH,OUTPUT_PATH
    plan={'source_commit':'a'*40,'source_fingerprint':'b'*64,'input_sha256':'c'*64}
    auth={'scope':SCOPE,'science_execution_authorized':True,**plan,'plan_sha256':'d'*64,
        'execution_branch':EXECUTION_BRANCH,'output_path':OUTPUT_PATH,'attempts':1,'retries':0,'mandatory_stop':True}
    validate_authorization(auth,plan,'d'*64)
    for key,value in [('science_execution_authorized',False),('scope','old_consumed_authorization'),('attempts',True),
                      ('retries',1),('input_sha256','0'*64),('execution_branch','root'),('output_path','other'),('mandatory_stop',False)]:
        bad=deepcopy(auth);bad[key]=value
        with pytest.raises(ValueError):validate_authorization(bad,plan,'d'*64)


def test_no_overwrite_and_output_cap(tmp_path,monkeypatch):
    from trottertracks.representation_exploration.n3_width_execution_v2 import Output,RESOURCES
    o=Output(tmp_path);o.write('one.json',{'x':1})
    with pytest.raises(FileExistsError):o.write('one.json',{'x':2})
    with pytest.raises(ValueError):o.write('../escape.json',{'x':2})
    monkeypatch.setitem(RESOURCES,'output_shard_bytes',8)
    with pytest.raises(ArithmeticError):o.write('large.json',{'long':'value'})


def test_saved_whole_pipeline_aggregation_and_adversarial_selection():
    import importlib.util
    from pathlib import Path
    from trottertracks.representation_exploration.n3_width_pipeline_v2 import acquire_case
    p=Path(__file__).resolve().parents[1]/'scripts/verify_n3_width_comparison_v2.py'
    spec=importlib.util.spec_from_file_location('saved_v2_checker',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=fixture();record=acquire_case(case);audit=module.verify_record(case,record)
    assert record['status']=='FINITE_COMPARISON_COMPLETE' and audit['certificates']>0
    bad=deepcopy(record);bad['blocks'][0]['endpoints']['OCC']['modes_removed']=2
    with pytest.raises(ValueError):module.verify_record(case,bad)
    bad=deepcopy(record);bad['blocks'][0]['endpoints']['OCC_OPERATIONAL_OR']['fallback_is_OCC_specific_gain']=True
    with pytest.raises(ValueError):module.verify_record(case,bad)
    with pytest.raises(ValueError):module.verify_record(case,record,'a'*64)


def script_module(name):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_preflight_is_identity_only_and_detects_mutation(tmp_path,monkeypatch):
    import json
    from trottertracks.representation_exploration.n3_width_execution_v2 import preflight,SCOPE,RESOURCES,file_sha
    monkeypatch.setattr(np.linalg,'eigh',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No eig in preflight')))
    case=fixture();pack={'schema_version':2,'recipe':{},'cases':[case]};path=tmp_path/'fixture.json';path.write_text(json.dumps(pack))
    plan={'scope':SCOPE,'methods':list(METHODS),'caps':CAPS,'resources':RESOURCES,'source_sha256':{},'source_fingerprint':digest({}),
          'environment_packages':{},'input_path':'fixture.json','input_sha256':file_sha(path),'input_ids':[case['id']],
          'case_sha256':{case['id']:digest(case)}}
    result=preflight(tmp_path,plan);assert result['new_solver_calls']==0 and result['new_eigensolver_calls']==0
    path.write_text(json.dumps({'mutated':True}))
    with pytest.raises(ValueError):preflight(tmp_path,plan)


def test_stop_record_reserve_remains_writable_after_data_budget(tmp_path,monkeypatch):
    from trottertracks.representation_exploration.n3_width_execution_v2 import Output,RESOURCES
    o=Output(tmp_path);limit=RESOURCES['total_output_bytes']-RESOURCES['failure_bytes']-RESOURCES['log_bytes'];o.total=limit
    with pytest.raises(ArithmeticError):o.write('extra.json',{'x':1})
    o.write('result.json',{'status':'TECHNICAL_STOP'},stop_record=True)
    o.write('MANDATORY_STOP.json',{'mandatory_stop':True},stop_record=True)


def test_parallel_synthetic_path_matches_serial_certificates():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from trottertracks.representation_exploration.n3_width_pipeline_v2 import acquire_case
    case=fixture();serial=acquire_case(case)
    # Use the production worker wrapper and actual spawn/resource settings, on synthetic literals.
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_n3_width_comparison_v2 import worker
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=[f.result() for f in [pool.submit(worker,case),pool.submit(worker,case)]]
    def scientific_view(r):
        return [[(name,[row['candidates'] for row in path['mode_records']]) for name,path in b['methods'].items()] for b in r['blocks']]
    assert all(r['status']=='FINITE_COMPARISON_COMPLETE' and scientific_view(r)==scientific_view(serial) for r in records)
