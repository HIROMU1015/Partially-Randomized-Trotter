"""Synthetic-only tests; registered saved numeric evaluation is not authorized."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys

import pytest

from trottertracks.representation_exploration import n3_saved_attribution_v1 as a
from trottertracks.representation_exploration import n3_saved_attribution_execution_v1 as gate


@pytest.fixture(autouse=True)
def refuse_registered_inputs(monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        if '/n3_width_comparison_v2/' in str(path) or path.name=='saved_input_registry.json':
            raise AssertionError('Registered saved inputs forbidden in tests')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)


def enc(x):return [F(x).numerator,F(x).denominator]


def candidate(k=F(0),q=F(1),kind='OCC'):
    # U0, c2, nu1; choose tau0,Z[q] so s=-q.
    params={'tau':[0,1],'Z':[[enc(q)]],'s':enc(-q)}
    if kind=='SOS':params={'ell':[0,1],'residual_bound':enc(q),'s':enc(-q)}
    return {'kind':kind,'kappa':enc(k),'s':enc(-q),'parameters':params,
            'width':{'low':[0,1],'high':[1,1],'positive_proof_delta':[1,1]}}


def mock_record():
    case={'id':'synthetic_only','nu':1};row={'external':0,'status':'CERTIFICATES_FOUND','bounds':{'upper':{'U':[0,1]},'c':[2,1]},
        'candidates':[candidate() for _ in range(3)],'refutations':[],'best_index':0,'wall_seconds':.5,'cpu_seconds':.4,
        'resources':{'phase_times':{'numerical_proposal':{'wall':.2,'cpu':.1}},'solver_events':[]}}
    methods={}
    for name in a.TARGETS:
        r=deepcopy(row);r['method']=name
        if name!='OCC':r['resources']['solver_events']=[{'status':'optimal','iterations':1,'reported_solve_time':.01,'reported_setup_time':.005} for _ in range(3)]
        methods[name]={'method':name,'mode_records':[r],'task_wall_including_selector_serialization':.6}
    record={'input_id':case['id'],'case_sha256':a.digest(case),'status':'FINITE_COMPARISON_COMPLETE','mandatory_stop':True,
            'blocks':[{'layer':'kernel','repetition':0,'methods':methods}]}
    return case,record


def test_metadata_does_not_evaluate_numbers_or_proofs(monkeypatch):
    case,record=mock_record()
    monkeypatch.setattr(a,'rational',lambda *args:(_ for _ in ()).throw(AssertionError('arithmetic')))
    out=a.metadata(case,record,'synthetic.json')
    assert out['new_numeric_posthoc_evaluations']==0 and out['new_PSD_certification_calls']==0
    assert out['candidates'][0]['repair_shift_r']=='NOT_SAVED'
    assert out['solver_events'][2]['tier_delta']==[1,400]


def test_metadata_owns_copies_and_preserves_pointers():
    case,record=mock_record();out=a.metadata(case,record,'synthetic.json');entry=out['candidates'][0]
    assert a.pointer(record,entry['pointer'])['kappa']==entry['kappa']
    entry['kappa'][0]=99
    assert a.pointer(record,entry['pointer'])['kappa']==[0,1]


@pytest.mark.parametrize('mutation',['identity','missing_tier','event_count','method'])
def test_metadata_refuses_ambiguous_bindings(mutation):
    case,r=mock_record()
    row=r['blocks'][0]['methods']['SDP_EQUIVALENT']['mode_records'][0]
    if mutation=='identity':r['case_sha256']='bad'
    if mutation=='missing_tier':row['candidates'].pop()
    if mutation=='event_count':row['resources']['solver_events'].pop()
    if mutation=='method':row['method']='OCC'
    with pytest.raises(ValueError):a.metadata(case,r,'synthetic.json')


@pytest.mark.parametrize('value',[[True,1],[1,False],[1,0],[1,-1],[1.0,1],[1<<4097,1]])
def test_rational_schema_and_bit_bound(value):
    with pytest.raises((ValueError,ArithmeticError)):a.rational(value)


@pytest.mark.parametrize('k,q,expected',[(0,1,'ROOT'),(3,1,'BOX'),(3,3,'BOX_ROOT_TIE'),(0,0,'ZERO_FLOOR')])
def test_fixed_candidate_box_root_attribution(k,q,expected):
    out=a.candidate_summary(candidate(F(k),F(q)),{'upper':{'U':[0,1]},'c':[2,1]},1)
    assert out['active_term']==expected
    assert out['repair_penalty_m_times_r']=='UNDETERMINED_R_NOT_SAVED'


def test_sos_saved_residual_decomposition_and_tampering():
    c=candidate(kind='SOS');out=a.candidate_summary(c,{'upper':{'U':[0,1]},'c':[2,1]},1)
    assert out['q_final']==[1,1] and out['residual_penalty_in_q']==[1,1]
    c['s']=[0,1]
    with pytest.raises(ValueError):a.candidate_summary(c,{'upper':{'U':[0,1]},'c':[2,1]},1)


def test_phase_residual_not_counterfactual_refutation_cost():
    case,r=mock_record();row=r['blocks'][0]['methods']['SDP_EQUIVALENT']['mode_records'][0]
    row['wall_seconds']=.1
    out=a.phase_summary(row)
    assert out['row_wall_minus_exclusive_phase_sum_unassigned']<0
    assert out['refutation_only_time']=='NOT_SEPARATELY_MEASURED'
    assert out['hypothetical_refutation_free_or_one_call_time']=='NOT_COMPUTED'


def test_old_or_false_authorization_refused():
    plan={'source_commit':'a'*40,'source_fingerprint':'b'*64,'registry_sha256':'c'*64}
    auth={'scope':gate.SCOPE,'posthoc_execution_authorized':False,**plan,'plan_sha256':'d'*64,
          'execution_branch':gate.BRANCH,'output_path':gate.OUTPUT,'attempts':1,'retries':0,'mandatory_stop':True}
    with pytest.raises(ValueError):gate.authorization(auth,plan,'d'*64)
    auth['posthoc_execution_authorized']=True;assert gate.authorization(auth,plan,'d'*64)
    auth['retries']=1
    with pytest.raises(ValueError):gate.authorization(auth,plan,'d'*64)


def test_path_and_output_overwrite_guards(tmp_path):
    with pytest.raises(ValueError):gate.confined(tmp_path,'../other')
    out=gate.Output(tmp_path);out.write('test.json',{'a':1})
    with pytest.raises(FileExistsError):out.write('test.json',{'a':2})
    with pytest.raises(ValueError):out.write('../test.json',{})


def test_existing_three_proof_recheck_on_synthetic_separation():
    from trottertracks.representation_exploration.n3_width_contract_v2 import SOS_LEVEL,digest
    from trottertracks.representation_exploration.n3_saved_width_v1 import interval
    from itertools import combinations
    h=[[enc(0) for _ in range(5)] for _ in range(5)];h[4][4]=enc(1);h[0][4]=h[4][0]=enc(1)
    case={'id':'synthetic_connection','family':'synthetic','seed':0,'modes':5,'nu':2,'h':h,
          'density':[[0,j,enc(1)] for j in range(1,5)],'candidate_modes':[0,4],'epsilon_proj':[1,100]}
    def bounds(d):
        active=[i for i in range(5) if i!=d];pool=[list(x) for x in combinations(active,2)]
        values=[sum((F(*h[i][i]) for i in o),F())+sum((F(1) for i,j in combinations(o,2) if i==0),F()) for o in pool]
        tau=0 if d==4 else 1
        return {'upper':{'occupations':pool,'energies':[enc(x) for x in values],'U':[0,1]},
                'external':{'tau':enc(tau),'Z':[[enc(0) for _ in range(4)] for _ in range(4)],'s':enc(tau)},'c':[1,1]}
    grams={name:[[enc(0) for _ in range(n)] for _ in range(n)] for name,n in [('particle',4),('hole',4),('D',6),('Q',6),('G',16)]}
    sos={'kind':'SOS','case_sha256':digest(case),'external':4,'kappa':[1,1],'s':[0,1],
         'parameters':{'level':SOS_LEVEL,'grams':grams,'multipliers':[[0,1],[1,1]]+[[0,1]]*9,'ell':[0,1],'residual_bound':[0,1],'s':[0,1]},
         'width':interval(F(1),F(1),F(0))}
    other={'kind':'OCC','case_sha256':digest(case),'external':0,'kappa':[1,1],'s':[0,1],
           'parameters':{'tau':[0,1],'Z':[[enc(0) for _ in range(4)] for _ in range(4)],'s':[0,1]},'width':interval(F(1),F(1),F(0))}
    gamma=[[enc(int(i==j and i<2)) for j in range(4)] for i in range(4)]
    ref={'delta':[1,400],'certificate':{'gamma':gamma,'t':[0,1],'objective':[1,1]}}
    bundle={'case':case,'target_delta':[1,400],'bounds_external4':bounds(4),'bounds_external0':bounds(0),
            'witnesses':[{'label':'SOS_external4','value':sos},{'label':'OCC_refutation_external4','value':ref},{'label':'OCC_winner_external0','value':other}]}
    assert a.recertify_connection(bundle)['status'].startswith('EXISTING_SOS_POSITIVE')
    bad=deepcopy(bundle);bad['witnesses'][1]['value']['certificate']['objective']=[0,1]
    with pytest.raises(ValueError):a.recertify_connection(bad)


def test_no_numeric_solver_modules_imported():
    assert not any(k in sys.modules for k in ('numpy','scipy','cvxpy','scs'))


def synthetic_parallel_worker(payload):
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(gate.RESOURCES['worker_AS_bytes'],gate.RESOURCES['worker_AS_bytes']))
    case,record=payload
    return a.analyze(case,record,'synthetic.json')


def test_two_spawn_workers_match_serial_synthetic_arithmetic():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    payload=mock_record();expected=a.analyze(*payload,'synthetic.json')
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        actual=list(pool.map(synthetic_parallel_worker,[payload,payload]))
    assert actual==[expected,expected]
