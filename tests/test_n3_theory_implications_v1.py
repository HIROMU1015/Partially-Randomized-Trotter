"""Rational synthetic fixtures; no repository evidence, solver or producer."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import pytest

from trottertracks.representation_exploration.n3_old_family_contract_v1 import encode, fingerprint
from trottertracks.representation_exploration.n3_theory_implications_v1 import (
    fixed_witness_accepts, norm_enclosure, saved_implication, uniform_exclusion,
)


def fixture(h=None, v=None, nu=2):
    return encode({'id': 'rational_synthetic', 'h_A': h or [[F(-1), F(0)], [F(0), F(1)]],
                   'v': v or [F(1), F(0)], 'nu': nu,
                   'U': F(0), 'c': F(1), 'delta': F(1, 10)})


W = encode([[F(1), F(0)], [F(0), F(0)]])


def test_uniform_sufficient_case():
    r = uniform_exclusion(fixture(), W)
    assert r['status'] == 'UNIFORM_OLD_EXCLUSION_PROVED'
    assert r['r'] == [-1, 1] and r['strict_intercept_margin'] == [1, 2]


def test_zero_slope_allowed():
    case = fixture([[F(0), F(0)], [F(0), F(1)]])
    assert uniform_exclusion(case, W)['status'] == 'UNIFORM_OLD_EXCLUSION_PROVED'


def test_target_dual_does_not_imply_uniform_lemma():
    # Both endpoint traces <0 at delta=.1, but r>0: cannot extrapolate to all k>=0.
    case = fixture([[F(1, 10), F(0)], [F(0), F(-1, 10)]])
    assert -1+F(11, 200) < 0 and F(11, 100)-1+F(11, 200) < 0
    assert uniform_exclusion(case, W)['status'] == 'SUFFICIENT_LEMMA_NOT_ESTABLISHED'


def test_small_intercept_even_with_negative_target_endpoints():
    case = fixture(v=[F(1, 2), F(1)])
    assert -F(1, 4)+F(11, 200) < 0
    assert uniform_exclusion(case, W)['status'] == 'SUFFICIENT_LEMMA_NOT_ESTABLISHED'


def test_strict_intercept_required():
    case = fixture(v=[F(1), F(1)])
    assert uniform_exclusion(case, W)['status'] == 'SUFFICIENT_LEMMA_NOT_ESTABLISHED'


def test_nu_one_cannot_meet_uniform_strict_intercept():
    assert uniform_exclusion(fixture(nu=1), W)['status'] == 'SUFFICIENT_LEMMA_NOT_ESTABLISHED'


def test_zero_coupling_has_no_strict_norm_improvement():
    case = fixture(); case['v'] = [[0, 1], [0, 1]]
    assert uniform_exclusion(case, W)['status'] == 'SUFFICIENT_LEMMA_NOT_ESTABLISHED'


@pytest.mark.parametrize('bad', [
    [[[0, 1], [0, 1]], [[0, 1], [0, 1]]],
    [[[2, 1], [0, 1]], [[0, 1], [0, 1]]],
    [[[-1, 1], [0, 1]], [[0, 1], [2, 1]]],
    [[[0, 1], [1, 1]], [[1, 1], [1, 1]]],
    [[[1, 1]]],
    [[[1, 1], [1, 1]], [[0, 1], [0, 1]]],
])
def test_malformed_or_non_psd_W_rejected(bad):
    with pytest.raises(ValueError): uniform_exclusion(fixture(), bad)


@pytest.mark.parametrize('key,value', [('nu', True), ('c', [0, 1]), ('delta', [0, 1]),
    ('U', [0.0, 1]), ('truth_gap', [1, 1])])
def test_bad_instance_rejected(key, value):
    case = fixture(); case[key] = value
    with pytest.raises(ValueError): uniform_exclusion(case, W)


def test_polynomial_enclosure_and_bad_brackets():
    assert norm_enclosure(F(1), F(1), [3, 5], [5, 8]) == (F(3, 5), F(5, 8))
    for lo, hi in [([5, 8], [3, 5]), ([7, 10], [4, 5]), ([1, 2], [3, 5]), ([0, 1], [1, 1])]:
        with pytest.raises(ValueError): norm_enclosure(F(1), F(1), lo, hi)


def test_witness_transport_to_stronger_c_and_larger_delta():
    k,s,U,c,d = F(1, 2),F(-1, 10),F(0),F(1),F(1, 10)
    assert fixed_witness_accepts(k,s,U,c,d)
    for cc in (c,c+1):
        for dd in (d,d+1): assert fixed_witness_accepts(k,s,U,cc,dd)
    assert not fixed_witness_accepts(k,s,U,c,F(1, 100))


def test_occupancy_bound_vs_scalar_relaxation():
    levels = [F(-2), F(1), F(3)]
    assert min(sum(levels[i] for i in occ) for occ in combinations(range(3),2)) == -1
    assert 2*min(levels) == -4  # violates the <=1 occupation constraint
    assert min(levels) == min(sum(levels[i] for i in occ) for occ in combinations(range(3),1))
    assert sum(levels) == min(sum(levels[i] for i in occ) for occ in combinations(range(3),3))


def test_fixed_fraction_domain_and_source_fixture_preserved():
    before=deepcopy(W); uniform_exclusion(fixture(),W); assert W==before
    with pytest.raises(ValueError): fixed_witness_accepts(0.5,F(0),F(0),F(1),F(1))


def saved_fixture():
    case=encode({'id':'saved_synthetic','h_A':[[F(-2),F(0),F(0)],[F(0),F(0),F(0)],
        [F(0),F(0),F(10)]],'v':[F(2),F(0),F(1)],'nu':2,
        'U':F(0),'c':F(999,10),'delta':F(1,20)})
    saved_W=encode([[F(1),F(0),F(0)],[F(0),F(0),F(0)],[F(0),F(0),F(0)]])
    record={'schema_version':1,'input_id':case['id'],'instance_sha256':fingerprint(case),
        'status':'CERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION','certificate':{
            'kind':'dual','W':saved_W,'trace_W':[1,1],
            'endpoint_traces':[[-1201,800],[-161121,800]],'origins':['synthetic_saved']},
        'NORM_at_target':False,'finite_failure_proves_infeasible':False,'events':[],
        'resources':{'counts':{'numeric_eigh_calls':0,'primal_PSD_probes':0,'ray_probes':0,'mixture_probes':0},
                     'arithmetic_operations':0,'max_monitored_rational_bits':0},
        'new_OCC_proposals':0,'new_U_c_bounds':0,'ground_or_gap_calls':0,
        'central_theme':None,'mandatory_stop':True,'next_stage_authorized':False}
    summary={'input_id':case['id'],'common_U':case['U'],'common_c_star':case['c'],
        'widths':{'NORM':{'low':[5001,100000],'high':[5003,100000]},
                  'OCC_RAW_ALL_SAVED_SPECTRA':{'high':[1,20],'positive_proof_delta':[1,20],'upper_witness_id':'saved'}},
        'winning_witnesses':{'OCC_RAW_ALL_SAVED_SPECTRA':{'id':'saved','kind':'OCC',
                             'kappa':[1,20],'s':[-999,200],'q':[999,200]}}}
    return case,record,summary


def test_saved_binding_and_exact_improvement():
    case,record,summary=saved_fixture()
    ans=saved_implication(case,record,summary)
    assert ans['achieved_relative_improvement_enclosure']==[[1,5001],[3,5003]]
    assert ans['new_certificate_proposals']==0 and ans['mandatory_stop'] is True


@pytest.mark.parametrize('mutation', ['U','id','delta','q','winner','target_trace','instance_hash'])
def test_changed_saved_provenance_rejected(mutation):
    case,record,summary=saved_fixture()
    if mutation=='U': summary['common_U']=[1,1]
    if mutation=='id': summary['input_id']='wrong'
    if mutation=='delta': summary['widths']['OCC_RAW_ALL_SAVED_SPECTRA']['high']=[1,11]
    if mutation=='q': summary['winning_witnesses']['OCC_RAW_ALL_SAVED_SPECTRA']['q']=[1,11]
    if mutation=='winner': summary['widths']['OCC_RAW_ALL_SAVED_SPECTRA']['upper_witness_id']='other'
    if mutation=='target_trace': record['certificate']['endpoint_traces'][0]=[0,1]
    if mutation=='instance_hash': record['instance_sha256']='0'*64
    with pytest.raises(ValueError): saved_implication(case,record,summary)
