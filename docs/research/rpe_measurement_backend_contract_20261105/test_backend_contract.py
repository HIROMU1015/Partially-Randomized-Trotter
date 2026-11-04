"""Contract fixtures only. No H4 callbacks, sampler, state action or backend.

Legacy interface/decoder remain unchanged. Strict int checking below is a TEST
fixture illustrating the callback precondition, not an adopted backend wrapper.
"""
import ast
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'docs/research/rpe_18round_measurement_connection_20261103'))
import measurement_adapter as old


def fixture(m=0, shots=3):
    return old.RoundSettings(3,m,1<<m,1<<(m%6),2,.125,
        old.AxisSettings(shots,'.01'),old.AxisSettings(shots,'.01'),
        '.01','.01','.3','.1',old.PHYSICAL,'attenuated_event_mean_signal',
        'ARTIFICIAL_STATE_ONLY','SYNTHETIC_ONLY')


def strict_int_fixture(callback):
    def checked(request,trajectory):
        value=callback(request,trajectory)
        if type(value) is not int or value not in (-1,1):
            raise ValueError('UNKNOWN/STOP: built-in int +/-1 required')
        return value
    return checked


def selected_functions(path,names,namespace):
    """Reuse unchanged production FunctionDefs on small descriptor stubs."""
    tree=ast.parse((ROOT/path).read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,*selected],type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)
    return namespace


def test_fixed_sources_and_saved_results_unchanged():
    spec=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for item in spec['files'].values():
        raw=subprocess.check_output(['git','show',item['commit']+':'+item['path']],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==item['sha256']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['blob']
        if item['current']:assert (ROOT/item['path']).read_bytes()==raw


def test_18_saved_rounds_literal_join_only():
    rows=json.loads((ROOT/'docs/research/rpe_18round_ideal_enclosure_preparation_20261104/fixed_inputs.json').read_bytes())['rounds']
    settings=old.load_fixed_rounds(3)
    assert len(settings)==len(rows)==18
    for s,row in zip(settings,rows,strict=True):
        assert (s.ld,s.m,s.q,s.r,s.k)==(3,row['m'],row['q'],row['r'],row['K'])
        assert s.delta.hex()==float(row['delta']).hex()
        assert (s.beta_pf,s.beta_rte,s.beta_stat)==tuple(row['beta'][k] for k in ('PF','RTE','stat'))
        assert s.physical_fingerprint=='a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90'
        for axis in old.AXES:
            a=getattr(s,axis)
            assert dict(N=str(a.shots),alpha=a.alpha,epsilon=s.epsilon_coordinate)==row['axes'][axis]
            assert 1<=a.shots<=2**52
    # Only metadata are read. Never collect/decode these H4 settings.


def test_all18_artificial_requests_propagate_and_remain_ordered():
    keys=[]
    for m in range(18):
        s=fixture(m)
        def factory(request):
            assert request.settings is s
            assert (request.settings.q,request.settings.r,request.settings.k,request.settings.delta)==(1<<m,1<<(m%6),2,.125)
            assert request.expected_event_count==s.q*s.r
            keys.append(request.trajectory_key)
            return ('ARTIFICIAL',request.trajectory_key)
        def measure(request,trajectory):
            assert trajectory==('ARTIFICIAL',request.trajectory_key)
            return 1 if request.shot==0 else -1
        result=old.collect_virtual_round(s,factory,strict_int_fixture(measure))
        assert type(result.cosine) is type(result.sine) is float
        assert Fraction.from_float(result.cosine)==Fraction.from_float(-1/3)
    assert len(keys)==len(set(keys))==18*2*3


def test_fresh_calls_are_not_cache_or_iid_proof():
    calls=[]
    same=object()
    def factory(request):calls.append(request.trajectory_key);return same
    result=old.collect_virtual_round(fixture(),factory,lambda *_:1)
    assert len(calls)==6 and len(set(calls))==6
    assert result.trajectory_policy=='fresh_per_shot'
    decoded=old.decode_measurements([result.settings],[result])
    assert decoded['statistical_iid_verified'] is False
    # This shows the API cannot inspect factory caching; repeated sampled words
    # can also occur legitimately. Distinct object/content is not required.


@pytest.mark.parametrize('value',[1.0,np.int64(1),np.bool_(True)])
def test_legacy_acceptance_exceeds_integer_sum_contract(value):
    result=old.collect_virtual_round(fixture(),lambda _:None,lambda *_:value)
    assert result.cosine==1
    with pytest.raises(ValueError,match='built-in int'):
        old.collect_virtual_round(fixture(),lambda _:None,strict_int_fixture(lambda *_:value))


@pytest.mark.parametrize('value',[True,0,2,float('nan')])
def test_invalid_callback_result_stops(value):
    with pytest.raises(ValueError):
        old.collect_virtual_round(fixture(),lambda _:None,strict_int_fixture(lambda *_:value))


@pytest.mark.parametrize('bit,expected',[(0,1),(1,-1)])
def test_bit_to_int_fixture(bit,expected):
    assert type(bit) is int and bit in (0,1)
    outcome=1-2*bit
    assert type(outcome) is int and outcome==expected


def test_integer_sum_N3_and_cmath_scalar_input(monkeypatch):
    s=fixture()
    def outcome(request,_):
        return (1 if request.shot==0 else -1) if request.axis=='cosine' else 1
    measured=old.collect_virtual_round(s,lambda _:None,strict_int_fixture(outcome))
    seen=[];original=old.cmath.phase
    def capture(value):
        assert type(value) is complex
        seen.append(value)
        return original(value)
    monkeypatch.setattr(old.cmath,'phase',capture)
    result=old.decode_measurements([s],[measured])
    assert seen==[complex(-1/3,1)]
    assert measured.cosine != 2.0/3-1.0  # count-grid is a different rounding path
    assert result['success_probability_evaluated'] is False


def test_zero_signed_pi_and_tie_diagnostics():
    s=fixture()
    assert old.decode_measurements([s],[old.MeasuredMeans(s,0,0)])['status']=='ZERO_SIGNAL'
    for sign in (0.0,-0.0):
        value=old.MeasuredMeans(s,-1,sign)
        assert math.copysign(1,old.cmath.phase(value.signal))==math.copysign(1,sign)
        assert old.decode_measurements([s],[value])['status']=='ALIAS_BOUNDARY'
    t=fixture(1)
    assert old.decode_measurements([s,t],[old.MeasuredMeans(s,1,0),old.MeasuredMeans(t,-1,0)])['status']=='AMBIGUOUS_BRANCH'


def test_callback_exception_and_round_reordering_stop():
    def abort(*_):raise RuntimeError('ARTIFICIAL backend failure')
    with pytest.raises(RuntimeError):old.collect_virtual_round(fixture(),abort,lambda *_:1)
    s,t=fixture(),fixture(1)
    with pytest.raises(ValueError):old.decode_measurements([t,s],[old.MeasuredMeans(t,1,0),old.MeasuredMeans(s,1,0)])


def test_source_run_policy_resets_at_step_boundary():
    namespace=dict(DFRTEApplicationBasisChoice=NS,DFRTEBasisPlan=NS)
    selected_functions('src/trotterlib/research_direction_sequence_policy.py',
        ('_flatten_applications','source_basis_run_lengths','_support_key','make_run_threshold_basis_plan'),namespace)
    namespace.update(replace=replace,POLICY_RUN_THRESHOLD=1)
    selected_functions('src/trotterlib/research_direction_full_scope.py',('apply_run_threshold_policy',),namespace)
    app=NS(is_identity=False,basis_id='A',basis_hash='Ahash',diagonal_pauli_support=(0,))
    event=NS(application_sequence=(app,))
    occurrence=NS(events=(event,))
    support={('A','Ahash',(0,)):NS(basis_id='A-support',basis_hash='selected-hash')}
    @dataclass(frozen=True)
    class Request:
        preparation:object
        rte_occurrences:tuple
        rte_basis_plans:tuple=()
    per_step=Request(NS(is_deterministic_only=False),(occurrence,occurrence))
    actual=namespace['apply_run_threshold_policy'](per_step,support,training_fingerprint='ARTIFICIAL')
    assert [p.choices[0].selected_basis_id for p in actual.rte_basis_plans]==['A-support','A-support']
    within_step=NS(events=(event,event))
    plan=namespace['make_run_threshold_basis_plan'](within_step,support,
            maximum_support_run_length=1,training_fingerprint='ARTIFICIAL')
    assert [c.selected_basis_id for c in plan.choices]==['A','A']


def test_numpy_cdf_right_boundary_without_random_sampler():
    # Artificial inverse-CDF boundary only: no RNG, H4 CDF or new sampler.
    cdf=np.array([.25,1.])
    u=np.array([.25-2**-53,.25,.25+2**-53])
    assert np.searchsorted(cdf,u,side='right').tolist()==[0,1,1]
