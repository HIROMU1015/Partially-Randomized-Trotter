"""Invented scalar/preparation metadata only; no molecular numerical leaf."""
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import diagnose_t2_scalar as Q

ROOT=Path(__file__).resolve().parents[1]


@dataclass
class Component:
    component_id:str='INVENTED_COMPONENT'
    probability:float=1.


@dataclass
class Metadata:
    tag:str='INVENTED_BASIS_OR_NORMALIZATION'


def preparation():
    tail=NS(lambda_r=1.,tail_hash='INVENTED_TAIL',components=[Component()],
            basis_definitions=[Metadata()],extraction_metadata=Metadata(),normalization_metadata=Metadata())
    return NS(extracted_identity_coefficient=.25,constant_coefficient=1.,
        hamiltonian_hash='INVENTED_H',partition_hash='INVENTED_PARTITION',preparation_hash='INVENTED_PREP',
        deterministic_blocks=[],
        deterministic_fragment_indices=(0,1),randomized_block_indices=tuple(range(2,10)),
        identity_policy='extract_identity_phase',coefficient_atol=0.,diagonal_sort='MOCK',
        basis_construction_policy='MOCK',rte_preparation=NS(symbolic_tail=tail))


class Runtime:
    def __init__(self,root,ham,**kwargs):
        self.ham=ham;self.calls=0;self.preparations={}
    def context(self,request):
        self.calls+=1
        prep=preparation();self.preparations['only']=prep
        scalar=self.ham.constant+prep.extracted_identity_coefficient
        lam=prep.rte_preparation.symbolic_tail.lambda_r
        if scalar.hex()!=float(request['physical_context']['expected_scalar']).hex():raise ValueError(Q.GUARD)
        return NS(prep=prep,scalar=scalar)
    def sample(self,*a,**kw):pytest.fail('No sampler is authorized')


def request():
    r=Q.T.L.request_for(ROOT,'PR-LD2',12,execution_source={'fixture':'METADATA_ONLY'})
    r['physical_context']['expected_scalar']=.25  # invented scalar, not H4 result
    r['context_key']=Q.T.D._key(r['physical_context'])
    r['request_id']=Q.T.D._key({k:v for k,v in r.items() if k!='request_id'})
    return r


def test_decimal_hex_and_exact_roundtrip():
    assert Q.number(1.25)==dict(value=1.25,decimal='1.25',decimal_exact='1.25',hex='0x1.4000000000000p+0')
    with pytest.raises(ValueError):Q.number(float('inf'))


@pytest.mark.parametrize('x',[.25,-.25])
def test_adjacent_float_ulp_count_for_both_signs(x):
    d=Q.difference(math.nextafter(x,math.inf),x)
    assert d['signed_adjacent_binary64_steps']==1
    assert d['exact_input_difference']['numerator']>0


def test_same_exception_frame_preserves_guard_and_request():
    r=request();before=deepcopy(r);runtime=Runtime(None,NS(constant=1.))
    result,local=Q.observe_context(runtime,r)
    assert result['guard_stop'] and result['error']==Q.GUARD
    assert local['scalar']==1.25 and local['prep'].extracted_identity_coefficient==.25
    assert runtime.calls==1 and r==before


def test_success_branch_still_calls_context_once():
    runtime=Runtime(None,NS(constant=1.));r={'physical_context':{'expected_scalar':1.25}}
    result,local=Q.observe_context(runtime,r)
    assert result['context_returned'] and not result['guard_stop'] and local['scalar']==1.25
    assert runtime.calls==1


def test_observation_failure_never_retries():
    class Broken(Runtime):
        def context(self,request):self.calls+=1;raise RuntimeError('BEFORE_PREPARATION')
    runtime=Broken(None,NS(constant=1.))
    result,local=Q.observe_context(runtime,request())
    assert local is None and result['error']=='BEFORE_PREPARATION' and runtime.calls==1


def test_one_restore_context_and_exclusive_output_then_stop(tmp_path):
    calls=[];r=request();before=deepcopy(r)
    def restore(*a):calls.append('restore');return NS(constant=1.),{'fixture':'NOT_H4'}
    class Counted(Runtime):
        def context(self,request):calls.append('context');return super().context(request)
    out=tmp_path/'one_run'
    result=Q.run_once(ROOT,'INVENTED_SOURCE',r,{}, {},out,descriptor={'scalar':.25},restore=restore,runtime_class=Counted)
    # The inherited context frame is from Runtime, so use the original class to
    # exercise complete capture; the wrapper case must still STOP without retry.
    assert calls==['restore','context'] and (out/'DIAGNOSTIC_RESULT.json').exists()
    assert all(result['counts'][k]==0 for k in ('sampling','compile','T2_retry','event_path_generation'))
    assert r==before
    with pytest.raises(FileExistsError):
        Q.run_once(ROOT,'INVENTED_SOURCE',r,{}, {},out,descriptor={'scalar':.25},restore=restore,runtime_class=Runtime)
    assert calls==['restore','context']


def test_full_mock_diagnostic_keeps_expected_scalar_unchanged(tmp_path):
    r=request();before=deepcopy(r)
    result=Q.run_once(ROOT,'INVENTED_SOURCE',r,{}, {},tmp_path/'one',descriptor={'scalar':.25},
        restore=lambda *a:(NS(constant=1.),{'fixture':'NOT_H4'}),runtime_class=Runtime)
    assert result['status']=='DIAGNOSTIC_COMPLETE_GUARD_STOP'
    assert result['scalars']['total_from_original_context']['value']==1.25
    assert result['scalars']['request_expected']['value']==.25
    assert result['diagnostic_checks']['extracted_equals_descriptor_hex']
    assert result['diagnostic_checks']['descriptor_plus_constant_equals_actual_hex']
    assert result['original_addition_roundoff_exact']==dict(numerator=0,denominator=1)
    assert result['preparation_cache_entries']==1 and r==before


def test_metadata_failure_preserves_scalars_without_second_context(tmp_path,monkeypatch):
    def fail(prep):raise ValueError('INVENTED_METADATA_FAILURE')
    monkeypatch.setattr(Q,'inventory',fail)
    result=Q.run_once(ROOT,'INVENTED_SOURCE',request(),{}, {},tmp_path/'one',descriptor={'scalar':.25},
        restore=lambda *a:(NS(constant=1.),{'fixture':'NOT_H4'}),runtime_class=Runtime)
    assert result['status']=='DIAGNOSTIC_STOP_EXCEPTION'
    assert result['scalars']['total_from_original_context']['value']==1.25
    assert result['counts']['context']==result['counts']['input_restore']==1
    assert (tmp_path/'one/DIAGNOSTIC_RESULT.json').exists()
