"""F2 scope/bridge regression. Saved JSON only; all numerical inputs invented."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys

import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_pr_aware_df_fixed_f2 as f2
from trottertracks import c_pr_aware_df_fixed_car as car
from trottertracks import c_pr_aware_df_stage_b as b
from trottertracks import c_pr_aware_df_zsxx_enclosure as z


def draft():
    return json.loads((ROOT/f2.DRAFT).read_text())


def candidate():
    c = draft()
    c.update(scope=f2.SCOPE, execution_binding=f2.VERSION, execution_base_commit=f2.BASE,
        executable=True, target_policy_status='ADOPTED_FIXED_F2_ONLY', execution_blockers=[],
        fresh_output=f2.FRESH, operator_norm='phase_sensitive_full512_operator2')
    return c


def test_new_contract_keeps_old_draft_blocked_and_same_numerical_semantics():
    old=draft(); c=candidate()
    f2.scope_gate(c,old)
    assert old['executable'] is False and c['executable'] is True
    assert c['call_limits']==dict(input_arrays=1,arm_replay=1,trajectory_load=1,
        CAR_reference=1,qpy_decode=1,raw_CAR_certificate=1,triangle_composition=1)
    assert c['status']=='UNAPPROVED'  # historical loader compatibility,not permission proof


@pytest.mark.parametrize('key,value',[
    ('experiment','F3'),('arm','proposed'),('trajectory_index',1),
    ('target','different_unitarization'),('target_policy_status','PROPOSED_NOT_ADOPTED'),
    ('precision_bits',256),('allowance_exact','1/1000'),('maximum_logical_units',8),
    ('retry',1),('resume',1),('global_phase_fit',True),('fresh_output','old/F2-run01'),
    ('operator_norm','sector36'),('execution_blockers',['not adopted'])])
def test_unregistered_scope_target_retry_norm_changes_rejected(key,value):
    c=candidate();c[key]=value
    with pytest.raises(ValueError,match='FIXED_F2_TARGET_SCOPE'):f2.scope_gate(c,draft())


def test_saved_conditions_and_inputs_cannot_change():
    c=candidate();c['additional_conditions']['h']='1/20'
    with pytest.raises(ValueError,match='POLICY_CHANGED'):f2.scope_gate(c,draft())
    c=candidate();c['saved_inputs'][c['export']]='a'*64
    with pytest.raises(ValueError,match='SAVED_INPUT_CHANGED'):f2.scope_gate(c,draft())


def prior_records():
    c=candidate()
    base=ROOT/'artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run02'
    B1=ROOT/'artifacts/research_c_pr_aware_df_guarantee_strategy/2026-10-27/B1-run01'
    return c,json.loads((base/'SOURCE.json').read_text()),json.loads((base/'STOP.json').read_text()),\
        json.loads((B1/'contract.json').read_text()),json.loads((B1/'kernel_compiled_raw.json').read_text())


def test_same_existing_raw_phase_dimension_and_B2_provenance_metadata_only():
    f2.prior_gate(*prior_records())


@pytest.mark.parametrize('key,value',[('phase_fit',True),('dimension',256),
    ('norm','max_entry'),('gate_status','BINARY64_DIAGNOSTIC'),('precision_bits',256)])
def test_triangle_different_scope_certificate_is_rejected(key,value):
    args=list(prior_records());args[-1][key]=value
    with pytest.raises(ValueError,match='TRIANGLE_RAW_PHASE_DIMENSION'):f2.prior_gate(*args)


def test_missing_SOURCE_hash_fails_before_numeric_loader(tmp_path,monkeypatch):
    monkeypatch.setattr(f2.execution,'run',lambda *a:pytest.fail('must not start science'))
    (tmp_path/'contract.json').write_text('{}')
    with pytest.raises(FileNotFoundError):f2.run(tmp_path,tmp_path/'contract.json')


def test_analytic_bridge_contains_validated_artificial_exact_polar_target():
    """Positive diagonal U has known exact right polar I; no numerical polar."""
    from flint import ctx
    from trotterlib.rte import _make_event
    h=np.array([[.125,.0625],[.0625,-.25]])
    J=np.array([[[.25,-.0625],[-.0625,-.125]]]*3)
    U=np.array([np.diag([1+1/1024,1-1/2048])]*3)
    def arm(rotations):
        return b.prepare_arm(h,np.zeros((2,)*4),-.125,J,rotations,role='baseline',
            source_identity='synthetic_JU',input_identity='synthetic_input',ld=2)
    a=arm(U);tail,config,law=b.configure(a,F(1,40),16,0)
    selected=next(i for i,c in enumerate(tail.components) if c.coefficient_sign<0)
    events=tuple(_make_event((selected,),tail.components,law,0) for _ in range(16))
    literal=car.build_reference(a,h,-.125,events,parallel=False)
    ideal=car.build_reference(arm(np.array([np.eye(2)]*3)),h,-.125,events,parallel=False)
    assert F(ideal['bridge']['upper_exact'])==0
    with ctx.workprec(192):
        distances=[F(z._norm(car.unpack(x['entries'],len(x['indices']))-
                    car.unpack(y['entries'],len(y['indices'])))['upper_exact'])
                   for x,y in zip(literal['blocks'],ideal['blocks'],strict=True)]
    assert max(distances)>0
    assert max(distances)<=F(literal['bridge']['upper_exact'])
    assert literal['bridge']['input_nonunitarity_included']
    assert not literal['bridge']['input_U_replaced']
