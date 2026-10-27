"""M4 invented sector arrays, saved JSON boundary and affine-price tests."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_m4 as m


def test_passive_and_fixed98_no_native_path():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_m4 as m;assert m.MAX_CALLS==98;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    assert [sum(x.values()) for x in (m.A_LIMITS,m.B_LIMITS,m.C_LIMITS)]==[39,48,10]
    assert not any(k in m.LIMITS for k in ('compile','kernel_build','trajectory_sample','exact_evolution','PF_kernel'))


def example():
    from test_research_c_pr_aware_df_stage_b2 import artificial
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks.c1_molecule_extension import SectorCAR
    arm,p=artificial(n=4,rank=8,role='proposed');sector=SectorCAR(4)
    G=sector.generators(**p,J=arm.J,U=arm.U)
    tail,c64,d=b.configure(arm,Fraction(1,10),64,0)
    _,c128,_=b.configure(arm,Fraction(1,10),128,0)
    return arm,p,G,tail,c64,c128


def test_RTE_subset_matches_existing_kernels_rounds_and_PF_is_r_independent():
    from threadpoolctl import threadpool_limits
    from trottertracks import c_pr_aware_df_stage_b as b
    with threadpool_limits(limits=1):
        arm,p,G,tail,c64,c128=example()
        old=b.kernels(arm,G,Fraction(1,10),c64);expected=b.kernels(arm,G,Fraction(1,10),c128)
        assert np.array_equal(old['pf'],expected['pf'])
        new=m.corrected_kernel(G,c128,ld=2,identity_tail=arm.plan.extracted_identity,saved_pf=old['pf'])
        np.testing.assert_array_equal(new['pf'],old['pf'])
        np.testing.assert_allclose(new['corrected'],expected['corrected'],atol=1e-14,rtol=0)
        H=sum(G)
        for i in range(13):
            ref=b.reference.reference_round(expected,(H,Fraction(1,10)*(1<<i)),repetitions=1<<i)
            with patch('scipy.linalg.expm',side_effect=AssertionError('no new exact-time evolution')):
                row=m.rte_round(new,i,ref['pf_operator_error'])
            assert row['pf_operator_error']==ref['pf_operator_error']
            assert row['rte_corrected_operator_error']==ref['rte_corrected_operator_error']
            assert row['attenuation']==ref['attenuation']
        saved=dict(M3=dict(A=dict(algebra=dict(tau_rational=str(Fraction(tail.lambda_r)*Fraction(1,10)/64),
            new_tau=c64.dimensionless_step_time,lambda_R=tail.lambda_r))))
        assert m.tau_gate(tail,c128,saved)['binary64_half']
        with pytest.raises(ValueError,match='R128_H'):m.corrected_kernel(G,c64,ld=2,identity_tail=0,saved_pf=old['pf'])
        with pytest.raises(ValueError,match='ROUND_INDEX'):m.rte_round(new,13,0.)


def synthetic_fixture(root):
    from test_research_c_pr_aware_df_m3 import a_fixture
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_stage_b_runner as loader
    c,saved=a_fixture(root);c['M3_sector']='M3_sector.npz'
    p=loader.load(root/c['input'],('h','g','constant'));d=loader.load(root/c['exports']['proposed'],('J','U'))
    arm=b.prepare_arm(**p,**d,role='proposed',source_identity='synthetic',input_identity='synthetic',ld=2)
    sec=loader.load(root/c['references']['proposed'],('generators','H','full_input_indices'))
    tail,config,_=b.configure(arm,Fraction(1,10),64,0);kernel=b.kernels(arm,sec['generators'],Fraction(1,10),config)
    np.savez(root/c['M3_sector'],**sec,pf=kernel['pf'])
    rounds=[dict(m=i,reference=b.reference.reference_round(kernel,(sec['H'],Fraction(1,10)*(1<<i)),repetitions=1<<i)) for i in range(13)]
    plans=[dict(m=i,shots=1000+i) for i in range(13)]
    def cost(value):return dict(totals={k:value for k in m.METRICS},cost_scope='serial_native_kernel_composition_without_state_preparation')
    saved['M3']=dict(A=dict(rounds=rounds,saved_DF_bound_Ha=1e-8,algebra=dict(
        tau_rational=str(Fraction(tail.lambda_r)*Fraction(1,10)/64),new_tau=config.dimensionless_step_time,lambda_R=tail.lambda_r)),
        B=dict(calibration=dict(sample_mean={k:100. for k in m.METRICS})),
        C=dict(lanes={lane:dict(rounds=plans,resources=cost(1e10)) for lane in ('ideal_exact_native','assumed_raw_signal_budget')}))
    saved['M2']=dict(candidates=dict(proposed=dict(cost=cost(1e9))))
    saved['C1']=dict(assumed_scenarios={lane:dict(proposed=dict(resources=cost(1e11))) for lane in ('ideal_exact_native','assumed_raw_signal_budget')})
    saved['shell_reports']={key:dict(metrics={k:value for k in m.METRICS}) for key,value in [('head',2),('X',3),('Y',4)]}
    # Exercise the same atom tuple/list boundary that failed M3 run01.
    m.io.save(root/'saved_M3.json',saved);return c,m.io.read(root/'saved_M3.json')


def test_artificial_end_to_end_reuses_PF_no_sampling_native_or_exact(tmp_path,monkeypatch):
    from threadpoolctl import threadpool_limits
    from trottertracks import c1_stage3_reference as ref
    from trottertracks import c_pr_aware_df_stage_b as b
    with threadpool_limits(limits=1):c,saved=synthetic_fixture(tmp_path)
    monkeypatch.setattr(m,'load_evidence',lambda *args:saved)
    monkeypatch.setattr(m,'identity_gate',lambda *args:dict(status='ARTIFICIAL_IDENTITY_PASS'))
    def forbidden(*args,**kw):raise AssertionError('forbidden scientific path')
    monkeypatch.setattr(ref,'reference_kernels',forbidden)
    monkeypatch.setattr(b,'independent_occurrence',forbidden);monkeypatch.setattr(b,'build_kernel',forbidden)
    out=tmp_path/'run';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):r=m.execute(tmp_path,c,out,call)
    assert call.counts==m.LIMITS and sum(call.counts.values())==98
    assert r['A']['status']=='POTENTIALLY_FEASIBLE' and r['A']['new_exact_evolution']==0
    assert r['B']['evidence']['shots']['total_shots'] is None
    assert r['C']['affine']['evidence']['W'] is None
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        b=r['B'][lane];a=r['C']['affine'][lane]
        assert b['energy']['same_accuracy'] and a['W']>0 and b['total_resources'] is None
        assert a['S']['rz_count']==sum(p['shots']*11 for p in b['rounds'])
        assert r['C']['comparisons'][lane+':M3_r64']['thresholds']['rz_count']['mu_break_even']>0
    assert r['r128_kernel_price'] is r['r128_total_resources'] is r['actual_CA_achieved'] is None
    assert not list(out.rglob('*.qpy'))
    for old,new in zip(saved['M3']['A']['rounds'],r['A']['rounds'],strict=True):
        assert old['reference']['pf_operator_error']==new['reference']['pf_operator_error']


def scalar_lane(shots=10):
    plans=[dict(m=i,shots=shots) for i in range(13)];old=[dict(m=i,shots=shots+1) for i in range(13)]
    energy=dict(same_accuracy=True)
    return dict(energy=energy,shots=m.shots_summary(plans,energy,old))


def shells():return {x:dict(metrics={k:v for k in m.METRICS}) for x,v in [('head',2),('X',3),('Y',4)]}


@pytest.mark.parametrize('target_relation',['positive','zero','negative'])
def test_affine_threshold_tie_strict_headroom_and_shell_once(target_relation):
    lane=scalar_lane();a=m.affine_accounting(lane,shells())
    assert a['W']==sum(20*(1<<m) for m in range(13)) and a['S']['rz_count']==13*10*11
    value=a['S']['rz_count']+dict(positive=1000,zero=0,negative=-100)[target_relation]
    target=dict(totals={k:value for k in m.METRICS},cost_scope='serial_native_kernel_composition_without_state_preparation')
    r=m.breakeven(a,target,{k:1. for k in m.METRICS},'artificial')
    row=r['thresholds']['rz_count'];mu=row['mu_break_even']
    assert a['W']*mu+a['S']['rz_count']==pytest.approx(value)
    assert row['nonnegative_price_can_strictly_win']==(target_relation=='positive')
    assert row['headroom_above_r64_kernel']==mu-1.
    assert r['actual_r128_kernel_price'] is r['actual_r128_total_cost'] is None
    if mu>0:assert a['W']*(mu/2)+a['S']['rz_count']<value


@pytest.mark.parametrize('status',['unqualified','infeasible'])
def test_unqualified_or_infeasible_does_not_invent_shots_threshold(status):
    energy=dict(same_accuracy=False);plans=[dict(m=i,shots=None) for i in range(13)]
    value=m.shots_summary(plans,energy,None)
    assert value['total_shots'] is None
    a=m.affine_accounting(dict(energy=energy,shots=value),shells());assert a['W'] is None
    assert m.breakeven(a,{},None,'x')['thresholds'] is None


def test_high_m_accounting_exact_and_not_independent_calibration():
    lane=scalar_lane();a=m.affine_accounting(lane,shells())
    hi=a['high_m_only_shots_swap'];old=a['old_M3_shots_accounting']
    assert old['W']-hi['W']==sum(2*(1<<i) for i in (10,11,12))
    assert old['S']['rz_count']-hi['S']['rz_count']==3*11
    assert lane['shots']['groups']['m10_to12']['old_shots']-lane['shots']['groups']['m10_to12']['new_shots']==6
    bad=deepcopy(a);bad['W']=0
    with pytest.raises(ValueError,match='AFFINE_OR_TARGET'):m.breakeven(bad,dict(totals={},cost_scope='serial_native_kernel_composition_without_state_preparation'),{},'bad')


def passive(root):
    from trottertracks.c_pr_aware_df_stage_c import SCENARIOS
    (root/'opaque.npz').write_bytes(b'closed artificial input')
    m.io.save(root/'source.json',dict(files={}))
    c=dict(scope=m.SCOPE,implementation=m.VERSION,base_commit=m.BASE,policy=m.POLICY,target=m.TARGET,
        allowances=m.ALLOWANCES,scenarios=list(SCENARIOS),stage_call_limits=dict(A=m.A_LIMITS,B=m.B_LIMITS,C=m.C_LIMITS),
        call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,retry=0,resume=0,
        new_fit=0,trajectory_sampling=0,build_compile=0,source_manifest='source.json',source_manifest_sha256=m.io.sha(root/'source.json'),
        saved_inputs={'opaque.npz':m.io.sha(root/'opaque.npz')},packages={})
    m.io.save(root/'contract.json',c);return c


def test_passive_input_source_and_fresh_mismatch(tmp_path):
    c=passive(tmp_path);assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'opaque.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')


def reuse_identity_fixture():
    from trottertracks import c_pr_aware_df_m3 as m3
    c=dict(input='inventedI',exports=dict(proposed='inventedJU'),grounds=dict(proposed='inventedGround'),
        original_input_sha256='SYNTHETIC_ONLY',saved_inputs=dict(inventedI='inputbytes',inventedJU='JUbytes'),
        pinned_M3_SOURCE={'invented.py':'SOURCEbytes'})
    old=dict(policy=deepcopy(m3.POLICY),target=m.TARGET,exports=c['exports'],input=c['input'],grounds=c['grounds'],
        original_input_sha256=c['original_input_sha256'],saved_inputs=dict(c['saved_inputs']),
        source_manifest_sha256='b375bb1f9f979a70de7577729a28f3d7755a66e6643e7b46366d74acc830d857')
    saved=dict(M3_contract=old,M3=dict(status='M3_A_B_C_COMPLETE',A=dict(status='POTENTIALLY_FEASIBLE',
        saved_DF_bound_Ha=1e-8,rounds=[dict(m=i,reference=dict(pf_operator_error=1e-4)) for i in range(13)]),
        C=dict(lanes={lane:dict(conditional_energy_assessment=dict(same_accuracy=True)) for lane in ('ideal_exact_native','assumed_raw_signal_budget')})),
        M3_STOP=dict(failed=False,scientific_units=471,call_counts=m3.LIMITS,retry=0,resume=0),
        M3_SOURCE=dict(execution_commit='5354edecc6aa995082440dd14be0310bdd9902e1',files={'invented.py':'SOURCEbytes'}),
        M3_ground=dict(status='PASS',alias_margin=1.6),M3_factors=[{} for i in range(9)],
        M3_tail=dict(status='PASS',catalog_identity=dict(status='PASS')))
    return saved,c


@pytest.mark.parametrize('fault',['h','scalar','frame_order','export_hash','reference_SOURCE','ground','PF_definition'])
def test_PF_reuse_identity_accepts_only_fixed_prerequisites(fault):
    saved,c=reuse_identity_fixture()
    assert m.identity_gate(saved,c)['reused_PF_rounds']==13
    if fault in ('h','scalar','frame_order'):saved['M3_contract']['policy'][fault]='changed'
    elif fault=='export_hash':saved['M3_contract']['saved_inputs']['inventedJU']='different'
    elif fault=='reference_SOURCE':saved['M3_SOURCE']['files']['invented.py']='different'
    elif fault=='ground':saved['M3_ground']['alias_margin']=-1
    else:saved['M3_contract']['policy']['PF']='different'
    with pytest.raises(ValueError):m.identity_gate(saved,c)


def test_failure_partial_STOP_and_no_retry(tmp_path,monkeypatch):
    c=passive(tmp_path)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kw:'ARTIFICIAL_SOURCE\n')
    def fail(root,c,out,call):
        m.io.save(out/'partial.json',dict(preserved=True))
        def fault():raise ValueError('ARTIFICIAL_M4_FAIL')
        return call('identity_gate','synthetic',fault)
    monkeypatch.setattr(m,'prepare_and_A',fail)
    with pytest.raises(ValueError,match='ARTIFICIAL_M4_FAIL'):m.run(tmp_path,tmp_path/'contract.json')
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed'] and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert m.io.read(tmp_path/m.FRESH/'A/partial.json')['preserved']
    with pytest.raises(ValueError,match='FRESH_M4'):m.run(tmp_path,tmp_path/'contract.json')
