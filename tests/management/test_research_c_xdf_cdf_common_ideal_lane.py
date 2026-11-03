"""Synthetic JSON/known scalar tests only; never join real H4 saved values."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_xdf_cdf_common_ideal_lane as m
from test_research_c_c1_h4_strong_ca_resources import artificial as strong_fixture
from test_research_c_c1_h4_ca_resources import artificial as legacy_artificial


@pytest.fixture
def case(tmp_path, monkeypatch, strong_fixture):
    def make(fault=None):
        old_path, _ = strong_fixture()
        old_contract = m.io.read(old_path)
        bundle = m.saved.load_bundle(tmp_path, old_contract)
        packets = {a: m.saved.attach_evidence(bundle, a) for a in m.ROUTES}
        source = m.io.read(tmp_path/old_contract['source_manifest'])
        records = dict(old_result=dict(status='H4_STRONG_BASELINE_S1_CA_RESOURCES_COMPLETE',
            scope=m.saved.SCOPE, candidates={p['id']:dict(id=p['id'],arm=a,s=1,aliases=p['aliases'])
                for a,p in packets.items()}), old_SOURCE=dict(source,
            execution_commit='1aeedd96178a3f50a15bc2701c3c332f4f391b64'),
            old_STOP=dict(failed=False,mandatory_stop=True,retry=0,resume=0,
                scientific_units=m.saved.MAX_CALLS,call_counts=m.saved.LIMITS),
            CDF_packet=packets['CDF'],XDF_packet=packets['XDF'])
        if fault=='candidate': records['XDF_packet']['arm']='CDF'
        if fault=='DF': records['CDF_packet']['df_error_with_guard']+=.00001
        if fault=='PF': records['CDF_packet']['PF_rounds'][0]['pf_operator_error']+=.001
        if fault=='metric': records['CDF_packet']['kernel_metrics']['rz_count']+=1
        if fault=='source': records['old_SOURCE']['execution_commit']='WRONG'
        if fault=='ground':
            old_legacy=m.io.read(tmp_path/old_contract['legacy_contract'])
            path=tmp_path/old_legacy['records']['XDF'];v=m.io.read(path)
            v['ground']['status']='FAIL';path.write_text(json.dumps(v))
        for key,value in records.items():
            (tmp_path/('ideal_'+key+'.json')).write_text(json.dumps(value))
        paths={k:'ideal_'+k+'.json' for k in records}
        paths['old_contract']=old_path.name
        (tmp_path/'oracle_source.py').write_text('# synthetic technical source\n')
        (tmp_path/'ideal_SOURCE_manifest.json').write_text(json.dumps(dict(
            files={'oracle_source.py':m.io.sha(tmp_path/'oracle_source.py')})))
        c=dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',base_commit=m.BASE,
            implementation=m.VERSION,conditions=m.CONDITIONS,target=m.TARGET,lane=m.LANE,
            call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,
            retry=0,resume=0,mandatory_stop=True,cost_scope=m.saved.COST_SCOPE,
            candidate_order=list(m.ROUTES.values()),records=paths,
            source_manifest='ideal_SOURCE_manifest.json',
            source_manifest_sha256=m.io.sha(tmp_path/'ideal_SOURCE_manifest.json'),
            compiler=old_contract['compiler'],gaussian=old_contract['gaussian'],
            compiler_phase=old_contract['compiler_phase'],packages={},
            **{k:0 for k in m.FORBIDDEN})
        c['saved_inputs']={str(p.relative_to(tmp_path)):m.io.sha(p) for p in tmp_path.rglob('*.json')}
        path=tmp_path/'ideal_contract.json';path.write_text(json.dumps(c))
        monkeypatch.setattr(m.saved.prior,'source_commit',lambda root:'synthetic')
        return path,c,packets
    return make


def test_opt_in_and_passive_import():
    with pytest.raises(ValueError,match='OPT_IN'):
        m.run('not opened','not opened')
    code=("import sys;sys.path.insert(0,'src');"
          "from trottertracks import c_xdf_cdf_common_ideal_lane as m;"
          "assert m.MAX_CALLS==61;"
          "assert not any(k in sys.modules for k in ['numpy','scipy','qiskit','ffsim'])")
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


@pytest.mark.parametrize('fault',['input','source','policy','missing'])
def test_passive_preflight_drift_no_output(case,tmp_path,fault):
    path,c,_=case()
    if fault in ('input','missing'):
        p=tmp_path/c['records']['CDF_packet']
        if fault=='missing':p.unlink()
        else:p.write_text('CHANGED')
    elif fault=='source':(tmp_path/'oracle_source.py').write_text('CHANGED')
    else:
        c['lane']=dict(c['lane'],synthesis_zero_is='measured');path.write_text(json.dumps(c))
    with pytest.raises((ValueError,FileNotFoundError)):
        m.preflight(tmp_path,path)
    assert not (tmp_path/m.FRESH).exists()


def test_synthetic_end_to_end_existing_solver_serial_and_qualifications(case,tmp_path):
    path,c,packets=case()
    assert m.preflight(tmp_path,path)==c
    old={p:(tmp_path/p).read_bytes() for p in c['saved_inputs']}
    blocked=AssertionError('NO_OLD_RUN_OR_SYNTHESIS')
    with patch.object(m.saved,'run',side_effect=blocked),patch.object(m.saved,'propagate_round',side_effect=blocked):
        result=m.run(tmp_path,path,execute_science=True)
    assert result['status']=='COMMON_IDEAL_LANE_COMPLETE'
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert not stop['failed'] and stop['scientific_units']==61 and stop['call_counts']==m.LIMITS
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    target=m.accuracy.AccuracyTarget(**m.TARGET)
    for arm,key in m.ROUTES.items():
        p=packets[arm];v=result['candidates'][key]
        assert v['actual_CA_achieved'] is v['rigorous_enclosure'] is v['guaranteed_resource_superiority'] is None
        assert v['status']=='CONDITIONAL_IDEAL_CA_PASS'
        assert len(v['rounds'])==13 and len(v['per_shot_costs'])==26
        for row,pf in zip(v['rounds'],p['PF_rounds'],strict=True):
            assert row['raw_signal_error']==pf['pf_operator_error']
            assert row['radius_lower']==1-pf['pf_operator_error']
            assert row['synthesis_zero_measured'] is False and row['law_model']=='deterministic'
            assert row['fixed_statistical_margin'] >= 0
            assert row['shots']==m.accuracy._legacy('rpe_resource_accounting')._hoeffding_shots(
                row['epsilon_coordinate'],target.alpha_axis)
        assert v['energy']['energy_error_bound_ha']==pytest.approx(
            p['df_error_with_guard']+target.beta_total/(target.base_time*(1<<12)))
        assert v['energy']['statistical_failure_upper']<=target.alpha_total
        expected_W=sum(2*r['shots']*(1<<r['m']) for r in v['rounds'])
        assert v['kernel_weight_W']==expected_W
        for metric in m.accuracy.METRICS:
            shell=sum(r['shots']*(2*p['shell_metrics']['head'][metric]
                +p['shell_metrics']['X'][metric]+p['shell_metrics']['Y'][metric]) for r in v['rounds'])
            assert v['cost']['totals'][metric]==expected_W*p['kernel_metrics'][metric]+shell
        assert v['cost']==m.accuracy.aggregate_totals(v['energy'],v['rounds'],v['per_shot_costs'],cost_scope=m.saved.COST_SCOPE)
    assert result['PR_XDF_statistical_superiority'] is result['equivalence'] is result['noninferiority'] is None
    assert all((tmp_path/p).read_bytes()==b for p,b in old.items())
    with pytest.raises(ValueError,match='FRESH'):
        m.run(tmp_path,path,execute_science=True)


@pytest.mark.parametrize('fault',['candidate','DF','PF','metric','source','ground'])
def test_saved_identity_and_provenance_fail_closed(case,tmp_path,fault):
    path,_,_=case(fault)
    with pytest.raises(ValueError):m.run(tmp_path,path,execute_science=True)
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['failed'] and stop['scientific_units']<=3
    assert not (out/'result.json').exists()
    assert stop['call_counts'].get('ideal_round_accuracy',0)==0


def test_infeasible_round_stops_dependent_cost_not_other_candidate(case,tmp_path,monkeypatch):
    path,_,_=case();original=m.ideal_round
    def assess(packet,pf,target):
        if packet['arm']=='CDF' and pf['m']==1:pf=dict(pf,pf_operator_error=.5)
        return original(packet,pf,target)
    monkeypatch.setattr(m,'ideal_round',assess)
    r=m.run(tmp_path,path,execute_science=True)
    a=r['candidates'][m.ROUTES['CDF']];b=r['candidates'][m.ROUTES['XDF']]
    assert len(a['rounds'])==2 and a['rounds'][-1]['status']=='INFEASIBLE_PHASE'
    assert a['cost']['totals'] is a['kernel_weight_W'] is None and a['per_shot_costs']==[]
    assert b['status']=='CONDITIONAL_IDEAL_CA_PASS'
    assert r['comparison']['comparison']['verdict']=='INDETERMINATE_MISSING_QUALIFICATION'
    assert not m.io.read(tmp_path/m.FRESH/'STOP.json')['failed']


def test_correctness_partial_stop_does_not_resume(case,tmp_path,monkeypatch):
    path,_,_=case()
    def fail(*args):raise ValueError('INJECTED_CORRECTNESS_FAIL')
    monkeypatch.setattr(m,'ideal_round',fail)
    with pytest.raises(ValueError,match='INJECTED_CORRECTNESS_FAIL|Dependent work stopped'):
        m.run(tmp_path,path,execute_science=True)
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed'] and stop['first_failure']['kind']=='ideal_round_accuracy'
    assert stop['call_counts'].get('round_serial_cost',0)==0
    assert stop['retry']==stop['resume']==0
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,path,execute_science=True)


def test_fixed_N_unqualified_stops_cost_without_inventing_more_shots(case,tmp_path,monkeypatch):
    path,_,_=case()
    monkeypatch.setattr(m.reused.det,'assess_fixed_round',lambda *a,**k:dict(status='INFEASIBLE_FIXED_SHOTS'))
    r=m.run(tmp_path,path,execute_science=True)
    for v in r['candidates'].values():
        assert len(v['rounds'])==1 and v['rounds'][0]['shots'] is None
        assert v['rounds'][0]['status']=='UNRESOLVED_INTEGER_SHOT_MARGIN'
        assert v['cost']['totals'] is None and v['per_shot_costs']==[]


def test_same_ideal_evidence_semantics_as_existing_same_JU_helper():
    from trottertracks import c_same_ju_strong_d as old
    packet=dict(id='invented',state_model='own_DF_ground_state_oracle',alias_verified=True,
                df_error_with_guard=1e-5)
    pf=[dict(m=i,pf_operator_error=.001*(i+1)/13) for i in range(13)]
    original=old.ideal_plans(pf,packet,lambda k,l,f:f())
    target=m.accuracy.AccuracyTarget(**m.TARGET)
    for row,want in zip(pf,original['rounds'],strict=True):
        got=m.ideal_round(packet,row,target)
        assert all(got[k]==v for k,v in want.items())
