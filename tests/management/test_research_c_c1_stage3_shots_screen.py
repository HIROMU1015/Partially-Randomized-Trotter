"""Synthetic scalar evidence only. Never reanalyse saved molecular values."""
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_stage3_shots_screen as s
from trottertracks import c1_stage3_accuracy as a


def target():
    # Registered parameters are metadata, not scientific saved-cell evidence.
    parent=json.loads((ROOT/'docs/research/research_c_c1_stage3_feasibility_contract_unapproved_20261025.json').read_text())
    return a.AccuracyTarget(**parent['proposal']['accuracy_target'])


def row(A=.5,pf=.01,rte=.02,m=0):
    return dict(m=m,attenuation=A,pf_operator_error=pf,rte_corrected_operator_error=rte)


def dataset(mixed=False):
    records=[]
    for c in s.integrated.candidates():
        bad=mixed and c.family=='PR' and c.K==0 and (c.s,c.r)!=(4,16)
        rows=[row(pf=.4 if bad else .01,m=m) for m in range(13)]
        decisions=[s.feasibility.classify_round(x,target()) for x in rows]
        records.append(dict(id=c.id,candidate=dict(arm=c.arm,family=c.family,s=c.s,r=c.r,K=c.K),
            status='INFEASIBLE' if bad else 'POTENTIALLY_FEASIBLE',df_error_bound_ha=0.,
            rounds=rows,feasibility=decisions,CA_achieved=None,synthesis_signal_error=None))
    return records


def test_regular_geometry_matches_existing_conditional_formula_without_actual_certificate():
    t=target();x=row();r=s.assess_optimistic_round(t,x,target_id='ARTIFICIAL')
    old=a.assess_round(t,a.RoundEvidence(m=0,attenuation=.5,pf_operator_error=.01,
        rte_corrected_operator_error=.02,synthesis_signal_error=0.,finite_law_signal_error=0.,
        target_id='ARTIFICIAL',state_model='own_DF_ground_state_oracle',alias_branch_verified=True,
        iid_per_shot=True,law_model='ideal_finite_RTE_law'))
    assert r['axes']['X']==r['axes']['Y']
    assert int(r['axes']['X']['optimistic_shots_decimal'])==old['shots']
    assert r['normalized_radius_lower']==pytest.approx(.97)
    assert r['beta_deterministic']==old['beta_deterministic']
    assert r['actual_synthesis_signal_error'] is r['actual_law_signal_error'] is None
    assert r['CA_achieved'] is r['guaranteed_actual_shots'] is None
    assert not r['universal_shots_lower_bound'] and 'Counterfactual' in r['assumption']


def test_attenuation_effect_and_pf_rte_errors_retained_in_no_attenuation_reference():
    t=target();r1=s.assess_optimistic_round(t,row(A=1.),target_id='T')
    r2=s.assess_optimistic_round(t,row(A=.5),target_id='T')
    n1=int(r1['axes']['X']['optimistic_shots_decimal']);n2=int(r2['axes']['X']['optimistic_shots_decimal'])
    assert 4*n1-3<=n2<=4*n1
    assert r2['axes']['X']['no_attenuation_shots_decimal']==str(n1)
    clean=s.assess_optimistic_round(t,row(A=1.,pf=0.,rte=0.),target_id='T')
    assert int(clean['axes']['X']['optimistic_shots_decimal'])<n1


@pytest.mark.parametrize('A',[1e-20,1e-160,1e-300,float.fromhex('0x0.0000000000001p-1022')])
def test_huge_counts_exact_rational_ceil_no_clip_and_json_loss(A):
    t=target();r=s.assess_optimistic_round(t,row(A=A),target_id='T');axis=r['axes']['X']
    n=int(axis['optimistic_shots_decimal'])
    assert n>2**63 and axis['integer_exceeds_int64']
    assert axis['arithmetic']=='rational_ceil_binary64_constants'
    epsilon=Fraction.from_float(r['unit_coordinate'])*Fraction.from_float(A)
    numerator=2*Fraction.from_float(math.log(2./t.alpha_axis))
    assert (n-1)*epsilon*epsilon<numerator<=n*epsilon*epsilon
    assert json.loads(json.dumps(r))['axes']['X']['optimistic_shots_decimal']==str(n)
    assert r['coordinate_underflow_binary64']==(A==float.fromhex('0x0.0000000000001p-1022'))


@pytest.mark.parametrize('x,status',[(row(A=0.,pf=None,rte=None),'INFEASIBLE_REGISTERED_BOUND'),
    (row(pf=.4),'INFEASIBLE_REGISTERED_BOUND'),(row(pf=None),'UNRESOLVED_SAVED_VALUES')])
def test_zero_saved_radius_missing_or_phase_failure_never_gets_fake_finite_shots(x,status):
    r=s.assess_optimistic_round(target(),x,target_id='T')
    assert r['status']==status and r['axes'] is None and r['guaranteed_actual_shots'] is None


@pytest.mark.parametrize('x',[row(A=float('nan')),row(A=1.1),row(pf=-1.),row(rte=float('inf'))])
def test_invalid_values_fail(x):
    with pytest.raises(ValueError):s.assess_optimistic_round(target(),x,target_id='T')


def test_normalized_geometry_phase_boundary_is_unresolved_without_threshold_adjustment():
    with patch.object(a,'assess_round',return_value=dict(status='INFEASIBLE_PHASE')):
        r=s.assess_optimistic_round(target(),row(),target_id='T')
    assert r['status']=='UNRESOLVED_NUMERICAL_PHASE_BOUNDARY' and r['axes'] is None


def test_artificial_e2e_all20_13rounds_520axes_282calls_and_no_cost_or_CA(tmp_path):
    forbidden=Mock(side_effect=AssertionError('Forbidden physics/circuit/cost path'))
    with patch.object(a,'assess_energy',forbidden),patch.object(a,'aggregate_totals',forbidden),\
            patch.object(s.feasibility.FeasibilityBackend,'prepare',forbidden):
        result=s.run(lambda:dataset(mixed=True),target(),tmp_path/'out')
    forbidden.assert_not_called()
    assert result['scientific_units']==282 and result['call_counts']==s.call_limits()
    assert result['candidate_status_counts']==dict(OPTIMISTIC_FORMULA_SCREENED=14,EXCLUDED_PRIOR_INFEASIBLE=6)
    assert [x['id'] for x in result['candidates']]==[x.id for x in s.integrated.candidates()]
    assert result['CA_achieved'] is result['guaranteed_actual_shots'] is result['total_resources'] is result['final_resource_ranking'] is None
    assert result['quantum_shots_drawn']==result['circuit_build']==result['new_fit']==result['retry']==0
    for c in result['candidates']:
        record=json.loads((tmp_path/'out/candidates'/(c['id']+'.json')).read_text())
        assert len(record['rounds'])==13
        if c['prior_feasibility']=='POTENTIALLY_FEASIBLE':
            assert int(c['optimistic_total_shots_decimal'])==sum(int(x['axes'][axis]['optimistic_shots_decimal']) for x in record['rounds'] for axis in ('X','Y'))
        else: assert c['optimistic_total_shots_decimal'] is None
    assert json.loads((tmp_path/'out/STOP.json').read_text())['mandatory_stop']
    assert 26*target().alpha_axis==pytest.approx(target().alpha_total)


def test_scientific_failure_preserves_partial_progress_no_retry_or_overwrite(tmp_path):
    records=dataset();records[1]['rounds'][3]['pf_operator_error']=float('nan')
    output=tmp_path/'out'
    with pytest.raises(ValueError):s.run(lambda:records,target(),output)
    progress=output/'candidates'/(records[1]['id']+'.progress.jsonl')
    assert len(progress.read_text().splitlines())==3
    stop=json.loads((output/'STOP.json').read_text());failure=json.loads((output/'failure.json').read_text())
    assert stop['failed'] and stop['scientific_units']==19 and stop['retry']==stop['resume']==0
    assert failure['first_failure']['kind']=='optimistic_round_assessment'
    loader=Mock(side_effect=AssertionError('Must not resume'))
    with pytest.raises(FileExistsError):s.run(loader,target(),output)
    loader.assert_not_called()


def test_unresolved_candidate_kept_and_no_fabricated_total(tmp_path):
    data=dataset();data[0]['rounds'][0]['pf_operator_error']=None
    data[0]['feasibility'][0]=dict(status='UNRESOLVED')
    data[0]['status']='UNRESOLVED'
    result=s.run(lambda:data,target(),tmp_path/'out')
    c=result['candidates'][0]
    assert c['status']=='EXCLUDED_PRIOR_UNRESOLVED' and c['optimistic_total_shots_decimal'] is None
    assert len(result['candidates'])==20 and result['scientific_units']==282
    r=json.loads((tmp_path/'out/candidates'/(c['id']+'.json')).read_text())
    assert r['rounds'][0]['status']=='UNRESOLVED_SAVED_VALUES' and r['rounds'][0]['axes'] is None


def test_saved_status_or_candidate_set_mismatch_stops_without_scientific_extension(tmp_path):
    data=dataset();data[0]['feasibility'][0]['status']='INFEASIBLE'
    with pytest.raises(ValueError,match='SAVED_FEASIBILITY_MISMATCH'):
        s.run(lambda:data,target(),tmp_path/'mismatch')
    with pytest.raises(ValueError,match='REGISTERED_CANDIDATE_ORDER'):
        s.run(lambda:dataset()[:-1],target(),tmp_path/'short')


def test_real_loader_on_artificial_files_only_and_schema_fail(tmp_path):
    data=dataset(mixed=True);folder=tmp_path/'saved';(folder/'candidates').mkdir(parents=True)
    ids=[x['id'] for x in data]
    result=dict(status='COMPLETE_FEASIBILITY_ONLY',candidate_ids=ids,scientific_units=639,
        candidate_statuses=[dict(id=x['id'],status=x['status']) for x in data],
        candidate_status_counts=dict(s.Counter(x['status'] for x in data)))
    prepared=dict(state_preparation='ORACLE_EXCLUDED',sampling_law=dict(model='ideal_finite_RTE_IID'),df_error_bounds_ha=dict(CDF=0.,XDF=0.))
    for name,payload in [('result',result),('prepared',prepared),('STOP',dict(failed=False,mandatory_stop=True))]:
        (folder/(name+'.json')).write_text(json.dumps(payload))
    for x in data:(folder/'candidates'/(x['id']+'.json')).write_text(json.dumps(x))
    verified=dict(root=tmp_path,contract=dict(saved_run='saved',candidate_ids=ids))
    assert s.load_saved(verified)==data
    data[0]['candidate']['K']=77
    (folder/'candidates'/(ids[0]+'.json')).write_text(json.dumps(data[0]))
    with pytest.raises(ValueError,match='SAVED_CANDIDATE_SCHEMA'):s.load_saved(verified)


def test_passive_import_and_cli_execute_requires_fresh_output():
    subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'src');from trottertracks import c1_stage3_shots_screen;assert not any(x in sys.modules for x in ['numpy','scipy','qiskit','ffsim'])"],cwd=ROOT,check=True)
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_research_c_c1_stage3_shots_screen.py'),'--contract','/NEVER_READ','--execute'],capture_output=True,text=True)
    assert p.returncode==2 and 'fresh' in p.stderr


def test_preflight_and_execute_dispatch_artificial_provenance_only(tmp_path):
    parent=json.loads((ROOT/'docs/research/research_c_c1_stage3_feasibility_contract_unapproved_20261025.json').read_text())
    (tmp_path/'parent.json').write_text(json.dumps(parent))
    ids=[x.id for x in s.integrated.candidates()]
    inputs={'saved/'+x:'evidence bytes only; not JSON' for x in ['result.json','prepared.json','STOP.json']}
    inputs.update({'saved/candidates/'+cid+'.json':'not decoded' for cid in ids})
    for path,text in inputs.items():
        p=tmp_path/path;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(text)
    (tmp_path/'source.json').write_text(json.dumps(dict(files={})))
    c=dict(scope=s.SCOPE,call_limits=s.call_limits(),maximum_logical_scientific_units=282,
        candidate_ids=ids,rounds=list(range(13)),retry=0,resume=0,new_fit=0,
        parent_contract='parent.json',parent_contract_sha256=s.io.sha(tmp_path/'parent.json'),
        accuracy_target=parent['proposal']['accuracy_target'],parent_proposal=parent['proposal'],
        runtime_required_packages=parent['runtime_required_packages'],
        physical_conditions=parent['physical_conditions'],source_manifest='source.json',
        source_manifest_sha256=s.io.sha(tmp_path/'source.json'),saved_run='saved',
        saved_inputs={p:s.io.sha(tmp_path/p) for p in inputs})
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(c))
    s.preflight(tmp_path,contract)  # Saved files deliberately not valid JSON.
    with patch.object(s,'load_saved',return_value=dataset()):
        out=tmp_path/s.FAMILY/'ARTIFICIAL'
        result=s.execute(tmp_path,contract,out)
    assert result['scientific_units']==282 and (out/'SOURCE.json').exists()
    with pytest.raises(ValueError,match='FRESH'):s.execute(tmp_path,contract,out)
    c['accuracy_target']=dict(c['accuracy_target'],alpha_axis=.001);contract.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='CHANGED_SCIENTIFIC_CONDITIONS'):s.preflight(tmp_path,contract)
