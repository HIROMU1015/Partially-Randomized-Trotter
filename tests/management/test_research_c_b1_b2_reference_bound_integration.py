"""Invented rational records; no H4 arrays or scientific engines."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
from unittest.mock import patch
import pytest
from trottertracks import c_b1_b2_reference_bound_integration as m

ROOT=Path(__file__).resolve().parents[2]


def fixture():
    from test_research_c_b1_b2_scope_integration import records,fixture as inherited
    packet,_,pins,cov=inherited();r=records();rows=[]
    for name,(leg,term) in m.PARTS.items():
        r[name].update(qualification='bound',numerical_qualification='rigorous_enclosure',
            included_error_terms=[term],reference_target='saved_CAR_raw_finite_RTE',proofs_verified_by_schema=False,
            provenance='invented Arb-shaped evidence;not actual H4',
            raw_conversion='uniform over all normalized sector36 states;no saved-state approximation',
            error_model=dict(targets=m.TARGETS,precision_bits=384,rounding_enclosed=True,saved_state_used=False,
                ground_oracle_precision_claimed=False,physical_input_error_included=False,
                generator_construction_error_included=False))
        for row in r[name]['rows']:row.update(bound='1/100000000',value='1/100000000',kind='rigorous_bound',rounding_enclosed=True)
    for i in range(13):
        leg=dict(lower_exact='0',upper_exact='1/100000000',norm='operator_2',upper_method='outward Frobenius;no numerical SVD')
        rows.append(dict(m=i,bridge=deepcopy(leg),reference_uncertainty=deepcopy(leg),total_upper_exact='1/50000000',
            direct_total_diagnostic=deepcopy(leg),rounding_enclosed=True,saved_state_used=False,
            norm_conversion='operator2<=outward_Frobenius;|psi*D psi|<=operator2 for every normalized psi'))
    b2=dict(status='B2_REFERENCE_OPERATOR_BOUNDS_COMPLETE',model=m.previous.MODEL,targets=m.TARGETS,
        records={k:r[k] for k in m.PARTS},rounds=rows,Stage_C_ready=False,
        categorical_unitary_trajectory_premise_qualified=False,scientific_input_uncertainty_covered=False,
        saved_ground_precision_qualified=False,generator_construction_uncertainty_covered=False,
        PCG64_joint_stream_qualified=False,actual_CA_achieved=None,mandatory_stop=True)
    a=dict(status='IID_GRID_REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE',model=m.previous.MODEL,
        candidate=m.previous.identity()['candidate'],PR_scientific_SOURCE=m.previous.identity()['PR_scientific_SOURCE'],
        PCG64_joint_stream_qualified=False,rounds=[dict(m=i,model=m.previous.MODEL,conditional_categorical=dict(m=i,
            events=32*(1<<i),random_model=m.previous.MODEL['id'],conditional_raw_bound_exact='1/1000000',
            reference_bridge_included=False,synthesis_included=False,actual_implemented_law_error=None)) for i in range(13)])
    return packet,b2,a,r['synthesis'],cov


def test_exact_endpoints_and_qualifications_preserved_no_calculation():
    p,b,a,s,cov=fixture();before=deepcopy(b)
    with patch.object(m.previous.iid,'qualify',side_effect=AssertionError('NO_NEW_BOUND_SUM')):
        attached,q=m.integrate(p,b,a,pins=m.publications(),coverage=cov)
    assert b==before and attached['error_evidence'] is None and not q['eligible']
    assert q['status']=='NOT_READY_BOUND_INPUTS' and q['Stage_C_calls']==0
    for name in m.PARTS:
        assert q['records'][name]['rows']==before['records'][name]['rows']
        assert q['records'][name]['numerical_qualification']=='rigorous_enclosure'
        assert q['records'][name]['space']=='sector36_reference'
    assert q['records']['synthesis'] is None and not q['categorical_unitary_trajectory_premise_qualified']
    with patch.object(m.previous.iid,'assess',side_effect=AssertionError('NO_STAGE_C')):
        assert m.previous.assess_if_ready(None,attached,q)['energy'] is None


def test_even_synthetic_P0_bound_and_true_external_flag_cannot_waive_fixed_categorical_gap():
    p,b,a,s,cov=fixture();cov['categorical_unitary_trajectory_premise_qualified']=True
    with patch.object(m.previous.iid,'qualify',side_effect=AssertionError('NO_PREMISE_ESCALATION')):
        _,q=m.integrate(p,b,a,pins=m.publications(),synthesis=s,coverage=cov)
    assert not q['eligible'] and 'categorical_unitary_trajectory_premise' in q['missing']


@pytest.mark.parametrize('fault',['candidate','PR_SOURCE','Stage_A_SOURCE','model','compiler','round','negative','float',
 'value','space','norm','conversion','estimate','rounding','total_component','total_value','overlap','bridge_includes_U','precision','input_error'])
def test_wrong_B2_identity_round_qualification_and_partition_rejected(fault):
    p,b,a,s,cov=fixture();r=b['records']['reference_bridge']
    if fault=='candidate':r['identity']['candidate']='WRONG'
    if fault=='PR_SOURCE':r['identity']['PR_scientific_SOURCE']='0'*40
    if fault=='Stage_A_SOURCE':r['identity']['stage_A_scientific_SOURCE']='0'*40
    if fault=='model':b['model']=dict(b['model'],id='PCG64')
    if fault=='compiler':r['identity']['compiler']=dict(r['identity']['compiler'],optimization_level=3)
    if fault=='round':r['rows'].pop()
    if fault=='negative':r['rows'][0]['bound']='-1/100'
    if fault=='float':r['rows'][0]['bound']=1e-8
    if fault=='value':r['rows'][0]['value']='1'
    if fault=='space':r['space']='full512_native'
    if fault=='norm':r['norm']='operator_2'
    if fault=='conversion':r['raw_conversion']='full512 with an unqualified non-normalized state'
    if fault=='estimate':r['rows'][0]['kind']='numerical_estimate'
    if fault=='rounding':r['rows'][0]['rounding_enclosed']=False
    if fault=='total_component':b['records']['total']=deepcopy(r)
    if fault=='total_value':b['rounds'][0]['total_upper_exact']='1'
    if fault=='overlap':b['records']['reference_uncertainty']['included_error_terms']=r['included_error_terms']
    if fault=='bridge_includes_U':r['rows'][0]['reference_uncertainty_included']=True
    if fault=='precision':r['error_model']['precision_bits']=53
    if fault=='input_error':r['error_model']['generator_construction_error_included']=True
    with pytest.raises(ValueError):m.b2_records(b,pins=m.publications())


@pytest.mark.parametrize('key',['B1_result','B2_result','B2_SOURCE','B2_implementation','Stage_A'])
def test_fixed_publication_source_roles(key):
    _,b,_,_,_=fixture();pins=m.publications();pins[key]='wrong'
    with pytest.raises(ValueError,match='SOURCE_ROLES'):m.b2_records(b,pins=pins)


@pytest.mark.parametrize('flag',['Stage_C_ready','categorical_unitary_trajectory_premise_qualified',
 'scientific_input_uncertainty_covered','saved_ground_precision_qualified',
 'generator_construction_uncertainty_covered','PCG64_joint_stream_qualified'])
def test_bound_does_not_cover_whole_physical_native_or_random_stream(flag):
    _,b,_,_,_=fixture();b[flag]=True
    with pytest.raises(ValueError,match='OVERCLAIM'):m.b2_records(b,pins=m.publications())


@pytest.mark.parametrize('fault',['candidate','model','SOURCE','round','double','unknown'])
def test_Stage_A_categorical_identity_and_single_leg(fault):
    _,_,a,_,_=fixture()
    if fault=='candidate':a['candidate']='other'
    if fault=='model':a['model']=dict(a['model'],independent_events=False)
    if fault=='SOURCE':a['PR_scientific_SOURCE']='other'
    if fault=='round':a['rounds'][0]['conditional_categorical']['m']=2
    if fault=='double':a['rounds'][0]['conditional_categorical']['reference_bridge_included']=True
    if fault=='unknown':a['rounds'][0]['conditional_categorical']['conditional_raw_bound_exact']=None
    with pytest.raises(ValueError):m.categorical_record(a)


def test_no_science_entry_and_passive_import():
    with pytest.raises(PermissionError):m.run_science()
    with pytest.raises(NotImplementedError):m.run_science(execute_science=True)
    code="import sys;from trottertracks import c_b1_b2_reference_bound_integration;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','flint','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


@pytest.mark.parametrize('fault',['none','fresh','resume','input','source'])
def test_passive_fresh_drift_and_retry_resume(tmp_path,monkeypatch,fault):
    p,b,a,s,cov=fixture()
    def save(path,x):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2)+'\n')
    paths=m.evidence_paths();source=dict(implementation_commit=m.B2_IMPLEMENTATION,files={})
    save(tmp_path/paths['technical_SOURCE'],source)
    save(tmp_path/paths['SOURCE'],dict(source,execution_commit=m.B2_SOURCE))
    contract=dict(implementation_source_commit=m.B2_IMPLEMENTATION,source_manifest_sha256=m.io.sha(tmp_path/paths['technical_SOURCE']),
        candidate_sha256=m.previous.identity()['candidate'],model=m.previous.MODEL,
        stage_A_scientific_SOURCE=m.previous.STAGE_A_SOURCE,stage_A_result=m.previous.STAGE_A,
        targets=m.TARGETS,saved_inputs={})
    save(tmp_path/paths['contract'],contract);save(tmp_path/paths['result'],b)
    save(tmp_path/paths['STOP'],dict(failed=False,scientific_units=27,call_counts=dict(rounds=27),first_failure=None,retry=0,resume=0,mandatory_stop=True))
    save(tmp_path/'a.json',a);save(tmp_path/'astop.json',dict(failed=False,scientific_units=38,retry=0,resume=0,mandatory_stop=True))
    save(tmp_path/'b1.json',dict(method_status='P0_UNIFORM_NOT_ESTABLISHED',actual_H4_scientific_calls=0))
    save(tmp_path/'source.json',dict(files={}))
    c=dict(m.fixed(),source_manifest='source.json',source_manifest_sha256=m.io.sha(tmp_path/'source.json'),B2_evidence=paths,
        B2_contract_sha256=m.io.sha(tmp_path/paths['contract']),stage_A_result='a.json',stage_A_STOP='astop.json',B1_STOP='b1.json',
        saved_inputs={str(x.relative_to(tmp_path)):m.io.sha(x) for x in tmp_path.rglob('*.json')})
    if fault=='fresh':(tmp_path/c['fresh_output']).mkdir(parents=True)
    if fault=='resume':c['resume']=1
    if fault=='input':save(tmp_path/'a.json',dict(a,candidate='other'))
    if fault=='source':save(tmp_path/'source.json',dict(files={},changed=True))
    save(tmp_path/'design.json',c)
    if fault=='none':assert m.preflight(tmp_path,tmp_path/'design.json')['Stage_C_calls']==0
    else:
        with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'design.json')
