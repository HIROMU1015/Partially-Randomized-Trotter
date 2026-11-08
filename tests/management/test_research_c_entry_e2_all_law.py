"""Artificial metadata only: never decode H4 or invoke numerical backends."""
from copy import deepcopy
import json
import pytest
from trottertracks import c_entry_e2_all_law_readiness as e
from trottertracks.c1_stage3_qualification import synthesis_envelope


def supplied(model=e.MODELS[0]):
    p = e.missing_packet()
    p['premises'] = {k:True for k in e.PREMISES}
    if model == e.MODELS[0]: p['premises']['actual_joint_stream'] = False
    for part in e.PARTS if model == e.MODELS[1] else e.PARTS[:-1]:
        p['components'][part] = dict(identity=e.target_identity(),law_model=model,
            norm='raw_complex_signal',included_terms=[part],kind='rigorous_bound',
            provenance='ARTIFICIAL_ONLY',coverage='all_trajectories_all_occurrences_and_shells'
            if part in ('synthesis','stream_model') else 'model_reference_all_rounds',
            rows=[dict(m=m,bound=1e-20) for m in range(13)])
    return p


@pytest.mark.parametrize('key,value', [('candidate','E1'),('arm','PR'),('candidate_sha256','CDF0'),
    ('r',32),('relocation','unrelocated'),('ld',3),('J_sha256','wrong'),('U_sha256','wrong'),
    ('original_kappa_export_sha256','wrong'),('original_scalar_array_sha256','wrong'),
    ('original_physical_input_sha256','wrong'),('scientific_SOURCE_commit','old'),
    ('code_SOURCE_sha256','wrong'),('sampling_SOURCE_sha256','wrong'),('compiler',{}),
    ('accuracy',{}),
    ('ground','HF'),('controlled_dimension',72),('rounds',list(range(12)))])
def test_wrong_identity(key,value):
    identity=e.target_identity();identity[key]=value
    with pytest.raises(ValueError,match='IDENTITY'):e.inspect_packet(identity,e.missing_packet())


@pytest.mark.parametrize('model',e.MODELS)
def test_artificial_complete_schema_cannot_qualify(model):
    result=e.inspect_packet(e.target_identity(),supplied(model),model=model)
    assert result['status']=='SCHEMA_ONLY_METHOD_NOT_ESTABLISHED'
    assert result['RPE_accuracy']=='UNQUALIFIED' and result['error_evidence'] is None
    assert result['actual_CA_achieved'] is None and not result['PCG64_joint_stream_qualified']
    assert result['mandatory_stop'] and not result['scientific_execution_ready']


def test_missing_preserved_and_legacy_gate_agrees():
    p=e.missing_packet();before=deepcopy(p)
    r=e.inspect_packet(e.target_identity(),p)
    assert p==before and all(x is None for row in r['round_errors'] for k,x in row.items() if k!='m')
    assert r['scientific_calls']==0 and r['error_evidence'] is None
    assert synthesis_envelope(whole_support_covered=False,numerical_diagnostic_max=1e-30)['synthesis_error_per_kernel'] is None


@pytest.mark.parametrize('premise',e.PREMISES[:-1])
def test_each_missing_premise_stays_unqualified(premise):
    p=supplied();p['premises'][premise]=False
    r=e.inspect_packet(e.target_identity(),p)
    assert premise in r['missing'] and not r['all_law_bound_qualified']


@pytest.mark.parametrize('key,value',[('norm','Ha'),('norm','corrected_signal'),
    ('included_terms',['synthesis','PF']),('included_terms',['synthesis','reference_bridge']),
    ('included_terms',['reference_uncertainty']),('kind','sample_maximum'),
    ('coverage','finite_four_words'),('identity',{}),('law_model','actual_implemented_law'),
    ('provenance',''),('rows',[dict(m=0,bound=-1)]),
    ('rows',[dict(m=m,bound=float('nan')) for m in range(13)])])
def test_invalid_bound_claims(key,value):
    p=supplied();p['components']['synthesis'][key]=value
    with pytest.raises(ValueError):e.inspect_packet(e.target_identity(),p)


@pytest.mark.parametrize('actual_part',['stream','premise'])
def test_iid_does_not_become_actual(actual_part):
    p=supplied()
    if actual_part=='stream':p['components']['stream_model']=supplied(e.MODELS[1])['components']['stream_model']
    else:p['premises']['actual_joint_stream']=True
    with pytest.raises(ValueError,match='IID_GRID'):e.inspect_packet(e.target_identity(),p)


def test_byte_binding_and_no_decode(tmp_path):
    # Deliberately invalid NPZ bytes: preflight must only hash them.
    (tmp_path/'toy.npz').write_bytes(b'not a scientific array')
    (tmp_path/'SOURCE.json').write_text(json.dumps({'files':{'toy.npz':e.io.sha(tmp_path/'toy.npz')}}))
    record=dict(identity=e.target_identity(),saved_files={'SOURCE.json':e.io.sha(tmp_path/'SOURCE.json')},
        code_manifests=['SOURCE.json'],bound_request=e.missing_packet(),case='B',
        whole_P0_scientific_calls=None,new_science_authorized=False)
    path=tmp_path/'readiness.json';path.write_text(json.dumps(record))
    assert e.preflight(tmp_path,'readiness.json')['files_checked']==2
    (tmp_path/'toy.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SAVED_BYTES'):e.preflight(tmp_path,'readiness.json')


def test_preparation_rejects_supplied_values(tmp_path):
    record=dict(identity=e.target_identity(),saved_files={},code_manifests=[],bound_request=supplied(),
        case='B',whole_P0_scientific_calls=None,new_science_authorized=False)
    (tmp_path/'record.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='PREPARATION_ONLY'):e.preflight(tmp_path,'record.json')
