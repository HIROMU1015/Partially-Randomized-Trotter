"""Invented numerical fixtures with the saved Stage A JSON schema."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from trottertracks import c_per_word_range_schema as m
from trottertracks.c_iid_grid_bridge import MODEL

ROOT = Path(__file__).resolve().parents[2]
STAGE_A = 'artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01'


def save(path, data):
    path.write_text(json.dumps(data) + '\n')


def metadata_fixture(root, atoms=None, candidate='invented'):
    # Keys and types follow the actual Stage A producer. No saved numerical
    # observation is copied into an artificial assertion or norm calculation.
    if atoms is None:
        atoms = [dict(atom_id=f'toy:{i}', frame_id='toy:F2', support=[0], coefficient='1/8')
                 for i in range(216)]
    rows = [dict(a, signed=True, hermitian={}, involution={}) for a in atoms]
    report = dict(status='K0_FINITE_MEAN_NUMERICAL_PASS', unitary_events=[], mean_identity={},
                  angle=.015625, normalization=1.0, scalar_phase_excluded=True,
                  reference_uncertainty=None, rigorous_operator_bound=None)
    stage_a = dict(status='IID_GRID_REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE', model=deepcopy(MODEL),
                   rounds=[], candidate=candidate, PR_scientific_SOURCE=m.previous.prior.integration.previous.identity()['PR_scientific_SOURCE'],
                   Stage_B='P0_UNIFORM_NOT_ESTABLISHED', Stage_C='NOT_READY_BOUND_INPUTS',
                   PCG64_joint_stream_qualified=False, PCG64_joint_stream_error=None,
                   all_law_synthesis=None, reference_numerical_uncertainty=None, error_evidence=None,
                   actual_CA_achieved=None, rigorous_CA=None, new_sampling=0, new_native_build_compile=0,
                   new_PF_RTE_generation=0, ground_eigensolve=0, shots_evaluation=0, mandatory_stop=True)
    counts = [(1 << 53) // 216] * 216
    counts[-1] += (1 << 53) - sum(counts)
    grid = dict(probabilities=[], cdf=[], counts=counts, denominator=1 << 53,
                probabilities_exact=[], zero_random_draws=True, mapping='searchsorted_right', bits=53)
    table = dict(status='INVENTED', reference='INVENTED', ideal_probabilities=[], first_probabilities=[],
                 sampler_probabilities=[], cdf=[], grid=grid, eta_exact='0', eta_upper_float=0.0,
                 stages_TV={}, lost_positive_ideal_atoms=[], pcg64_IID_proven=False,
                 coefficient_generation_error=None, reference_bridge_error=None,
                 angle_normalization_error=None, actual_implemented_law_error=None)
    records = dict(event_mean=dict(model='ideal_saved_atom', report=report),
                   grid_event_mean=dict(model=MODEL['id'], report=deepcopy(report)), stage_A_result=stage_a,
                   grid_table=table, atoms=dict(status='INVENTED', atoms=rows, raw_tail={}, exact_mass='1',
                   float_mass=1.0, coefficient_identity_rounding_enclosed=False, reference_uncertainty=None))
    c = dict(identity=deepcopy(m.previous.prior.integration.previous.identity()))
    c['identity']['candidate'] = candidate
    for role, record in records.items():
        c[role] = role + '.json'
        save(root / c[role], record)
    return c


def test_artificial_fixture_matches_actual_saved_JSON_schema(tmp_path):
    c = metadata_fixture(tmp_path)
    paths = dict(event_mean='ideal_event_mean.json', grid_event_mean='grid_event_mean.json',
                 stage_A_result='result.json', grid_table='categorical_table.json', atoms='atoms.json')
    for role, name in paths.items():
        actual = m.io.read(ROOT / STAGE_A / name)
        toy = m.io.read(tmp_path / c[role])
        assert {k: type(v) for k, v in toy.items()} == {k: type(v) for k, v in actual.items()}
        if 'report' in toy:
            assert {k: type(v) for k, v in toy['report'].items()} == {k: type(v) for k, v in actual['report'].items()}
    assert m.saved_metadata(tmp_path, c)['angle'].hex() == (.015625).hex()


@pytest.mark.parametrize('role,path,value,error', [
    ('event_mean', ('model',), 'WRONG_TAG', 'IDEAL_EVENT_TAG'),
    ('event_mean', ('model',), MODEL, 'IDEAL_EVENT_TAG'),
    ('event_mean', ('model',), None, 'IDEAL_EVENT_TAG'),
    ('grid_event_mean', ('model',), 'WRONG_IID', 'GRID_EVENT_MODEL_ID'),
    ('grid_event_mean', ('model',), MODEL, 'GRID_EVENT_MODEL_ID'),
    ('stage_A_result', ('model',), MODEL['id'], 'STAGE_A_IID_MODEL'),
    ('stage_A_result', ('model', 'id'), 'WRONG_IID', 'STAGE_A_IID_MODEL'),
    ('stage_A_result', ('model', 'bits'), 53.0, 'STAGE_A_IID_MODEL'),
    ('stage_A_result', ('model', 'independent_events'), 1, 'STAGE_A_IID_MODEL'),
    ('stage_A_result', ('model', 'independent_events'), False, 'STAGE_A_IID_MODEL'),
    ('stage_A_result', ('model', 'extra'), True, 'STAGE_A_IID_MODEL'),
    ('stage_A_result', ('candidate',), 'different', 'STAGE_A_TARGET_IDENTITY'),
    ('event_mean', ('report', 'angle'), '.015625', 'REPORT_SCHEMA'),
    ('event_mean', ('report', 'angle'), 1, 'REPORT_SCHEMA'),
    ('event_mean', ('report', 'angle'), float('nan'), 'REPORT_SCHEMA'),
    ('event_mean', ('report',), [], 'REPORT_SCHEMA'),
    ('grid_event_mean', ('report', 'angle'), .03125, 'ANGLE_IDENTITY'),
    ('grid_event_mean', ('report', 'scalar_phase_excluded'), False, 'REPORT_SCHEMA'),
    ('grid_table', ('grid', 'bits'), 53.0, 'GRID_TABLE_SCHEMA'),
    ('grid_table', ('grid', 'denominator'), str(1 << 53), 'GRID_TABLE_SCHEMA'),
    ('grid_table', ('grid', 'mapping'), 'searchsorted_left', 'GRID_TABLE_SCHEMA'),
    ('grid_table', ('grid', 'zero_random_draws'), 1, 'GRID_TABLE_SCHEMA'),
])
def test_wrong_event_IID_model_or_JSON_type_is_explicit_FAIL(tmp_path, role, path, value, error):
    c = metadata_fixture(tmp_path)
    data = m.io.read(tmp_path / c[role])
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    save(tmp_path / c[role], data)
    with pytest.raises(ValueError, match=error):
        m.saved_metadata(tmp_path, c)


@pytest.mark.parametrize('model', [MODEL['id'], dict(MODEL, bits=53.0), dict(MODEL, independent_events=1)])
def test_contract_model_types_are_not_coerced(tmp_path, model):
    c = metadata_fixture(tmp_path)
    c['identity']['model'] = model
    with pytest.raises(ValueError, match='CONTRACT_IID_MODEL'):
        m.saved_metadata(tmp_path, c)


@pytest.mark.parametrize('fault', ['duplicate', 'bad_support_type', 'count_bool', 'counts_length'])
def test_adjacent_catalog_grid_schema_gates(tmp_path, fault):
    c = metadata_fixture(tmp_path)
    role = 'atoms' if fault in ('duplicate', 'bad_support_type') else 'grid_table'
    data = m.io.read(tmp_path / c[role])
    if fault == 'duplicate':
        data['atoms'][-1]['atom_id'] = data['atoms'][0]['atom_id']
    if fault == 'bad_support_type':
        data['atoms'][0]['support'] = [True]
    if fault == 'count_bool':
        data['grid']['counts'][0] = True
    if fault == 'counts_length':
        data['grid']['counts'].pop()
    save(tmp_path / c[role], data)
    with pytest.raises(ValueError):
        m.saved_metadata(tmp_path, c)


def fake_contract(root):
    c = metadata_fixture(root)
    # Use the fixed candidate metadata identity; there is no candidate NPZ.
    candidate = m.previous.prior.integration.previous.identity()['candidate']
    stage_a = m.io.read(root / c['stage_A_result'])
    stage_a['candidate'] = candidate
    save(root / c['stage_A_result'], stage_a)
    c['identity']['candidate'] = candidate
    (root / 'code').write_text('invented source\n')
    save(root / 'SOURCE.json', dict(files={'code': m.io.sha(root / 'code')}))
    c.update(base_commit=m.BASE, schema=m.SCHEMA, target=m.previous.method.TARGET,
             precision_bits=192, retry=0, resume=0, mandatory_stop=True, packages={},
             maximum_logical_units=m.MAX_CALLS, source_manifest='SOURCE.json',
             source_manifest_sha256=m.io.sha(root / 'SOURCE.json'),
             saved_inputs={c[role]: m.io.sha(root / c[role]) for role in
                           ('atoms', 'event_mean', 'grid_event_mean', 'grid_table', 'stage_A_result')},
             stages=dict(R=dict(scope=m.SCOPE, call_limits=m.LIMITS, maximum_logical_units=m.MAX_CALLS,
                                fresh_output=m.FAMILY + '/stageR-run01', executable=True,
                                entry=m.ENTRY + ' --execute-stage-R'),
                         P=dict(executable=False), C=dict(executable=False), A=dict(executable=False)))
    save(root / 'contract.json', c)
    return c


@pytest.mark.parametrize('fault', ['none', 'source', 'input', 'event_tag', 'model', 'fresh',
                                  'retry', 'resume', 'old_contract', 'unpin_metadata', 'Stage_C'])
def test_passive_preflight_rejects_before_scientific_import(tmp_path, fault):
    c = fake_contract(tmp_path)
    if fault == 'source':
        (tmp_path / 'code').write_text('changed')
    if fault == 'input':
        (tmp_path / c['event_mean']).write_text('{}')
    if fault == 'event_tag':
        data = m.io.read(tmp_path / c['event_mean'])
        data['model'] = deepcopy(MODEL)
        save(tmp_path / c['event_mean'], data)
        c['saved_inputs'][c['event_mean']] = m.io.sha(tmp_path / c['event_mean'])
    if fault == 'model':
        c['identity']['model']['bits'] = 53.0
    if fault == 'fresh':
        (tmp_path / c['stages']['R']['fresh_output']).mkdir(parents=True)
    if fault in ('retry', 'resume'):
        c[fault] = 1
    if fault == 'old_contract':
        c['stages']['R']['scope'] = m.previous.SCOPE
    if fault == 'unpin_metadata':
        c['saved_inputs'].pop(c['grid_event_mean'])
    if fault == 'Stage_C':
        c['stages']['C']['executable'] = True
    save(tmp_path / 'contract.json', c)
    if fault == 'none':
        r = m.preflight(tmp_path, tmp_path / 'contract.json')
        assert r['scientific_calls'] == r['arrays_decoded'] == 0 and r['Stage_C_ready'] is False
    else:
        with pytest.raises(ValueError):
            m.preflight(tmp_path, tmp_path / 'contract.json')


def test_real_frozen_preflight_cold_with_numerical_imports_and_execution_blocked():
    code = """
import sys
class Block:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','flint','ffsim','pyscf','openfermion'):
   raise AssertionError('NO_SCIENTIFIC_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
from pathlib import Path
from trottertracks import c_per_word_range_schema as m
m.run=lambda *a,**k: (_ for _ in ()).throw(AssertionError('NO_SCIENTIFIC_EXECUTION'))
r=m.preflight(Path.cwd(),m.CONTRACT)
assert r['scientific_calls']==r['arrays_decoded']==0
assert r['Stage_R_maximum_logical_units']==256 and r['Stage_C_ready'] is False
"""
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


def test_call_adapter_reuses_orchestration_and_replaces_only_event_catalog(tmp_path, monkeypatch):
    c = metadata_fixture(tmp_path)
    arm = NS(plan=NS(atoms=tuple(NS(**row) for row in m.io.read(tmp_path / c['atoms'])['atoms'])))
    observed = []

    def legacy(root, contract, out, call):
        call('native_model', 'toy', lambda: arm)
        call('evidence_load', 'unchanged', lambda: 7)
        return call('event_catalog', 'V2', lambda: pytest.fail('OLD_SCHEMA_CLOSURE_MUST_NOT_RUN'))

    def call(kind, label, fn):
        observed.append(kind)
        return fn()

    monkeypatch.setattr(m.previous, 'execute_range', legacy)
    result = m.execute_range(tmp_path, c, tmp_path, call)
    assert observed == ['native_model', 'evidence_load', 'event_catalog']
    assert len(result['catalog']) == 216
    assert all(event.rotation_angle == .015625 for event in result['catalog'].values())


@pytest.mark.parametrize('fault', ['tag', 'signed_atom'])
def test_catalog_failure_stops_dependent_calls_and_preserves_failure(tmp_path, monkeypatch, fault):
    from trottertracks.c1_stage3_integrated import Calls
    c = metadata_fixture(tmp_path)
    rows = m.io.read(tmp_path / c['atoms'])['atoms']
    arm = NS(plan=NS(atoms=tuple(NS(**row) for row in rows)))
    if fault == 'tag':
        data = m.io.read(tmp_path / c['event_mean'])
        data['model'] = MODEL
        save(tmp_path / c['event_mean'], data)
    else:
        arm.plan.atoms[0].coefficient = '1/4'

    def legacy(root, contract, out, call):
        call('native_model', 'toy', lambda: arm)
        call('event_catalog', 'V2', lambda: pytest.fail('NO_OLD_SCHEMA_FALLBACK'))
        call('event_norm', 'dependent', lambda: pytest.fail('NO_DEPENDENT_WORK'))

    monkeypatch.setattr(m.previous, 'execute_range', legacy)
    calls = Calls(tmp_path, m.LIMITS, m.MAX_CALLS)
    with pytest.raises(ValueError):
        m.execute_range(tmp_path, c, tmp_path, calls)
    assert calls.stop.is_set() and calls.first_failure['kind'] == 'event_catalog'
    assert calls.counts == dict(native_model=1, event_catalog=1)


@pytest.mark.parametrize('failure', [False, True])
def test_mock_lifecycle_saves_STOP_and_no_retry(tmp_path, monkeypatch, failure):
    c = fake_contract(tmp_path)
    monkeypatch.setattr(m, 'preflight', lambda *a: {})

    def toy(root, contract, out, calls):
        def body():
            if failure:
                raise ValueError('INVENTED_CORRECTNESS_FAIL')
            return {'status': 'INVENTED_COMPLETE'}
        return calls('event_catalog', 'invented only', body)

    if failure:
        with pytest.raises(ValueError, match='INVENTED_CORRECTNESS_FAIL'):
            m.run(tmp_path, tmp_path / 'contract.json', execute_science=True, executor=toy)
    else:
        assert m.run(tmp_path, tmp_path / 'contract.json', execute_science=True, executor=toy)['status'] == 'INVENTED_COMPLETE'
    out = tmp_path / c['stages']['R']['fresh_output']
    stop = m.io.read(out / 'STOP.json')
    assert stop['scientific_units'] == 1 and stop['mandatory_stop'] is True
    assert stop['retry'] == stop['resume'] == 0
    assert bool(stop['failed']) is failure
    with pytest.raises(FileExistsError):
        m.run(tmp_path, tmp_path / 'contract.json', execute_science=True, executor=toy)
    with pytest.raises(PermissionError):
        m.run(tmp_path, tmp_path / 'contract.json')


def test_artificial_complete_stage_R_uses_actual_schema_and_exact_256_units(tmp_path, monkeypatch):
    import numpy as np
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks.c1_stage3_integrated import Calls
    J = np.full((8, 4, 4), .0009765625)
    U = np.array([np.eye(4)] * 8)
    p = dict(h=np.diag([.125, .25, .375, .5]), g=np.zeros((4,) * 4), constant=np.asarray(.125))
    np.savez(tmp_path / 'input.npz', **p)
    np.savez(tmp_path / 'candidate.npz', J=J, U=U)
    digest = m.io.sha(tmp_path / 'candidate.npz')
    arm = b.prepare_arm(**p, J=J, U=U, role='proposed', source_identity=digest, input_identity='invented', ld=2)
    atoms = [dict(atom_id=a.atom_id, frame_id=a.frame_id, support=list(a.support), coefficient=str(Fraction(a.coefficient)))
             for a in arm.plan.atoms]
    c = metadata_fixture(tmp_path, atoms, digest)
    c.update(parent_roles=dict(input='input.npz', candidate='candidate.npz', sector='sector.npz', basis='basis.npz'),
             saved_inputs={'input.npz': m.io.sha(tmp_path / 'input.npz')})
    np.savez(tmp_path / 'sector.npz', generators=np.zeros((9, 36, 36)))
    np.savez(tmp_path / 'basis.npz', sector_basis_indices=np.arange(36))
    monkeypatch.setattr(m.previous.prior, 'evidence', lambda *a: {'status': 'INVENTED'})
    monkeypatch.setattr(b, 'build_kernel', lambda *a, **k: pytest.fail('NO_NATIVE_BUILD'))
    out = tmp_path / 'output'
    out.mkdir()
    calls = Calls(out, m.LIMITS, m.MAX_CALLS)
    result = m.execute_range(tmp_path, c, out, calls)
    assert calls.counts == m.LIMITS and sum(calls.counts.values()) == 256
    assert calls.first_failure is None and result['population']['d_range_upper_exact'] == '2'
    assert result['head_binding']['additional_head_bridge_required'] is True
    assert result['new_sampling'] == result['native_build_compile'] == 0 and result['Stage_C_ready'] is False
