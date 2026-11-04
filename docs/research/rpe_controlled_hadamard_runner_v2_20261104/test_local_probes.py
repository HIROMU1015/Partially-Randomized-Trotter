"""Only explicit 2-mode artificial fixtures, never construct_context()/run()."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from flint import ctx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT/'src')]
import probe_adapter as adapter
import reference_adapter as ref
import run_local_probes as runner


def definition(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def toy():
    previous = definition('previous_artificial_circuit_tests',
        ROOT/'docs/research/rpe_controlled_hadamard_preparation_20261104/test_circuit_semantics.py')
    exporter = definition('prior_export_definitions_only',
        ROOT/'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/export_adapter.py')
    h, prep = previous.artificial_preparation.__wrapped__()
    ops = previous.signed_operators(prep)
    indices = previous.noncommuting_pair(prep, ops)
    # Deliberately unnormalized raw vector tests the normalized-ray convention.
    export = exporter.export_preparation(prep, np.array([1., 1j]), [1, 2],
        n_qubits=2, n_electrons=1, delta=.17)
    # Match real JSON serialization (tuples become lists).
    export = json.loads(json.dumps(export))
    from trotterlib.research_direction_sequence_policy import register_support_restricted_bases
    support, _ = register_support_restricted_bases(h, prep)
    return prep, export, indices, support


def parent(prep, r):
    tau = prep.rte_preparation.symbolic_tail.lambda_r*(.17/r)
    return {'m': 0, 'r': r, 'q': 131072, 'K': 2, 'delta': '.17',
            'signal_tau64': str(tau), 'model_tau64': str(math.nextafter(tau, math.inf)),
            'axes': {'cosine': {'N': '7'}, 'sine': {'N': '7'}}, 'beta': {'stat': '.2'}}


def job(toy, r=2):
    prep, export, indices, support = toy
    record = {'counts_started': {}}
    result = runner.prepare_jobs(prep, support, export, [parent(prep, r)], record, indices=indices)
    return result[0], record


@pytest.mark.parametrize('r', (1, 2, 4, 8, 16, 32))
def test_fixed_profile_order_and_signal_literal(toy, r):
    prep, _, indices, _ = toy
    row = parent(prep, r)
    config, law = adapter.signal_config(prep, row)
    assert config.dimensionless_step_time.hex() == float(row['signal_tau64']).hex()
    assert config.dimensionless_step_time.hex() != float(row['model_tau64']).hex()
    ((q0, name0, zero), (q2, name2, stress)) = adapter.profiles(prep.rte_preparation.symbolic_tail.components, law, r, indices)
    assert (q0, name0, len(zero)) == (1, 'all0', r)
    assert (q2, name2, len(stress)) == (2, 'stress', 2*r)
    assert all(e.taylor_order == 0 for e in zero)
    assert sum(e.taylor_order == 2 for e in stress) == (2 if r == 1 else 3)
    cs = prep.rte_preparation.symbolic_tail.components
    assert stress[0].rotation_component_id == cs[indices[0]].component_id
    assert stress[0].product_component_ids == tuple(cs[i].component_id for i in reversed(indices))
    assert stress[-1].product_component_ids == (cs[indices[1]].component_id,)*2


def test_tau_close_enough_for_legacy_is_still_rejected(toy):
    prep = toy[0]
    row = parent(prep, 2); row['signal_tau64'] = row['model_tau64']
    with pytest.raises(ValueError, match='no model-tau fallback'):
        adapter.signal_config(prep, row)


@pytest.mark.parametrize('r', (1, 2, 4, 8, 16, 32))
def test_complete_generic_pipeline_and_exact_call_counts(toy, r):
    prep, _, _, support = toy
    supplied, record = job(toy, r)
    before = supplied['psi'].copy()
    output = adapter.evaluate_group(supplied, preparation=prep, support=support)
    assert output['pass'], output.get('error')
    np.testing.assert_array_equal(supplied['psi'], before)
    assert output['counts_started'] == {'uncontrolled_builds': 2, 'uncontrolled_actions': 2,
        'controlled_raw_concatenation_builds': 2, 'controlled_boundary_optimized_builds': 2,
        'controlled_branch_actions': 8, 'wrapper_builds': 4, 'ideal_mean_actions': 4}
    assert record['counts_started'] == {'reference_half_pairs': 1, 'reference_components': 2,
                                       'config_replays': 1, 'reference_trajectories': 2}
    assert len(output['_arrays']) == 16
    assert output['probes'][0]['parent']['q'] == 131072  # Metadata preserved, never used to build.
    assert [p['q_probe'] for p in output['probes']] == [1, 2]
    assert all(p['pass'] for p in output['probes'])


def test_reference_does_not_use_diagonal_primitive_generator(toy, monkeypatch):
    from trotterlib.df_trotter import ops
    def forbidden(*args, **kwargs):
        raise AssertionError('primitive generator reached independent reference')
    monkeypatch.setattr(ops, 'one_body_diagonal_primitives', forbidden)
    monkeypatch.setattr(ops, 'df_squared_diagonal_primitives', forbidden)
    supplied, _ = job(toy, 2)
    assert np.isfinite(supplied['probes'][0]['target']).all()


def test_negative_or_changed_reference_phase_is_not_aligned_away(toy):
    supplied, _ = job(toy)
    supplied['probes'][0]['target'] *= -1
    output = adapter.evaluate_group(supplied, preparation=toy[0], support=toy[3])
    assert not output['pass']
    assert not output['probes'][0]['checks']['uncontrolled']['pass']
    assert 'r2_all0_uncontrolled' in output['_arrays']
    assert 'controlled_branch_actions' not in output['counts_started']


def test_wrong_sine_signal_stops_and_preserves_failed_mean(toy):
    supplied, _ = job(toy)
    supplied['probes'][0]['signal']['imag'] *= -1
    output = adapter.evaluate_group(supplied, preparation=toy[0], support=toy[3])
    assert not output['pass']
    assert not output['probes'][0]['axes']['sine']['pass']
    assert output['counts_started']['ideal_mean_actions'] == 2
    assert len(output['probes']) == 1


def test_reference_uncertainty_cannot_be_ignored():
    assert ref.scalar_residual(.1, .1, '1e-20')['pass']
    assert not ref.scalar_residual(.1, .1, '1e-9')['pass']
    assert not ref.vector_residual(np.array([1j]), np.array([1j]), '1e-9')['pass']
    with pytest.raises(ValueError):
        ref.scalar_residual(math.nan, 0., '0')


def test_runtime_gate_parameter_substitution_is_rejected(toy):
    prep, export, indices, _ = toy
    bad = deepcopy(export)
    used = prep.rte_preparation.symbolic_tail.components[indices[0]].basis_id
    bad['bases'][used]['operations'][0]['params'][0] = (.8).hex()
    with pytest.raises(ValueError, match='reference gate'):
        ref.build_reference(prep, bad, indices, .17)


def test_normalized_ray_scale_and_raw_bytes(toy):
    prep, export, indices, _ = toy
    ctx.prec = 96
    original = deepcopy(export)
    scaled = deepcopy(export)
    scaled['state'] = [[(3*float.fromhex(re)).hex(), (3*float.fromhex(im)).hex()] for re, im in export['state']]
    a = ref.build_reference(prep, export, indices, .17)
    b = ref.build_reference(prep, scaled, indices, .17)
    av, _ = ref.to_float_vector(a['state'], a); bv, _ = ref.to_float_vector(b['state'], b)
    np.testing.assert_array_equal(av, bv)
    assert export == original


def test_sibling_stop_prevents_next_action(toy):
    supplied, _ = job(toy)
    stop = threading.Event(); stop.set()
    output = adapter.evaluate_group(supplied, preparation=toy[0], support=toy[3], stop=stop)
    assert not output['pass']
    assert output['counts_started'] == {}


def test_spawn_group_matches_serial_artificial_only(toy):
    supplied, _ = job(toy)
    prep, _, _, support = toy
    serial = adapter.evaluate_group(supplied, preparation=prep, support=support)
    context = multiprocessing.get_context('spawn'); stop = context.Event()
    with ProcessPoolExecutor(max_workers=2, mp_context=context,
            initializer=adapter.initialize_worker, initargs=(prep, support, stop)) as pool:
        parallel = pool.submit(adapter.evaluate_group, supplied).result()
    assert parallel['pass'], parallel.get('error')
    assert parallel['probes'] == serial['probes']
    assert parallel['counts_started'] == serial['counts_started']


def test_default_main_cannot_enter_scientific_execution(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('real execution reached')
    monkeypatch.setattr(runner, 'run', forbidden)
    monkeypatch.setattr(runner, 'construct_context', forbidden)
    monkeypatch.setattr(runner, 'static_inputs', lambda: (None, [], None))
    assert runner.main([]) == 0
    assert json.loads(capsys.readouterr().out)['new_H4_scientific_calls'] == 0


def test_long_q_request_and_missing_output_are_rejected(toy):
    prep, _, indices, _ = toy
    config, law = adapter.signal_config(prep, parent(prep, 1))
    events = adapter.profiles(prep.rte_preparation.symbolic_tail.components, law, 1, indices)[0][2]
    with pytest.raises(ValueError):
        adapter.request_for_probe(prep, config, law, events, 131072, controlled=True)
    with pytest.raises(SystemExit):
        runner.main(['--run'])


def test_full_runner_coordination_with_artificial_inputs_only(toy, monkeypatch, tmp_path):
    prep, export, indices, support = toy
    parents = [parent(prep, r) for r in (1, 2, 4, 8, 16, 32)]
    monkeypatch.setattr(runner, 'static_inputs', lambda: ({}, parents, {}))
    monkeypatch.setattr(runner, 'frozen_source', lambda: ('ARTIFICIAL_ONLY', {}))
    monkeypatch.setattr(runner, 'construct_context', lambda manifest, raw, record: (
        record['counts_started'].update(hamiltonian_loads=1, state_loads=1, preparations=1, support_registrations=1)
        or (prep, support, export)))
    monkeypatch.setattr(runner, 'resource_context', lambda: {'workers': 2, 'artificial_fixture_only': True})
    original = runner.prepare_jobs
    monkeypatch.setattr(runner, 'prepare_jobs', lambda prep, support, export, rows, record, **kwargs:
        original(prep, support, export, rows, record, indices=indices, **kwargs))
    output = tmp_path/'synthetic_run'
    result = runner.run(output)
    assert result['status'] == 'LOCAL_TWELVE_PROBES_CONSISTENT_MANDATORY_STOP', result.get('error')
    assert result['total_counts_started'] == runner.CAPS
    assert len(list(output.glob('*'))) == 8
    saved = json.loads((output/'RESULT.json').read_text())
    assert saved['source_commit'] == 'ARTIFICIAL_ONLY'
    assert saved['scientific_status'] == 'UNDETERMINED'
    assert not saved['RPE_tolerance_certified']
    assert all('vectors' in g and len(g['probes']) == 2 for g in saved['groups'])


def test_angle_hex_is_saved_before_any_reference_action(toy, monkeypatch, tmp_path):
    prep, export, indices, support = toy
    path = tmp_path/'parameters.json'
    def stop_after_parameters(*args, **kwargs):
        saved = json.loads(path.read_text())
        assert saved[0]['order_angles_hex']['2'] == math.atan(float(parent(prep, 2)['signal_tau64'])/3).hex()
        raise RuntimeError('artificial reference boundary')
    monkeypatch.setattr(ref, 'build_reference', stop_after_parameters)
    with pytest.raises(RuntimeError, match='artificial reference boundary'):
        runner.prepare_jobs(prep, support, export, [parent(prep, 2)], {'counts_started': {}},
                            indices=indices, parameter_path=path)


def test_existing_output_is_preserved_before_context_loading(monkeypatch, tmp_path):
    marker = tmp_path/'existing.txt'; marker.write_text('preserve')
    monkeypatch.setattr(runner, 'static_inputs', lambda: pytest.fail('input processing reached'))
    with pytest.raises(FileExistsError):
        runner.run(tmp_path)
    assert marker.read_text() == 'preserve'
    assert list(tmp_path.iterdir()) == [marker]


def saved_export():
    selection = json.loads((HERE/'selector.json').read_text())
    return json.loads((ROOT/selection['export_path']).read_text())


def test_real_saved_payload_membership_is_checked_in_default_mode(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('real H4 context reached by static membership test')
    monkeypatch.setattr(runner, 'construct_context', forbidden)
    assert adapter.validate_saved_selection(saved_export()) == ('df-fragment-3:Z0Z2', 'df-fragment-4:Z0Z1')
    assert runner.describe()['new_H4_scientific_calls'] == 0


@pytest.mark.parametrize('field,value', (('sign',1), ('basis_id','missing'), ('support',[]),
                                        ('coefficient_abs',0.), ('probability64',0.)))
def test_saved_selected_payload_mutation_is_rejected(field, value):
    export = saved_export()
    item = next(c for c in export['components'] if c['id']=='df-fragment-3:Z0Z2')
    item[field] = value
    with pytest.raises(ValueError):
        adapter.validate_saved_selection(export)


def test_runtime_selector_matches_saved_metadata_with_original_indices():
    export = saved_export()
    names = adapter.validate_saved_selection(export)
    by_id = {c['id']:c for c in export['components']}
    items = []
    for name in reversed(names):
        c = by_id[name]
        items.append(SimpleNamespace(component_id=name, coefficient_sign=c['sign'], basis_id=c['basis_id'],
            diagonal_pauli_support=tuple(c['support']), coefficient_abs=c['coefficient_abs'],
            probability=c['probability64'], basis_hash=export['bases'][c['basis_id']]['basis_hash']))
    fake = SimpleNamespace(rte_preparation=SimpleNamespace(symbolic_tail=SimpleNamespace(components=items)))
    assert adapter.select_fixed_components(fake, export) == (1,0)
    items[0].basis_hash = 'wrong'
    with pytest.raises(ValueError, match='identity differs'):
        adapter.select_fixed_components(fake, export)
