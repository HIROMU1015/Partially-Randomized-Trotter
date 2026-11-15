"""LD3 extension checks: artificial operators, saved JSON metadata only.

Never loads saved arrays, builds circuits, evaluates a mean or prices.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
package = ModuleType('trotterlib'); package.__path__ = [str(ROOT/'src/trotterlib')]
sys.modules.setdefault('trotterlib', package)
from trottertracks import df_head_pauli_tail_ld3 as A
import run_df_head_pauli_tail_ld3 as R

p0, p1 = A.p0, A.p1
pauli = p0.pauli_api()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

F = module('LD3_B_fixture', 'tests/check_pr_rpe_signal_plan.py')


def fixture():
    """Three artificial modes, five signed complex fragments (LD3 tail = 3, 4)."""
    rng = np.random.default_rng(20261116)
    def herm(scale):
        x = scale*(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
        return (x+x.conj().T)/2
    h = herm(.1)
    gs = tuple(herm(.3) for _ in range(5))
    ham = SimpleNamespace(n_qubits=3, n_blocks=5, constant=.17, one_body=h,
        lambdas=np.array([.9, -.2, .6, .35, -.15]), g_matrices=gs)
    sector = SimpleNamespace(basis_indices=np.array([1, 2, 4]))
    state = np.array([1., .2, .3j]); state /= np.linalg.norm(state)
    full_state = np.zeros(8, complex); full_state[sector.basis_indices] = state
    # Tr(A(g)^2)/2^n = ((Tr g)^2+Tr(g^2))/4, independent of the JW collector.
    tails = {str(p): dict(scalar=math.fsum(float(w*((np.trace(g)**2+np.trace(g@g)).real)/4)
        for w, g in zip(ham.lambdas[p:], ham.g_matrices[p:], strict=True)),
        lambda_R=1., source='ARTIFICIAL_DESCRIPTOR') for p in (1, 2, 3, 4)}
    meta = dict(saved_tail_descriptors=tails, binding=dict(system='ARTIFICIAL_ONLY'))
    return ham, sector, full_state, state, meta


def run(function, args, **kw):
    result = {}; products = {}
    function(ROOT, *args, result, lambda name, value: products.update({name: deepcopy(value)}), **kw)
    return result, products


def same_products(test, left, right, names):
    for name in names:
        a, b = left[name], right[name]
        if name.endswith('.npz'):
            test.assertEqual(set(a), set(b))
            for k in a:
                test.assertTrue(np.array_equal(np.asarray(a[k]), np.asarray(b[k])), name+':'+k)
        else:
            test.assertEqual(a, b, name)


class LD3Tests(unittest.TestCase):
    def setUp(self):
        for guard in (patch.object(np, 'load', side_effect=AssertionError('No saved arrays')),
            patch.object(pauli, 'QuantumCircuit', side_effect=AssertionError('No circuits'))):
            guard.start(); self.addCleanup(guard.stop)

    def test_default_prefixes_reproduce_fixed_diagnose_bitwise(self):
        fixed, fixed_products = run(p0.diagnose, fixture())
        new, new_products = run(A.diagnose, fixture(), prefixes=(1, 2))
        self.assertEqual(fixed, new)
        self.assertEqual(set(fixed_products), set(new_products))
        same_products(self, fixed_products, new_products, fixed_products)

    def test_LD3_head_tail_split_and_LD2_control_unchanged(self):
        fixed, fixed_products = run(p0.diagnose, fixture())
        new, products = run(A.diagnose, fixture(), prefixes=(2, 3))
        self.assertEqual([r['prefix'] for r in new['prefixes']], [2, 3])
        self.assertEqual(fixed['prefixes'][1], new['prefixes'][0])
        same_products(self, fixed_products, products, ['LD2_DICTIONARY.json', 'LD2_JW_COEFFICIENTS.json',
            'LD2_OPERATORS.npz', 'COMMON_OPERATORS.npz'])
        ham, sector, full_state, state, meta = fixture()
        car = p0.car_api(ROOT)(ham, SimpleNamespace(basis_indices=np.arange(8)), full_state,
            meta['saved_tail_descriptors'])
        ops = products['LD3_OPERATORS.npz']
        self.assertTrue(np.array_equal(ops['head'], sum(car.matrices[:4], np.zeros((8, 8), complex))))
        self.assertTrue(np.array_equal(ops['tail'], sum(car.matrices[4:], np.zeros((8, 8), complex))))
        row = new['prefixes'][1]; payload = products['LD3_DICTIONARY.json']
        self.assertIn('GATES_PASS', row['status'])
        self.assertEqual(payload['prefix'], 3)
        self.assertEqual(row['lambda_P'], math.fsum(abs(t['coefficient']) for t in payload['terms']))
        self.assertEqual(row['dictionary_sha256'], A.canonical_sha(payload))
        self.assertNotEqual(row['dictionary_sha256'], new['prefixes'][0]['dictionary_sha256'])
        self.assertTrue(all(c['PASS'] for c in new['checks']))
        self.assertLess(row['identity_metrics']['Pauli_tail_full_fro'], 2e-14)

    def test_LD3_scalar_mismatch_stops_without_repair(self):
        args = fixture(); args[-1]['saved_tail_descriptors']['3']['scalar'] += .01
        result = {}
        with self.assertRaisesRegex(ValueError, 'LD3_Pauli_tail_saved_scalar_fro'):
            A.diagnose(ROOT, *args, result, lambda *v: None, prefixes=(2, 3))
        self.assertFalse(result['checks'][-1]['PASS'])
        self.assertEqual(result['prefixes'][1]['status'], 'STARTED')

    def test_prefix_domain_and_changed_fixed_loop_are_rejected(self):
        for bad in ((0,), (5,), (2, 2), ()):
            with self.assertRaises(ValueError):
                A.diagnose(ROOT, *fixture(), {}, lambda *v: None, prefixes=bad)
        original = p0.blob
        def changed(root, commit, path):
            raw = original(root, commit, path)
            return raw.replace(b'for prefix in (1, 2):', b'for prefix in (1, 2, 3):') if path == A.P0_MODULE else raw
        with patch.object(p0, 'blob', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'FIXED_P0_PREFIX_LOOP_NOT_FOUND'):
                A.diagnose(ROOT, *fixture(), {}, lambda *v: None, prefixes=(3,))

    def test_P1_identity_default_reproduces_fixed_and_locator_follows_dictionary(self):
        b = p1.b_api(ROOT); _, original, _, _ = F.fixture(prefix=1, M=12, kappa='2')
        item = dict(source=dict(fixture='ARTIFICIAL_ONLY'), dictionary_sha256='ARTIFICIAL_DICTIONARY',
            payload=dict(prefix=1, lambda_P=.7, saved_DF_scalar=0., mapping='ARTIFICIAL'),
            P0_result_source=dict(commit=p1.P0, path=p1.P0DIR+'run01/result.json'))
        self.assertEqual(A.generate_configuration(ROOT, b, original, item), p1.generate_configuration(b, original, item))
        moved = dict(item, P0_result_source=dict(commit='ARTIFICIAL_P0', path='ARTIFICIAL/result.json'))
        c = A.generate_configuration(ROOT, b, original, moved)
        self.assertNotEqual(c['id'], p1.generate_configuration(b, original, item)['id'])
        self.assertEqual(c['spec']['split']['tail_dictionary']['P0_operator_residual_source'], moved['P0_result_source'])
        self.assertEqual(c['spec']['law']['tail_dictionary'], c['spec']['split']['tail_dictionary'])

    def test_P1_LD3_artificial_common_kappa_schedule(self):
        b = p1.b_api(ROOT); _, original, _, _ = F.fixture(prefix=3, M=12, kappa='2000')
        def config(lam):
            item = dict(source=dict(fixture='ARTIFICIAL_ONLY'), dictionary_sha256='ARTIFICIAL_'+str(lam),
                payload=dict(prefix=3, lambda_P=lam, saved_DF_scalar=0., mapping='ARTIFICIAL'),
                P0_result_source=dict(fixture='ARTIFICIAL_ONLY'))
            return A.generate_configuration(ROOT, b, original, item)
        big, small = config(.7), config(.35)
        for c in (big, small):
            b.validate(c); self.assertEqual(len(c['rounds']), 13)
            self.assertEqual(c['spec']['kappa']['value'], original['spec']['kappa']['value'])
            self.assertTrue(all(r['candidate']['prefix'] == 3 for r in c['rounds']))
        self.assertEqual(R.spec_difference(big, small), ['law', 'split'])
        rows = R.table(big, small)
        self.assertEqual([r['m'] for r in rows], list(range(13)))
        self.assertAlmostEqual(rows[-1]['events_ratio_LD3_over_LD2'], .25, places=3)
        for r in rows:
            self.assertEqual(r['events_LD3'], r['q']*r['R_LD3'])
            self.assertGreaterEqual(r['events_LD3'], r['paper_total_target_LD3'])

    def test_real_metadata_reproduces_saved_LD1_LD2_configurations(self):
        meta = A.p0_metadata(ROOT, (1, 2, 3))
        self.assertEqual(meta['saved_tail_descriptors']['3'], dict(lambda_R=0.5832704246339966, scalar=0.09261516589219046))
        originals = A.original_df_configurations(ROOT, (1, 2))
        for p in (1, 2):
            self.assertEqual(originals[p], A.read(ROOT, p0.MENU, p0.DIRECTORY+'CONFIGURATION_LD'+str(p)+'.json'))
        saved = A.read(ROOT, A.SAVED_PAULI['commit'], A.SAVED_PAULI['path'])['common_kappa_configurations']
        b = p1.b_api(ROOT)
        for p in (1, 2):
            item = A.dictionary_item(ROOT, A.SAVED_P0['commit'], A.SAVED_P0['directory'], p, meta)
            self.assertEqual(item['P0_result_source'], dict(commit=p1.P0, path=p1.P0DIR+'run01/result.json'))
            config = A.generate_configuration(ROOT, b, originals[p], item)
            self.assertEqual(config, saved[str(p)]); self.assertEqual(config['id'], A.SAVED_PAULI_IDS[p])

    def test_runner_p0_entry_on_artificial_fixture(self):
        ham, sector, full_state, state, meta = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)/'context.json'; context.write_text('{}')
            meta.update(inputs={}, context_source=dict(path=str(context),
                sha256=hashlib.sha256(context.read_bytes()).hexdigest()))
            _, fixed_products = run(p0.diagnose, fixture())
            seen = {}
            def control(root, result, held, prefix=2):
                seen.update(held)
                return dict(PASS=all(np.array_equal(held['LD2_OPERATORS.npz'][k], fixed_products['LD2_OPERATORS.npz'][k])
                    for k in held['LD2_OPERATORS.npz']) and held['LD2_DICTIONARY.json'] == fixed_products['LD2_DICTIONARY.json'])
            for outcome in ('pass', 'fail'):
                out = Path(tmp)/outcome
                check = control if outcome == 'pass' else (lambda *a, **k: dict(PASS=False))
                with patch.object(R, 'head', return_value='ARTIFICIAL_SOURCE'), \
                        patch.object(R, 'own_sources', return_value=[]), \
                        patch.object(R.fixed_p0_runner, 'sources', return_value=[]), \
                        patch.object(A, 'p0_metadata', return_value=deepcopy(meta)), \
                        patch.object(p0, 'load_saved', return_value=(ham, sector, full_state, state, {})), \
                        patch.object(A, 'saved_control', side_effect=check):
                    result = R.execute_p0(out, 'ARTIFICIAL_SOURCE')
                names = {p.name for p in out.iterdir()}
                if outcome == 'pass':
                    self.assertEqual(result['status'], A.LD3_P0_SUCCESS)
                    self.assertTrue({'LD3_DICTIONARY.json', 'LD3_OPERATORS.npz', 'LD3_JW_COEFFICIENTS.json',
                        'FIXED_INPUTS.json', 'result.json'} <= names)
                    self.assertFalse({'LD2_DICTIONARY.json', 'LD2_OPERATORS.npz', 'COMMON_OPERATORS.npz'} & names)
                    self.assertEqual(set(seen), {'COMMON_OPERATORS.npz', 'LD2_JW_COEFFICIENTS.json',
                        'LD2_DICTIONARY.json', 'LD2_OPERATORS.npz'})
                    saved = json.loads((out/'LD3_DICTIONARY.json').read_text())
                    row = result['prefixes'][1]
                    self.assertEqual(A.canonical_sha(saved), row['dictionary_sha256'])
                else:
                    self.assertEqual(result['status'], 'STOP_P0_LD3_INPUT_OR_NUMERICAL_FAILURE')
                    self.assertIn('CONTROL_LD2_DICTIONARY.json', names)
                with self.assertRaises(ValueError):
                    R.execute_p0(out, 'ARTIFICIAL_SOURCE')

    def test_runner_configure_entry_with_artificial_LD3_dictionary(self):
        meta = A.p0_metadata(ROOT, (1, 2, 3))
        real_item, real_read = A.dictionary_item, A.read
        def item(root, commit, directory, prefix, meta):
            if prefix != 3:
                return real_item(root, commit, directory, prefix, meta)
            value = real_item(root, A.SAVED_P0['commit'], A.SAVED_P0['directory'], 2, meta)
            value = deepcopy(value); value['payload'].update(prefix=3, lambda_P=.7,
                saved_DF_scalar=meta['saved_tail_descriptors']['3']['scalar'])
            value.update(dictionary_sha256='ARTIFICIAL_LD3', source=dict(commit='ARTIFICIAL_P0', path='LD3.json'),
                P0_result_source=dict(commit='ARTIFICIAL_P0', path='result.json'))
            return value
        def read(root, commit, path):
            if commit == 'ARTIFICIAL_P0':
                return dict(status=A.LD3_P0_SUCCESS, scientific_source='ARTIFICIAL_SOURCE', control=dict(PASS=True))
            return real_read(root, commit, path)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(R, 'head', return_value='ARTIFICIAL_P0'), \
                patch.object(R, 'is_ancestor', return_value=True), \
                patch.object(R, 'own_sources', return_value=[]), \
                patch.object(A, 'dictionary_item', side_effect=item), patch.object(A, 'read', side_effect=read):
            out = Path(tmp)/'config'
            result = R.execute_configure(out, 'ARTIFICIAL_SOURCE', 'ARTIFICIAL_P0')
            self.assertEqual(result['status'], R.CONFIG_SUCCESS, result.get('error'))
            self.assertEqual(result['checks']['LD3_spec_differs_from_LD2_only_in'], ['L_D', 'law', 'split'])
            table = json.loads((out/'SCHEDULE_LD2_LD3.json').read_text())
            self.assertEqual([r['R_LD2'] for r in table['rows']],
                [2, 3, 5, 9, 17, 33, 66, 132, 263, 526, 1052, 2103, 4205])
            self.assertEqual(table['LD2_configuration']['id'], A.SAVED_PAULI_IDS[2])
            self.assertTrue((out/'SCHEDULE_LD2_LD3.csv').exists())
            self.assertEqual(json.loads((out/'CONFIGURATION_LD3.json').read_text())['spec']['L_D'], 3)


if __name__ == '__main__':
    unittest.main()
