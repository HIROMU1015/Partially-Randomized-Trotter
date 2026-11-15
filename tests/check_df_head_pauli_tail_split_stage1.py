"""Split stage-1 runner (LD5) on artificial operators and saved JSON metadata only.

Never loads saved arrays, builds circuits, evaluates a mean or prices.
"""
from copy import deepcopy
import hashlib
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
import run_df_head_pauli_tail_split_stage1 as S  # noqa: E402

A, R3 = S.A, S.R3
p0 = A.p0
pauli = p0.pauli_api()


def fixture(fragments=7):
    """Three artificial modes, seven signed complex fragments (LD5 tail = fragments 5, 6)."""
    rng = np.random.default_rng(20261116)
    def herm(scale):
        x = scale*(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
        return (x+x.conj().T)/2
    gs = tuple(herm(.3) for _ in range(fragments))
    ham = SimpleNamespace(n_qubits=3, n_blocks=fragments, constant=.17, one_body=herm(.1),
        lambdas=np.array([.9, -.2, .6, .35, -.15, .25, .12][:fragments]), g_matrices=gs)
    sector = SimpleNamespace(basis_indices=np.array([1, 2, 4]))
    state = np.array([1., .2, .3j]); state /= np.linalg.norm(state)
    full_state = np.zeros(8, complex); full_state[sector.basis_indices] = state
    tails = {str(p): dict(scalar=math.fsum(float(w*((np.trace(g)**2+np.trace(g@g)).real)/4)
        for w, g in zip(ham.lambdas[p:], ham.g_matrices[p:], strict=True)), lambda_R=1., source='ARTIFICIAL')
        for p in range(1, fragments)}
    return ham, sector, full_state, state, dict(saved_tail_descriptors=tails, binding=dict(system='ARTIFICIAL_ONLY'))


class SplitStage1Tests(unittest.TestCase):
    def setUp(self):
        for guard in (patch.object(np, 'load', side_effect=AssertionError('No saved arrays')),
                      patch.object(pauli, 'QuantumCircuit', side_effect=AssertionError('No circuits'))):
            guard.start(); self.addCleanup(guard.stop)

    def test_p0_entry_prefix5_on_artificial_fixture(self):
        ham, sector, full_state, state, meta = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)/'context.json'; context.write_text('{}')
            meta.update(inputs={}, context_source=dict(path=str(context), sha256=hashlib.sha256(b'{}').hexdigest()))
            seen = {}
            def control(root, result, held, prefix=2):
                seen.update(held); return dict(PASS=True)
            out = Path(tmp)/'p0'
            with patch.object(R3, 'head', return_value='ARTIFICIAL'), patch.object(S, 'own_sources', return_value=[]), \
                    patch.object(S.fixed_p0_runner, 'sources', return_value=[]), \
                    patch.object(A, 'p0_metadata', return_value=deepcopy(meta)), \
                    patch.object(p0, 'load_saved', return_value=(ham, sector, full_state, state, {})), \
                    patch.object(A, 'saved_control', side_effect=control):
                result = S.execute_p0(out, 'ARTIFICIAL', 5)
            self.assertEqual(result['status'], S.success(5), result.get('error'))
            self.assertEqual([r['prefix'] for r in result['prefixes']], [2, 5])
            names = {p.name for p in out.iterdir()}
            self.assertTrue({'LD5_DICTIONARY.json', 'LD5_OPERATORS.npz', 'LD5_JW_COEFFICIENTS.json'} <= names)
            self.assertFalse({'LD2_DICTIONARY.json', 'COMMON_OPERATORS.npz'} & names)
            self.assertIn('LD2_DICTIONARY.json', seen)
            # head = one-body + fragments 0..4 (six blocks); tail = fragments 5, 6
            car = p0.car_api(ROOT)(ham, SimpleNamespace(basis_indices=np.arange(8)), full_state, meta['saved_tail_descriptors'])
            with patch.object(np, 'load', wraps=np.lib.npyio.load):
                ops = dict(np.lib.npyio.load(out/'LD5_OPERATORS.npz'))
            self.assertTrue(np.array_equal(ops['head'], sum(car.matrices[:6], np.zeros((8, 8), complex))))
            self.assertTrue(np.array_equal(ops['tail'], sum(car.matrices[6:], np.zeros((8, 8), complex))))
            self.assertTrue(all(c['PASS'] for c in result['checks']))

    def test_configure_entry_reproduces_LD3_and_tabulates_with_artificial_LD5(self):
        real_item, real_read = A.dictionary_item, A.read
        def item(root, commit, folder, prefix, meta, status):
            value = deepcopy(real_item(root, S.LD3['p0_commit'], S.LD3['p0_directory'], 3, meta))
            value['payload'].update(prefix=prefix, lambda_P=.1, saved_DF_scalar=meta['saved_tail_descriptors'][str(prefix)]['scalar'])
            value.update(dictionary_sha256='ARTIFICIAL_LD5', source=dict(commit='ARTIFICIAL_P0', path='LD5.json'),
                         P0_result_source=dict(commit='ARTIFICIAL_P0', path='result.json'))
            return value
        def read(root, commit, path):
            if commit == 'ARTIFICIAL_P0':
                return dict(status=S.success(5), scientific_source='ARTIFICIAL_SOURCE', control=dict(PASS=True))
            return real_read(root, commit, path)
        with tempfile.TemporaryDirectory() as tmp, patch.object(R3, 'head', return_value='ARTIFICIAL_P0'), \
                patch.object(R3, 'is_ancestor', return_value=True), patch.object(S, 'own_sources', return_value=[]), \
                patch.object(S, 'dictionary_item', side_effect=item), patch.object(A, 'read', side_effect=read):
            out = Path(tmp)/'config'
            result = S.execute_configure(out, 'ARTIFICIAL_SOURCE', 'ARTIFICIAL_P0', 5)
            self.assertEqual(result['status'], 'LD5_CONFIGURATION_AND_SCHEDULE_COMPLETE_NOT_SIGNAL_OR_PRICE', result.get('error'))
            self.assertTrue(result['checks']['Pauli_LD3_equal_saved_dd5b881d'])
            self.assertTrue(result['checks']['original_DF_LD3_equal_saved_421258d40'])
            self.assertEqual(result['checks']['spec_differs_from_LD3_only_in'], ['L_D', 'law', 'split'])
            table = json.loads((out/'SCHEDULE_LD3_LD5.json').read_text())
            self.assertEqual([r['R_LD3'] for r in table['rows']], [1, 1, 1, 2, 3, 6, 11, 21, 42, 83, 165, 330, 660])
            self.assertEqual([r['m'] for r in table['rows'] if r['R_floor_1_LD3']], [0, 1, 2])
            w = table['shot_weighted_events']
            self.assertEqual(w['LD3']['VP']['events'], 73947076)
            for r in table['rows']:
                self.assertEqual(r['events_LD5'], r['q']*r['R_LD5'])
                self.assertGreaterEqual(r['events_minus_target_LD5'], 0)
            self.assertEqual(w['LD5']['FULL_PF__B_PRICE']['N_per_axis'][-2:], [8, 2])


if __name__ == '__main__':
    unittest.main()
