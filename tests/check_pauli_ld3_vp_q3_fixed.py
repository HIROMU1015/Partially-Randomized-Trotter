"""LD3 passive intake plus artificial worker; never evaluate H4 risk here."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
spec = importlib.util.spec_from_file_location('_ld3_test_caller', ROOT/'scripts/run_pauli_ld3_vp_q3_fixed.py')
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)
from check_rpe_pauli_integration import forbid_science
from check_pf_shortlist_q3_caller import artificial, MOCK


class LD3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with forbid_science(), patch.object(C.T, 'evaluate', side_effect=AssertionError('No H4 risk')):
            cls.inputs = C.build_inputs()
            cls.binding, cls.view = C.metadata(cls.inputs['cases'][0])

    def test_fixed_VP_and_LD2_contract_exactly_preserved(self):
        old = C.document(C.PAULI, C.PAULI_DIR+'/INPUTS.json'); inputs = self.inputs
        self.assertEqual(len(inputs['cases']), 1); self.assertEqual(inputs['maximum_scientific_calls'], 1)
        case = inputs['cases'][0]
        self.assertEqual(case['allocation']['shots'], old['cases'][1]['allocation']['shots'])
        for k in ('statistical_contract', 'options', 'adopted_evaluation', 'acceleration'):
            self.assertEqual(inputs[k], old[k])
        self.assertEqual(case['evaluation_target'], old['cases'][1]['evaluation_target'])
        self.assertNotEqual(case['total_configuration_id'], C.CONFIGURATION_ID)
        self.assertEqual(inputs['finite_scope_preflight'], old['finite_scope_preflight'])

    def test_raw_dictionary_SOURCE_scalar_once_and_unknowns_preserved(self):
        handoff = C.document(C.A.LD3_HANDOFF['commit'], C.A.LD3_HANDOFF['path']); before = deepcopy(handoff)
        packet = self.view['signal_packet']; original = packet['origin_signal_packet']
        self.assertEqual(original['configuration_id'], C.CONFIGURATION_ID)
        self.assertEqual(original['dictionary_identity'], handoff['packets'][0]['dictionary_identity'])
        self.assertEqual(packet['target'], original['target'])
        self.assertFalse(packet['target']['ground_state_certified'])
        self.assertFalse(packet['target']['truth_proxy_adopted_for_new_simulation'])
        for m, mean in enumerate(packet['means']):
            row = handoff['packets'][0]['rounds'][m]; rec = row['mean_record']
            for k in ('raw', 'corrected', 'reference'):
                self.assertEqual(mean[k], rec['signal'][k])
            self.assertEqual(mean['normalization']['B'], rec['signal']['B'])
            self.assertEqual(mean['qualification'], 'UNQUALIFIED')
            self.assertEqual(mean['implementation_source'], rec['source'])
            self.assertTrue(rec['scalar_once'])
        for k in ('sufficient_N', 'global_RMSE_bound', 'G', 'winner'):
            self.assertIsNone(packet[k])
        self.assertEqual(handoff, before)

    def test_old_dictionary_helper_rejected_and_A_adapter_exact(self):
        h = C.document(C.A.LD3_HANDOFF['commit'], C.A.LD3_HANDOFF['path']); packet = h['packets'][0]
        store = C.A.p.SavedArtifacts(ROOT)
        records = {r['m']: store.read(r['mean_source'])['record'] for r in packet['rounds']}
        original = self.view['signal_packet']['origin_signal_packet']
        def bind(p, helpers):
            return C.A.bind_packet(p['configuration'], p, records, store.read(C.A.LD3_DICTIONARY),
                target=original['target'], target_source=original['target_view_source'],
                assembly_source=C.A.LD3_HANDOFF, helpers=helpers)
        with forbid_science():
            with self.assertRaisesRegex(ValueError, 'dictionary/hash/bit order'):
                bind(packet, C.A._metadata_helpers(str(ROOT)))
            bind(packet, C.A._ld3_metadata_helpers(str(ROOT)))
            for key in ('candidate_id', 'law_binding'):
                bad = deepcopy(packet); bad['rounds'][0][key] = 'OTHER'
                with self.assertRaises(ValueError): bind(bad, C.A._ld3_metadata_helpers(str(ROOT)))
            bad = deepcopy(packet); bad['rounds'][0]['mean_source']['scientific_source'] = 'OTHER'
            with self.assertRaises(ValueError): bind(bad, C.A._ld3_metadata_helpers(str(ROOT)))

    def test_changed_view_and_unpinned_handoff_rejected(self):
        with forbid_science():
            packet = deepcopy(self.view['signal_packet']['origin_signal_packet'])
            packet['means'][0]['raw'] = [0, 0]
            with self.assertRaises(ValueError): C.T.bind_signal(ROOT, self.binding, packet)
            bad = dict(C.A.LD3_HANDOFF, commit='0'*40)
            with self.assertRaises(ValueError):
                C.A.accept_saved(ROOT, integration=C.T.C.interfaces(str(ROOT)).pipeline, handoff_source=bad)

    def test_actual_worker_uses_supplied_metadata_and_saves_before_return(self):
        case, inputs, binding, view = artificial()
        with tempfile.TemporaryDirectory() as folder, patch.object(C, 'metadata', return_value=(binding, view)),\
                patch.object(C, 'check_source'):
            receipt = C.worker(case, inputs, MOCK, folder); path = Path(folder)
            result = json.loads((path/'RESULT.json').read_text())
            self.assertEqual(json.loads((path/'API_RETURN.json').read_text()), [result['record']])
            self.assertEqual(receipt['scientific_evaluation_calls'], 1)
            self.assertTrue(receipt['resource_saved']); self.assertIsNone(result['sufficient_N'])
            self.assertIsNone(result['G_qualified'])


if __name__ == '__main__': unittest.main(verbosity=2)
