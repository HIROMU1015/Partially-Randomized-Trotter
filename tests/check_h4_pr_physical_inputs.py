"""Artificial mutations and exact saved-field copying; no molecular reanalysis."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import h4_pr_physical_inputs as P
from trottertracks import h4_pr_rmse_completion as completion
from trottertracks import pr_rpe_rmse as A


class PhysicalInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = P.load(ROOT)

    def fixture(self):
        # Keep physical metadata, use artificial complex coordinates. No signal
        # arithmetic, gate reevaluation or nominal-shot generation is performed.
        config, packet, entries, truth, index, _ = deepcopy(self.saved)
        for e in entries:
            for key in ('reference', 'raw', 'corrected', 'exact_tail'):
                e['diagnostics']['signal'][key] = dict(real=.4, imag=-.2)
            e['model']['signal'] = deepcopy(e['diagnostics']['signal'])
        return config, packet, entries, truth, index

    def test_all13_saved_fields_are_copied_without_science(self):
        with (patch.object(A, 'nominal_axis_shots', side_effect=AssertionError('N solver')),
              patch.object(A, 'decode_supplied_coordinates', side_effect=AssertionError('decoder')),
              patch.object(completion, 'load_context', side_effect=AssertionError('H/state load')),
              patch.object(completion, 'diagnose', side_effect=AssertionError('reanalysis')),
              patch.object(completion.power, 'evolve', side_effect=AssertionError('mean'))):
            result = P.build(ROOT)
        config, packet, entries, truth, *_ = self.saved
        by_m = {e['m']:e for e in entries}
        self.assertEqual(len(result['rounds']), 13)
        for m, row in enumerate(result['rounds']):
            self.assertEqual(row['diagnostics'], by_m[m]['diagnostics'])
            self.assertEqual(row['law_binding'], config['rounds'][m]['law_binding'])
            self.assertEqual(row['nominal_N_X'], packet['rounds'][m]['nominal_N_X'])
            self.assertEqual(row['nominal_N_Y'], packet['rounds'][m]['nominal_N_Y'])
            self.assertIsNone(row['sufficient_N_X'])
            self.assertIsNone(row['energy_bias_ha'])
            self.assertIsNone(row['energy_RMSE_bound_ha'])
        self.assertEqual(result['target']['reference_solver_energy_ha'], truth['reference_solver_energy_ha'])
        self.assertIsNone(result['global_RMSE_bound'])
        self.assertEqual(result['new_science_calls'], 0)
        self.assertFalse(result['measurement_contract']['probabilities_evaluated'])

    def test_missing_duplicate_or_incomplete_evidence_rejected(self):
        for variant in ('missing', 'duplicate', 'field'):
            args = self.fixture()
            if variant == 'missing': args[2].pop()
            elif variant == 'duplicate': args[2][-1] = deepcopy(args[2][0])
            else: del args[2][0]['diagnostics']['signal']['raw']['imag']
            with self.subTest(variant=variant), self.assertRaises((ValueError, KeyError)):
                P.assemble(*args)

    def test_source_candidate_law_policy_and_whole_identity_rejected(self):
        for variant in ('source', 'carrier', 'candidate', 'law', 'M', 'kappa', 'root_M', 'root_kappa', 'policy', 'nominal'):
            args = self.fixture(); config, packet, entries, *_ = args
            if variant == 'source': entries[0]['scientific_source'] = '0'*40
            elif variant == 'carrier': entries[0]['source']['commit'] = '0'*40
            elif variant == 'candidate': entries[0]['diagnostics']['candidate']['R_outer'] += 1
            elif variant == 'law': packet['rounds'][0]['law_binding']['foreign'] = True
            elif variant in ('M', 'kappa'): packet['rounds'][0]['identity']['foreign_'+variant] = True
            elif variant == 'root_M': config['spec']['A_input']['M'] = 11
            elif variant == 'root_kappa': config['spec']['A_input']['kappa_paper'] = '1'
            elif variant == 'policy': entries[0]['model']['policy_id'] = 'HOEFFDING'
            else: entries[0]['model']['nominal_N_X'] += 1
            with self.subTest(variant=variant), self.assertRaises(ValueError): P.assemble(*args)

    def test_m3_m4_conditional_norm_missing_and_no_invented_observation(self):
        args = self.fixture(); result = P.assemble(*args)
        for m in (3,4):
            d = result['rounds'][m]['diagnostics']
            self.assertIsNone(d['norm_evidence']['observed_error'])
            self.assertNotIn('exact_tail_norm_error', d['signal']['allowance_evidence'])
        entry = next(e for e in args[2] if e['m'] == 3)
        entry['diagnostics']['norm_evidence']['observed_error'] = 0.
        with self.assertRaises(ValueError): P.assemble(*args)

    def test_failed_gate_sufficient_N_or_unknown_error_cannot_be_promoted(self):
        for variant in ('gate', 'sufficient', 'law_error', 'global', 'state'):
            args = self.fixture(); _, packet, entries, truth, _ = args
            if variant == 'gate': entries[0]['diagnostics']['numerical_gates_passed'] = False
            elif variant == 'sufficient': packet['rounds'][0]['N_X'] = 1
            elif variant == 'law_error': entries[0]['diagnostics']['actual_implementation']['law_error'] = 0.
            elif variant == 'global': packet['rmse_evidence']['rmse_bound_ha'] = .001
            else: truth['ground_state_certified'] = True
            with self.subTest(variant=variant), self.assertRaises(ValueError): P.assemble(*args)

    def test_nonfinite_saved_field_is_not_exportable(self):
        args = self.fixture()
        args[2][0]['diagnostics']['signal']['raw']['real'] = float('nan')
        args[2][0]['model']['signal'] = deepcopy(args[2][0]['diagnostics']['signal'])
        with self.assertRaises(ValueError): P.assemble(*args)

    def test_reused_B_identity_validator_matches_pinned_B(self):
        path = P.V.legacy.previous.B_PATH
        functions = []
        for commit in (P.V.B_RELEASE, P.B):
            data = subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)
            functions.append({n.name:ast.dump(n, include_attributes=False) for n in ast.parse(data).body
                              if isinstance(n, ast.FunctionDef)})
        for name in ('_id', '_positive', 'comparison_method', '_coverage', '_assemble', '_validate'):
            self.assertEqual(functions[0][name], functions[1][name])

    def test_cli_export_is_passive_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory(prefix='physical-input-copy-') as tmp:
            out = Path(tmp)/'inputs.json'
            command = [sys.executable, '-B', 'scripts/export_h4_pr_physical_inputs.py', '--out', str(out)]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
            before = out.read_bytes()
            self.assertEqual(json.loads(before)['new_science_calls'], 0)
            self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)
            self.assertEqual(out.read_bytes(), before)
        subprocess.run([sys.executable, '-B', '-c',
            "import sys;sys.path.insert(0,'src');from trottertracks import h4_pr_physical_inputs;"
            "assert not any(x in sys.modules for x in ('numpy','scipy','qiskit'))"], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
