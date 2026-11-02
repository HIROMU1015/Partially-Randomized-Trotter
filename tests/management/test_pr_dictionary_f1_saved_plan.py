"""Management checks only: no research imports, events, circuits or numerics."""
from pathlib import Path
import collections
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class PlanManagementTests(unittest.TestCase):
    def test_id_plan_matches_published_B_counts_and_dependencies_without_events(self):
        plan=load('saved_plan','src/trottertracks/pr_dictionary_validation/f1_saved_plan.py')
        doc=json.loads((ROOT/'docs/research/pr_algorithm_improvement_phase2d_binding_execution_20261023/F1_COUNT_DESCRIPTORS.json').read_bytes())
        counts=collections.Counter(); seen=set()
        for case in plan.iter_cases():
            self.assertNotIn(case['id'],seen)
            self.assertTrue(set(case['requires']).issubset(seen))
            self.assertFalse({'macros','products','child','atom_id'} & case.keys())
            seen.add(case['id']);counts[case['call_group']]+=1
        self.assertEqual(dict(counts),{x['family']:x['resolved_count'] for x in doc['families']})
        self.assertNotIn('trotterlib',sys.modules)
        self.assertNotIn('qiskit',sys.modules)

    def test_existing_gate_rejects_non_F1_draft_before_inputs_or_science(self):
        auth=load('old_gate','src/trottertracks/pr_dictionary_validation/authorization.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'authorization.json').write_text(json.dumps(dict(stage='I',authority='USER',status='DRAFT')))
            with self.assertRaises(PermissionError): auth.verify_packet(root,ROOT,root/'output')
            self.assertFalse((root/'output').exists())
            self.assertEqual(len(list(root.iterdir())),1)
        self.assertNotIn('trotterlib',sys.modules)


if __name__=='__main__': unittest.main()
