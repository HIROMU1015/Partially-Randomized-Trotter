"""Artificial circuits only. Never loads a saved H4 input or calls a compiler."""
import copy
import json
from pathlib import Path
import sys
import unittest

# Reuse the existing artificial fixtures/private import and original adapter API.
from test_h4_d1_adapter import AdapterTests, A, event, toy
from d1_private import h4_d1_reproducibility as R
from d1_private.rte_compiled_cost import _circuit_payload


class ReproducibilityTests(unittest.TestCase):
    def payload(self, circuit):
        return _circuit_payload(circuit)

    def links(self):
        # Artificial fixture only; no evidence claims about real H4.
        return dict.fromkeys(R.REQUIRED_LINKS, True)

    def classify(self, left, right):
        names = [('instructions', 1, 'operation', 'name'),
                 ('instructions', 1, 'operation', 'base_gate', 'name')]
        return R.classify_payload_difference(self.payload(left), self.payload(right),
                                             display_name_paths=names)

    def test_auto_names_change_old_hash_but_are_separate_candidates(self):
        a = A.api()
        left = a.QuantumCircuit(1); left.x(0)
        right = a.QuantumCircuit(1); right.x(0)
        self.assertNotEqual(left.name, right.name)
        lc = a.hadamard_wrapper(left, 'cosine', measure=False)
        rc = a.hadamard_wrapper(right, 'cosine', measure=False)
        self.assertNotEqual(a.canonical_qiskit_circuit_fingerprint(lc),
                            a.canonical_qiskit_circuit_fingerprint(rc))
        diff = self.classify(lc, rc)
        self.assertTrue(diff); self.assertEqual({d['difference_class'] for d in diff}, {2})
        from qiskit.quantum_info import Operator
        self.assertTrue(a.np.array_equal(Operator(lc).data, Operator(rc).data))
        result = R.assess(historical_match=False, source_links=self.links(),
                          classified_differences=diff)
        self.assertEqual(result['historical_status'], 'LEGACY_FINGERPRINT_MISMATCH')
        self.assertTrue(result['research_GO_candidate'])
        self.assertFalse(result['dispatch_enabled'])

    def test_phase_survives_control_and_stops(self):
        a=A.api(); body=a.QuantumCircuit(1);body.x(0)
        changed=body.copy();changed.global_phase=.31
        left=a.hadamard_wrapper(body,'cosine',measure=False)
        right=a.hadamard_wrapper(changed,'cosine',measure=False)
        self.assertIn(1, {d['difference_class'] for d in self.classify(left,right)})
        from qiskit.quantum_info import Operator
        # Gate's first qarg is the control (least significant matrix bit).
        old=Operator(body.to_gate().control(1)).data
        new=Operator(changed.to_gate().control(1)).data
        self.assertTrue(a.np.allclose(old[0::2,0::2],new[0::2,0::2]))
        self.assertFalse(a.np.allclose(old[1::2,1::2],new[1::2,1::2]))

    def test_event_order_change_stops(self):
        a=A.api();left=a.QuantumCircuit(1);right=a.QuantumCircuit(1)
        left.x(0);left.rz(.27,0);right.rz(.27,0);right.x(0)
        diff=R.classify_payload_difference(self.payload(left),self.payload(right))
        self.assertEqual({d['difference_class'] for d in diff},{1})

    def test_control_state_change_stops(self):
        a=A.api();body=a.QuantumCircuit(1);body.rz(.19,0)
        left=a.QuantumCircuit(2);right=a.QuantumCircuit(2)
        left.append(body.to_gate().control(1,ctrl_state=1),[0,1])
        right.append(body.to_gate().control(1,ctrl_state=0),[0,1])
        diff=R.classify_payload_difference(self.payload(left),self.payload(right))
        self.assertTrue(any('ctrl_state' in d['path'] and d['difference_class']==1 for d in diff))

    def test_same_named_gate_different_definition_stops(self):
        a=A.api();l=a.QuantumCircuit(1,name='fixed');l.x(0)
        r=a.QuantumCircuit(1,name='fixed');r.z(0)
        left=a.QuantumCircuit(1);right=a.QuantumCircuit(1)
        left.append(l.to_gate(),[0]);right.append(r.to_gate(),[0])
        diff=R.classify_payload_difference(self.payload(left),self.payload(right))
        self.assertTrue(diff);self.assertEqual({d['difference_class'] for d in diff},{1})

    def test_saved_angle_and_emitted_angle_difference_stops(self):
        a=A.api();saved=event(0);altered=copy.deepcopy(saved);altered['rotation_angle']=.29
        labels={'0':'Z','1':'X'};left=a.QuantumCircuit(1);right=a.QuantumCircuit(1)
        a.append_pauli_event(left,A.event_from_record(saved),labels)
        a.append_pauli_event(right,A.event_from_record(altered),labels)
        diff=R.classify_payload_difference(self.payload(left),self.payload(right))
        self.assertTrue(any('params' in d['path'] and d['difference_class']==1 for d in diff))
        status=R.assess(historical_match=False,source_links=self.links(),classified_differences=diff)
        self.assertEqual(status['source_semantic_status'],'SCIENTIFIC_DIFFERENCE_STOP')

    def test_primitive_name_cannot_be_waived(self):
        a=A.api();c=a.QuantumCircuit(1);c.rz(.2,0)
        left=self.payload(c);right=copy.deepcopy(left)
        path=('instructions',0,'operation','name');right['instructions'][0]['operation']['name']='rx'
        diff=R.classify_payload_difference(left,right,display_name_paths=[path])
        self.assertEqual(diff[0]['difference_class'],1)

    def test_missing_source_link_and_unknown_phase_stop_even_if_hash_matches(self):
        links=self.links();links.pop('controlled_phase')
        r=R.assess(historical_match=True,source_links=links)
        self.assertEqual(r['source_semantic_status'],'UNRESOLVED_CORRESPONDENCE')
        self.assertFalse(r['research_GO_candidate'])
        r=R.assess(historical_match=False,source_links=self.links(),
                   unresolved_semantic_fields=['phase'])
        self.assertFalse(r['research_GO_candidate'])

    def test_qpy_self_match_does_not_supply_source_links(self):
        r=R.assess(historical_match=False,source_links={})
        self.assertEqual(len(r['missing_source_links']),12)
        self.assertFalse(r['research_GO_candidate'])


if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ReproducibilityTests),
                             unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if len(sys.argv)>1:
        Path(sys.argv[1]).write_text(json.dumps(dict(
            scope='ARTIFICIAL_ONLY_NOT_H4_EQUIVALENCE',tests=result.testsRun,
            failures=len(result.failures),errors=len(result.errors),reused_adapter_tests=8,
            cost_compile=0,real_H4_build=0),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
