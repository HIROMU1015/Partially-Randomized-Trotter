"""Artificial/mock guards; no frozen real diagonals are loaded here."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('site_validation',ROOT/'scripts/validate_h46_s4_primitive_sites.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class SiteTests(unittest.TestCase):
    def test_input_hash_rejected_before_json_parsing(self):
        with patch.object(Path,'is_file',return_value=True),patch.object(Path,'is_symlink',return_value=False),\
             patch.object(Path,'read_bytes',return_value=b'broken-json'):
            with self.assertRaisesRegex(ValueError,'FROZEN_INPUT_BYTES_SHA'):v.checked_input('H4','fixture')

    def test_symlink_rejected(self):
        with patch.object(Path,'is_symlink',return_value=True):
            with self.assertRaisesRegex(ValueError,'FROZEN_INPUT_PATH'):v.checked_input('H4','fixture')

    def test_authorization_rejected_before_git(self):
        with patch.object(v.subprocess,'check_output') as git:
            with self.assertRaisesRegex(ValueError,'SCOPE_NOT_AUTHORIZED'):v.verify(dict(approved=False))
            git.assert_not_called()

    def test_source_identity_rejected(self):
        plan=dict(approved=True,runtime_authorization=True,max_primitive_evaluations=66,automatic_retry=False,SOURCE='a'*40)
        with patch.object(v.subprocess,'check_output',return_value='b'*40):
            with self.assertRaisesRegex(ValueError,'SOURCE_BINDING'):v.verify(plan)

    def test_environment_version_drift_rejected(self):
        plan=dict(approved=True,runtime_authorization=True,max_primitive_evaluations=66,automatic_retry=False,
                  SOURCE='a'*40,own_source_hashes={},environment=dict(python_version=[0,0,0]))
        with patch.object(v.subprocess,'check_output',return_value='a'*40):
            with self.assertRaisesRegex(ValueError,'ENVIRONMENT_VERSION_BINDING'):v.verify(plan)

    def test_receipt_records_pruning_and_hex(self):
        pp=types.SimpleNamespace(global_phase=-0.,rz=((1,1e-30),),rzz=((0,1,-0.),(1,2,.1)))
        r=v.primitive_receipt(pp,3)
        self.assertEqual(r['zero_rz_qubits'],[0,2]);self.assertEqual(r['nonzero_rzz_pairs'],[[1,2]])
        self.assertEqual(r['phase_hex'],'-0x0.0p+0');self.assertFalse(r['phase_nonzero'])
        self.assertEqual(r['rz']['1'],(1e-30).hex())

    def test_nonfinite_rejected(self):
        pp=types.SimpleNamespace(global_phase=0.,rz=((1,float('nan')),),rzz=())
        with self.assertRaisesRegex(ValueError,'NONFINITE_PRIMITIVE'):v.primitive_receipt(pp,3)

    def fake_graph(self,fragment_order=range(12),preparation=None):
        blocks=[dict(one=True,diagonal=[.2]*8,fragment=None)]+[dict(one=False,diagonal=[.3]*8,fragment=i) for i in fragment_order]
        fields=dict(preparation_hash=preparation or v.CONFIG['H4']['preparation'],num_system_qubits=8,
                    deterministic_fragment_indices=list(range(12)),randomized_block_indices=[],deterministic_blocks=blocks)
        class Graph:
            def __init__(self,*args):pass
            def fields(self,*args):return fields
            def scalar(self,x):return x
            def sequence(self,x):return x
            def node(self,x):
                return dict(type='DFDeterministicOneBodySpec' if x['one'] else 'DFDeterministicFragmentSpec',
                            fields=dict(diagonal_eigenvalues=x['diagonal'],diagonal_eta=x['diagonal'],lam=.1,
                                        basis_id='artificial',original_fragment_index=x['fragment']))
        for i,b in enumerate(blocks):b['ref']=str(i)
        return dict(nodes={},roots={'faithful:p12':{}}),types.SimpleNamespace(FrozenMetadataGraph=Graph,digest=lambda x:'fake')

    def test_metadata_selects_only_fixed_blocks(self):
        archive,graph=self.fake_graph();blocks=v.selected_blocks('H4',archive,graph)
        self.assertEqual(len(blocks),13);self.assertTrue(blocks[0]['one'])
        self.assertEqual([b['fragment'] for b in blocks[1:]],list(range(12)))

    def test_wrong_preparation_rejected(self):
        archive,graph=self.fake_graph(preparation='changed')
        with self.assertRaisesRegex(ValueError,'FULL_PREPARATION_BINDING'):v.selected_blocks('H4',archive,graph)

    def test_wrong_fragment_order_rejected(self):
        archive,graph=self.fake_graph(fragment_order=reversed(range(12)))
        with self.assertRaisesRegex(ValueError,'BLOCK_INVENTORY'):v.selected_blocks('H4',archive,graph)

    def test_two_reservations_per_block_no_repeated_q_axes(self):
        audit=v.load_module('site_artificial_audit',ROOT/'scripts/audit_h46_s4_count_mismatch.py');f=audit.functions()
        reservations=[]
        b=dict(index=0,one=False,diagonal=[1.,2.,-3.],lam=1.,basis_id='artificial',fragment=0,node_sha256='fake')
        result=v.evaluate_system('H4',[b],f,reservations.append)
        self.assertEqual(len(reservations),2)
        self.assertEqual([r['role'] for r in reservations],['model','native'])
        self.assertEqual(result['net_per_outer_step'],dict(rz=1,rzz=0,phase=0))
        self.assertEqual(result['inferred_full_saved_delta'],dict(raw=4,rz=2,cx=2,sx=0,x=0))

    def test_budget_reservation_failure_precedes_producer(self):
        f=types.SimpleNamespace(m=types.SimpleNamespace(pf_schedule=lambda *a:[types.SimpleNamespace(time=-.2)]),
              schedule=types.SimpleNamespace(deterministic_control_plan=lambda *a:[types.SimpleNamespace(time=-.2)]))
        b=dict(index=0,one=False,diagonal=[1.,2.,-3.],lam=1.,basis_id='artificial',fragment=0)
        def reserve(row):raise ValueError('budget')
        with self.assertRaisesRegex(ValueError,'budget'):v.evaluate_system('H4',[b],f,reserve)


if __name__=='__main__':unittest.main(verbosity=2)
