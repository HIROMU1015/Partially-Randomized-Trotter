"""Fabricated metadata only; never open scientific inputs or build circuits."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


binding = load_file('_metadata_connection_schema', ROOT / 'src/trotterlib/native_df_metadata_binding.py')
model = load_file('_metadata_connection_model', ROOT / 'src/trotterlib/native_df_gate_model.py')


def synthetic_basis():
    ops = [dict(name='rz', qubits=[0], parameters=['-0.0'], matrix_sha256='a'*64),
           dict(name='xx_plus_yy', qubits=[1, 0], parameters=['0.3', '1.57'], matrix_sha256='b'*64)]
    payload = dict(num_system_qubits=2, operations=ops, hash_policy='ordered_local_basis_operations_v1')
    sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(basis_id='synthetic-basis', basis_hash=sha, num_system_qubits=2,
                phase_policy='native_gate_list_only', control_scope='uncontrolled', operations=ops)


def synthetic_binding():
    v = {key:'a'*64 for key in binding._IDENTITY_FIELDS}
    v.update(execution_source='1'*40, dataset_source='2'*40, model_source='3'*40,
             candidate_id='FABRICATED_PR', family='PR', formula='2nd', T=0.8, prefix=1, q=2, r=3, K=4, R=6,
             num_system_qubits=2, ancilla=2, control_policy='symmetric_directional', control_state=1, axis='cosine',
             wrapper_scope='measured_hadamard_state_preparation_excluded', sample_scope='confirmation32',
             epsilon=0.01, allowance_factor=1, tau=0.125,
             tau_semantics='exact_rte_lambda_r*T/(q*r)', model='B', basis_policy='registered_full_basis',
             rte_reuse_policy='raw_adjacent_equal_basis',
             occurrence_policy='closed_each_occurrence_and_prefix_half')
    return v


class MetadataBindingTests(unittest.TestCase):
    def setUp(self):
        self.b = synthetic_basis()
        self.identity = synthetic_binding()

    def check_basis(self, value):
        return binding.runtime_basis_metadata(value, expected_basis_id='synthetic-basis',
                                              expected_basis_hash=self.b['basis_hash'])

    def test_basis_id_alias_rejected_even_for_same_operations(self):
        self.b['basis_id']='different-alias'
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_artificial_bridge_to_existing_adapter_without_cost_call(self):
        runtime = self.check_basis(self.b)
        result = model.basis_from_runtime(self.b['basis_id'], self.b['basis_hash'], runtime)
        self.assertEqual(result.operations[1].qubits, (1, 0))
        self.assertEqual(result.operations[0].angle.hex(), '-0x0.0p+0')
        self.assertNotEqual(result.operation_digest, self.b['basis_hash'])

    def test_changed_operation_same_registry_hash_rejected(self):
        self.b['operations'][0]['parameters'] = ['0.0']
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_changed_matrix_claim_same_registry_hash_rejected(self):
        self.b['operations'][0]['matrix_sha256'] = 'c'*64
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_reordered_qubits_rejected(self):
        self.b['operations'][1]['qubits'].reverse()
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_reordered_operations_rejected(self):
        self.b['operations'].reverse()
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_invalid_system_qubit_rejected(self):
        self.b['operations'][0]['qubits'] = [2]
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_unsupported_opaque_gate_rejected(self):
        self.b['operations'][1]['name'] = 'unitary'
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_symbolic_nonfinite_parameters_rejected(self):
        for p in ('theta', 'nan', 'inf'):
            with self.subTest(p=p):
                v=copy.deepcopy(self.b); v['operations'][0]['parameters']=[p]
                with self.assertRaises(ValueError): self.check_basis(v)

    def test_missing_phase_rejected(self):
        del self.b['phase_policy']
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_controlled_gaussian_rejected(self):
        self.b['control_scope']='controlled'
        with self.assertRaises(ValueError): self.check_basis(self.b)

    def test_declared_binding_is_schema_only_not_permission(self):
        r=binding.validate_connection_identity(self.identity, expected=copy.deepcopy(self.identity))
        self.assertFalse(r['cost_evaluated'])
        self.assertFalse(r['runtime_authorization'])
        self.assertFalse(r['receipt_contents_certified'])

    def test_source_input_profile_mismatch_rejected(self):
        for key in ('execution_source','dataset_source','input_sha256','df_hash','profile_hash','compiler_hash'):
            with self.subTest(key=key):
                v=copy.deepcopy(self.identity);v[key]='f'*len(v[key])
                with self.assertRaises(ValueError): binding.validate_connection_identity(v, expected=self.identity)

    def test_tau_cutoff_component_law_mismatch_rejected(self):
        for key,value in [('tau',0.25),('K',2),('component_law_sha256','f'*64),('tau_semantics','T/q')]:
            with self.subTest(key=key):
                v=copy.deepcopy(self.identity);v[key]=value
                with self.assertRaises(ValueError): binding.validate_connection_identity(v, expected=self.identity)

    def test_bad_dimensions_and_type_rejected_even_when_expected_matches(self):
        for key,value in [('R',7),('K',3),('q',True),('ancilla',0),('formula','4th'),('control_state',0)]:
            with self.subTest(key=key):
                v=copy.deepcopy(self.identity);v[key]=value
                with self.assertRaises(ValueError): binding.validate_connection_identity(v, expected=copy.deepcopy(v))

    def test_missing_axis_phase_shot_receipt_rejected(self):
        for key in ('axis','phase_receipt_sha256','shot_receipt_sha256'):
            with self.subTest(key=key):
                v=copy.deepcopy(self.identity);del v[key]
                with self.assertRaises(ValueError): binding.validate_connection_identity(v, expected=self.identity)

    def test_wrong_reuse_policy_rejected(self):
        v=copy.deepcopy(self.identity);v['rte_reuse_policy']='disabled'
        with self.assertRaises(ValueError): binding.validate_connection_identity(v, expected=self.identity)

    def test_support_history_or_flattened_occurrences_rejected(self):
        for key,value in [('basis_policy','support_restricted'),('occurrence_policy','flatten_all'),('model','C')]:
            with self.subTest(key=key):
                v=copy.deepcopy(self.identity);v[key]=value
                with self.assertRaises(ValueError): binding.validate_connection_identity(v, expected=copy.deepcopy(v))

    def test_exploration_and_confirmation_never_join(self):
        v=copy.deepcopy(self.identity);v['sample_scope']='exploration8'
        with self.assertRaises(ValueError): binding.validate_connection_identity(v, expected=self.identity)

    def test_pf_explicitly_has_no_rte_law(self):
        v=copy.deepcopy(self.identity);v.update(family='PF',sample_scope='deterministic')
        for key in ('r','R','K','law_receipt_sha256','component_law_sha256','tau','tau_semantics'):v[key]=None
        self.assertEqual(binding.validate_connection_identity(v,expected=copy.deepcopy(v))['status'],'SCHEMA_ONLY_VALIDATED')
        v['tau']=0.0
        with self.assertRaises(ValueError): binding.validate_connection_identity(v,expected=copy.deepcopy(v))


if __name__ == '__main__':
    unittest.main()
