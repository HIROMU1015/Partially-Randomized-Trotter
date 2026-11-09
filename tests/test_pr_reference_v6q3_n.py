"""Artificial finite-audit tests only; no H4 cache, circuit or science run."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load('scripts/run_pr_reference_v6q3_n.py', 'audit_driver')
checker = load('scripts/check_pr_reference_v6q3_n.py', 'audit_checker')
provider = load('src/trottertracks/pr_reference_v6q.py', 'unchanged_provider')


class FakeRZ:
    name = 'rz'
    num_qubits = 1
    num_clbits = 0
    condition = None

    def __init__(self):
        self.params = [.125]


class AuditTests(unittest.TestCase):
    def record(self, key='angle', role='angle', value=.125):
        return driver.generator(key, role, value, provider._binary_input)

    def test_signed_half_integer_grid(self):
        rows = [self.record(), self.record('phase', 'phase', -.5)]
        grid = driver.lattice_result(rows)
        self.assertEqual(grid['ell'], 4)
        checker.check_grid(rows, 4)

    def test_missing_half_is_detected(self):
        rows = [self.record()]
        rows[0]['required_ell'] = 3
        with self.assertRaises(ValueError):
            checker.check_grid(rows, 3)

    def test_corrupt_ratio_and_duplicate_are_detected(self):
        row = self.record()
        corrupted = copy.deepcopy(row)
        corrupted['numerator'] = '9'
        with self.assertRaises(ValueError):
            checker.check_grid([corrupted], 4)
        with self.assertRaises(ValueError):
            checker.check_grid([row, row], 4)

    def test_subnormal_generator_is_inconclusive_not_counterexample(self):
        row = self.record(value=float.fromhex('0x0.0000000000001p-1022'))
        grid = driver.lattice_result([row])
        self.assertEqual(grid['ell'], 1075)
        self.assertFalse(grid['sufficient'])
        checker.check_grid([row], 1075)

    def test_unsupported_or_nonfinite_literal_reject(self):
        for value in (True, complex(1, 0), float('nan'), float('inf')):
            with self.assertRaises((TypeError, ValueError)):
                self.record(value=value)

    def basis(self, gate=None, wires=(0,), metadata=None):
        gate = gate or FakeRZ()
        meta = metadata or NS(name='rz', qubits=(0,), parameters=('0.125',))
        return driver.audit_basis('artificial', [(gate, wires)], [meta], provider._binary_input,
                                  {'rz':FakeRZ})

    def test_basis_metadata_and_acceptance(self):
        result = self.basis()
        self.assertEqual(result['parameter_count'], 1)
        self.assertEqual(result['operations'][0]['parameters'][0]['hex'], (.125).hex())

    def test_basis_shape_and_condition_reject(self):
        for wires in ((8,), (0, 0)):
            with self.assertRaises(ValueError):
                self.basis(wires=wires)
        gate = FakeRZ()
        gate.condition = ('classical', 1)
        with self.assertRaises(ValueError):
            self.basis(gate=gate)

    def test_basis_metadata_mismatch_reject(self):
        with self.assertRaises(ValueError):
            self.basis(metadata=NS(name='rz', qubits=(0,), parameters=('0.25',)))


if __name__ == '__main__':
    unittest.main()
