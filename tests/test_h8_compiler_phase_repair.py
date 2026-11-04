"""Three-qubit artificial integration through the original seven checks."""
from contextlib import contextmanager, redirect_stdout
import io
import math
import gzip
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h8_long_time_adapter as a
import h8_long_time_runtime as runtime
import repair_h8_compiler_phase as repair


class RepairTests(unittest.TestCase):
    def test_original_checks_and_phase_provenance_with_artificial_inputs(self):
        import h46_long_time_native as native
        row = a.candidate('PR', '2nd', 8, 32, 32, 4)
        endpoint = dict(candidate=row, status='ELIGIBLE', N_cos=2, N_sin=3,
                        log_B=0., log_B_margin=1e-12)
        initial = np.array([1., 0., 0., 0.], complex)
        evolution = QuantumCircuit(3)
        evolution.append(XXPlusYYGate(.37, .19), [0, 1])
        evolution.append(XXPlusYYGate(.41, .19), [0, 1])

        def wrapper(evo, *, axis, **kwargs):
            qc = QuantumCircuit(3, 1, name=axis)
            qc.h(2)
            qc.compose(evo.circuit, inplace=True)
            if axis == 'sine':
                qc.sdg(2)
            qc.h(2)
            qc.measure(2, 0)
            return qc

        def compile_raw(qc, fingerprint, representative):
            compiled = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'],
                optimization_level=1, seed_transpiler=17, num_processes=1)
            value = dict(compiler=a.COMPILER, fingerprint=fingerprint,
                metrics=dict(rz_count=compiled.count_ops().get('rz', 0)))
            cache = native.PRIVATE / 'compiled_cache'
            a.save(cache / (fingerprint + '.json'), value)
            for label, circuit in [('wrapper', qc), ('compiled', compiled)]:
                stream = io.BytesIO()
                qpy.dump(circuit, stream)
                (cache / (fingerprint + '_' + label + '.qpy.gz')).write_bytes(gzip.compress(stream.getvalue()))
            return value, compiled, False

        def fingerprint(qc):
            return a.digest([qc.num_qubits, qc.num_clbits, float(qc.global_phase),
                [[i.operation.name, [float(p) for p in i.operation.params],
                  [qc.find_bit(q).index for q in i.qubits], [qc.find_bit(c).index for c in i.clbits]] for i in qc.data]])

        @contextmanager
        def reuse():
            yield NS(fingerprint=fingerprint)

        # Load the circuit module before replacing its simulator. Its real
        # numerical backend is never called in this artificial test.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
        import trotterlib.df_trotter.circuit as circuits
        ctx = NS(np=np, qstate=initial, preps={8: NS(preparation_hash='artificial', num_system_qubits=2)},
                 scalar=lambda row: 0., native=NS(build_native_hadamard_wrapper=wrapper,
                    NativeEvolution=lambda *args: NS(circuit=args[0])))
        with tempfile.TemporaryDirectory() as tmp:
            private, output = Path(tmp) / 'private', Path(tmp) / 'new'
            output.mkdir()
            a.save(private / 'runs/stage_B02/prices' / row['id'] / '0/events_PRIVATE.json',
                dict(seeds=['artificial'], events=[], conditional=False, cohort='EXPLORATION'))
            names = {'trottertracks.resource_applicability.h6_hash_reuse_v1': NS(use_fingerprint_reuse=reuse)}
            with patch.dict(sys.modules, names), patch.object(repair, 'PRIVATE', private), \
                 patch.object(repair, 'OUTPUT', output), \
                 patch.object(repair, 'SUPPLEMENT', dict(SOURCE=repair.PREVIOUS_SOURCE, source_files={})), \
                 patch.dict(repair.CONFIGS, {'stage_B02': {}}), patch.object(runtime, 'CONTEXT', ctx), \
                 patch.object(runtime, 'independent_trajectory', return_value=initial.copy()), \
                 patch.object(circuits, 'simulate_statevector', side_effect=lambda qc, v: Statevector(v).evolve(qc).data), \
                 patch.object(native, 'sample_requests', return_value=([], ['artificial'], [])), \
                 patch.object(native, 'compose', return_value=NS(circuit=evolution)), \
                 patch.object(native, 'compile_cached', side_effect=compile_raw), \
                 patch.object(repair.os, 'sched_getaffinity', return_value={0}), redirect_stdout(io.StringIO()):
                value = repair.finish_trial((row, endpoint, 'EXPLORATION', 0, 'stage_B02'))
            self.assertEqual(value['status'], 'PASS')
            self.assertEqual(len(value['checks']), 7)
            self.assertTrue(all(c['error'] < c['tolerance'] and not c['phase_alignment'] for c in value['checks']))
            self.assertEqual(value['compiler_phase_supplement']['axes']['cosine']['delta_phase_quarters'], 2)
            self.assertEqual(len(list((output / row['id'] / 'checkpoints').glob('*.npy'))), 7)
            self.assertFalse((private / 'runs/stage_B02/result.json').exists())

            # Resume a complete artificial checkpoint set to test the same
            # machinery used after a technical interruption. A corrected
            # vector must not receive the global phase a second time.
            previous = private / 'phase_repair01'
            output.rename(previous)
            supplement = dict(SOURCE=repair.PREVIOUS_SOURCE, source_files={})
            a.save(previous / 'scope.json', dict(supplement=supplement,
                jobs=[dict(candidate=row, index=0, batch='stage_B02')]))
            output.mkdir()
            with patch.dict(sys.modules, names), patch.object(repair, 'PRIVATE', private), \
                 patch.object(repair, 'OUTPUT', output), patch.object(repair, 'SUPPLEMENT', dict(SOURCE='second-artificial')), \
                 patch.dict(repair.CONFIGS, {'stage_B02': {}}), patch.object(runtime, 'CONTEXT', ctx), \
                 patch.object(runtime, 'independent_trajectory', side_effect=AssertionError('must reuse algebraic vector')), \
                 patch.object(circuits, 'simulate_statevector', side_effect=AssertionError('must reuse vectors')), \
                 patch.object(native, 'sample_requests', return_value=([], ['artificial'], [])), \
                 patch.object(native, 'compose', side_effect=AssertionError('must restore saved evolution')), \
                 patch.object(native, 'compile_cached', side_effect=AssertionError('must reuse raw QPY')), \
                 patch.object(repair.phase, 'check_gate', side_effect=AssertionError('must reuse local operator tests')), \
                 patch.object(repair.os, 'sched_getaffinity', return_value={0}), redirect_stdout(io.StringIO()):
                resumed = repair.finish_trial((row, endpoint, 'EXPLORATION', 0, 'stage_B02'))
            self.assertEqual(resumed['status'], 'PASS')
            self.assertEqual(len(resumed['checks']), 7)
            for f in (output / row['id'] / 'checkpoints').glob('*.json'):
                self.assertTrue(a.read(f)['reused_previous_repair_vector'])

    def test_cache_search_uses_saved_recovery_qpy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / 'original', root / 'recovery'
            first.mkdir(); second.mkdir()
            (first / 'fp.json').write_text('{}')
            self.assertIsNone(repair.find_raw_cache([first], 'fp'))
            (second / 'fp.json').write_text('{}')
            (second / 'fp_compiled.qpy.gz').write_bytes(b'artificial')
            self.assertEqual(repair.find_raw_cache([first, second], 'fp'),
                             (second / 'fp.json', second / 'fp_compiled.qpy.gz'))

    def test_compiled_identity_includes_parameters_and_global_phase(self):
        left = QuantumCircuit(2)
        left.rz(.1, 0)
        left.cx(0, 1)
        right = left.copy()
        self.assertTrue(repair.same_compiled_circuit(left, right))
        right.global_phase = math.pi
        self.assertFalse(repair.same_compiled_circuit(left, right))
        right = QuantumCircuit(2)
        right.rz(.2, 0)
        right.cx(0, 1)
        self.assertFalse(repair.same_compiled_circuit(left, right))


if __name__ == '__main__':
    unittest.main()
