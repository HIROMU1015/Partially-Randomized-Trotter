"""Artificial native QPY export tests; no transpilation or state action."""
import io
import json
import os
from fractions import Fraction as Q
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from qiskit import QuantumCircuit, qpy

ROOT = Path(__file__).resolve().parents[1]
B = runpy.run_path(str(ROOT / 'scripts/inspect_h4_arb_qpy.py'))['B']


class QpyTests(unittest.TestCase):
    def test_native_roundtrip_exact_params_phase_wires_both_axes(self):
        for axis in ('cosine','sine'):
            c = QuantumCircuit(3,1,global_phase=.3)
            c.sx(2); c.rz(.1,2); c.cx(2,0); c.x(1); c.sx(2); c.measure(2,0)
            data = io.BytesIO(); qpy.dump(c,data); data.seek(0)
            loaded = qpy.load(data)[0]
            with patch('qiskit.quantum_info.Statevector.evolve',side_effect=AssertionError('STRUCTURAL_ONLY')):
                p = B.descriptor_from_circuit(loaded, axis=axis, domain='ARTIFICIAL', fixture_id='TOY_QPY')
            self.assertEqual(Q(p['global_phase_exact']), Q(float(loaded.global_phase)))
            self.assertEqual(Q(p['instructions'][1]['params_exact'][0]), Q(.1))
            self.assertEqual(p['instructions'][2]['qubits'],[2,0])
            self.assertEqual(p['axis'],axis)
            self.assertEqual(len(p['instructions']),5)

    def test_native_sx_matrix_matches_installed_qiskit(self):
        from qiskit.circuit.library import SXGate
        self.assertEqual(SXGate().to_matrix().tolist(), [[.5+.5j,.5-.5j],[.5-.5j,.5+.5j]])

    def test_qpy_sha_fingerprint_binding_and_no_action(self):
        c = QuantumCircuit(2,1); c.sx(1); c.cx(1,0); c.measure(1,0)
        from arb_bridge_private import h4_r0_empirical as R
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'toy.qpy'
            with path.open('wb') as f: qpy.dump(c,f)
            kwargs = dict(axis='cosine', expected_sha256=R.C.sha(path), expected_fingerprint=R.api().fingerprint(c))
            with patch('qiskit.quantum_info.Statevector.evolve',side_effect=AssertionError('NO_ACTION')):
                p = B.inspect_qpy(path, **kwargs)
            self.assertEqual(p['domain'],'SAVED_QPY_NO_EXECUTION')
            self.assertEqual(p['origin']['state_actions'],0)
            for field in ('expected_sha256','expected_fingerprint'):
                bad = dict(kwargs, **{field:'WRONG'})
                with self.assertRaises(ValueError): B.inspect_qpy(path,**bad)

    def test_midmeasurement_condition_and_unsupported_gate_rejected(self):
        for kind in ('h','reset','mid_measure','wrong_wire','conditional_measure'):
            c = QuantumCircuit(2,1)
            if kind == 'mid_measure': c.measure(0,0)
            elif kind in ('h','reset'): getattr(c,kind)(0)
            c.sx(1)
            measurement = c.measure(0 if kind=='wrong_wire' else 1,0)
            if kind == 'conditional_measure': measurement.c_if(c.cregs[0],0)
            with self.assertRaises(ValueError): B.descriptor_from_circuit(c,axis='cosine')

    def test_native_name_does_not_authorize_custom_gate_semantics(self):
        from qiskit.circuit import Gate
        c = QuantumCircuit(2,1); c.append(Gate('sx',1,[]),[1]); c.measure(1,0)
        with self.assertRaisesRegex(ValueError,'STANDARD_NATIVE_GATE_CLASS'):
            B.descriptor_from_circuit(c,axis='cosine')

    def test_artificial_qpy_to_exact_descriptor_to_arb_end_to_end(self):
        arb_python = '/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-reference-fix-20261023/.c-reference-fix-venv/bin/python'
        for n in (2,3):
            c = QuantumCircuit(n,1,global_phase=.3)
            c.rz(.125,n-1); c.sx(n-1); c.sx(n-1); c.x(n-1); c.rz(-.125,n-1)
            c.cx(n-1,0); c.cx(n-1,0); c.measure(n-1,0)
            stream = io.BytesIO(); qpy.dump(c,stream); stream.seek(0)
            p = B.descriptor_from_circuit(qpy.load(stream)[0],axis='sine',domain='ARTIFICIAL',fixture_id='TOY_NATIVE')
            payload = dict(program=p,raw=[[float(i+1)/8,.125] for i in range(1 << (n-1))])
            code = """import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('toy_bridge',Path('src/trotterlib/h4_arb_bridge.py'))
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
d=json.load(sys.stdin)
signal,record=B.evaluate_artificial(d['program'],d['raw'],bits=64,fixture_id='TOY_NATIVE')
assert signal.contains(1)
print(record['status'])
"""
            result = subprocess.run([arb_python,'-B','-c',code],cwd=ROOT,input=json.dumps(payload),text=True,
                capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('ARTIFICIAL_VALIDATED_SIGNAL_NO_H4_RESULT',result.stdout)


if __name__ == '__main__': unittest.main(verbosity=2)
