"""Tiny synthetic circuits/mocks only; never read a real H4 numerical endpoint."""
import copy
from fractions import Fraction as F
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('rho_test_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['rho_test_private']=pkg
from rho_test_private import h4_central_rho_plan as P
from rho_test_private.h4_d2_pauli_control import central_control_body
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator,SparsePauliOp


def body(n=2):
    b=QuantumCircuit(n)
    label='YZ' if n==2 else 'XYZ'
    b.append(PauliEvolutionGate(SparsePauliOp.from_list([(label,-.4)]),time=.2),range(n))
    b.y(0);b.x(0);b.global_phase=.37
    return b


class PreparationTests(unittest.TestCase):
    def test_original_wrapper_positive_real_imag_same_ray(self):
        for n in (2,3):
            b=body(n);controlled=central_control_body(b).body
            state=np.arange(1,1+(1<<n))+1j*np.arange(1<<n)
            z=np.vdot(state,Operator(b).data@state)/np.vdot(state,state)
            for axis,value in (('cosine',z.real),('sine',z.imag)):
                signal=P.toy_signal(P.toy_wrapper(controlled,axis),state,fixture_id='TOY_PHASE')
                self.assertAlmostEqual(signal['signal'],value,places=12)
                self.assertIsNone(signal['numerical_error_upper'])

    def test_sine_wrong_S_sign_detected(self):
        b=QuantumCircuit(2);b.global_phase=.43;c=central_control_body(b).body
        good=P.toy_wrapper(c,'sine');wrong=QuantumCircuit(3,1)
        wrong.h(2);wrong.compose(c,inplace=True);wrong.s(2);wrong.h(2);wrong.measure(2,0)
        s=[1,0,0,0]
        a=P.toy_signal(good,s,fixture_id='TOY_SDG')['signal'];v=P.toy_signal(wrong,s,fixture_id='TOY_SDG')['signal']
        self.assertAlmostEqual(a,-v);self.assertGreater(abs(a-v),.1)

    def test_controlled_phase_matters_full_global_does_not(self):
        b=QuantumCircuit(2);b.global_phase=.4;c=central_control_body(b).body
        a=P.toy_wrapper(c,'sine');w=a.copy();w.global_phase+=.7
        s=[1,0,0,0]
        self.assertAlmostEqual(P.toy_signal(a,s,fixture_id='TOY_GLOBAL')['signal'],P.toy_signal(w,s,fixture_id='TOY_GLOBAL')['signal'])
        b.global_phase=0;wrong=P.toy_wrapper(central_control_body(b).body,'sine')
        self.assertGreater(abs(P.toy_signal(a,s,fixture_id='TOY_REL')['signal']-P.toy_signal(wrong,s,fixture_id='TOY_REL')['signal']),.1)

    def test_raw_state_not_normalized_or_mutated(self):
        s=np.array([2+.1j,1j,0,1]);before=s.copy();w=P.toy_wrapper(central_control_body(body()).body,'cosine')
        a=P.toy_signal(w,s,fixture_id='TOY_NORM');b=P.toy_signal(w,7j*s,fixture_id='TOY_NORM')
        self.assertTrue(np.array_equal(s,before));self.assertAlmostEqual(a['signal'],b['signal'])
        self.assertGreater(a['raw_norm'],1)

    def test_compiler_gap_not_C_target_gap(self):
        w=P.toy_wrapper(central_control_body(body()).body,'cosine')
        out=P.toy_compare(w,w,[1,0,0,0],'cosine',B_saved=1.3,c_target_value=.5,fixture_id='TOY_SCOPE')
        self.assertEqual(out['delta_input_output'],0);self.assertGreater(out['selected_C_target_gap'],.01)
        self.assertIsNone(out['rho_new']);self.assertFalse(out['PR_law_guaranteed'])

    def test_B_saved_applied_once(self):
        b=QuantumCircuit(2);a=P.toy_wrapper(central_control_body(b).body,'cosine');b.global_phase=.2
        c=P.toy_wrapper(central_control_body(b).body,'cosine')
        x=P.toy_compare(a,c,[1,0,0,0],'cosine',B_saved=2,fixture_id='TOY_SCALE')
        self.assertAlmostEqual(x['delta_input_output'],2*(1-math.cos(.2)))

    def test_H4_shape_refused_before_propagation(self):
        fake=types.SimpleNamespace(num_qubits=9,num_clbits=1)
        with patch('qiskit.quantum_info.Statevector.evolve',side_effect=AssertionError('REAL_PROPAGATION')):
            with self.assertRaisesRegex(ValueError,'ARTIFICIAL_PROGRAM_SHAPE'):
                P.toy_signal(fake,[0]*256,fixture_id='TOY_NOT_H4')
        with self.assertRaisesRegex(ValueError,'ARTIFICIAL_FIXTURE'):
            P.toy_signal(fake,[],fixture_id='SAVED_H4')

    def test_QPY_structure_toy_no_matrix_or_propagation(self):
        c=QuantumCircuit(3,1);c.rz(.1,2);c.sx(2);c.cx(2,0);c.measure(2,0)
        buffer=io.BytesIO();qpy.dump(c,buffer);buffer.seek(0)
        with patch('qiskit.quantum_info.Statevector.evolve',side_effect=AssertionError('NO_PROPAGATION')):
            restored=qpy.load(buffer)[0];p=P.toy_program_profile(restored,compiled=True)
        self.assertEqual(p['ancilla'],2)
        wrong=QuantumCircuit(3,1);wrong.h(2);wrong.measure(2,0)
        with self.assertRaisesRegex(ValueError,'COMPILED_BASIS'):P.toy_program_profile(wrong,compiled=True)

    def test_wrong_measure_reset_or_mid_measure_rejected(self):
        for wrong in ('wrong_wire','reset','mid_measure'):
            c=QuantumCircuit(3,1)
            if wrong=='reset':c.reset(0)
            if wrong=='mid_measure':c.measure(0,0)
            c.measure(0 if wrong=='wrong_wire' else 2,0)
            with self.assertRaises(ValueError):P.toy_program_profile(c)

    def test_source_fingerprint_profile_mismatch_rejected(self):
        compiler=dict(backend_name=None,basis_gates=['rz','sx','x','cx'],coupling_map=None,layout_method=None,
            optimization_level=1,qiskit_version='1.3.0',routing_method=None,transpiler_seed=17)
        inp=dict(SOURCE='TOY_SRC',input_fingerprint='fp');m=dict(inp,compiler=compiler)
        P.check_saved_tags(inp,m,'TOY_SRC')
        for field in ('SOURCE','input_fingerprint','compiler'):
            bad=copy.deepcopy(m);bad[field]='corrupt'
            with self.assertRaises(ValueError):P.check_saved_tags(inp,bad,'TOY_SRC')

    def test_all_drafts_disabled_and_no_escalation_default(self):
        ts=[dict(key=f'{m}{i}_{a}',method=m,index=i,axis=a) for m,i,_ in P.TARGETS for a in P.AXES]
        ds=P.drafts(ts);P.require_draft(ds)
        self.assertEqual([len(x['targets']) for x in ds['candidates']],[2,4,0])
        for d in ds['candidates']:
            P.require_draft(d);self.assertFalse(d['numerical_path']['interval_tiers_default_enabled'])
        bad=copy.deepcopy(ds);bad['execution_authorized']=True
        with self.assertRaises(ValueError):P.require_draft(bad)
        with self.assertRaises(PermissionError):P.execution_request(bad)

    def test_CLI_execution_denied(self):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_h4_central_rho.py'),'--execute'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('NO_H4_EXECUTOR',r.stderr)

    def test_mock_interval_and_precision_outcomes(self):
        kw=dict(B=1,guard=F(1,100),width_target=F(1,10000),fixture_id='TOY_INTERVAL')
        self.assertEqual(P.mock_interval_diagnostic((0,0),(0,0),**kw)['status'],'LOCAL_BUDGET_MET_MOCK')
        self.assertEqual(P.mock_interval_diagnostic((0,0),(1,1),**kw)['status'],'LOCAL_BUDGET_EXCEEDED_MOCK')
        self.assertEqual(P.mock_interval_diagnostic((0,0),(0,F(1,100)),**kw)['status'],'WIDTH_INSUFFICIENT')
        self.assertEqual(P.mock_interval_diagnostic((0,0),(F(9999,1000000),F(10001,1000000)),**kw)['status'],'BUDGET_OVERLAP_STOP')

    def test_nonrefundable_technical_failure_no_duplicate_call(self):
        with tempfile.TemporaryDirectory() as td:
            b=P.C.CallBudget(Path(td),dict(toy_diagnostic=1));b.take('toy_diagnostic')
            try:raise RuntimeError('MOCK_TECHNICAL_INTERRUPTION')
            except RuntimeError:pass
            with self.assertRaises(ValueError):b.take('toy_diagnostic')
            with self.assertRaises(ValueError):b.take('compile')

    def test_no_compile_or_sampler_in_toy_diagnostic(self):
        with patch('qiskit.transpile',side_effect=AssertionError('NO_COMPILE')):
            w=P.toy_wrapper(central_control_body(body()).body,'cosine')
            x=P.toy_compare(w,w,[1,0,0,0],'cosine',B_saved=1,fixture_id='TOY_NO_COMPILE')
        self.assertEqual(x['delta_input_output'],0);self.assertFalse(x['formal_certificate_accepted'])


if __name__=='__main__':unittest.main(verbosity=2)
