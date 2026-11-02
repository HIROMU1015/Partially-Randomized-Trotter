"""Synthetic small states/programs only; no real H4 endpoint evaluation."""
import copy
from fractions import Fraction as F
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('r0_test_private');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['r0_test_private']=pkg
from r0_test_private import h4_r0_empirical as R
from r0_test_private.h4_d2_pauli_control import central_control_body
from r0_test_private.h4_central_rho_plan import toy_wrapper
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator,SparsePauliOp


def toy_body(n=2):
    b=QuantumCircuit(n);b.append(PauliEvolutionGate(SparsePauliOp.from_list([('YZ' if n==2 else 'XYZ',-.4)]),time=.2),range(n))
    b.x(0);b.y(1);b.global_phase=.37
    return b


def selected_fixture():
    comp=[dict(label='XI',sign=-1,coefficient_abs=.2),dict(label='YZ',sign=1,coefficient_abs=.3)]
    cfg=dict(T=.8,q=8,r=4,K=4,method='PR')
    model=dict(representation='pauli_jw',step_time=.1,cases=dict(PR=dict(config=cfg,
        blocks=[dict(label='XY',coefficient=.31),dict(label='ZZ',coefficient=-.27)],
        components=comp,native_scalar_phases=[.037],law=dict(angles=[.19,.07,.03]))))
    events=[]
    for step in range(8):
        for j in range(4):
            cid=str((step+j)%2);c=comp[int(cid)]
            events.append(dict(outer_step_index=step,inner_RTE_position=j,taylor_order=0,
                rotation_angle=.19,phase=dict(real=1,imag=0),rotation_component_id=cid,product_component_ids=[],
                application_sequence=[dict(component_id=cid,coefficient_sign=c['sign'],coefficient_abs=c['coefficient_abs'])]))
    return model,dict(method='PR',trajectory_id='TOY_SELECTED_NOT_LAW_MEAN',events=events)


class R0Tests(unittest.TestCase):
    def test_QPY_source_control_wrapper_match_without_evolution(self):
        b=toy_body();d=central_control_body(b);w=toy_wrapper(d.body,'cosine')
        buf=io.BytesIO();qpy.dump([b,d.body,w],buf);buf.seek(0);b,c,w=qpy.load(buf)
        with patch('qiskit.quantum_info.Statevector.evolve',side_effect=AssertionError('STRUCTURE_ONLY')):
            R.validate_controlled(b,c,w,json.loads(json.dumps(d.sidecar)))
            R.validate_program(w,compiled=False,n=2)
            self.assertEqual(R.api().fingerprint(w),R.api().fingerprint(w.copy()))

    def test_wrong_relative_phase_or_wrapper_rejected(self):
        b=toy_body();d=central_control_body(b);w=toy_wrapper(d.body,'cosine')
        bad=d.body.copy();bad.data.pop()
        with self.assertRaises(ValueError):R.validate_controlled(b,bad,w,d.sidecar)
        bad=toy_wrapper(d.body,'sine')
        with self.assertRaises(ValueError):R.validate_controlled(b,d.body,bad,d.sidecar)

    def test_fixed_basis_and_measurement_guards(self):
        c=QuantumCircuit(3,1);c.rz(.1,2);c.sx(2);c.cx(2,0);c.measure(2,0)
        R.validate_program(c,compiled=True,n=2)
        for kind in ('wrong_measure','reset','mid_measure','bad_basis'):
            bad=QuantumCircuit(3,1)
            if kind=='reset':bad.reset(0)
            if kind=='mid_measure':bad.measure(0,0)
            if kind=='bad_basis':bad.h(0)
            bad.measure(0 if kind=='wrong_measure' else 2,0)
            with self.assertRaises(ValueError):R.validate_program(bad,compiled=True,n=2)

    def test_unclipped_raw_ray_observable_and_no_state_replacement(self):
        for n in (2,3):
            b=toy_body(n);w=toy_wrapper(central_control_body(b).body,'cosine')
            raw=np.arange(1,1+(1<<n))+1j*np.arange(1<<n);before=raw.copy()
            encoded=[[float(z.real),float(z.imag)] for z in raw];nu=R.raw_norm(encoded)
            out,r=R.propagate(w,raw,nu)
            expected=(np.vdot(raw,Operator(b).data@raw)/float(nu)).real
            self.assertAlmostEqual(r['signal'],expected,places=12);self.assertTrue(np.array_equal(raw,before))
            self.assertFalse(r['probability_API_used']);self.assertFalse(r['output_norm_division']);self.assertEqual(len(out),2*len(raw))

    def test_selected_C_action_noncommuting_schedule_and_sign(self):
        model,join=selected_fixture();raw=np.array([1,2j,3,.7j]);nu=R.raw_norm([[z.real,z.imag] for z in raw])
        out,r=R.selected_C_action(model,join,raw,nu)
        # Independent small dense operator composition, no real input, no sampler.
        a=R.api();I=np.eye(4);case=model['cases']['PR'];forward=[]
        for b in case['blocks']:
            angle=float(F(b['coefficient'])*F(.1)/2)
            forward.append(math.cos(angle)*I-1j*math.sin(angle)*a.pauli(b['label']))
        U=I.copy()
        for step in range(8):
            for gate in forward:U=gate@U
            for e in join['events'][4*step:4*step+4]:
                comp=case['components'][int(e['rotation_component_id'])];P=comp['sign']*a.pauli(comp['label'])
                U=(math.cos(.19)*I-1j*math.sin(.19)*P)@U
            for gate in reversed(forward):U=gate@U
            U=np.exp(.037j)*U
        self.assertTrue(np.allclose(out,U@raw,atol=2e-13,rtol=0))
        self.assertAlmostEqual(r['signal'],(np.vdot(raw,U@raw)/float(nu)).real,places=12)
        self.assertFalse(r['PR_law_mean_used']);self.assertIsNone(r['numerical_error_upper'])

    def test_C_event_sign_and_angle_mismatch_stop(self):
        model,join=selected_fixture()
        for field,value in [('coefficient_sign',1),('coefficient_abs',.8)]:
            bad=copy.deepcopy(join);bad['events'][0]['application_sequence'][0][field]=value
            with self.assertRaises(ValueError):R.selected_C_action(model,bad,np.array([1,0,0,0]),F(1))
        bad=copy.deepcopy(join);bad['events'][0]['rotation_angle']=.2
        with self.assertRaises(ValueError):R.selected_C_action(model,bad,np.array([1,0,0,0]),F(1))

    def test_saved_PF_interval_respected_without_new_interval_backend(self):
        c=dict(implemented_law=dict(PF=dict(corrected_encoded_mean=dict(real=dict(lower='1',upper='3')))),state_norm=dict(lower='2',upper='2'))
        target,r=R.saved_PF_target(c,F(2));self.assertEqual(target,1);self.assertEqual(r['normalized_interval_exact'],['1/2','3/2'])
        self.assertFalse(r['compiled_signal_error_enclosed'])

    def test_delta_eta_direct_d_and_B_once(self):
        r=R.compare_signals(F(1,10),.3,.31,2,F(1,100))
        self.assertAlmostEqual(r['delta'],.02);self.assertAlmostEqual(r['eta'],.4);self.assertAlmostEqual(r['d'],.42)
        self.assertTrue(r['triangle_verified']);self.assertEqual(r['status'],'EMPIRICAL_ONLY_NO_RHO_UPPER')
        self.assertIsNone(r['rho_upper']);self.assertFalse(r['LOCAL_BUDGET_MET'])

    def test_small_empirical_gap_never_certified(self):
        r=R.compare_signals(0,0,0,1,F(1,1000));self.assertEqual(r['empirical_relation'],'BELOW_SAVED_BUDGET')
        self.assertFalse(r['formal_certificate_accepted']);self.assertIsNone(r['PR_law_rho'])

    def test_nonrefundable_limits_five_state_actions(self):
        self.assertEqual(sum(R.ACTION_LIMITS.values()),5)
        with tempfile.TemporaryDirectory() as td:
            b=R.C.CallBudget(Path(td),R.LIMITS);b.take('PF_input')
            with self.assertRaises(ValueError):b.take('PF_input')
            with self.assertRaises(ValueError):b.take('sine')
            with self.assertRaises(ValueError):b.take('compile')


if __name__=='__main__':unittest.main(verbosity=2)
