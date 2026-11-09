"""Artificial-only emitter/lowering/run tests. No transpile, H4 or sampling."""
import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_track_d_h4_reference as bootstrap
import qiskit
with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden at bootstrap')),
      patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('P0 forbidden at bootstrap'))):
    bootstrap.bootstrap(os.environ['TRACK_D_ORIGIN'])
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit import Gate,Parameter,Measure
from qiskit.circuit.library import RZGate,SXGate,XGate,CXGate,RZZGate,CRZGate,XXPlusYYGate,CSwapGate,PhaseGate,HGate,SdgGate
from trottertracks import track_d_c7_2_toy as toy
from trottertracks.track_d_native_resources import NativeInstruction as I,NativeCounter,count_native,DEPTHS
from trottertracks.track_d_native_emitter import NativeLowerer,NativeSink,StreamBlock,circuit_chunk,legacy_to_stream
from trottertracks.track_d_p0_1q_bridge import optimize_native


def native_circuit(ops,nq,nc=0,phase=0.):
    qc=QuantumCircuit(nq,nc);qc.global_phase=phase
    classes=dict(rz=RZGate,sx=SXGate,x=XGate,cx=CXGate,measure=Measure)
    for op in ops:qc.append(classes[op.name](*op.params),list(op.qubits),list(op.clbits))
    return qc


def same(actual,expected):
    delta=Operator(actual).data-Operator(expected).data
    if np.max(abs(delta))>1e-10:raise AssertionError(float(np.max(abs(delta))))


def artificial_context():
    spec=json.loads((ROOT/'docs/research/track_d_c7_2_cx_structure_20261107/CASES.json').read_text())
    support=bootstrap.load_file('native_stream_existing_support',ROOT/'scripts/native_df_qiskit_validation_support.py')
    # Existing 2-mode fixture preparation/reference checks only; not H4 or a
    # CAR target in the stream path. Its small oracle is reused unchanged.
    return toy.context(spec,support.exterior_fock_lift)[0]


class LoweringTests(unittest.TestCase):
    def test_standard_definitions_and_declared_phase(self):
        gates=[HGate(),PhaseGate(.31),SdgGate(),RZZGate(-.23),CRZGate(.27),
               RZZGate(.17).control(1),XXPlusYYGate(.41,-.13),CSwapGate()]
        lower=NativeLowerer()
        import qiskit
        with patch.object(qiskit,'transpile',side_effect=AssertionError('new P0 forbidden')):
            for gate in gates:
                with self.subTest(gate=gate.name):
                    raw=QuantumCircuit(gate.num_qubits);raw.global_phase=.19
                    raw.append(gate,range(gate.num_qubits))
                    word=[];sink=NativeSink(gate.num_qubits,consumer=word.append,lowerer=lower)
                    sink.global_phase=.19;sink.append(gate,range(gate.num_qubits))
                    same(native_circuit(word,gate.num_qubits,phase=sink.global_phase),raw)
                    self.assertTrue(all(op.name in ('rz','sx','x','cx') for op in word))

    def test_actual_inverse_and_mapping_without_inverse_matrix(self):
        raw=QuantumCircuit(2);raw.global_phase=.17;raw.append(XXPlusYYGate(.37,.23),[0,1]);raw.rz(.29,0)
        word=[];sink=NativeSink(3,consumer=word.append)
        sink.compose(circuit_chunk(raw).inverse(),[2,0],inplace=True)
        expected=QuantumCircuit(3);expected.compose(raw.inverse(),[2,0],inplace=True)
        same(native_circuit(word,3,phase=sink.global_phase),expected)

    def test_stream_to_counter_without_word_retention(self):
        c=NativeCounter(3,1);sink=NativeSink(3,1,consumer=c.feed)
        sink.h(0);sink.cx(0,2);sink.rz(.17,2);sink.measure(2,0)
        self.assertFalse(hasattr(sink,'data'))
        self.assertEqual(c.result()['metrics']['measurement_count'],1)
        self.assertGreater(c.result()['metrics']['circuit_size'],4)

    def test_unsupported_parameter_custom_and_nonunitary(self):
        sink=NativeSink(3,consumer=lambda _:None)
        for gate in [Gate('opaque',2,[]),RZGate(Parameter('theta')),Gate('rz',1,[.2])]:
            with self.assertRaises(ValueError):sink.append(gate,range(gate.num_qubits))
        with self.assertRaises(ValueError):sink.compose(StreamBlock(1,lambda _:0.),[0])
        with self.assertRaises(ValueError):StreamBlock(1,lambda _:0.).inverse()


class EmitterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.context=artificial_context()

    def test_existing_head_relative_gaussian_and_step_boundary(self):
        c=self.context;module=toy.c3
        for kind in ('first','boundary','last'):
            with self.subTest(kind=kind):
                recipe=module.head_recipe(kind);raw=module.mixed_head(c,recipe)
                word=[]
                sink=legacy_to_stream(module.mixed_head,c,recipe,consumer=word.append)
                self.assertFalse(hasattr(sink,'data'))
                same(native_circuit(word,3,phase=sink.global_phase),raw)
                self.assertTrue(c.builder.relative_fusion)
        self.assertIs(module.QuantumCircuit,QuantumCircuit)

    def test_pr_head_tail_sequence_without_whole_wrapper_or_target(self):
        c=self.context;module=toy.c3
        events=(toy.event(c.arm.plan.atoms[0],math.atan(.2)),toy.event(c.arm.plan.atoms[-1],math.atan(.2)))
        # The unchanged tail emitter imports QC locally: bounded step fallback.
        tail=module.base.build_event_sequence(events,c.arm.plan.atoms,c.provider,optimized=True)
        with self.assertRaises(ValueError):
            legacy_to_stream(module.base.build_event_sequence,events,c.arm.plan.atoms,c.provider,
                             optimized=True,consumer=lambda _:None)
        counter=NativeCounter(3);lower=NativeLowerer()
        sink=NativeSink(3,consumer=counter.feed,lowerer=lower)
        for kind in ('first','boundary','last'):
            head=legacy_to_stream(module.mixed_head,c,module.head_recipe(kind),consumer=sink._send,lowerer=lower)
            sink.global_phase=math.fsum((sink.global_phase,head.global_phase))
            if kind!='last':sink.compose(circuit_chunk(tail,lower),inplace=True)
        self.assertFalse(hasattr(sink,'data'))
        self.assertGreater(counter.result()['metrics']['cx_count'],0)
        # No C3 prepare_steps, full wrapper, CAR target or H4 input was used.

    def test_reference_shell_actual_adjoint_chi_XY_emission(self):
        c=self.context;module=toy.c3
        events=(toy.event(c.arm.plan.atoms[0],math.atan(.2)),toy.event(c.arm.plan.atoms[-1],math.atan(.2)))
        tail=module.base.build_event_sequence(events,c.arm.plan.atoms,c.provider,optimized=True)
        with patch.object(module.base,'N',2):
            a=module.physical_segment(c,[tail]);b=module.physical_segment(c,[tail])
        # Existing artificial deterministic blocks have vacuum energy zero.
        # The tail vacuum phase is minus the signed event angles, plus scalar.
        chi=2*(c.scalar_phase-sum(e.unsigned_rotation_angle for e in events))
        vacuum=np.array([1.,0.,0.,0.],complex)
        self.assertLess(float(np.max(abs(Operator(a).data@Operator(b).data@vacuum
                        -np.exp(1j*chi)*vacuum))),1e-10)
        a_block,b_block=circuit_chunk(a),circuit_chunk(b)
        def emit_reference(consumer):
            return legacy_to_stream(module.base.reference_gadget,a_block,b_block,chi,consumer=consumer).global_phase
        reference=StreamBlock(5,emit_reference)
        original=module.base.reference_gadget(a,b,chi)
        for axis in ('X','Y'):
            expected=module.base.artificial.hadamard_wrapper(original,axis)
            word=[];sink=legacy_to_stream(module.base.artificial.hadamard_wrapper,reference,axis,consumer=word.append)
            actual=native_circuit(word,5,1,sink.global_phase)
            same(actual.remove_final_measurements(inplace=False),expected.remove_final_measurements(inplace=False))
            self.assertFalse(hasattr(sink,'data'))


class OneQTests(unittest.TestCase):
    def test_zero_opposite_signed_and_phase_identity(self):
        for word in [[I('rz',(0,),params=(0.,))],
                     [I('rz',(0,),params=(.23,)),I('rz',(0,),params=(-.23,))],
                     [I('rz',(0,),params=(2*math.pi,))],
                     [I('sx',(0,)),I('sx',(0,)),I('x',(0,))]]:
            result=optimize_native(iter(word),1,global_phase=.19)
            same(native_circuit(result['operations'],1,phase=result['global_phase']),native_circuit(word,1,phase=.19))
            self.assertEqual(result['output']['metrics']['circuit_size'],0)

    def test_oneq_removal_exposes_cx_but_no_rz_commuting(self):
        cx=I('cx',(0,1))
        result=optimize_native(iter([cx,I('rz',(0,),params=(0.,)),cx]),2)
        self.assertEqual(result['output']['metrics']['circuit_size'],0)
        for q in (0,1):
            kept=optimize_native(iter([cx,I('rz',(q,),params=(.23,)),cx]),2)
            self.assertEqual(kept['output']['metrics']['cx_count'],2)

    def test_chunk_invariance_and_official_synthesis_only(self):
        import qiskit
        from itertools import chain
        from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import Optimize1qGatesDecomposition
        word=[I('rz',(0,),params=(.31,)),I('x',(0,)),I('x',(0,)),I('cx',(0,1)),
              I('rz',(1,),params=(0.,)),I('cx',(0,1)),I('rz',(0,),params=(-.31,))]
        with (patch.object(qiskit,'transpile',side_effect=AssertionError('P0 forbidden')),
              patch.object(Optimize1qGatesDecomposition,'run',side_effect=AssertionError('full pass forbidden'))):
            expected=optimize_native(iter(word),2)
            for cut in range(len(word)+1):
                actual=optimize_native(chain(iter(word[:cut]),iter(word[cut:])),2)
                self.assertEqual(actual['operations'],expected['operations'])
                self.assertEqual(actual['global_phase'],expected['global_phase'])
        same(native_circuit(expected['operations'],2,phase=expected['global_phase']),native_circuit(word,2))


if __name__=='__main__':
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('new P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('new P0 forbidden'))):
        unittest.main(verbosity=2)
