"""Artificial gates/rationals/sectors only; fixed real files use byte/header checks."""
from fractions import Fraction as Q
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from flint import arb, acb, acb_mat, ctx

import ball_adapter as b
import mean_adapter as m
import export_adapter as e
import run_mean_enclosure as run


def toy_export():
    return {'n_qubits': 2, 'n_electrons': 1, 'ordered_sector': [1,2],
            'state': [['0x1p0','0x0p0'],['0x0p0','0x0p0']],
            'bases': {'u': {'operations': [{'name':'xx_plus_yy','qubits':[0,1], 'params':[.25.hex(), .125.hex()]},
                                         {'name':'rz','qubits':[1], 'params':[.125.hex()]}]}},
            'components': [{'id':'a','basis_id':'u','support':[0],'sign':1,'probability64':.75},
                           {'id':'b','basis_id':'u','support':[1],'sign':-1,'probability64':.25}],
            'deterministic_blocks': [], 'step_phase_angles':[.0625.hex(),(-.03125).hex()]}


def toy_literals():
    return {'q':'4','r':'2','saved_mean':{'real':'0.9','imag':'-0.1'},
            'axes': {axis:{'N':'100000','alpha':'0.1','epsilon':'0.5'} for axis in ['cosine','sine']}}


class EnclosureTests(unittest.TestCase):
    def setUp(self):
        self.old_prec, self.old_threads = ctx.prec, ctx.threads
        ctx.prec, ctx.threads = 96, 1

    def tearDown(self):
        ctx.prec, ctx.threads = self.old_prec, self.old_threads

    def test_exact_rational_and_binary64_differ(self):
        self.assertTrue(b.real(Q(1,3)).contains(b.real('1/3')))
        self.assertEqual(b.dyadic(b.real(.1)), Q.from_float(.1))
        self.assertFalse(b.real(.1).contains(b.real('1/10')))
        for value in [float('nan'),float('inf'),None, True]:
            with self.assertRaises(ValueError): b.real(value)

    def test_outward_endpoints_and_midpoint_radius_retained(self):
        x=b.real(Q(1,3))
        ep=b.endpoints(x)
        self.assertLessEqual(Q(ep['lower']),Q(1,3))
        self.assertGreaterEqual(Q(ep['upper']),Q(1,3))
        a=b.recenter(acb_mat([[x]]),norm=1)
        self.assertGreater(a.radius,0)
        self.assertLessEqual(abs(b.dyadic(a.center[0,0].real)-Q(1,3)),b.dyadic(a.radius))

    def test_cdf_exact_grid_and_boundary_side_right(self):
        law=m.categorical([.25,.25,.5])
        self.assertEqual(law['grid_masses'],['1/4','1/4','1/2'])
        law=m.categorical([.1,.2,.7])
        masses=[Q(x) for x in law['grid_masses']]
        self.assertEqual(sum(masses),1)
        for x in law['cdf64_hex']:
            self.assertLessEqual(abs(Q.from_float(float.fromhex(x))*(1<<53)-math.ceil(Q.from_float(float.fromhex(x))*(1<<53))),1)

    def test_component_renormalization_matches_source(self):
        raw=[.25,.75-1e-16]
        law=m.categorical(raw,component=True)
        self.assertEqual(law['choice_input64_hex'],[(x/math.fsum(raw)).hex() for x in raw])
        self.assertEqual(sum(Q(x) for x in law['grid_masses']),1)
        self.assertIsNone(m.categorical(raw)['component_fsum64_hex'])

    def test_invalid_categorical_stops_without_rng(self):
        for raw in [[],[float('nan')],[.5,-.5],[.2,.2],[1],[True]]:
            with self.assertRaises(ValueError):m.categorical(raw)

    def test_gate_analytic_formula_matches_qiskit_on_toy(self):
        from qiskit.circuit.library import RZGate,XXPlusYYGate
        for gate,wires in [(RZGate(.25),[0]),(XXPlusYYGate(.25,.125),[0,1])]:
            matrix=m.gate_matrix({'name':gate.name,'params':[float(x).hex() for x in gate.params],'qubits':wires})
            numeric=np.asarray(gate)
            for i in range(len(numeric)):
                for j in range(len(numeric)):
                    # Independent binary64 reference tolerance; not a rigorous gate test.
                    self.assertLess(abs(complex(matrix[i,j])-numeric[i,j]),1e-15)
            residual=b.dagger(matrix)*matrix-b.identity(len(numeric))
            self.assertTrue(residual.contains(acb_mat(len(numeric),len(numeric))))

    def test_sector_invariance_and_little_endian_row_action(self):
        export=toy_export()
        basis=m.basis_matrix(export['bases']['u'],[1,2],2,1)
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from qiskit.circuit.library import XXPlusYYGate
        circuit=QuantumCircuit(2);circuit.append(XXPlusYYGate(.25,.125),[0,1]);circuit.rz(.125,1)
        expected=np.asarray(Operator(circuit).data)[np.ix_([1,2],[1,2])]
        for i in range(2):
            for j in range(2):self.assertLess(abs(complex(basis.center[i,j])-expected[i,j]),1e-15)
        self.assertEqual(basis.norm,1)
        for states in [[1],[1,1],[0,1]]:
            with self.assertRaises(ValueError):m.basis_matrix(export['bases']['u'],states,2,1)
        with self.assertRaises(ValueError):m.gate_matrix({'name':'x','params':[],'qubits':[0]})

    def test_signed_component_mixture_and_unsupported_support(self):
        x=toy_export();bases={'u':m.basis_matrix(x['bases']['u'],[1,2],2,1)}
        a=m.component_average(x['components'],[Q(3,4),Q(1,4)],bases,[1,2])
        u=bases['u'];z=b.unit(acb_mat([[-1,0],[0,1]]))
        expected=b.product(b.product(u,z),b.adjoint(u))
        self.assertLess(b.frobenius(a.center-expected.center),a.radius+expected.radius+b.real('1e-25'))
        wrong=copy.deepcopy(x);wrong['components'][0]['support']=[2]
        with self.assertRaises(ValueError):m.enclose(wrong,m.categorical([.75,.25]),m.categorical([1.,0.]),q=1,r=1,orders=[0,2],angles=[0.,0.])

    def test_noncommuting_signed_event_average(self):
        x=b.unit(acb_mat([[0,1],[1,0]]));z=b.unit(acb_mat([[-1,0],[0,1]]))
        a=b.weighted_sum([(Q(1,3),x),(Q(2,3),z)],norm=1)
        event=m.event_average(a,[0,2],[Q(3,4),Q(1,4)],[0.,0.])
        exact=b.identity(2)*b.real(Q(3,4))-a.center*a.center*b.real(Q(1,4))
        delta=b.frobenius(event.center-exact)
        self.assertLessEqual(delta,event.radius+b.real('1e-25'))

    def test_builder_pi_phase_transfer_includes_float_additions(self):
        bound=m.builder_phase_bound(1,2)
        for count in range(7):
            accumulated=0.
            for _ in range(count):accumulated += math.pi
            difference=b.upper(abs(b.real(accumulated)-count*arb.pi()))
            self.assertLessEqual(difference,bound)
        self.assertGreater(bound,0)
        with self.assertRaises(ValueError):m.builder_phase_bound(0,2)

    def test_scalar_half_phase_uses_source_separate_parameters(self):
        export=toy_export();export['bases']={};export['deterministic_blocks']=[]
        f,r=m.half_operators(export,{})
        with ctx.workprec(224):
            expected=acb(0,b.real(export['step_phase_angles'][0])).exp()*acb(0,b.real(export['step_phase_angles'][1])).exp()
        enclosure=f.center[0,0]+acb(arb(0,f.radius),arb(0,f.radius))
        self.assertTrue(enclosure.contains(expected))
        self.assertEqual(r.center,b.identity(2))

    def test_no_extra_normalization_r_lives_inside_q(self):
        a=b.unit(acb_mat([[-1,0],[0,1]]))
        event=m.event_average(a,[0,2],[Q(1),Q(0)],[.125,.125])
        result=b.power(b.power(event,2),4)
        with ctx.workprec(224):expected=acb(0,b.real(1)).exp()
        observed=result.center[0,0]+acb(arb(0,result.radius),arb(0,result.radius))
        self.assertTrue(observed.contains(expected))

    def test_noncommuting_product_order_and_power(self):
        a=b.recenter(acb_mat([[1,1],[0,1]]),norm=2)
        z=b.unit(acb_mat([[-1,0],[0,1]]))
        correct=b.product(a,z);reverse=b.product(z,a)
        self.assertNotEqual(correct.center,reverse.center)
        result=b.power(a,5)
        self.assertEqual(result.center,acb_mat([[1,5],[0,1]]))
        for exponent in [-1,1.5,True]:
            with self.assertRaises(ValueError):b.power(a,exponent)

    def test_global_radius_product_uses_general_norms(self):
        a=b.recenter(acb_mat([[2]]),b.real('1/100'),norm=3)
        result=b.power(a,3)
        self.assertLessEqual(abs(Q(201,100)**3-8),b.dyadic(result.radius))
        self.assertGreater(result.radius,3*a.radius)  # Cannot assume contraction for arbitrary matrices.

    def test_large_q_with_global_radius_and_direct_interval(self):
        # Dense artificial unitary, unrelated to any physical input or sector.
        size=8
        h=acb_mat([[b.real((-1)**((i&j).bit_count()))/b.real(size).sqrt() for j in range(size)] for i in range(size)])
        d=acb_mat(size,size)
        for i in range(size):d[i,i]=acb(0,b.real(Q(i+1,32))).exp()
        raw=h*d*b.dagger(h)
        with ctx.workprec(224):
            hi_h=acb_mat([[b.real((-1)**((i&j).bit_count()))/b.real(size).sqrt() for j in range(size)] for i in range(size)])
            hi_d=acb_mat(size,size)
            for i in range(size):hi_d[i,i]=acb(0,b.real(Q((i+1)*131072,32))).exp()
            exact=hi_h*hi_d*b.dagger(hi_h)
        global_result=b.power(b.unit(raw),131072)
        direct=raw**131072
        rectangle=global_result.center[0,0]+acb(arb(0,global_result.radius),arb(0,global_result.radius))
        self.assertTrue(rectangle.contains(exact[0,0]))
        self.assertLess(global_result.radius,b.real('1e-15'))
        self.assertGreaterEqual(b.upper(direct[0,0].real.rad()),0)
        self.large_q_comparison={'global_radius':str(global_result.radius),'direct_entry_radius':str(direct[0,0].real.rad())}

    def test_measurement_re_im_and_normalized_saved_ray(self):
        operator=b.unit(acb_mat([[acb(0,1),0],[0,1]]))
        value, info=b.expectation(operator,[[2.,0.],[0.,0.]])
        self.assertTrue(value.contains(acb(0,1)))
        self.assertEqual(info['saved_vector_norm_squared']['lower'],'4')
        self.assertEqual((1+value.real)/2,b.real(Q(1,2)))
        self.assertTrue(((1+value.imag)/2).contains(1))
        with self.assertRaises(ValueError):b.expectation(operator,[[0.,0.],[0.,0.]])

    def test_toy_hadamard_sdg_convention_and_plus_probability(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        for axis, expected in [('cosine',math.cos(.25)),('sine',math.sin(.25))]:
            circuit=QuantumCircuit(2);circuit.x(0);circuit.h(1);circuit.cp(.25,1,0)
            if axis=='sine':circuit.sdg(1)
            circuit.h(1)
            probability=Statevector.from_instruction(circuit).probabilities(qargs=[1])[0]
            self.assertAlmostEqual(2*probability-1,expected,places=14)
            self.assertAlmostEqual(probability,(1+expected)/2,places=14)

    def test_strict_budget_unknown_bridge_and_ambiguous_interval(self):
        # Compute a known artificial endpoint with a downward epsilon -> strict exclusion.
        shot=(2*b.real(20).log()/100).sqrt()
        exact_epsilon=Q(b.dyadic(shot.lower()))
        result=b.classify_interval(arb(0),0.,100,Q(1,10),exact_epsilon)
        self.assertEqual(result['categorical_decision'],'SUFFICIENT_CONDITION_EXCLUDED')
        self.assertIsNone(result['U_exec_upper'])
        self.assertEqual(result['exec_decision'],'UNDETERMINED')
        ambiguous=b.classify_interval(arb(0,1),0.,100,.1,.5)
        self.assertEqual(ambiguous['categorical_decision'],'UNDETERMINED')
        yes=b.classify_interval(arb(0),0.,100,.1,.5,bridge=Q(1,100))
        self.assertEqual(yes['exec_decision'],'SUFFICIENT_CONDITION_CONFIRMED')
        with self.assertRaises(ValueError):b.classify_interval(arb(0),0.,100,.1,.5,bridge=-1)

    def test_nonfinite_and_missing_input_stops(self):
        with self.assertRaises(ValueError):b.recenter(acb_mat([[acb('nan')]]))
        with self.assertRaises(ValueError):b.recenter(acb_mat([[1]]),error=-1)
        with self.assertRaises(ValueError):m.event_average(b.unit(b.identity(2)),[1],[Q(1)],[0.])

    def test_basis_export_on_artificial_gate_registry(self):
        sys_path=run.ROOT/'src'
        import sys
        sys.path.insert(0,str(sys_path))
        from trotterlib.df_rte_tail import DFBasisRegistry
        from qiskit.circuit.library import RZGate,XXPlusYYGate,XGate
        registry=DFBasisRegistry()
        definition=registry.register([(RZGate(.25),(0,)),(XXPlusYYGate(.125,.5),(0,1))],num_system_qubits=2)
        result=e.export_basis(definition.basis_id,definition.basis_hash,definition.metadata.operations,definition.runtime_operations,n_qubits=2)
        self.assertEqual(len(result['operations']),2)
        desc=registry.register([(XGate(),(0,))],num_system_qubits=2)
        with self.assertRaises(ValueError):e.export_basis(desc.basis_id,desc.basis_hash,desc.metadata.operations,desc.runtime_operations,n_qubits=2)

    def test_full_preparation_export_on_synthetic_object(self):
        import sys
        sys.path.insert(0,str(run.ROOT/'src'))
        from trotterlib.df_rte_tail import DFBasisRegistry
        from qiskit.circuit.library import RZGate
        registry=DFBasisRegistry();d=registry.register([(RZGate(.125),(0,))],num_system_qubits=2)
        component=SimpleNamespace(component_id='toy',coefficient_sign=-1,coefficient_abs=.5,
            coefficient=-.5,probability=1.,basis_id=d.basis_id,basis_hash=d.basis_hash,
            diagonal_pauli_support=(0,),df_fragment_id='toy-fragment')
        block=SimpleNamespace(block_id='toy-block',block_kind='one_body',basis_id=d.basis_id,
            basis_hash=d.basis_hash,basis_change_operations=d.metadata.operations,
            runtime_basis_operations=d.runtime_operations,diagonal_eigenvalues=(.25,.5))
        extraction=SimpleNamespace(components=[component],tail_hash='toy-tail',basis_definition=lambda _:d)
        prep=SimpleNamespace(rte_preparation=SimpleNamespace(symbolic_tail=SimpleNamespace(components=[component])),
            tail_extraction=extraction,deterministic_blocks=[block],preparation_hash='toy-prep',
            exact_rte_lambda_r=.5,constant_coefficient=.125,extracted_identity_coefficient=.25,
            identity_policy='extract_identity_phase',coefficient_atol=1e-12,
            deterministic_fragment_indices=[0],randomized_block_indices=[1])
        result=e.export_preparation(prep,np.array([1,0],complex),[1,2],n_qubits=2,n_electrons=1,delta=.02)
        self.assertIn('SOURCE_REPLAY_GENERATED',result['provenance'])
        self.assertEqual(result['components'][0]['coefficient'],-.5)
        self.assertEqual(len(result['step_phase_angles']),2)
        json.dumps(result,allow_nan=False)

    def test_default_does_not_load_real_arrays_or_construct_cdf(self):
        with patch.object(np,'load',side_effect=AssertionError('No real arrays')), patch.object(m,'categorical',side_effect=AssertionError('No real CDF')), patch.object(e,'replay',side_effect=AssertionError('No SOURCE replay')), patch.object(run,'execute',side_effect=AssertionError('No science')),contextlib.redirect_stdout(io.StringIO()) as output:
            run.main([])
        self.assertEqual(json.loads(output.getvalue())['real_numeric_calls'],0)

    def test_precision_selection_is_bounded_and_driven_by_interval(self):
        export=toy_export();law=m.categorical([.75,.25]);order=m.categorical([1.,0.])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);record={'enclosures':[],'scientific_calls':{'complex_enclosure_started':0}}
            with patch.object(run,'classify_both',return_value=({},'UNDETERMINED')):
                run.evaluate(export,law,order,[.125,.0625],toy_literals(),path,record)
            self.assertEqual([item['bits'] for item in record['enclosures']],[96,160,224])
            self.assertEqual(record['scientific_calls']['complex_enclosure_started'],3)
            with zipfile.ZipFile(path/'enclosure_operators.zip') as z:self.assertEqual(len(z.namelist()),3)
        ctx.threads=1

    def test_driver_synthetic_success_failure_and_fresh_output(self):
        export=toy_export()
        meta={'inputs':[], 'old_result':{'context':{'config':{'lambda_r':.5},'distribution':{'order_probabilities':[1.,0.]}},'fixed_literals':toy_literals()},
              'b1_result':{'scalar_trace':{'theta64':[.125,.0625]}}}
        export['lambda_r64']=.5
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'new'
            with patch.object(run,'source_metadata',return_value={'commit':'synthetic'}), patch.object(run,'fixed_metadata',return_value=meta),patch.object(e,'replay',return_value=export),patch.object(run,'classify_both',return_value=({},'CONDITIONAL_IDEAL_CATEGORICAL_SUFFICIENT_CONDITION_CONFIRMED')):
                run.execute(path)
                with self.assertRaises(FileExistsError):run.execute(path)
            result=json.loads((path/'RESULT.json').read_text())
            self.assertEqual(len(result['enclosures']),1)
            self.assertFalse(result['RPE_tolerance_certified'])
            self.assertIsNone(result['bridges']['virtual_float_clip_Bernoulli'])
            failed=Path(temp)/'failed'
            with patch.object(run,'source_metadata',return_value={'commit':'synthetic'}),patch.object(run,'fixed_metadata',return_value=meta),patch.object(e,'replay',side_effect=ValueError('synthetic missing gate')):
                with self.assertRaises(ValueError):run.execute(failed)
            self.assertEqual(json.loads((failed/'RESULT.json').read_text())['execution_status'],'INTERRUPTED_OR_INVALID_MANDATORY_STOP')
        ctx.threads=1

    def test_run_requires_output_before_science(self):
        with patch.object(run,'execute') as called,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):run.main(['--run'])
        called.assert_not_called()


if __name__ == '__main__':
    unittest.main()
