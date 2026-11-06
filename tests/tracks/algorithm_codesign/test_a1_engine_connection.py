"""Invented gates/states only. No fixed molecular record is numerically used."""
from dataclasses import asdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
from trottertracks.algorithm_codesign import a1_engine_connection as a


class EngineConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global QuantumCircuit,Operator,Statevector,RZGate,XXPlusYYGate,HGate,PhaseGate,RZZGate,CZGate,SdgGate
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator,Statevector
        from qiskit.circuit.library import RZGate,XXPlusYYGate,HGate,PhaseGate,RZZGate,CZGate,SdgGate
        cls.dep=a.n2.n1.dependencies()
        d=cls.dep
        cls.h=d.hamiltonian.DFHamiltonian(.125,np.array([[.5,.25],[.25,-.25]],complex),np.array([.5,-.25]),
            (np.array([[1.,.25],[.25,-.5]],complex),np.array([[.75,-.5],[-.5,-.25]],complex)),{'fixture':'invented-engine-only'})
        cls.partition=d.split.split_df_hamiltonian_by_ld(cls.h,1)
        cls.prep=d.partial.prepare_df_partial_s2(cls.h,cls.partition,identity_policy='extract_identity_phase',coefficient_atol=0.,diagonal_sort='descending_abs',basis_construction_policy='fermionic_gaussian_jw')
        cls.saved=a.n2.n1.preparation_payload(cls.h,cls.partition,cls.prep,d.partial._block_hash_payload)
        cls.saved.update(constant_coefficient_hex=cls.prep.constant_coefficient.hex(),identity_coefficient_hex=cls.prep.extracted_identity_coefficient.hex())
        cls.definitions,proofs=d.policy.register_support_restricted_bases(cls.h,cls.prep)
        cls.policy=a.n2.n1.policy_payload(cls.prep,cls.definitions,proofs,source_commit='invented')
        cls.config,cls.distribution=d.rte.make_rte_config(cls.prep.rte_preparation.symbolic_tail,evolution_time=.02,rte_steps=1,finite_taylor_order=2,truncation_tolerance=1.)
        cls.config_record=dict(config=asdict(cls.config),distribution=asdict(cls.distribution))
        cls.law=dict(label_masses=[dict(label=c['component_id'],exact='1/'+str(len(cls.saved['components']))) for c in cls.saved['components']])
        cls.state=dict(num_system_qubits=2,qiskit_full_complex_hex=[[float(re).hex(),float(im).hex()] for re,im in [(1,0),(0,1),(0,0),(1,0)]])

    def make_engine(self):return a.SavedEngine(self.saved,self.policy,self.config_record,self.law,self.state,step_time=.02)

    def test_rehydration_preserves_each_named_matrix_and_basis_identity(self):
        e=self.make_engine();self.assertEqual(len(e.blocks),len(self.prep.deterministic_blocks))
        self.assertEqual({x.basis_hash for x in e.registry._definitions.values()},
            {x['definition']['basis_hash'] for x in self.policy['definitions']}|set(self.saved['basis_operations']))

    def test_changed_saved_matrix_hash_stops(self):
        row=copy.deepcopy(next(iter(self.saved['basis_operations'].values())))
        row[0]['matrix_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'matrix/metadata'):a.restore_operations(row)

    def test_missing_selected_basis_coverage_stops(self):
        policy=copy.deepcopy(self.policy);policy['definitions'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):a.SavedEngine(self.saved,policy,self.config_record,self.law,self.state,step_time=.02)

    def test_all_artificial_paths_exact_gate_order_matches_existing_full_builder(self):
        e=self.make_engine();labels=list(e.index);d=self.dep;interface=a.module('df_rte_circuit');scope=a.module('research_direction_full_scope')
        def signature(c):return [(i.operation.name,tuple(float(p).hex() for p in i.operation.params),tuple(c.find_bit(q).index for q in i.qubits)) for i in c.data]
        for n in (0,2):
            for path in product(labels,repeat=n+1):
                event=d.rte._make_event([e.index[path[-1]],*(e.index[x] for x in path[:-1])],e.components,self.distribution,(0,2).index(n))
                occurrence=interface.DFRTEEventSequenceCircuitRequest((event,),e.specs,controlled=True,ancilla_qubit=2,
                    tail_id=self.config.tail_id,tail_hash=self.config.tail_hash,occurrence_rte_steps=1)
                request=d.repeated.DFPartialS2RepeatedRequest(preparation=self.prep,step_time=.02,repetition_count=1,rte_config=self.config,rte_distribution=self.distribution,rte_occurrences=(occurrence,),step_seeds=(0,),controlled=True,ancilla_qubit=2,construction_policy='boundary_optimized')
                selected=scope.apply_run_threshold_policy(request,self.definitions,maximum_support_run_length=1,training_fingerprint=self.policy['policy_fingerprint'])
                evolution=d.repeated.QiskitDFPartialS2RepeatedCircuitBuilder().build(selected)
                for axis in ('X','Y'):
                    expected=d.hadamard.QiskitRPEHadamardInterrogationBuilder().build(d.hadamard.RPEHadamardInterrogationRequest(evolution,'cosine' if axis=='X' else 'sine',include_measurement=False)).circuit
                    observed=e.selected_circuit(n,path,axis)
                    self.assertEqual(signature(expected),signature(observed));self.assertEqual(float(observed.global_phase),0.)

    def test_exact_local_application_qubit_order_matches_statevector(self):
        v=[(F(i+1,8),F(7-i,16)) for i in range(8)]
        for gate,qs in [(XXPlusYYGate(.37,-.51),(2,0)),(RZZGate(.23).control(1),(1,2,0)),(RZGate(-.2).control(1),(2,0)),(CZGate(),(2,1))]:
            _,matrix,_=a.matrix_record(gate);out=a.apply_exact(v,matrix,qs)
            c=QuantumCircuit(3);c.append(gate,list(qs));expected=Statevector([complex(float(x),float(y)) for x,y in v]).evolve(c).data
            np.testing.assert_allclose([complex(float(x),float(y)) for x,y in out],expected,atol=1e-14,rtol=0)

    def test_exact_normalization_without_sqrt_and_scale_invariance(self):
        v=[(F(3),F(0)),a.ZERO,(F(0),F(4)),a.ZERO]
        self.assertEqual(a.norm_squared(v),25);self.assertEqual(a.born_probability(v,1),F(9,25))
        self.assertEqual(a.born_probability([(7*x,7*y) for x,y in v],1),F(9,25))
        with self.assertRaisesRegex(ValueError,'nonpositive'):a.born_probability([a.ZERO]*4,1)

    def test_probability_range_and_nonfinite_reject_without_clip(self):
        for p in (-F(1,10),F(11,10),float('nan'),float('inf')):
            with self.assertRaises((ValueError,OverflowError)):a.require_probability(p)

    def test_exact_kernel_XY_sign_for_known_controlled_phase(self):
        import math
        for theta in (.37,-.23):
            for axis in ('X','Y'):
                c=QuantumCircuit(2);c.h(1);c.p(theta,1)
                if axis=='Y':c.sdg(1)
                c.h(1);v=[(F(1),F(0)),a.ZERO,a.ZERO,a.ZERO]
                errors=[]
                for inst in c.data:
                    row=a.certify_gate(inst.operation);errors.append(row['error_upper'])
                    v=a.apply_exact(v,row['matrix'],[c.find_bit(q).index for q in inst.qubits])
                observed=2*a.born_probability(v,1)-1
                expected=math.cos(theta) if axis=='X' else math.sin(theta)
                self.assertAlmostEqual(float(observed),expected,places=14)
                self.assertLess(F(a.compose_error(max(errors),len(c.data),0)['gate_and_final_normalization_coordinate_upper']),F(1,10**12))

    def test_grid_endpoints_and_exact_threshold_count(self):
        self.assertEqual(a.threshold_count(0),0);self.assertEqual(a.threshold_count(1),2**53)
        for p in [F(0),F(1),F(1,16),F(1,3),F(1,2),F(7,8)]:
            self.assertEqual(a.threshold_count(p,3),sum(F(k,8)<p for k in range(8)))
            self.assertLessEqual(2*abs(F(a.threshold_count(p,3),8)-p),F(2,8))

    def test_interval_XXYY_formula_matches_independent_high_precision_exponential(self):
        import mpmath as mp
        mp.mp.dps=150;t=F(float(.37));b=F(float(-.51));theta=mp.mpf(t.numerator)/t.denominator;beta=mp.mpf(b.numerator)/b.denominator
        h=mp.matrix(4);h[1,2]=h[2,1]=2
        u=mp.expm(-1j*theta*h/4);r=mp.diag([mp.exp(1j*beta/2) if i%2==0 else mp.exp(-1j*beta/2) for i in range(4)])
        u=r*u*r.H;actual=a.interval_matrix(XXPlusYYGate(.37,-.51));mp.mp.dps=150
        for i in range(4):
            for j in range(4):
                for ball,point in ((actual[i][j].real,u[i,j].real),(actual[i][j].imag,u[i,j].imag)):
                    lo,hi=a.bounds(ball);tol=mp.mpf('1e-140')
                    self.assertLessEqual(mp.mpf(lo.numerator)/lo.denominator-tol,point);self.assertGreaterEqual(mp.mpf(hi.numerator)/hi.denominator+tol,point)

    def test_supported_gate_matrices_have_finite_conservative_enclosures(self):
        gates=[HGate(),SdgGate(),CZGate(),PhaseGate(.13),RZGate(-.3),RZGate(.7).control(1),RZZGate(.4).control(1),XXPlusYYGate(.4,.9)]
        for g in gates:
            row=a.certify_gate(g);self.assertGreaterEqual(row['error_upper'],0);self.assertLess(row['error_upper'],F(1,10**12))
        self.assertEqual(a.certify_gate(SdgGate())['error_upper'],0)
        with self.assertRaisesRegex(ValueError,'closed control'):a.gate_key(RZGate(.2).control(1,ctrl_state=0))

    def test_uniform_bound_includes_actual_probability_and_all_toy_gates(self):
        e=self.make_engine();certificate=e.qualify(0)
        labels=list(e.index)
        for n in (0,2):
            for path in product(labels,repeat=n+1):
                for axis in ('X','Y'):
                    c=e.selected_circuit(n,path,axis)
                    self.assertLessEqual(len(c.data),certificate['max_gate_count'])
                    self.assertTrue(all(a.gate_key(i.operation) in e.table for i in c.data))
        # Test probabilities on specified invented paths only; the all-path
        # envelope above is proved structurally, not fitted to these values.
        for n,path in [(0,(labels[0],)),(2,(labels[0],labels[-1],labels[0]))]:
            for axis in ('X','Y'):
                p=e.probability(n,path,axis);self.assertTrue(0<=p<=1)
                c=e.selected_circuit(n,path,axis);raw=np.array([1,1j,0,1,0,0,0,0],complex)
                v=Statevector(raw/np.linalg.norm(raw)).evolve(c);reference=float(v.probabilities([2])[0])
                self.assertAlmostEqual(float(p),reference,places=13)

    def test_final_renormalization_error_is_charged_and_long_bound_serializes(self):
        delta=F(1,1024);m=[[(1+delta,F(0)),a.ZERO],[a.ZERO,(F(1),F(0))]];v=[(F(1),F(0)),(F(1),F(0))]
        for _ in range(3):v=a.apply_exact(v,m,(0,))
        observed=abs(2*a.born_probability(v,0)-1);bound=a.compose_error(delta,3,0)
        self.assertLessEqual(observed,F(bound['gate_and_final_normalization_coordinate_upper']))
        json.dumps(a.compose_error(F(1,2**200),100000,0))
        with self.assertRaisesRegex(ValueError,'positive approximate'):a.compose_error(F(1),2,0)

    def test_runtime_rejects_uncertified_or_changed_gates(self):
        e=self.make_engine();label=next(iter(e.index))
        with self.assertRaisesRegex(ValueError,'certificate missing'):e.probability(0,(label,),'X')
        e.qualify(0)
        for row in e.table.values():row['matrix_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'matrix changed'):e.probability(0,(label,),'X')
        with self.assertRaisesRegex(ValueError,'positive Q'):e.selected_circuit(0,('unknown',),'X')

    def test_unsupported_basis_gate_or_gate_support_stops(self):
        with self.assertRaisesRegex(ValueError,'unsupported saved'):a.restore_operations([dict(name='unitary',parameters=[],qubits=[0],matrix_sha256='x')])
        with self.assertRaisesRegex(ValueError,'support'):a.apply_exact([a.ZERO]*4,[[(F(1),F(0))]],(0,0))

    def test_identity_mismatch_stops_before_any_numerical_processing(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'mismatch.json';path.write_text('changed')
            with patch.object(a.subprocess,'check_output',side_effect=['mismatch.json\n',b'original']),patch.object(a,'from_fixed_bundle',side_effect=AssertionError('no real decode')),patch.object(a.SavedEngine,'qualify',side_effect=AssertionError('no bound')):
                with self.assertRaisesRegex(ValueError,'byte mismatch'):a.inspect_fixed_bundle(tmp)

    def test_no_RNG_compile_or_eigensolver_in_artificial_engine_route(self):
        e=self.make_engine();label=next(iter(e.index))
        with patch.object(np.random,'default_rng',side_effect=AssertionError('RNG')),patch.object(np.linalg,'eigh',side_effect=AssertionError('eigh')),patch.object(self.dep.compiler,'transpile_and_measure_cost',side_effect=AssertionError('compile')),patch.object(self.dep.hamiltonian,'solve_df_ground_state',side_effect=AssertionError('solve')),patch.object(self.dep.partial,'prepare_df_partial_s2',side_effect=AssertionError('preparation')):
            e.qualify(0);e.probability(0,(label,),'Y')

if __name__=='__main__':unittest.main()
