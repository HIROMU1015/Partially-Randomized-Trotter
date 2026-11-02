"""Artificial-only production assembly and validated arithmetic regression."""
import copy
from fractions import Fraction as Q
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from flint import arb,acb,acb_mat,ctx,fmpq
ROOT=Path(__file__).resolve().parents[1]
def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
e=load('production_ball','src/trotterlib/h4_ball_evidence.py')
c=load('production_control','src/trotterlib/h4_certificate_control.py')

class MemoryBudget:
    def __init__(self):self.used={k:0 for k in c.LIMITS}
    def take(self,k):
        if k not in self.used or self.used[k]>=c.LIMITS[k]:raise ValueError('BUDGET')
        self.used[k]+=1

def pairs(rows):return [[[float(x),0.] for x in row] for row in rows]
def basis():return dict(basis_hash=e.canonical(dict(num_system_qubits=2,operations=[],hash_policy='ordered_local_basis_operations_v1')),operations=[])
def fixture(rep='pauli_jw'):
    axes=[dict(epsilon_sig=eps,axis=axis,signed_bias=0.,target_exact='1/100') for eps in [.05,.01,.005,.001,.0005] for axis in ['cosine','sine']]
    model=dict(schema='h4_frozen_C_export_v1',representation=rep,n_qubits=2,qubit_order='qiskit_little_endian',
        state=[[1.,0.],[0.,0.],[0.,0.],[0.,0.]],preparation_verified=True,operator_agreement_limit=1e-9,step_time=.1,cases={})
    if rep=='pauli_jw':
        model['hamiltonian']=dict(constant=.125,terms=[['ZI',.25],['IX',.125]])
        model['saved_full_matrix']=pairs([[.375,.125,0,0],[.125,.375,0,0],[0,0,-.125,.125],[0,0,.125,-.125]])
        blocks=[dict(operator_key=str(i),label=l,coefficient=v) for i,(l,v) in enumerate(model['hamiltonian']['terms'])]
        components=[dict(label='IX',sign=1,coefficient_abs=.125)]
        extracted=0.
    else:
        model['hamiltonian']=dict(constant=.125,one_body=pairs([[.125,0],[0,.25]]),lambdas=[.25],g_matrices=[pairs([[1,0],[0,0]])])
        model['saved_full_matrix']=pairs([[.125,0,0,0],[0,.5,0,0],[0,0,.375,0],[0,0,0,.75]])
        b=basis()
        blocks=[dict(operator_key='one_body',basis=b,primitives=dict(global_phase=-.05*(.125+.25)/2,rz=[[0,-.05*.125],[1,-.05*.25]],rzz=[])),
                dict(operator_key='0',basis=b,primitives=dict(global_phase=-.05*.25/2,rz=[[0,-.05*.25]],rzz=[]))]
        components=[dict(sign=-1,coefficient_abs=.125,support=[0],basis=b)]
        extracted=.125
    for method in ['PF','PR']:
        case=dict(metadata=dict(method=method),expected_metadata=dict(method=method),config=dict(T=.8,q=8,r=4,K=4,method=method),
            blocks=blocks if method=='PF' else blocks[:1],components=[] if method=='PF' else components,
            extracted_identity=0. if method=='PF' else extracted,native_scalar_phases=[-.0125-(.1*extracted if method=='PR' else 0)],saved_axes=copy.deepcopy(axes),
            saved_normalization=dict(B=1.,log_B=0.,log_B_upper=1.,lambda_r=.125),semantic_error_bound_exact='0',circuit_correspondence_proved=True,exact_event_range_proved=True)
        if method=='PR':
            tau=.125*.1/4;weights=[tau**k/math.factorial(k)*math.sqrt(1+(tau/(k+1))**2) for k in (0,2,4)];BK=math.fsum(weights)
            ops=[w/BK for w in weights]
            case['law']=dict(numpy_version='1.26.4',uniform_bits=53,component_probabilities=[1.],component_cdf=[1.],order_probabilities=ops,order_cdf=e.cdf_binary64(ops),angles=[math.atan(tau/(k+1)) for k in (0,2,4)])
            case['saved_normalization']['B']=math.exp(32*math.log(BK))
        if rep=='pauli_jw':case['metadata']['retained_labels']=[b['label'] for b in case['blocks']]
        else:case['metadata']['deterministic_fragment_indices']=[int(b['operator_key']) for b in case['blocks'] if b['operator_key']!='one_body']
        case['expected_metadata']=copy.deepcopy(case['metadata'])
        model['cases'][method]=case
    model['state_hash']=e.array_hash(model['state'],vector=True);model['binding_hash']=e.canonical(model)
    return model

def rebind(m):m['binding_hash']=e.canonical({k:v for k,v in m.items() if k!='binding_hash'});return m

class Arithmetic(unittest.TestCase):
    def test_exact_binary(self):
        with ctx.workprec(128):self.assertTrue(e.real(.1).contains(fmpq(*(.1).as_integer_ratio())));self.assertFalse(e.real(.1).contains(fmpq(1,10)))
    def test_nonfinite(self):
        for x in [math.inf,math.nan]:
            with self.assertRaisesRegex(ValueError,'NONFINITE'):e.real(x)
    def test_endpoint_roundtrip(self):
        x=e.real(Q(1,3));bounds=e.endpoints(x);self.assertLessEqual(Q(bounds['lower']),Q(1,3));self.assertGreaterEqual(Q(bounds['upper']),Q(1,3))
    def test_exponential_nilpotent(self):self.assertTrue(acb_mat([[0,fmpq(1,4)],[0,0]]).exp().contains(acb_mat([[1,fmpq(1,4)],[0,1]])))
    def test_exponential_rotation(self):
        x=e.real(Q(1,8));A=acb_mat([[0,-x],[x,0]]).exp();self.assertTrue((A[0,0]-x.cos()).contains(0));self.assertTrue((A[1,0]-x.sin()).contains(0))
    def test_degree5(self):
        got=e.polynomial5(acb_mat([[e.real(Q(1,3))]]))[0,0];wanted=sum((Q(1,3)**k/math.factorial(k) for k in range(6)),Q(0));self.assertTrue(got.contains(fmpq(wanted.numerator,wanted.denominator)))
    def test_noncommuting_order(self):
        X=e.pauli('X');Z=e.pauli('Z');psi=acb_mat([[1],[0]])
        self.assertTrue(e.evolve(psi,[],Z*X,acb(1),1,1).contains(Z*X*psi));self.assertFalse((X*Z).contains(Z*X))
    def test_little_endian(self):self.assertEqual(e.pauli('IX')[1,0],1);self.assertEqual(e.pauli('XI')[2,0],1)
    def test_one_body_fermionic_sign(self):
        A=e.one_body(pairs([[0,1],[0,0]]));self.assertEqual(A[1,2],1);self.assertTrue(A[3,3].is_zero())
    def test_local_embedding(self):self.assertTrue(e.embed(e.pauli('X'),[0],2).contains(e.pauli('IX')))
    def test_support_reject(self):
        with self.assertRaisesRegex(ValueError,'SUPPORT'):e.embed(e.pauli('X'),[2],2)
    def test_nonunitary_norm(self):self.assertGreater(e.contraction_norm(acb_mat([[2]])),1)
    def test_boundary(self):self.assertEqual(e.endpoint_compare(e.real(Q(1,8)),'1/8'),'TARGET_MET');self.assertEqual(e.endpoint_compare(e.real(Q(1,4)),'1/8'),'TARGET_NOT_MET')
    def test_interval_not_midpoint(self):self.assertEqual(e.endpoint_compare(arb(0,1),'1/8'),'WIDTH_INSUFFICIENT')
    def test_finite_normalization(self):
        p=e.finite_normalization(e.real(Q(1,8)),e.real(Q(1,10)),4,8);self.assertGreater(p['B_K'].lower(),1);self.assertTrue((p['B'].log()-p['log_B']).contains(0))
    def test_cdf_matches_numpy(self):
        import numpy as np
        for p in [[.1,.2,.7],[.3,.3,.4],[.0,.5,.5]]:
            a=np.array(p).cumsum();a/=a[-1];self.assertEqual(e.cdf_binary64(p),a.tolist())
    def test_grid_ties(self):self.assertEqual(e.grid_masses([.25,.5,1.],bits=4),[Q(1,4),Q(1,4),Q(1,2)])
    def test_nonrepresentable_grid(self):self.assertEqual(e.grid_masses([.3,1.],bits=4),[Q(5,16),Q(11,16)])
    def test_bad_cdf(self):
        for p in [[.4,.3,1.],[.1,.9]]:
            with self.assertRaises(ValueError):e.grid_masses(p)
    def test_cdf_tamper(self):
        x=fixture()['cases']['PR']['law'];x['order_cdf'][0]=0.
        with self.assertRaisesRegex(ValueError,'ROUNDING_GRAPH'):e.verify_probability_mapping(x)
    def test_finite_enumeration(self):
        I=e.identity(2);comps=[e.pauli('X'),e.pauli('Z')];ps=[Q(1,4),Q(3,4)];pis=[Q(1,2),Q(1,4),Q(1,4)];angles=[e.real(Q(1,k)) for k in [8,16,32]]
        rs=[[I*a.cos()+P*acb(0,-a.sin()) for P in comps] for a in angles]
        F=e.finite_implemented_mean(comps,rs,ps,pis);expected=acb_mat(2,2)
        for index,k in enumerate((0,2,4)):
            for ids in product(range(2),repeat=k+1):
                prod=I
                for j in ids[1:]:prod=comps[j]*prod
                probability=pis[index]
                for j in ids:probability*=ps[j]
                expected+=rs[index][ids[0]]*prod*e.real(probability)*((-1)**(k//2))
        self.assertTrue((F-expected).contains(acb_mat(2,2)))

class ProductionFixture(unittest.TestCase):
    def test_pauli_end_to_end(self):
        budget=MemoryBudget();out=e.EnclosureEngine(fixture(),budget).tier(64);self.assertEqual(out['status'],'TARGET_MET');self.assertEqual(len(out['axes']),20);self.assertFalse(out['formal_certificate_accepted']);self.assertEqual(budget.used['dense_matrix_exponential_requests'],1)
    def test_df_end_to_end(self):
        budget=MemoryBudget();out=e.EnclosureEngine(fixture('df_native'),budget).tier(64);self.assertEqual(out['status'],'TARGET_MET');self.assertEqual(budget.used['dense_matrix_exponential_requests'],3)
    def test_state_defect(self):
        m=fixture();m['state'][0]=[1.000001,0.];m['state_hash']=e.array_hash(m['state'],vector=True)
        out=e.EnclosureEngine(rebind(m),MemoryBudget()).tier(96);self.assertGreater(Q(out['state_norm_defect']['lower']),0)
    def test_zero_state(self):
        m=fixture();m['state']=[[0.,0.]]*4;m['state_hash']=e.array_hash(m['state'],vector=True)
        with self.assertRaisesRegex(ValueError,'STATE_NORMALIZATION'):e.EnclosureEngine(rebind(m),MemoryBudget()).tier(64)
    def test_hash_rejection(self):
        m=fixture();m['step_time']=.2
        with self.assertRaisesRegex(ValueError,'BINDING'):e.EnclosureEngine(m,MemoryBudget())
    def test_state_hash_rejection(self):
        m=fixture();m['state_hash']='bad'
        with self.assertRaisesRegex(ValueError,'STATE_HASH'):e.EnclosureEngine(rebind(m),MemoryBudget())
    def test_metadata_rejection(self):
        m=fixture();m['cases']['PR']['expected_metadata']={}
        with self.assertRaisesRegex(ValueError,'PREPARATION'):e.EnclosureEngine(rebind(m),MemoryBudget())
    def test_preparation_missing(self):
        m=fixture();m['preparation_verified']=False
        with self.assertRaisesRegex(ValueError,'PREPARATION'):e.EnclosureEngine(rebind(m),MemoryBudget())
    def test_operator_mismatch(self):
        m=fixture();m['saved_full_matrix'][0][0]=[3.,0.]
        with self.assertRaisesRegex(ValueError,'RECONSTRUCTION'):e.EnclosureEngine(rebind(m),MemoryBudget()).tier(64)
    def test_no_sector_projection(self):
        m=fixture();m['state']=m['state'][:2]
        with self.assertRaisesRegex(ValueError,'SHAPE'):e.EnclosureEngine(rebind(m),MemoryBudget())
    def test_normalization_upper_failure(self):
        m=fixture();m['cases']['PR']['saved_normalization']['log_B_upper']=-1.
        self.assertEqual(e.EnclosureEngine(rebind(m),MemoryBudget()).tier(64)['status'],'MODEL_UNRESOLVED')
    def test_unproved_circuit_refusal(self):
        m=fixture();m['cases']['PR']['circuit_correspondence_proved']=False
        self.assertEqual(e.EnclosureEngine(rebind(m),MemoryBudget()).tier(64)['status'],'MODEL_UNRESOLVED')
    def test_basis_hash_rejection(self):
        b=basis();b['basis_hash']='bad'
        with self.assertRaisesRegex(ValueError,'BASIS_HASH'):e.basis_operators(b,2)
    def test_local_matrix_hash_rejection(self):
        b=basis();b['operations']=[dict(matrix=pairs([[1,0],[0,1]]),inverse_matrix=pairs([[1,0],[0,1]]),qubits=[0],description=dict(matrix_sha256='bad',qubits=[0]))]
        with self.assertRaisesRegex(ValueError,'MATRIX_HASH'):e.basis_operators(b,2)
    def test_precision_context_restored(self):
        original=ctx.prec;e.EnclosureEngine(fixture(),MemoryBudget()).tier(96);self.assertEqual(ctx.prec,original)
    def test_duplicate_axis_rejected(self):
        m=fixture();m['cases']['PF']['saved_axes'][0]=m['cases']['PF']['saved_axes'][1]
        with self.assertRaisesRegex(ValueError,'AXIS_TARGETS'):e.EnclosureEngine(rebind(m),MemoryBudget())
    def test_nonpositive_target_rejected(self):
        m=fixture();m['cases']['PF']['saved_axes'][0]['target_exact']='0'
        with self.assertRaisesRegex(ValueError,'POSITIVE_TARGET'):e.EnclosureEngine(rebind(m),MemoryBudget())
    def test_pauli_block_order_rejected(self):
        m=fixture();m['cases']['PF']['blocks'].reverse()
        with self.assertRaisesRegex(ValueError,'BLOCK_ORDER'):e.EnclosureEngine(rebind(m),MemoryBudget())
    def test_df_block_order_rejected(self):
        m=fixture('df_native');m['cases']['PF']['blocks'].reverse()
        with self.assertRaisesRegex(ValueError,'BLOCK_ORDER'):e.EnclosureEngine(rebind(m),MemoryBudget())
    def test_completed_cases_not_recomputed(self):
        b=MemoryBudget();engine=e.EnclosureEngine(fixture(),b);engine.tier(64)
        before=b.used['nominal_law_mean_enclosures'];engine.completed.pop('PR')
        engine.tier(96);self.assertEqual(b.used['nominal_law_mean_enclosures'],before+1)
    def test_rzz_phase_parity(self):
        spec=dict(global_phase=0.,rz=[],rzz=[[0,1,.25]])
        D=e.primitive_diagonal(spec,2);self.assertTrue((D[0,0]-e.phase(e.real(-.125))).contains(0));self.assertTrue((D[1,1]-e.phase(e.real(.125))).contains(0))
    def test_invalid_hermitian_input(self):
        m=fixture('df_native');m['hamiltonian']['one_body'][0][1]=[.1,0.]
        with self.assertRaisesRegex(ValueError,'HERMITICITY'):e.EnclosureEngine(rebind(m),MemoryBudget()).tier(64)

class ReferenceContraction(unittest.TestCase):
    """Analytic small-model oracles exercise the actual production tier path."""
    def with_state(self,rep,values):
        model=fixture(rep);model['state']=values
        model['state_hash']=e.array_hash(values,vector=True)
        return rebind(model)

    def assert_encloses(self,encoded,expected):
        # Compare exact saved endpoints to a separate 256-bit scalar formula.
        # A wide interval or a matching midpoint alone must not pass.
        for axis in ('real','imag'):
            lo=Q(encoded[axis]['lower']);hi=Q(encoded[axis]['upper'])
            wanted=getattr(expected,axis)
            self.assertLessEqual(lo,e.dyadic(wanted.lower()))
            self.assertGreaterEqual(hi,e.dyadic(wanted.upper()))
            self.assertLess(hi-lo,Q(1,10**14))

    def pauli_reference(self,values):
        T=e.real(.8);answer=acb(0)
        for j,energy in ((0,Q(3,8)),(2,Q(-1,8))):
            a=e.complex_ball(values[j]);b=e.complex_ball(values[j+1])
            norm=(a.conjugate()*a+b.conjugate()*b).real
            x=(a.conjugate()*b+b.conjugate()*a).real
            angle=T/8
            answer+=e.phase(-e.real(energy)*T)*(norm*angle.cos()+acb(0,-x*angle.sin()))
        return answer

    def df_reference(self,values):
        return sum((abs(e.complex_ball(z))**2*e.phase(-e.real(energy)*e.real(.8))
                    for z,energy in zip(values,(Q(1,8),Q(1,2),Q(3,8),Q(3,4)),strict=True)),acb(0))

    def test_signal_rejects_square_operator(self):
        with self.assertRaisesRegex(ValueError,'SIGNAL_STATE_VECTOR_SHAPE'):
            e.signal(acb_mat([[0],[1]]),e.identity(2))

    def test_signal_rejects_row_vector(self):
        with self.assertRaisesRegex(ValueError,'SIGNAL_STATE_VECTOR_SHAPE'):
            e.signal(acb_mat([[0],[1]]),acb_mat([[0,1]]))

    def test_signal_rejects_multicolumn_state(self):
        with self.assertRaisesRegex(ValueError,'SIGNAL_STATE_VECTOR_SHAPE'):
            e.signal(e.identity(2),acb_mat([[0],[1]]))

    def test_signal_rejects_mismatched_rows(self):
        with self.assertRaisesRegex(ValueError,'SIGNAL_STATE_VECTOR_SHAPE'):
            e.signal(acb_mat([[0],[1]]),acb_mat([[0],[1],[0]]))

    def test_signal_nonvacuum_identity(self):
        state=acb_mat([[0],[1]])
        self.assertEqual(e.signal(state,e.identity(2)*state),1)

    def test_signal_complex_conjugation(self):
        self.assertEqual(e.signal(acb_mat([[1],[acb(0,2)]]),acb_mat([[3],[acb(0,4)]])),11)
        self.assertEqual(e.signal(acb_mat([[acb(0,2)]]),acb_mat([[3]])),acb(0,-6))

    def test_pauli_all_basis_states_all_precisions(self):
        for index in range(4):
            values=[[float(j==index),0.] for j in range(4)]
            for bits in (64,96,128):
                with self.subTest(index=index,bits=bits),ctx.workprec(256):
                    out=e.EnclosureEngine(self.with_state('pauli_jw',values),MemoryBudget()).tier(bits)
                    self.assert_encloses(out['reference'],self.pauli_reference(values))

    def test_df_all_basis_states_all_precisions(self):
        for index in range(4):
            values=[[float(j==index),0.] for j in range(4)]
            for bits in (64,96,128):
                with self.subTest(index=index,bits=bits),ctx.workprec(256):
                    out=e.EnclosureEngine(self.with_state('df_native',values),MemoryBudget()).tier(bits)
                    self.assert_encloses(out['reference'],self.df_reference(values))

    def test_pauli_complex_superposition(self):
        values=[[.5,0.],[0.,.5],[.5,0.],[0.,-.5]]
        with ctx.workprec(256):
            out=e.EnclosureEngine(self.with_state('pauli_jw',values),MemoryBudget()).tier(96)
            self.assert_encloses(out['reference'],self.pauli_reference(values))

    def test_df_complex_superposition(self):
        values=[[.5,0.],[0.,.5],[.5,0.],[0.,-.5]]
        with ctx.workprec(256):
            out=e.EnclosureEngine(self.with_state('df_native',values),MemoryBudget()).tier(96)
            self.assert_encloses(out['reference'],self.df_reference(values))

    def test_reference_global_phase_invariance(self):
        values=[[.5,0.],[0.,.5],[.5,0.],[0.,-.5]]
        rotated=[[-im,re] for re,im in values]
        for rep,oracle in (('pauli_jw',self.pauli_reference),('df_native',self.df_reference)):
            with self.subTest(rep=rep),ctx.workprec(256):
                for state in (values,rotated):
                    out=e.EnclosureEngine(self.with_state(rep,state),MemoryBudget()).tier(96)
                    self.assert_encloses(out['reference'],oracle(values))

    def test_noncommuting_complex_state_reference(self):
        values=[[.5,0.],[0.,.5],[.5,0.],[0.,.5]]
        m=self.with_state('pauli_jw',values)
        m['hamiltonian']['terms'][0][0]='IZ'
        m['saved_full_matrix']=pairs([[.375,.125,0,0],[.125,-.125,0,0],[0,0,.375,.125],[0,0,.125,-.125]])
        for case in m['cases'].values():
            case['blocks'][0]['label']='IZ';case['metadata']['retained_labels'][0]='IZ'
            case['expected_metadata']=copy.deepcopy(case['metadata'])
        with ctx.workprec(256):
            out=e.EnclosureEngine(rebind(m),MemoryBudget()).tier(96)
            radius=(e.real(Q(1,4))**2+e.real(Q(1,8))**2).sqrt()
            expected=e.phase(-e.real(.8)/8)*(radius*e.real(.8)).cos()
            self.assert_encloses(out['reference'],expected)

    def test_encoded_state_norm_not_silently_changed(self):
        values=[[0.,0.],[2.,0.],[0.,0.],[0.,0.]]
        for rep,oracle in (('pauli_jw',self.pauli_reference),('df_native',self.df_reference)):
            with self.subTest(rep=rep),ctx.workprec(256):
                m=self.with_state(rep,values);original=copy.deepcopy(m['state'])
                out=e.EnclosureEngine(m,MemoryBudget()).tier(96)
                self.assert_encloses(out['reference'],oracle(values))
                self.assertEqual(out['state_norm'],dict(lower='4',upper='4'))
                self.assertEqual(m['state'],original)

    def test_biases_use_the_correct_reference_without_extra_calls(self):
        # Pauli blocks commute; both methods and both bias axes have an analytic
        # zero-bias oracle at this artificial tolerance. The defective first
        # column would instead give a reference of zero for this state.
        values=[[0.,0.],[0.,0.],[1.,0.],[0.,0.]]
        budget=MemoryBudget()
        out=e.EnclosureEngine(self.with_state('pauli_jw',values),budget).tier(96)
        for method in ('PF','PR'):
            for axis in ('real','imag'):
                bias=out['nominal_law'][method]['signed_bias'][axis]
                self.assertLess(max(abs(Q(bias['lower'])),abs(Q(bias['upper']))),Q(1,10**10))
                physical=out['implemented_law'][method]['same_ray_physical_bias'][axis]
                self.assertLess(max(abs(Q(physical['lower'])),abs(Q(physical['upper']))),Q(1,10**10))
        self.assertEqual(out['status'],'TARGET_MET')
        self.assertFalse(out['formal_certificate_accepted'])
        self.assertEqual(budget.used['reference_signal_enclosures'],1)
        self.assertEqual(budget.used['dense_matrix_exponential_requests'],1)

if __name__=='__main__':unittest.main()
