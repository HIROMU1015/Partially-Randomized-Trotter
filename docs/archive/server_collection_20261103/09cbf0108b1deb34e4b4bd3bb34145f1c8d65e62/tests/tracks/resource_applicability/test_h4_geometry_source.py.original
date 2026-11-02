"""Artificial identities, matrices and circuits only. No real input fixtures."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import numpy as np
from qiskit import QuantumCircuit,transpile
from qiskit.quantum_info import Operator,Statevector
from qiskit.circuit import Parameter,Gate,Instruction
from trottertracks.resource_applicability.h4_geometry import identity as ident
from trottertracks.resource_applicability.h4_geometry import gates,inputs,signal,circuits,resources,ledger,execution
from trottertracks.resource_applicability.h4_geometry import review as sensitivity
from trottertracks.resource_applicability.h4_geometry import workers

COUNTS={}
OPTIONS=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
FAKE='a'*64
SOURCE='b'*40


class LocalAssertions:
    """Avoid numpy.testing's implicit CPU capability subprocess."""
    @staticmethod
    def assert_allclose(a,b,atol=1e-8,rtol=1e-5):
        if not np.allclose(a,b,atol=atol,rtol=rtol):
            raise AssertionError('operator/numerical mismatch: '+str(float(np.max(np.abs(np.asarray(a)-np.asarray(b))))))

    @staticmethod
    def assert_array_equal(a,b):
        if not np.array_equal(a,b):
            raise AssertionError('array order/identity mismatch')


assertions=LocalAssertions()


def fake_identity():
    return dict(geometry='0.70',H=FAKE,DF='c'*64,state='d'*64,input='e'*64,template='f'*64,
                source=SOURCE,compiler='1'*64,environment='2'*64,wrapper_semantics='ARTIFICIAL_TEST_ONLY')


def fake_stage(stage='input_generation'):
    plan={'stage':stage,'run_id':gates.RUN_ID,'base_commit':gates.BASE,'contract_plan_fingerprint':gates.PLAN_FP,
          'source_commit':SOURCE,'source_hashes':{'src/fake.py':FAKE},'source_root':'/tmp/artificial-source',
          'artifact_anchor':gates.ARTIFACT_ANCHOR,'output_root':gates.OUTPUT,'distances':list(gates.DISTANCES),
          'requested_workers':1,'binding':'SOURCE_BOUND','inputs':None,'generation_freeze_digest':None,
          'schema_version':'h4-native-execution-plan-v1','templates':[],
          'compiler_fingerprint':FAKE,'environment_fingerprint':FAKE,'source_audit_sha256':FAKE}
    if stage=='signal_compile':
        plan.update(binding='INPUT_BOUND',generation_freeze_digest=FAKE,
                    inputs={d:dict(file='input-'+d+'.npz',bytes_sha256=FAKE,input=FAKE,H=FAKE,DF=FAKE,state=FAKE) for d in gates.DISTANCES})
    auth={'stage':stage,'run_id':gates.RUN_ID,'permission':stage,'one_shot':True,'allowed_cpus':[0],
          'result_prior':True,'plan_fingerprint':ident.fingerprint('h4-execution-plan-v1',plan),
          'schema_version':'h4-native-authorization-v1'}
    review={'stage':stage,'run_id':gates.RUN_ID,'approved':True,'plan_fingerprint':auth['plan_fingerprint'],
            'authorization_digest':ident.fingerprint('h4-authorization-v1',auth),
            'schema_version':'h4-native-stage-review-v1'}
    return plan,auth,review


class IdentityTests(unittest.TestCase):
    def test_signed_zero_and_phase(self):
        self.assertNotEqual(ident.canonical(-0.),ident.canonical(0.))
        self.assertEqual(ident.canonical(1.),b'{"real64_hex":"0x1.0000000000000p+0"}')
        self.assertNotEqual(ident.canonical(1.),ident.canonical(1))

    def test_complex_and_order(self):
        self.assertEqual(ident.canonical({'b':2,'a':1j}),ident.canonical({'a':1j,'b':2}))

    def test_seed_independent_reference(self):
        fields=fake_identity()
        payload={'domain':'h4-trajectory-v1','master_seed':20261006,**fields,'index':7}
        expected=int.from_bytes(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()[:8],'big')
        actual=ident.trajectory_seed(fields,7,actual_inputs_frozen=True,signal_launch=True)
        self.assertEqual(expected,actual)

    def test_axis_shared_seeds_wrapper_distinct(self):
        fields=fake_identity();seeds=signal.trajectory_seeds(fields)
        self.assertEqual(len(set(seeds)),32)
        self.assertNotEqual(ident.wrapper_key(fields,'cosine',seeds[0],0),ident.wrapper_key(fields,'sine',seeds[0],0))

    def test_step_streams(self):
        values={ident.step_seed(7,o,s,k,d) for o in range(2) for s in range(2) for k in range(5) for d in ('order','component')}
        self.assertEqual(len(values),40)

    def test_seed_before_freeze_or_launch(self):
        for frozen,launch in ((False,True),(True,False),(False,False)):
            with self.assertRaises(ident.Stop):
                ident.trajectory_seed(fake_identity(),0,actual_inputs_frozen=frozen,signal_launch=launch)

    def test_baseline_seed_pair(self):
        with self.assertRaises(ident.Stop):
            ident.wrapper_key(fake_identity(),'cosine',7,None)


class GateTests(unittest.TestCase):
    def test_production_structure_separate_from_semantics(self):
        p,a,r=fake_stage();gates.structural_gate(p,a,r)
        with self.assertRaises(ident.Stop):gates.structural_gate({**p,'unexpected':True},a,r)
        with self.assertRaises(ident.Stop):gates.structural_gate({**p,'requested_workers':True},a,r)
        with self.assertRaises(ident.Stop):gates.structural_gate(p,{**a,'schema_version':'synthetic-contract-example'},r)
    def test_generation_and_signal_artificial_permits(self):
        for stage in ('input_generation','signal_compile'):
            permit=gates.authorize(stage,*fake_stage(stage),explicit_launch=True)
            self.assertEqual(permit.stage,stage)

    def test_no_generation_authorization_reuse(self):
        with self.assertRaises(ident.Stop):
            gates.authorize('signal_compile',*fake_stage(),explicit_launch=True)

    def test_negative_no_private_boundary(self):
        with patch.object(execution,'checkout_gate',side_effect=AssertionError('checkout touched')),patch.object(execution,'generation_stage',side_effect=AssertionError('data/output touched')),patch.object(execution,'signal_stage',side_effect=AssertionError('data/output touched')):
            with self.assertRaises(ident.Stop):
                execution.launch('input_generation',*fake_stage(),explicit_launch=False)

    def test_missing_input_hashes_not_sealed(self):
        plan,auth,review=fake_stage('signal_compile');plan['inputs']['0.70']['H']=None
        with self.assertRaises(ident.Stop):
            gates.authorize('signal_compile',plan,auth,review,explicit_launch=True)

    def test_six_distances_exact(self):
        self.assertEqual(len(gates.DISTANCES),6)
        self.assertNotIn('1.00',gates.DISTANCES)
        self.assertNotIn('1.30',gates.DISTANCES)

    def test_compiler_metadata_matches_pinned(self):
        root=Path(__file__).absolute().parents[3]
        p=root/gates.CONTRACT/'zero_compute_plan_v2.json'
        plan=json.loads(p.read_text())
        options=gates.compiler_matches(plan['compiler_environment_reference']['compiler'])
        self.assertEqual(options['num_processes'],1)
        counts={k:sum(t['method']==k for t in plan['templates']) for k in ('B0','B1','B2','B3')}
        self.assertEqual(counts,dict(B0=20,B1=4,B2=145,B3=49))
        self.assertEqual([t['template_id'] for t in plan['templates'] if t['r']==64],['B3-rank0-q8-r64-K2','B2-rank3-q1-r64-K2'])


class InputTests(unittest.TestCase):
    def test_no_save_constructor_state_mock(self):
        from types import SimpleNamespace
        mol=SimpleNamespace(_built=True,verbose=0,max_memory=4000,stdout=None)
        with patch.object(tempfile,'NamedTemporaryFile',side_effect=AssertionError('implicit temp')):
            mf=inputs.initialize_no_save(SimpleNamespace(),mol)
        self.assertIsNone(mf.chkfile);self.assertIsNone(mf._chkfile);self.assertIsNone(mf._eri)

    def test_prefix_equal_weight_significant_STOP(self):
        vals=np.r_[np.zeros(12),1.,4.,5.,6.];vectors=np.eye(16);vectors[0,12]=1;vectors[12,12]=1
        with patch.object(np.linalg,'eigh',return_value=(vals,vectors)):
            with self.assertRaisesRegex(ident.Stop,'prefix weight tie'):
                inputs.factorize(None,chemist_transform=lambda _x,spin_basis:(np.zeros((8,8)),np.zeros((4,)*4)))

    def test_freeze_synthetic_bytes_closure(self):
        p=gates.authorize('input_generation',*fake_stage(),explicit_launch=True)
        H=np.diag(np.arange(256.)).astype(complex);a={'H':H,**inputs.ground_state(H),
             'lambdas':np.arange(12.),'G':np.zeros((12,8,8)),'correction':np.zeros((8,8)),
             'permutation':np.arange(16),'raw_indices':np.arange(12),'raw_eigenvalues':np.arange(16.),
             'raw_eigenvectors':np.eye(16),'signs':np.ones(16)}
        data,record,identity=inputs.freeze_input(p,'0.70',a)
        self.assertEqual(record['bytes_sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(identity['arrays']['G']['shape'],[12,8,8])
        self.assertEqual(set(record),{'file','bytes_sha256','input','H','DF','state'})

    def test_frozen_state_no_resolve_or_substitution(self):
        H=np.diag(np.arange(256.)).astype(complex);a={'H':H,**inputs.ground_state(H)}
        with patch.object(np.linalg,'eigh',side_effect=AssertionError('new solve')):
            inputs.validate_frozen_state(a)
        bad=copy.deepcopy(a);bad['state']*=1.01
        with self.assertRaises(ident.Stop):inputs.validate_frozen_state(bad)
        bad=copy.deepcopy(a);bad['qiskit_state']=bad['state']
        with self.assertRaises(ident.Stop):inputs.validate_frozen_state(bad)
    def test_coordinates_order_exact(self):
        coords=inputs.coordinates('0.70')
        self.assertEqual([c[1][2] for c in coords],[(i-1.5)*float('0.70') for i in range(4)])

    def test_unknown_distance_stop(self):
        with self.assertRaises(ident.Stop):inputs.coordinates('1.00')

    def test_controls_mock_no_files(self):
        mf=Mock();cls=type('FakeCDIIS',(),{'__init__':lambda self,*a,**kw:None})
        inputs.apply_scf_controls(mf,cls)
        for k,v in inputs.SCF_CONTROLS.items():self.assertEqual(getattr(mf,k),v)
        diis=mf.DIIS();self.assertTrue(diis.incore);self.assertTrue(mf.diis)
        self.assertIsNone(mf.chkfile);self.assertEqual(mf.max_cycle,50)

    def test_strict_scf_extra_cycle_insufficient(self):
        inputs.scf_accept(True,[(1e-10,1e-6)],1e-6,[np.ones(2)])
        for conv,cycles,grad,arrays in ((False,[(0,0)],0,[1]),(True,[(1e-8,0)],0,[1]),
                                       (True,[(0,1e-4)],0,[1]),(True,[(0,0)],1e-4,[1]),
                                       (True,[(0,0)],0,[float('nan')])):
            with self.assertRaises(ident.Stop):inputs.scf_accept(conv,cycles,grad,arrays)

    def test_mo_phase_and_degeneracy(self):
        c=inputs.canonical_mos(np.arange(4),-np.eye(4))
        assertions.assert_array_equal(c,np.eye(4))
        with self.assertRaises(ident.Stop):inputs.canonical_mos(np.array([0.,1.,1.,2.]),np.eye(4))

    def test_df_raw_order_and_sign(self):
        # This artificial 16x16 diagonal tensor has no molecular meaning.
        values=np.arange(1,17,dtype=float)
        result=inputs.factorize(None,chemist_transform=lambda _x,spin_basis:(np.zeros((8,8)),np.diag(values).reshape((4,)*4)))
        self.assertEqual(len(result['lambdas']),12)
        assertions.assert_array_equal(result['permutation'],np.argsort(4*values)[::-1])
        self.assertEqual(result['raw_eigenvectors'].shape,(16,16))
        self.assertEqual(result['raw_indices'].shape,(12,))

    def test_df_null_modes_not_removed(self):
        vals=np.r_[np.zeros(10),np.arange(1.,7.)]
        result=inputs.factorize(None,chemist_transform=lambda _x,spin_basis:(np.zeros((8,8)),np.diag(vals).reshape((4,)*4)))
        self.assertEqual(len(result['lambdas']),12)
        self.assertEqual(np.count_nonzero(result['lambdas']==0),6)

    def test_df_significant_degeneracy_stop(self):
        with self.assertRaises(ident.Stop):
            inputs.factorize(None,chemist_transform=lambda _x,spin_basis:(np.zeros((8,8)),np.eye(16).reshape((4,)*4)))

    def test_df_symmetry_stop(self):
        x=np.diag(np.arange(16.)).astype(complex);x[0,1]=1
        with self.assertRaises(ident.Stop):inputs.factorize(None,chemist_transform=lambda _x,spin_basis:(0,x.reshape((4,)*4)))

    def test_sector_phase_order_and_bit_reverse(self):
        idx=inputs.sector_indices();self.assertEqual(len(idx),36);self.assertEqual(idx,sorted(idx))
        H=np.diag(np.arange(256.)).astype(complex)
        state=inputs.ground_state(H)
        self.assertEqual(state['energy'],idx[0]);self.assertAlmostEqual(np.linalg.norm(state['state']),1)
        self.assertEqual(state['state'][idx[0]],1)
        self.assertEqual(state['qiskit_state'][inputs.reverse_bits(idx[0])],1)
        self.assertTrue(all(inputs.reverse_bits(inputs.reverse_bits(i))==i for i in range(256)))

    def test_sector_gap_and_original_residual_stop(self):
        H=np.diag(np.arange(256.)).astype(complex);idx=inputs.sector_indices()
        H[idx[1],idx[1]]=H[idx[0],idx[0]]
        with self.assertRaises(ident.Stop):inputs.ground_state(H)
        H=np.diag(np.arange(256.)).astype(complex);H[idx[0],0]=H[0,idx[0]]=1e-5
        with self.assertRaises(ident.Stop):inputs.ground_state(H)

    def test_original_hermiticity_and_imaginary_stop(self):
        H=np.diag(np.arange(256.)).astype(complex);H[0,1]=.01
        with self.assertRaises(ident.Stop):inputs.ground_state(H)
        H=np.diag(np.arange(256.)+1j*.1)
        with self.assertRaises(ident.Stop):inputs.ground_state(H)

    def test_generation_wrong_stage_before_import(self):
        permit=gates.authorize('signal_compile',*fake_stage('signal_compile'),explicit_launch=True)
        with self.assertRaises(ident.Stop):inputs.generate_input(permit,'0.70',lambda:None)

    def test_array_bytes_signed_zero(self):
        self.assertNotEqual(inputs.array_identity(np.array([0.])),inputs.array_identity(np.array([-0.])))


class CircuitTests(unittest.TestCase):
    def test_rte_event_full_controlled_relative_phase(self):
        from scipy.linalg import expm
        h=QuantumCircuit(1);h.h(0);empty=QuantumCircuit(1)
        xb={'basis':h,'coefficients':{(0,):1.}};zb={'basis':empty,'coefficients':{(0,):1.}}
        x={'block':xb,'support':(0,),'sign':-1};z={'block':zb,'support':(0,),'sign':1}
        event={'order':2,'rotation':x,'products':[z,x],'angle':.17}
        q=circuits.build_evolution({'n':1,'constant':.13,'deterministic':[]},
                                   {'T':.8,'delta':.8,'q':1},[[event]])
        X=np.array([[0,1],[1,0]],complex);Z=np.diag([1,-1]);expected=np.eye(4,dtype=complex)
        expected[2:,2:]=np.exp(-1j*.8*.13)*(-1)*expm(1j*.17*X)@(-X)@Z
        assertions.assert_allclose(Operator(q).data,expected,atol=1e-12)
    def test_custom_control_full_roundtrip(self):
        q=QuantumCircuit(1);q.global_phase=.14;q.rx(.29,0)
        outer=QuantumCircuit(2);outer.append(q.to_gate(label='fake').control(ctrl_state=0),[1,0])
        record=circuits.serialize(outer,'sine');new=circuits.deserialize(record)
        self.assertEqual(record,circuits.serialize(new,'sine'));self.compare(outer,new)

    def test_gaussian_dense_port_fake_fock(self):
        U=np.array([[np.exp(1j*.31)]])
        with patch.object(inputs,'one_body_operator',side_effect=lambda anti:np.diag([0,anti[0,0]])):
            basis=circuits.gaussian_basis(U)
        assertions.assert_allclose(Operator(basis).data,np.diag([1,np.exp(1j*.31)]),atol=1e-12)

    def test_phase_exact_fingerprint_difference(self):
        a=QuantumCircuit(1);a.rz(.2,0);b=a.copy();b.global_phase=math.pi
        self.assertNotEqual(circuits.numerical_fingerprint(a,'cosine'),circuits.numerical_fingerprint(b,'cosine'))
    def compare(self,left,right):
        COUNTS['synthetic_operator_checks']+=1
        assertions.assert_allclose(Operator(left).data,Operator(right).data,atol=1e-12,rtol=0)

    def compare_compiled_wrapper(self,left,right):
        # Both matrices include the control qubit. A single overall wrapper phase
        # may change; branch-relative phase must agree across the entire operator.
        COUNTS['synthetic_operator_checks']+=1
        a,b=Operator(left).data,Operator(right).data
        pivot=np.unravel_index(np.argmax(np.abs(a)),a.shape)
        common=b[pivot]/a[pivot]
        self.assertAlmostEqual(abs(common),1,places=12)
        assertions.assert_allclose(b,common*a,atol=1e-12,rtol=0)

    def test_roundtrip_exact_parameters_phase(self):
        q=QuantumCircuit(2);q.global_phase=-0.125;q.rz(.17,0);q.cx(0,1);q.rzz(-.31,0,1)
        record=circuits.serialize(q,'cosine');rebuilt=circuits.deserialize(record)
        self.assertEqual(record,circuits.serialize(rebuilt,'cosine'));self.compare(q,rebuilt)

    def test_custom_definition_closure(self):
        child=QuantumCircuit(1);child.rx(.13,0);child.global_phase=.07
        q=QuantumCircuit(2);q.append(child.to_gate(label='fake-custom'),[0]);q.h(1)
        record=circuits.serialize(q,'cosine');rebuilt=circuits.deserialize(record)
        self.assertEqual(record,circuits.serialize(rebuilt,'cosine'));self.compare(q,rebuilt)

    def test_controlled_relative_phase(self):
        child=QuantumCircuit(1);child.global_phase=.28;child.rz(.21,0)
        q=QuantumCircuit(2);q.append(child.to_gate().control(),[1,0])
        rebuilt=circuits.deserialize(circuits.serialize(q,'cosine'));self.compare(q,rebuilt)
        no_phase=child.copy();no_phase.global_phase=0
        other=QuantumCircuit(2);other.append(no_phase.to_gate().control(),[1,0])
        self.assertGreater(np.linalg.norm(Operator(q).data-Operator(other).data),.1)

    def test_condition_measurement_and_bits(self):
        q=QuantumCircuit(2,2);q.x(1).c_if(q.cregs[0],1);q.measure(0,1)
        record=circuits.serialize(q,'sine');self.assertEqual(record,circuits.serialize(circuits.deserialize(record),'sine'))
        self.assertNotEqual(circuits.numerical_fingerprint(q,'sine'),circuits.numerical_fingerprint(q,'cosine'))

    def test_nonfinite_symbolic_unsupported(self):
        q=QuantumCircuit(1);q.rz(Parameter('a'),0)
        with self.assertRaises(ident.Stop):circuits.serialize(q,'cosine')
        with self.assertRaises(ident.Stop):circuits.number(float('nan'))
        q=QuantumCircuit(1);q.append(Gate('unknown',1,[]),[0])
        with self.assertRaises(ident.Stop):circuits.serialize(q,'cosine')

    def test_unitary_array_parameter(self):
        q=QuantumCircuit(1);q.unitary(np.array([[0,1],[1,0]],complex),[0])
        self.compare(q,circuits.deserialize(circuits.serialize(q,'cosine')))

    def test_open_control_state(self):
        from qiskit.circuit.library import XGate
        q=QuantumCircuit(2);q.append(XGate().control(ctrl_state=0),[0,1])
        record=circuits.serialize(q,'cosine');self.compare(q,circuits.deserialize(record))

    def test_full_wrapper_sine_sign(self):
        evolution=QuantumCircuit(2);phase=-.38;evolution.p(phase,1)
        for axis,expected in (('cosine',math.cos(phase)),('sine',math.sin(phase))):
            q=circuits.wrapper(evolution,axis).remove_final_measurements(inplace=False)
            psi=Statevector.from_instruction(q)
            measured=sum((1 if i<2 else -1)*abs(a)**2 for i,a in enumerate(psi.data))
            self.assertAlmostEqual(measured,expected,places=12)

    def test_diagonal_expansion_independent(self):
        eta=np.array([.2,-.4]);lam=.3
        co=circuits.diagonal_coefficients(eta,lam)
        for i in range(4):
            direct=lam*sum(eta[p]*((i>>p)&1) for p in range(2))**2
            expanded=sum(c*(-1)**sum((i>>p)&1 for p in support) for support,c in co.items())
            self.assertAlmostEqual(direct,expanded,places=14)

    def test_controlled_block_operator_and_transpile(self):
        basis=QuantumCircuit(1);basis.ry(.4,0)
        block={'basis':basis,'coefficients':{():.13,(0,):-.21}}
        q=QuantumCircuit(2);circuits.append_block(q,block,.3)
        U=Operator(basis).data;H=U@np.diag([-.08,.34])@U.conj().T
        from scipy.linalg import expm
        expected=np.eye(4,dtype=complex);expected[2:,2:]=expm(-1j*.3*H)
        assertions.assert_allclose(Operator(q).data,expected,atol=1e-12)
        compiled=transpile(q,**OPTIONS);self.compare_compiled_wrapper(q,compiled)

    def test_paired_wrapper_transpile(self):
        evolution=QuantumCircuit(2);evolution.crz(.17,1,0);evolution.p(-.23,1)
        for axis in ('cosine','sine'):
            q=circuits.wrapper(evolution,axis);compiled=transpile(q,**OPTIONS)
            self.assertEqual(compiled.count_ops()['measure'],1)
            self.compare_compiled_wrapper(q.remove_final_measurements(inplace=False),compiled.remove_final_measurements(inplace=False))

    def test_noncommuting_pf_order(self):
        basis=QuantumCircuit(1);basis.h(0);identity=QuantumCircuit(1)
        det=[{'basis':basis,'coefficients':{(0,):.2}},{'basis':identity,'coefficients':{(0,):-.1}}]
        prep={'n':1,'constant':.05,'deterministic':det}
        template={'T':.8,'delta':.4,'q':2}
        q=circuits.build_evolution(prep,template,[[],[]])
        from scipy.linalg import expm
        X=np.array([[0,1],[1,0]],complex);Z=np.diag([1,-1])
        short=np.exp(-1j*.4*.05)*expm(-1j*.4/2*.2*X)@expm(-1j*.4*-.1*Z)@expm(-1j*.4/2*.2*X)
        expected=np.eye(4,dtype=complex);expected[2:,2:]=short@short
        assertions.assert_allclose(Operator(q).data,expected,atol=1e-12)


class SignalTests(unittest.TestCase):
    def test_common_P_affine_envelope_exact_ties(self):
        a={'normalization':1.,'bias':{'cosine':0,'sine':0},'paired_costs':[[10,10]]}
        b={'normalization':2.,'bias':{'cosine':0,'sine':0},'paired_costs':[[1,1]]}
        r=sensitivity.common_P_envelope({'A':a,'A_tie':a,'B':b},.05)
        self.assertEqual(r['boundaries'][0]['point_minimum_candidates'],['B'])
        self.assertEqual(r['intervals'][-1]['point_minimum_candidates'],['A','A_tie'])
        self.assertIsNone(r['research_decision']);self.assertFalse(r['next_stage_authorized'])

    def test_candidate_signal_cost_input_identity(self):
        inp={'geometry':'0.70',**{k:FAKE for k in ('H','DF','state','input')}}
        template={'template_id':'ARTIFICIAL','T':.8,'q':1,'delta':.8}
        a=signal.candidate_identity(inp,template,SOURCE,FAKE,FAKE)
        b=signal.candidate_identity({**inp,'geometry':'0.80','H':'c'*64},template,SOURCE,FAKE,FAKE)
        self.assertNotEqual(ident.fingerprint('h4-candidate-v1',a),ident.fingerprint('h4-candidate-v1',b))
        expected=ledger.wire(a,'cosine',None,None,FAKE)
        self.assertEqual(expected['candidate_fingerprint'],ident.fingerprint('h4-candidate-v1',a))
    def test_finite_event_expectation_independent(self):
        # Enumerate two artificial involutions; analytic mean must be Taylor degree K+1.
        import itertools
        from scipy.linalg import expm
        X=np.array([[0,1],[1,0]],complex);Z=np.diag([1,-1]);ops=[X,Z];p=[.3,.7];tau=.25;K=4
        orders,probs,B=signal.distribution(tau,K);mean=np.zeros((2,2),complex)
        for order,prob in zip(orders,probs):
            for chosen in itertools.product(range(2),repeat=order+1):
                product=np.eye(2,dtype=complex)
                for j in chosen[1:]:product=ops[j]@product
                event=(-1)**(order//2)*expm(-1j*math.atan(tau/(order+1))*ops[chosen[0]])@product
                mean+=prob*math.prod(p[j] for j in chosen)*event
        assertions.assert_allclose(B*mean,signal.finite_polynomial(.3*X+.7*Z,tau,K),atol=1e-12)

    def test_sampler_step_plan_rules_and_axis_pair(self):
        components=[{'abs_coefficient':.3,'name':'A'},{'abs_coefficient':.7,'name':'B'}]
        t={'q':2,'r':3,'K':2,'delta':.4}
        events,B=signal.sample_events(components,t,7)
        self.assertEqual(events,signal.sample_events(components,t,7)[0])
        self.assertEqual([len(x) for x in events],[3,3]);self.assertGreater(B,1)
        seed=ident.step_seed(7,0,0,0,'order');orders,p,_=signal.distribution(1*.4/3,2)
        order=int(np.random.Generator(np.random.PCG64(seed)).choice(orders,p=p))
        self.assertEqual(events[0][0]['order'],order)

    def test_duplicate_step_stream_stop(self):
        with patch.object(signal,'step_seed',return_value=7):
            with self.assertRaises(ident.Stop):signal.sample_events([{'abs_coefficient':1}],{'q':1,'r':1,'K':2,'delta':.1},7)

    def test_signal_normalization_and_delta(self):
        Z=np.diag([1.,-1.]);state=np.array([1.,0.]);t={'q':2,'T':.8,'delta':.4,'r':2,'K':2,'method':'B2'}
        result=signal.corrected_signal([.2*Z],(.3,Z),.1,state,t)
        expected=np.exp(-1j*.8*(.2+.1))*signal.finite_polynomial(Z,.3*.4/2,2)[0,0]**4
        self.assertAlmostEqual(abs(result['corrected']-expected),0,places=12)
        self.assertEqual(result['raw']*result['normalization'],result['corrected'])

    def test_display302_saved_only(self):
        saved={'normalization':1.2,'bias':{'cosine':.001,'sine':.002},'paired_costs':[[3,7]]*32}
        with patch.object(signal,'sample_events',side_effect=AssertionError('new samples')):
            points=signal.display_map(saved)
        self.assertEqual(len(points),302);self.assertIn(.05,[p['epsilon'] for p in points])

    def test_shot_strict_boundary_and_null(self):
        epsilon=.05;bias=epsilon/math.sqrt(2)
        self.assertIsNone(signal.shots(1,bias,epsilon))
        saved={'normalization':1.,'bias':{'cosine':bias,'sine':0},'paired_costs':[[1,1]]}
        result=signal.resource_point(saved,epsilon)
        self.assertFalse(result['eligible']);self.assertIsNone(result['shots']);self.assertIsNone(result['work'])

    def test_paired_covariance_weights_P_and_ties(self):
        samples=[[i,2*i+3] for i in range(32)];saved={'normalization':1.,'bias':{'cosine':.001,'sine':.002},'paired_costs':samples}
        result=signal.resource_point(saved,.05);n=np.asarray(result['shots'])
        expected=math.sqrt(float(n@np.cov(samples,rowvar=False,ddof=1)@n)/32)
        self.assertAlmostEqual(result['SE'],expected)
        with_P=signal.resource_point(saved,.05,5)
        self.assertAlmostEqual(with_P['work']-result['work'],sum(n)*5)
        self.assertEqual(len(signal.exact_ties([result,dict(result)])),2)
        self.assertEqual(signal.resource_point({**saved,'paired_costs':[[1,2]]},.05)['SE'],0)


class ResourceTests(unittest.TestCase):
    def test_owned_pipe_protocol_and_interrupted_frame(self):
        import io
        stream=io.BytesIO();value={'name':'ARTIFICIAL','numbers':[1,2,3]}
        workers.write_frame(stream,value);stream.seek(0)
        self.assertEqual(workers.read_frame(stream),value)
        for data in (b'',(workers.MAX_FRAME+1).to_bytes(8,'big'),(20).to_bytes(8,'big')+b'x'):
            with self.assertRaises(ident.Stop):workers.read_frame(io.BytesIO(data))

    def test_owned_worker_separate_stage_before_science(self):
        p=gates.authorize('input_generation',*fake_stage(),explicit_launch=True)
        with patch.object(execution,'_compile_worker',side_effect=AssertionError('private compile boundary')):
            with self.assertRaises(ident.Stop):workers.private_dispatch(p,'_compile_worker',(None,{}),{})
        p=gates.authorize('signal_compile',*fake_stage('signal_compile'),explicit_launch=True)
        with patch.object(execution,'_generate_worker',side_effect=AssertionError('private molecular boundary')):
            with self.assertRaises(ident.Stop):workers.private_dispatch(p,'_generate_worker',(p,'0.70'),{})

    def test_owned_worker_logs_bounded_before_write(self):
        text=workers.CappedText();text.write('x'*8192)
        with self.assertRaises(ident.Stop):text.write('x')
    def test_hidden_ancestor_limits_STOP(self):
        with self.assertRaises(ident.Stop):
            resources.cgroup_directories('0::/hidden/owned\n','1 2 0:1 /hidden /sys/fs/cgroup rw - cgroup2 cgroup rw\n')

    def test_negative_cgroup_and_stale_monitor(self):
        with self.assertRaises(ident.Stop):resources.effective_available(100,[(50,51)])
        obs={'available':64*resources.GiB,'observed_at':0,'oom_events':{},'psi_full_avg10':0}
        with self.assertRaises(ident.Stop):resources.Monitor(1,obs,clock=lambda:6)

    def test_foreign_pid_never_signalled(self):
        obs={'available':64*resources.GiB,'observed_at':10,'oom_events':{},'psi_full_avg10':0}
        m=resources.Monitor(1,obs,clock=lambda:10)
        raw='123 (fake process) '+ ' '.join(['S',str(os.getpid()+1)]+['0']*20)
        with patch.object(Path,'read_text',return_value=raw),patch.object(os,'kill') as kill:
            with self.assertRaises(ident.Stop):m.own_child(123)
            m.stop_children();kill.assert_not_called()

    def test_byte_budget_handoff_not_reset(self):
        with tempfile.TemporaryDirectory(prefix='h4-synthetic-') as t:
            root=Path(t)/'own';a=resources.OutputBudget(root,cap=300);a.write('a',b'x');a.close()
            b=resources.OutputBudget(root,cap=300,handoff=True)
            try:
                with self.assertRaises(ident.Stop):b.write('b',b'x'*50)
                self.assertFalse((root/'b').exists())
            finally:b.close()
    def test_admission_64_max5_and_120_max12(self):
        self.assertEqual(resources.admission(64*resources.GiB,10,12,range(20),range(20),now=10),5)
        self.assertEqual(resources.admission(120*resources.GiB,10,12,range(20),range(20),now=10),12)
        self.assertEqual(resources.admission(120*resources.GiB,10,12,[0,1],range(20),now=10),2)

    def test_low_stale_and_no_permission_stop(self):
        for available,at,explicit in ((31*resources.GiB,10,[0]),(120*resources.GiB,0,[0]),(120*resources.GiB,10,[])):
            with self.assertRaises(ident.Stop):resources.admission(available,at,12,explicit,[0],now=10)

    def test_ancestor_cgroup_minimum(self):
        self.assertEqual(resources.effective_available(200,[(None,12),(180,100),(100,75)]),25)
        dirs=resources.cgroup_directories('0::/user.slice/owned.scope\n','1 2 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
        self.assertEqual([str(p) for p,v2 in dirs],['/sys/fs/cgroup/user.slice/owned.scope','/sys/fs/cgroup/user.slice','/sys/fs/cgroup'])

    def test_v1_memory_ancestors(self):
        dirs=resources.cgroup_directories('2:memory:/a/b\n','1 2 0:1 / /sys/fs/cgroup/memory rw - cgroup memory rw,memory\n')
        self.assertEqual(len(dirs),3);self.assertFalse(dirs[0][1])

    def test_address_space_own_process_only(self):
        with patch.object(resources.resource,'getrlimit',return_value=(-1,-1)),patch.object(resources.resource,'setrlimit') as setting:
            resources.limit_owned_address_space()
            setting.assert_called_once_with(resources.resource.RLIMIT_AS,(8*resources.GiB,8*resources.GiB))

    def test_monitor_pressure_oom_rss_stop(self):
        obs={'available':64*resources.GiB,'observed_at':10,'oom_events':{'fake':0},'psi_full_avg10':0.}
        m=resources.Monitor(2,obs,clock=lambda:10)
        for changed in ({**obs,'available':15*resources.GiB},{**obs,'oom_events':{'fake':1}},{**obs,'psi_full_avg10':.001}):
            with self.assertRaises(ident.Stop):m.check(changed,[0])
        with self.assertRaises(ident.Stop):m.check(obs,[resources.ROLE_CAP+1])

    def test_cumulative_wall_no_reset(self):
        now=[10];w=resources.WallBudget(123,clock=lambda:now[0]);now[0]=20
        self.assertEqual(w.consumed(),133)
        now[0]+=resources.WALL_CAP
        with self.assertRaises(ident.Stop):w.consumed()

    def test_output_prebudget_exclusive_symlink(self):
        with tempfile.TemporaryDirectory(prefix='h4-synthetic-') as t:
            root=Path(t)/'own';b=resources.OutputBudget(root,cap=500)
            try:
                b.write('file.json',b'123')
                with self.assertRaises(FileExistsError):b.write('file.json',b'456')
                with self.assertRaises(ident.Stop):b.write('../escape',b'x')
                with self.assertRaises(ident.Stop):b.write('large',b'x'*500)
                self.assertFalse((root/'large').exists())
            finally:b.close()
            with self.assertRaises(FileExistsError):resources.OutputBudget(root)
            link=Path(t)/'link';link.symlink_to(root,target_is_directory=True)
            with self.assertRaises(ident.Stop):resources.OutputBudget(link)


class LedgerTests(unittest.TestCase):
    def test_ledger_chain_tamper_and_missing_owner(self):
        with tempfile.TemporaryDirectory(prefix='h4-synthetic-') as t:
            b=resources.OutputBudget(Path(t)/'own');l=ledger.Ledger(b)
            try:
                a,c=self.make_record(),self.make_record(1);l.register(a);l.register(c);l.reserve(a['wrapper_key'])
                l.complete(a['wrapper_key'],dict.fromkeys(ledger.METRICS,1));l.complete(c['wrapper_key'],{},owner_key=a['wrapper_key'])
                p=b.root/'ledger-000001.json';j=json.loads(p.read_text());j['reservations'][a['wrapper_key']]['invocation']='tampered';p.write_text(json.dumps(j))
                with self.assertRaises(ident.Stop):l.audit()
                l.entries.pop(a['wrapper_key'])
                with self.assertRaises(ident.Stop):l.read(a['wrapper_key'])
            finally:l.close();b.close()

    def test_compile_exception_reserved_no_retry(self):
        with tempfile.TemporaryDirectory(prefix='h4-synthetic-') as t:
            b=resources.OutputBudget(Path(t)/'own');l=ledger.Ledger(b)
            try:
                a=self.make_record();l.register(a);l.reserve(a['wrapper_key'])
                with self.assertRaises(MemoryError):
                    raise MemoryError('synthetic AS allocation failure after reservation')
                with self.assertRaises(ident.Stop):l.audit()
                self.assertEqual(l.reservations[a['wrapper_key']]['status'],'RESERVED')
            finally:l.close();b.close()
    def make_record(self,index=0,axis='cosine',numerical=FAKE):
        return ledger.wire(fake_identity(),axis,7+index,index,numerical)

    def test_reservation_complete_reuse_weights(self):
        with tempfile.TemporaryDirectory(prefix='h4-synthetic-') as t:
            b=resources.OutputBudget(Path(t)/'own');l=ledger.Ledger(b)
            try:
                a=self.make_record();c=self.make_record(1);l.register(a);l.register(c)
                l.reserve(a['wrapper_key']);metrics=dict.fromkeys(ledger.METRICS,3)
                l.complete(a['wrapper_key'],metrics)
                cached=l.complete(c['wrapper_key'],{},owner_key=a['wrapper_key'])
                self.assertEqual(cached['sample_weight'],1/32)
                self.assertEqual(l.audit(),{'logical_wrappers':2,'actual_invocations':1})
                self.assertEqual(sum(l.read(k)['sample_weight'] for k in l.entries),2/32)
            finally:l.close();b.close()

    def test_crash_window_consumed_orphan_STOP(self):
        with tempfile.TemporaryDirectory(prefix='h4-synthetic-') as t:
            b=resources.OutputBudget(Path(t)/'own');l=ledger.Ledger(b)
            try:
                a=self.make_record();l.register(a);l.reserve(a['wrapper_key'])
                with self.assertRaises(ident.Stop):l.complete(a['wrapper_key'],dict.fromkeys(ledger.METRICS,1),interrupt_after_record=True)
                with self.assertRaises(ident.Stop):l.audit()
                self.assertEqual(len(l.reservations),1)
                with self.assertRaises(ident.Stop):l.reserve(a['wrapper_key'])
            finally:l.close();b.close()

    def test_cap_before_compile(self):
        with tempfile.TemporaryDirectory(prefix='h4-synthetic-') as t:
            b=resources.OutputBudget(Path(t)/'own');l=ledger.Ledger(b,cap=1)
            try:
                a,c=self.make_record(),self.make_record(1);l.register(a);l.register(c);l.reserve(a['wrapper_key'])
                with self.assertRaises(ident.Stop):l.reserve(c['wrapper_key'])
            finally:l.close();b.close()

    def test_external_digest_registry_owner_and_scope(self):
        a,c=self.make_record(),self.make_record(1)
        owner={**a,'status':'COMPLETE','metrics':dict.fromkeys(ledger.METRICS,1),'cache_reuse':False,'cache_owner_wrapper_key':None,'actual_transpile_invocation_id':'fake-1'}
        registry={a['wrapper_key']:FAKE,c['wrapper_key']:FAKE};digest=ledger.completion_digest(owner)
        self.assertEqual(ledger.reuse_owner(owner,a,c,registry,digest),owner['metrics'])
        for field,value in (('geometry','0.80'),('axis','sine'),('candidate_template','other'),('cache_reuse',True),
                             ('status','RESERVED'),('status','AMBIGUOUS'),('actual_transpile_invocation_id',None)):
            altered={**owner,field:value}
            with self.assertRaises(ident.Stop):ledger.reuse_owner(altered,a,c,registry,digest)
        with self.assertRaises(ident.Stop):ledger.reuse_owner(owner,a,a,registry,digest)
        with self.assertRaises(ident.Stop):ledger.reuse_owner(owner,a,c,{},digest)
        with self.assertRaises(ident.Stop):ledger.reuse_owner(owner,a,c,registry,'0'*64)

    def test_modified_complete_or_ledger_stop(self):
        with tempfile.TemporaryDirectory(prefix='h4-synthetic-') as t:
            b=resources.OutputBudget(Path(t)/'own');l=ledger.Ledger(b)
            try:
                a=self.make_record();l.register(a);l.reserve(a['wrapper_key']);l.complete(a['wrapper_key'],dict.fromkeys(ledger.METRICS,1))
                p=b.root/('record-'+a['wrapper_key']+'.json');changed=json.loads(p.read_text());changed['metrics']['rz_count']=2;p.write_text(json.dumps(changed))
                with self.assertRaises(ident.Stop):l.read(a['wrapper_key'])
            finally:l.close();b.close()


def reject_number(value):
    def test(self):
        with self.assertRaises(ident.Stop):ident.canonical(value)
    return test


for name,value in [('nan',float('nan')),('inf',float('inf')),('complex_inf',complex(0,float('inf'))),('symbolic',object()),('nonstring_key',{1:2})]:
    setattr(IdentityTests,'test_reject_'+name,reject_number(value))


def mutation_test(target,key,value):
    def test(self):
        p,a,r=fake_stage();documents=[p,a,r];documents[target][key]=value
        with self.assertRaises(ident.Stop):gates.authorize('input_generation',p,a,r,explicit_launch=True)
    return test


for name,target,key,value in [('root',0,'output_root','/tmp/other'),('source',0,'source_commit','x'),
    ('placeholder',0,'inputs',{'fake':None}),('workers',0,'requested_workers',0),('stage',1,'stage','signal_compile'),
    ('cpu',1,'allowed_cpus',[]),('approval',2,'approved',False),('digest',2,'authorization_digest',FAKE),
    ('once',1,'one_shot',False),('result_prior',1,'result_prior',False)]:
    setattr(GateTests,'test_reject_'+name,mutation_test(target,key,value))
