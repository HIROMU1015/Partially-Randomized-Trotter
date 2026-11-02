"""Artificial arrays/circuits and injected providers only. No H5 science."""
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# Avoid the legacy plotting/import-all initializer, as native launchers do.
if 'trotterlib' not in sys.modules:
    package=types.ModuleType('trotterlib');package.__path__=[str(ROOT/'src/trotterlib')]
    sys.modules['trotterlib']=package

import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Operator
from trotterlib.df_hamiltonian import DFHamiltonian
from trottertracks.resource_applicability.h5_open_shell import sector as s, contract as c, input as inp
from trottertracks.resource_applicability.h8_sparse_reference_v1 import OccupationCSR


def spin_matrix(spatial):return np.kron(np.asarray(spatial,dtype=complex),np.eye(2))


def toy_ham(m=2):
    rng=np.random.default_rng(71);a=rng.normal(size=(m,m));g=rng.normal(size=(m,m));h=rng.normal(size=(m,m))
    return DFHamiltonian(.137,spin_matrix((a+a.T)*.04),np.array([.11,-.07]),
        (spin_matrix((g+g.T)*.1),spin_matrix((h+h.T)*.1)),{})


class SectorTests(unittest.TestCase):
    def test_H5_equal_dimensions_different_bases(self):
        sector=s.sector_for(5,3,2);self.assertEqual(sector.dimension,100)
        primitive=s.SpinPrimitive(spin_matrix(np.diag([.1,.2,.4,.8,1.6])),sector)
        self.assertEqual(primitive.sides,(10,10));self.assertNotEqual(primitive.alpha_basis,primitive.beta_basis)
        self.assertFalse(np.allclose(primitive.alpha_values,primitive.beta_values))
    def test_H7_geometry_only_sector_dimension(self):
        self.assertEqual(s.sector_for(7,4,3).dimension,1225)
    def test_one_and_squared_cross_term_against_occupation_expm(self):
        rng=np.random.default_rng(38);a=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3));a=(a+a.conj().T)/2
        sector=s.sector_for(3,2,1);matrix=spin_matrix(a);one=OccupationCSR(sector.basis_indices,6).one(matrix).toarray()
        x=rng.normal(size=9)+1j*rng.normal(size=9);x/=np.linalg.norm(x)
        for weight in (None,.37,-.29):
            action=s.SpinPrimitive(matrix,sector,weight=weight)
            operator=one if weight is None else weight*one@one
            for time in (.8,-.8,0.):
                np.testing.assert_allclose(action(x,time),expm(-1j*time*operator)@x,atol=2e-12)
        eigen_a=np.linalg.eigvalsh(s.small_one_body(a,action.alpha_basis))
        eigen_b=np.linalg.eigvalsh(s.small_one_body(a,action.beta_basis))
        np.testing.assert_allclose(action.energies,-.29*(eigen_a[:,None]+eigen_b[None,:])**2)
    def test_basis_reordering_rejected(self):
        from dataclasses import replace
        sector=s.sector_for(3,2,1)
        with self.assertRaises(ValueError):s.SpinPrimitive(np.eye(6),replace(sector,basis_indices=sector.basis_indices[::-1]))
    def test_spin_structure_and_nonfinite_fail_closed(self):
        sector=s.sector_for(2,1,0)
        for name in ('cross','unequal','nonhermitian','nan'):
            a=np.eye(4,dtype=complex)
            if name=='cross':a[0,1]=1e-30
            elif name=='unequal':a[1,1]=2
            elif name=='nonhermitian':a[0,2]=a[1,3]=.2
            else:a[0,0]=float('nan')
            with self.assertRaises(ValueError):s.SpinPrimitive(a,sector)
    def test_cache_eviction_and_norm_preservation(self):
        sector=s.sector_for(3,2,1);a=s.SpinPrimitive(np.eye(6),sector,cache_size=2)
        x=np.ones(9,dtype=complex)*2
        for time in (0.,.1,.2,.3):self.assertAlmostEqual(np.linalg.norm(a(x,time)),np.linalg.norm(x))
        self.assertEqual(len(a.phases),2)
    def test_S2_against_full_Fock_OpenFermion(self):
        from openfermion import s_squared_operator, jordan_wigner, get_sparse_operator
        for m,na,nb in ((2,1,0),(3,2,1),(3,3,0)):
            sector=s.sector_for(m,na,nb);indices=sector.basis_indices
            full=get_sparse_operator(jordan_wigner(s_squared_operator(m)),n_qubits=2*m)
            independent=full[indices,:][:,indices].toarray()
            np.testing.assert_allclose(s.spin_squared(sector).toarray(),independent,atol=1e-13)
    def test_artificial_100dim_reference_and_doublet(self):
        ham=DFHamiltonian(.2,spin_matrix(np.diag(np.arange(5))),np.array([.01]),(np.eye(10),),{})
        sector=s.sector_for(5,3,2);H=s.independent_df_matrix(ham,sector)
        x=np.zeros(100,dtype=complex);x[int(np.argmin(H.diagonal().real))]=1
        checks,_=s.validate_state(ham,sector,x)
        self.assertAlmostEqual(checks['S_squared'],.75);self.assertLess(checks['action_difference'],1e-12)
        with self.assertRaises(ValueError):s.validate_state(ham,sector,x*2)
        with self.assertRaises(ValueError):s.validate_state(ham,sector,x,maximum_matvecs=99)
    def test_high_spin_is_not_relabelled_doublet(self):
        ham=DFHamiltonian(0.,np.zeros((6,6)),np.array([1.]),(np.eye(6),),{})
        sector=s.sector_for(3,3,0);x=np.ones(1,dtype=complex)
        with self.assertRaisesRegex(ValueError,'DOUBLET'):s.validate_state(ham,sector,x)


class InputTests(unittest.TestCase):
    def test_neutral_ROHF_provider_arguments_and_no_save(self):
        seen={}
        class Mol:
            nelectron=5;nelec=(3,2)
            def build(self):seen.update(charge=self.charge,spin=self.spin,geometry=self.atom,basis=self.basis,symmetry=self.symmetry)
            def nao_nr(self):return 5
            def intor_symmetric(self,_):return np.eye(5)
            def energy_nuc(self):return .1
        class MF:
            mo_coeff=np.eye(5);converged=True;e_tot=-1.
            def __init__(self,mol):seen['ROHF']=True
            def kernel(self):seen['kernel']=seen.get('kernel',0)+1;seen['chkfile']=self.chkfile
        provider=(Mol,MF,lambda mol,mf:(np.eye(5),np.zeros((5,)*4)),
            lambda a,b:(spin_matrix(a),np.zeros((10,)*4)))
        scratch=Path(__file__).resolve().parents[1]/'.h5-artificial-no-write'
        # This fake provider performs no IO; existing worktree itself is the scratch.
        arrays,receipt=inp.rohf_integrals(ROOT,c.Calls(c.call_limits(3)),provider=provider)
        self.assertEqual((seen['charge'],seen['spin'],seen['kernel']),(0,1,1));self.assertTrue(seen['ROHF'])
        self.assertFalse(receipt['state_is_ROHF_determinant']);self.assertEqual(arrays['one_body'].shape,(10,10))
    def test_tol_only_signed_order_rank_and_projection(self):
        weights=np.array([.2,-.7,.03]);raw=np.stack([np.eye(4)*v for v in (.1,.2,.4)])
        seen={}
        def decomp(two,**kwargs):seen.update(kwargs);return weights,raw,np.zeros((4,4)),1e-9
        ham,receipt,_=inp.weighted_df(0.,np.zeros((4,4)),np.zeros((4,)*4),c.Calls(c.call_limits(3)),decomposer=decomp)
        self.assertEqual(seen,{'truncation_threshold':1e-8});np.testing.assert_array_equal(ham.lambdas,weights)
        self.assertEqual(receipt['actual_rank'],3);self.assertTrue(receipt['all_returned_fragments'])
    def test_weighted_projection_rejects_large_change(self):
        g=np.eye(4,dtype=complex);g[0,2]=g[1,3]=1e-3
        def decomp(*a,**k):return np.array([1.]),g[None],np.zeros((4,4)),0.
        with self.assertRaisesRegex(ValueError,'WEIGHTED'):
            inp.weighted_df(0.,np.zeros((4,4)),np.zeros((4,)*4),c.Calls(c.call_limits(1)),decomposer=decomp)


class ContractTests(unittest.TestCase):
    def test_scope_and_two_phase_counts(self):
        reg=c.registry(17);self.assertEqual(reg[0]['prefix'],5)
        initial=c.tasks(17,'1'*40,'2'*64)
        extra=c.tasks(17,'1'*40,'2'*64,extension=True,eligible=[r['id'] for r in reg if r['R']])
        self.assertEqual((len(initial),len(extra),2*(len(initial)+len(extra))),(6,14,40))
        seeds=[r['seed'] for r in initial+extra if r['seed'] is not None]
        self.assertEqual(len(set(seeds)),16)
    def test_namespace_source_and_input_change_seed(self):
        self.assertNotEqual(c.seed('1'*40,'2'*64,'toy',0),c.seed('3'*40,'2'*64,'toy',0))
        self.assertNotEqual(c.seed('1'*40,'2'*64,'toy',0),c.seed('1'*40,'4'*64,'toy',0))
    def test_no_eligible_extension_and_no_rank_guess(self):
        self.assertEqual(c.tasks(17,'1'*40,'2'*64,extension=True),[])
        for value in (None,0,-1,True):
            with self.assertRaises(ValueError):c.registry(value)
        draft=c.draft();self.assertIsNone(draft['actual_rank']);self.assertFalse(draft['approved'])
    def test_old_grant_and_unapproved_rejected(self):
        for grant in (None,{},dict(kind='H6_TECHNICAL',approved=True)):
            with self.assertRaises(ValueError):c.verify_grant({},grant,'1'*40)
    def test_new_grant_full_binding_and_mutations(self):
        from copy import deepcopy
        manifest=dict(SOURCE='1'*40,approved=True,runtime_authorization=True,profile_digest='2'*64,
            carry_digest='3'*64,target_digest=c.digest(c.target()),compiler_hash=c.COMPILER_HASH,
            contract_digest=c.digest(c.draft()),output='/home/AbeHiromu/projects/h5-handoff-evidence/UNUSED_TEST_ONLY')
        grant=dict(kind='H5_NEUTRAL_OPEN_SHELL_ONE_SHOT_V1',SOURCE='1'*40,approved=True,
            runtime_authorization=True,manifest_digest=c.digest(manifest),maximum_compile=40,
            one_shot=True,automatic_retry=False,**{k:manifest[k] for k in ('profile_digest','carry_digest','target_digest','output')})
        self.assertEqual(c.verify_grant(manifest,grant,'1'*40),Path(manifest['output']))
        for key,value in (('SOURCE','4'*40),('approved',False),('maximum_compile',42),('carry_digest','5'*64),('automatic_retry',True)):
            bad=deepcopy(grant);bad[key]=value
            with self.assertRaises(ValueError):c.verify_grant(manifest,bad,'1'*40)
    def test_call_failures_are_not_refunded(self):
        calls=c.Calls({'compile':1});calls.take('compile')
        with self.assertRaises(RuntimeError):calls.take('compile')
        self.assertEqual(calls.used['compile'],1)
    def test_pinned_bounded_solver_counter_interface(self):
        from trottertracks.resource_applicability.ax2b_h6_controller import bounded_solver
        calls=c.Calls({'solver_matvec':10});recorded=[]
        calls.recorder=lambda key,used:recorded.append((key,dict(used)))
        def solver(op,**kwargs):
            np.testing.assert_allclose(op@np.array([1.,0.,0.]),[1.,0.,0.])
            return np.array([1.]),np.array([[1.],[0.],[0.]])
        bounded_solver(lambda x:x,3,np.array([1.,0.,0.]),calls,solver=solver)
        self.assertEqual(calls.used['solver_matvec'],1)
        self.assertEqual(recorded[0][0],'solver_matvec')
        calls.caps={'solver_matvec':1};self.assertEqual(calls.limits,calls.caps)
        with self.assertRaises(RuntimeError):calls.take('solver_matvec')


class NativeTests(unittest.TestCase):
    def setUp(self):
        def forbidden(*args,**kwargs):raise AssertionError('REAL_INPUT_OR_TRANSPILE_FORBIDDEN')
        self.patches=[patch('numpy.load',forbidden),patch('qiskit.transpile',forbidden),
            patch('trotterlib.rte_compiled_cost.transpile',forbidden)]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        from trottertracks.resource_applicability.h5_open_shell import science
        self.science=science;self.ham=toy_ham();self.prepared=science.preparations(self.ham)
    def ordinary(self,unitary):
        z=np.zeros_like(unitary);return np.block([[np.eye(len(unitary)),z],[z,unitary]])
    def independent(self,cell):
        ham=self.ham;n=ham.n_qubits;oracle=OccupationCSR(tuple(range(1<<n)),n)
        selected=len(ham.lambdas) if cell['method']=='B1' else cell['prefix']
        matrices=[oracle.one(ham.one_body).toarray()]+[w*(oracle.one(g)@oracle.one(g)).toarray()
            for w,g in zip(ham.lambdas[:selected],ham.g_matrices[:selected])]
        reverse=[int(f'{i:0{n}b}'[::-1],2) for i in range(1<<n)]
        matrices=[a[np.ix_(reverse,reverse)] for a in matrices]
        def s2(t):
            U=np.eye(1<<n,dtype=complex)
            for a in matrices+matrices[::-1]:U=expm(-1j*t/2*a)@U
            return U
        if cell['formula']=='4th':
            w1=1/(2-2**(1/3));w0=-2**(1/3)*w1
            U=s2(.8*w1)@s2(.8*w0)@s2(.8*w1)
        else:U=s2(.8)
        scalar=ham.constant+(self.science.tail_identity(ham,cell['prefix']) if cell['method']=='B0I' else 0.)
        return np.exp(-.8j*scalar)*U
    def test_S2_S4_and_B0I_full_system_controlled_phases(self):
        for cell in c.registry(2)[:4]:
            task=dict(cell=cell,replica=0,seed=None,validate=True)
            evolution,events=self.science.build_group(self.ham,self.prepared,task,c.Calls(c.call_limits(2)))
            self.assertIsNone(events)
            np.testing.assert_allclose(Operator(evolution.circuit).data,self.ordinary(self.independent(cell)),atol=2e-11)
    def test_PR_native_equals_legacy_explicit_event_circuit(self):
        from trotterlib.df_partial_s2 import make_df_partial_s2_step_request,QiskitDFPartialS2CircuitBuilder
        for K in (2,4):
            cell=next(x for x in c.registry(2) if x['K']==K)
            task=dict(cell=cell,replica=0,seed=283,validate=True)
            evolution,events=self.science.build_group(self.ham,self.prepared,task,c.Calls(c.call_limits(2)))
            request=make_df_partial_s2_step_request(self.prepared['partial'],step_time=.8,rte_steps=2,
                finite_taylor_order=K,seed=283,controlled=True,ancilla_qubit=4)
            legacy=QiskitDFPartialS2CircuitBuilder().build_step(request)
            np.testing.assert_allclose(Operator(evolution.circuit).data,Operator(legacy.circuit).data,atol=2e-11)
            self.assertEqual(len(events),2)
    def test_RTE_distribution_and_component_weights_unchanged(self):
        from trotterlib.rte import finite_rte_distribution
        for K in (2,4):
            law=finite_rte_distribution(.17,K)
            weights=[.17**k/math.factorial(k)*math.hypot(1.,.17/(k+1)) for k in range(0,K+1,2)]
            np.testing.assert_allclose(law.order_probabilities,np.array(weights)/sum(weights),atol=1e-15)
        self.assertEqual(self.prepared['partial'].coefficient_atol,0.)
    def test_PR_registered_cutoff_above_default_residual(self):
        from trotterlib.df_partial_s2 import make_df_partial_s2_step_request,QiskitDFPartialS2CircuitBuilder
        from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
        from trotterlib.rte import step_taylor_truncation_residual_bound
        ham=DFHamiltonian(self.ham.constant,self.ham.one_body,self.ham.lambdas*1000,self.ham.g_matrices,{})
        prepared=self.science.preparations(ham);p=prepared['partial'];tau=p.exact_rte_lambda_r*.8/2
        self.assertGreater(step_taylor_truncation_residual_bound(tau,2),1e-10)
        for K in (2,4):
            cell=next(x for x in c.registry(2) if x['K']==K)
            task=dict(cell=cell,replica=0,seed=283,validate=True)
            evolution,events=self.science.build_group(ham,prepared,task,c.Calls(c.call_limits(2)))
            request=make_df_partial_s2_step_request(p,step_time=.8,rte_steps=2,finite_taylor_order=K,
                truncation_tolerance=_explicit_cutoff_tolerance(tau,K),seed=283,controlled=True,ancilla_qubit=4)
            self.assertEqual(events,[e.to_dict() for e in request.rte_occurrence.events])
            legacy=QiskitDFPartialS2CircuitBuilder().build_step(request)
            np.testing.assert_allclose(Operator(evolution.circuit).data,Operator(legacy.circuit).data,atol=2e-11)
    def test_wrapper_cos_sin_expected_signal_and_fingerprint(self):
        from qiskit.quantum_info import Statevector
        from trottertracks.resource_applicability.ax2b_native_df_v5 import build_native_hadamard_wrapper
        cell=c.registry(2)[3];task=dict(cell=cell,replica=0,seed=None,validate=True)
        evolution,_=self.science.build_group(self.ham,self.prepared,task,c.Calls(c.call_limits(2)))
        rng=np.random.default_rng(91);x=rng.normal(size=16)+1j*rng.normal(size=16);x/=np.linalg.norm(x)
        z=np.vdot(x,self.independent(cell)@x);v=np.r_[x,np.zeros_like(x)]
        for axis,expected in (('cosine',z.real),('sine',z.imag)):
            qc=build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=100000)
            out=Statevector(v).evolve(qc).data;observed=np.linalg.norm(out[:16])**2-np.linalg.norm(out[16:])**2
            self.assertAlmostEqual(observed,expected,places=11)
        evolution.circuit.p(.1,4)
        with self.assertRaises(ValueError):build_native_hadamard_wrapper(evolution,axis='cosine',max_instructions=100000)
    def test_paired_cost_interface_mock_compile_only(self):
        from trotterlib.rte_compiled_cost import compiler_settings_hash
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint
        seen=[]
        def fake(qc,settings):
            seen.append(qc)
            return types.SimpleNamespace(rz_count=7,cx_count=3,rz_depth=5,cx_depth=2,total_depth=9,circuit_size=17,
                compiler_settings_hash=compiler_settings_hash(settings),actual_circuit_fingerprint=canonical_qiskit_circuit_fingerprint(qc))
        sector=s.sector_for(2,1,0);x=np.zeros(2,dtype=complex);x[0]=1
        full=np.zeros(16,dtype=complex);full[sector.basis_indices]=x
        task=dict(cell=c.registry(2)[0],replica=0,seed=None,validate=True)
        calls=c.Calls(c.call_limits(2))
        result=self.science.cost_group(self.ham,sector,full,x,self.prepared,task,calls,compile_fn=fake)
        self.assertEqual(len(seen),2);self.assertEqual(calls.used['compile'],2)
        self.assertEqual([r['axis'] for r in result['rows']],['cosine','sine'])
        self.assertTrue(all(r['compiler_hash']==c.COMPILER_HASH for r in result['rows']))
    def test_artificial_signal_Horner_forward_and_B0I_reference(self):
        ham=self.ham;sector=s.sector_for(2,1,0)
        H=s.independent_df_matrix(ham,sector);_,v=np.linalg.eigh(H);x=v[:,0]
        checks,H=s.validate_state(ham,sector,x)
        records=self.science.panel_signals(ham,sector,x,H,checks,self.prepared,c.Calls(c.call_limits(2)))
        self.assertEqual(len(records),6)
        b0=next(r for r in records if r['cell']['method']=='B0');bi=next(r for r in records if r['cell']['method']=='B0I')
        self.assertEqual(b0['reference'],bi['reference'])
        z=lambda v:complex(v['real'],v['imag'])
        np.testing.assert_allclose(z(bi['signals']['corrected']),z(b0['signals']['corrected'])*np.exp(-.8j*self.science.tail_identity(ham,1)))
    def test_generalized_primitives_vs_native_full_system(self):
        ham=toy_ham(3);sector=s.sector_for(3,2,1)
        H=s.independent_df_matrix(ham,sector);_,v=np.linalg.eigh(H);x=v[:,0]
        full=np.zeros(64,dtype=complex);full[sector.basis_indices]=x
        result=self.science.validate_primitives(ham,sector,full,x,self.science.preparations(ham),c.Calls(c.call_limits(2)))
        self.assertLess(result['max_error'],1e-10)
        self.assertTrue(any(r['time']<0 for r in result['rows']))


class ExecutionTests(unittest.TestCase):
    def test_cleanup_ESRCH_first_middle_all_and_FDs(self):
        import errno,os
        from trottertracks.resource_applicability.h5_open_shell.execution import Child,cleanup
        for gone in ({0},{1},{0,1,2}):
            children=[Child(300+i,dict(start_ticks=i),100+i,ROOT,dict(id=i)) for i in range(3)]
            sent=[];closed=[]
            def send(fd,sig):
                sent.append(fd)
                if fd-100 in gone:raise ProcessLookupError(errno.ESRCH,'gone')
            with patch('signal.pidfd_send_signal',side_effect=send),patch('os.waitid',return_value=object()),\
                    patch('os.waitpid',side_effect=lambda pid,flags:(pid,0)),patch('os.close',side_effect=closed.append):
                errors=cleanup(children)
            self.assertEqual(errors,[]);self.assertEqual(sent,[100,101,102]);self.assertEqual(closed,[100,101,102])
            self.assertTrue(all(ch.reaped for ch in children))
    def test_non_ESRCH_retained_and_next_cleanup_continues(self):
        import errno
        from trottertracks.resource_applicability.h5_open_shell.execution import Child,cleanup
        children=[Child(300+i,dict(start_ticks=i),100+i,ROOT,{}) for i in range(3)]
        def send(fd,sig):
            if fd==100:raise PermissionError(errno.EPERM,'denied')
        with patch('signal.pidfd_send_signal',side_effect=send) as sent,patch('os.waitid',return_value=object()),\
                patch('os.waitpid',side_effect=lambda pid,flags:(pid,0)),patch('os.close') as closed:
            errors=cleanup(children)
        self.assertEqual(sent.call_count,3);self.assertEqual(len(errors),1);self.assertEqual(errors[0]['operation'],'terminate')
        self.assertEqual(closed.call_count,3)
    def test_double_reap_and_missing_identity_handle_rejected(self):
        from trottertracks.resource_applicability.h5_open_shell.execution import Child
        child=Child(300,dict(start_ticks=1),None,ROOT,{})
        with self.assertRaisesRegex(ValueError,'PIDFD'):child.terminate()
        child.reaped=True
        with self.assertRaisesRegex(ValueError,'DOUBLE'):child.reap()
    def test_unapproved_execution_does_not_create_output(self):
        from trottertracks.resource_applicability.h5_open_shell.execution import execute
        with patch('pathlib.Path.mkdir') as mkdir,patch('os.fork') as fork:
            with self.assertRaises(ValueError):execute(ROOT,dict(SOURCE='1'*40),dict(kind='H6_OLD'))
        mkdir.assert_not_called();fork.assert_not_called()
    def test_metadata_CLI_default_has_no_science_import(self):
        import ast
        path=ROOT/'src/trottertracks/resource_applicability/h5_open_shell/contract.py'
        imports=[n.names[0].name for n in ast.parse(path.read_text()).body if isinstance(n,ast.Import)]
        self.assertFalse(set(imports)&{'numpy','qiskit','pyscf','scipy','openfermion'})
    def test_primitive_call_budget_is_actual_registered_coverage(self):
        from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
        for rank in (2,9,17,25):
            times={(i,t) for cell in c.registry(rank) if cell['method']!='B0I'
                for i,t in primitive_time_schedule(cell,T=.8)['unique_primitive_times']}
            self.assertEqual(c.call_limits(rank)['primitive'],3*len(times))
    def test_supervisor_mock_completion_and_first_STOP_cleanup(self):
        import os,time
        from trottertracks.resource_applicability.h5_open_shell.execution import supervise,write_json
        memory=dict(host_available_bytes=1000,effective_available_bytes=1000,oom={},full_psi_avg10={'host':0})
        limits=dict(headroom_bytes=1,worker_RSS_bytes=100,AS_bytes=100,aggregate_RSS_bytes=400,
            output_bytes=2**20,minimum_free_bytes=1)
        base=Path('/home/AbeHiromu/projects/h5-handoff-evidence/20261016/open-shell-readiness')
        class Fake:
            def __init__(self,pid,directory,task,code):
                self.pid=pid;self.identity={'pid':pid,'start_ticks':pid};self.directory=directory;self.task=task
                self.reaped=False;self.exit_code=None;self.code=code;self.closed=False;self.killed=False
            def exited(self):return True
            def reap(self):self.exit_code=self.code;self.reaped=True;self.closed=True;return self.code
            def terminate(self):self.killed=True
            def close(self):self.closed=True
        for fail in (False,True):
            out=base/f'mock-supervisor-{os.getpid()}-{time.monotonic_ns()}';out.mkdir()
            spawned=[]
            def spawn(directory,task,function,cpu,limits):
                directory.mkdir();code=1 if fail and not spawned else 0
                write_json(directory/'terminal.json',dict(status='STOP' if code else 'COMPLETE'))
                write_json(directory/'result.json',dict(task=task))
                child=Fake(1000+len(spawned),directory,task,code);spawned.append(child);return child
            with patch('time.sleep'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'FAIL_CLOSED'):
                        supervise(out,[dict(id=i) for i in range(5)],None,[1,2,3,4],limits,memory,spawn_fn=spawn,observe=lambda:memory)
                else:
                    result=supervise(out,[dict(id=i) for i in range(5)],None,[1,2,3,4],limits,memory,spawn_fn=spawn,observe=lambda:memory)
                    self.assertEqual(len(result['completed']),5)
            summary=json.loads((out/'summary.json').read_bytes())
            self.assertTrue(all(ch.reaped and ch.closed for ch in spawned))
            self.assertEqual(len(spawned),4 if fail else 5)
            self.assertEqual(summary['cleanup_errors'],[])
            if fail:self.assertIn('WORKER_STOP',summary['first_stop'])


if __name__=='__main__':unittest.main(verbosity=2)
