"""Non-molecular implementation tests; never generate the registered H4 pack."""
import copy
import importlib.util
import itertools
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from trottertracks.representation_exploration import qc_mixed_access_math_v1 as m
from trottertracks.representation_exploration import qc_mixed_access_checker_v1 as c
from trottertracks.representation_exploration import qc_mixed_access_resource_v1 as cost
from trottertracks.representation_exploration import qc_mixed_access_execution_v1 as gate
from trottertracks.representation_exploration import qc_mixed_access_pipeline_v1 as pipeline

ROOT=Path(__file__).resolve().parents[1]


def fixture(ns=2):
    # Hand-written symmetric density factors, including signed and literal-zero
    # channels. No geometries, molecular integrals, SCF, or scientific packs.
    h=np.diag(np.linspace(-.3,.2,ns))
    for i in range(ns-1): h[i,i+1]=h[i+1,i]=.07
    f=np.diag(np.linspace(.4,.2,ns))
    for i in range(ns-1): f[i,i+1]=f[i+1,i]=.06
    g=np.zeros((ns,ns)); g[0,-1]=g[-1,0]=.11
    eri=np.einsum('pq,rs->pqrs',f,f)-.3*np.einsum('pq,rs->pqrs',g,g)
    return h,eri,.13


def actor_reference(node,model):
    kind,index,t,adj=node; dim=1<<model['n']
    A=np.diag(model['diagonal']+model['scalar'])
    if kind=='SCALAR': return np.eye(dim)*np.exp(-1j*t*model['scalar'])
    if kind=='A': return expm(-1j*t*A)
    if kind=='D':
        actor=model['gaussian'][index]; frame=m.fock_frame(m.spin_matrix(actor['orbital_matrix']))
        H=(frame*actor['energy'][None,:])@frame.conj().T
        return expm(-1j*t*H)
    if kind=='P':
        x,z,g=model['paulis'][index]
        paulis=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[1,0],[0,-1]]),np.array([[0,-1j],[1j,0]])]
        H=np.ones((1,1))
        for i in reversed(range(model['n'])):
            H=np.kron(H,paulis[((x>>i)&1)+2*((z>>i)&1)])
        return expm(-1j*t*g*H)
    group=model['groups'][index]; B=np.zeros_like(A,dtype=complex)
    a,b=group['dst'],group['src']; B[a,b]=group['amplitude']; B[b,a]=group['amplitude']
    if kind=='E': return expm(-1j*t*B)
    C=expm(1j*t*A)@expm(-1j*t*(A+B))
    return C.conj().T if adj else C


def independent_seed(method,h,model):
    # Explicit algebra; do not use the generator's stage scheduler.
    dim=1<<model['n']
    if method.endswith('4'):
        s=1/(4-4**(1/3)); result=np.eye(dim,dtype=complex)
        for weight in (s,s,1-4*s,s,s): result=independent_seed(method[0]+'2',h*weight,model)@result
        return result
    if method=='R2':
        def P(t):
            result=actor_reference(('A',0,t,False),model)
            for i in range(len(model['groups'])): result=result@actor_reference(('R',i,t,False),model)
            return result
        return P(h/2)@P(-h/2).conj().T
    actors=([('A',0)]+[('E',i) for i in range(len(model['groups']))]) if method=='E2' else (
            [('D',i) for i in range(len(model['gaussian']))] if method=='D2' else [('P',i) for i in range(len(model['paulis']))])
    result=np.eye(dim,dtype=complex)
    for kind,index in actors+list(reversed(actors)):
        result=actor_reference((kind,index,h/2,False),model)@result
    if method=='D2': result*=np.exp(-1j*h*model['scalar'])
    return result


class MathematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model=m.build_model(*fixture())

    def test_car_contractions_and_adjoint(self):
        for degree in (2,4):
            for modes in itertools.product(range(2),repeat=degree):
                ops=tuple(zip(modes,([0,1] if degree==2 else [0,1,0,1])))
                raw=m.term_matrix(2,{ops:1.})
                canonical=m.term_matrix(2,dict(m.normal_form(ops)))
                np.testing.assert_allclose(raw,canonical,atol=1e-14)

    def test_raw_car_independent_two_and_three_spatial(self):
        for ns in (2,3):
            model=m.build_model(*fixture(ns)); expected=c.raw_hamiltonian(*fixture(ns))
            np.testing.assert_allclose(model['H'],expected,atol=1e-13)
            self.assertGreater(len(model['groups']),0)
            self.assertTrue(any(g['g_weights'] for g in model['groups']))

    def test_full_df_and_correction_signed_no_truncation(self):
        model=self.model
        self.assertEqual(len(model['factors']),4)
        self.assertTrue(any(lam<0 for lam,f in model['factors']))
        np.testing.assert_allclose(pipeline.derived_df_matrix(model),c.raw_hamiltonian(*fixture()),atol=1e-12)

    def test_gaussian_frame_reflection_and_eight_modes(self):
        for ns in (2,4):
            u=np.eye(ns); u[:2,:2]=[[.8,.6],[.6,-.8]]
            if ns==4: u=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])/2
            expected=m.fock_frame(m.spin_matrix(u)); actual=m.gaussian_apply(np.eye(len(expected)),m.qr_plan(u))
            np.testing.assert_allclose(actual,expected,atol=1e-12)
            np.testing.assert_allclose(m.gaussian_apply(actual,m.qr_plan(u),True),np.eye(len(expected)),atol=1e-12)

    def test_every_actor_dense_exponential_negative_zero_times(self):
        model=self.model; nodes=[('A',0),('SCALAR',0)]+[('P',i) for i in range(len(model['paulis']))]+[('D',i) for i in range(len(model['gaussian']))]+[(kind,i) for kind in ('E','R') for i in range(len(model['groups']))]
        for kind,i in nodes:
            for t in (-.21,0.,.19):
                for adj in ([False,True] if kind=='R' else [False]):
                    node=(kind,i,t,adj)
                    np.testing.assert_allclose(m.apply_stage(np.eye(16),node,model),actor_reference(node,model),atol=2e-13)

    def test_relative_negative_is_not_adjoint(self):
        group=next(i for i,g in enumerate(self.model['groups']) if np.any(g['delta']*g['amplitude']!=0))
        plus=m.apply_stage(np.eye(16),('R',group,.8,False),self.model)
        minus=m.apply_stage(np.eye(16),('R',group,-.8,False),self.model)
        self.assertGreater(np.linalg.norm(minus-plus.conj().T),1e-5)

    def test_zero_relative_branch(self):
        model=copy.deepcopy(self.model); group=model['groups'][0]
        group['amplitude'][:]=0; group['delta'][:]=0
        np.testing.assert_allclose(m.apply_stage(np.eye(16),('R',0,-.2,True),model),np.eye(16),atol=0)

    def test_all_seven_formulas_independent_scheduler_and_convergence(self):
        for method in ('P2','E2','E4','D2','D4','R2','R4'):
            seed=m.seed_matrix(method,.4,self.model)
            np.testing.assert_allclose(seed,independent_seed(method,.4,self.model),atol=3e-12)
            errors=[np.linalg.norm(m.power_doubling(m.seed_matrix(method,.4/r,self.model),r)-expm(-.4j*self.model['H']),'fro') for r in (1,2,4)]
            self.assertLess(errors[-1],errors[0]/5)

    def test_pauli_sector_leakage_full_rows_and_phase(self):
        model=self.model; cols=[i for i in range(16) if i.bit_count()==2]; other=[i for i in range(16) if i.bit_count()!=2]
        # A single JW X/Y word can leak, even when the complete H conserves nu.
        leaks=[np.linalg.norm(m.apply_stage(np.eye(16),('P',i,.3,False),model)[np.ix_(other,cols)]) for i in range(len(model['paulis']))]
        self.assertGreater(max(leaks),1e-5)
        exact=expm(-.3j*model['H']); shifted=np.exp(.04j)*exact
        self.assertGreater(np.linalg.norm((shifted-exact)[:,cols]),.04)

    def test_taylor_reference_and_power_count(self):
        budget=m.Budget({'reference_matrices':1,'dense_power_products':4})
        reference,info=m.taylor_reference(self.model['H'],4.,budget)
        np.testing.assert_allclose(reference,expm(-4j*self.model['H']),atol=2e-13)
        np.testing.assert_allclose(m.power_doubling(reference,16,budget),np.linalg.matrix_power(reference,16),atol=2e-13)
        self.assertEqual(budget.counts['dense_power_products'],4)
        with self.assertRaises(m.ContractError): m.power_doubling(reference,3)

    def test_bad_shapes_complex_and_nan_rejected(self):
        h,e,s=fixture()
        for badh,bade in ((h.astype(complex),e),(h,e[:1]),(h*np.nan,e),(np.zeros((5,5)),np.zeros((5,)*4))):
            with self.assertRaises(m.ContractError): m.build_model(badh,bade,s)


class EvidenceAndResourceTests(unittest.TestCase):
    def test_norm_certificate_no_checker_svd_and_tamper(self):
        error=np.array([[1,.2j],[0,.03],[.1,.04]],complex)
        norm,sigma,vh=m.norm_certificate(error)
        with patch('numpy.linalg.svd',side_effect=AssertionError('checker SVD forbidden')):
            report=c.check_norm(error,sigma,vh,norm)
            self.assertEqual(report['SVD_calls'],0)
        for badsigma,badvh,declared in ((sigma*.5,vh,norm),(sigma,vh*.9,norm),(sigma,vh,norm*.5)):
            with self.assertRaises(c.EvidenceError): c.check_norm(error,badsigma,badvh,declared)

    def test_eligibility_tail_and_boundary(self):
        def rows(errors): return [dict(method='P2',T=1,r=r,error=e) for r,e in zip([1,2,4,8],errors)]
        args=('P2',1,.4,[1,2,4,8],.001,1e-8)
        good,boundary=cost.eligible_steps(rows([.01,.0003,.0002,.0001]),*args)
        self.assertEqual(good,[2,4]); self.assertEqual(boundary,'GRID_CHECKED')
        good,boundary=cost.eligible_steps(rows([.01,.0003,.0007,.0001]),*args)
        self.assertEqual(good,[]); self.assertEqual(boundary,'BOUNDARY_UNCONFIRMED')

    def test_unknown_function_not_zero_and_equal_rotation_price(self):
        model=m.build_model(*fixture())
        for method in ('P2','E2','D2','R2','R4'):
            row=cost.ledger(method,1,8,.6,model)
            self.assertGreater(row['per_step']['RZ'],0)
            value,tr=cost.modeled_total(row,6)
            self.assertEqual(tr,math.ceil(6*math.log2(1/row['eta_RZ'])))
            self.assertGreater(value,0)
            if method[0]=='R': self.assertEqual(row['F_remaining'],'UNKNOWN'); self.assertEqual(row['total_workspace'],'UNKNOWN')

    def test_cap_counts_abort(self):
        budget=m.Budget({'norm_SVD_including_checks':0})
        with self.assertRaises(m.ContractError): m.norm_certificate(np.eye(2),budget)

    def test_s1_s2_word_bounds_and_center_theta_sharing(self):
        model=m.build_model(*fixture())
        for method in ('R2','R4'):
            row=cost.ledger(method,1,8,.6,model); precision=row['group_precision']
            for group,bits in zip(model['groups'],precision['by_j']):
                f=bits[2]; z=sum(v!=0 for v in group['delta_weights'].values()); zg=sum(v!=0 for v in group['g_weights'].values())
                bmax=abs(group['g0'])+sum(abs(v) for v in group['g_weights'].values())
                self.assertLessEqual(bmax*(z+1)*2**(-f-1)*precision['S2']/2,precision['per_group_detuning_budget'])
                self.assertLessEqual((zg+1)*2**(-f-1)*precision['S1'],precision['per_group_amplitude_budget'])
            original=m.stages(method,1/8,model); compressed=cost.resource_nodes(original)
            self.assertEqual(sum(node[0]=='RPAIR' for node in compressed),1 if method=='R2' else 5)
            self.assertEqual(row['relative_blocks_per_step'],sum(node[0] in ('R','RPAIR') for node in compressed))
        # Direct two-dimensional verification of the five-rotation center.
        d,b,t=.21,-.07,.13; rho=math.hypot(d,b); theta=math.atan2(b,d)
        rz=lambda phi:np.diag([np.exp(-.5j*phi),np.exp(.5j*phi)])
        ry=lambda phi:np.array([[math.cos(phi/2),-math.sin(phi/2)],[math.sin(phi/2),math.cos(phi/2)]])
        C=lambda t:rz(-2*t*d)@ry(theta)@rz(2*t*rho)@ry(-theta)
        expected=rz(-2*t*d)@ry(theta)@rz(4*t*rho)@ry(-theta)@rz(-2*t*d)
        np.testing.assert_allclose(C(t)@C(-t).conj().T,expected,atol=1e-14)

    def test_parallel_probe_reference_accounting_and_saved_checker(self):
        model=m.build_model(*fixture()); methods=['P2','E2','E4','D2','D4','R2','R4']; times=[.2,.4]; steps=[1,2,4]
        references={t:expm(-1j*t*model['H']) for t in times}
        registered=json.loads((ROOT/gate.CONTRACT).read_text())
        contract=copy.deepcopy(registered); contract['methods']=methods
        contract['target'].update(spin_orbitals=4,particles=2)
        contract['grid'].update(times_atomic=times,steps=steps)
        contract['limits'].update(full_Fock_dimension=16,nu_input_columns=6)
        with tempfile.TemporaryDirectory() as output:
            path=Path(output)
            jobs=[(model,name,times,steps,references,2,output,{}) for name in methods]
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
                workers=list(pool.map(pipeline.worker_probe,jobs))
            reservation=pipeline.reserved_counts(model,methods,times,steps)
            actual={key:sum(w['counts'].get(key,0) for w in workers) for key in reservation}
            self.assertEqual(reservation,actual)
            probes=[p for w in workers for p in w['probes']]
            np.savez(path/'input.npz',h=model['h'],eri=model['eri'],constant=.13)
            np.savez(path/'derived.npz',H=model['H'])
            pipeline.write_json(path/'probes.json',probes)
            pipeline.write_json(path/'raw_seal.json',{'status':'SEALED_BEFORE_SELECTION','files':{p.name:pipeline.file_sha(p) for p in path.iterdir()}})
            self.assertEqual(c.check_raw(path,contract)['probes'],42)
            # Serial/dense reference agrees; independent scheduler, same full outputs.
            cols=[i for i in range(16) if i.bit_count()==2]
            for row in probes:
                expected=(np.linalg.matrix_power(independent_seed(row['method'],row['T']/row['r'],model),row['r'])-references[row['T']])[:,cols]
                with np.load(path/row['file']) as saved: np.testing.assert_allclose(saved['error'],expected,atol=5e-12)
            result=cost.evaluate_resources(model,probes,contract,0.)
            self.assertEqual(len(result['rows']),7*2*3*3*3)
            ambiguous=copy.deepcopy(probes); ambiguous[0]['error']=.0004
            undecided=cost.evaluate_resources(model,ambiguous,contract,0.)
            self.assertEqual(undecided['status'],'UNDETERMINED_BOUNDARY_OR_COST')
            self.assertTrue(undecided['numerical_ambiguities'])
            file=path/probes[0]['file']; file.write_bytes(file.read_bytes()+b'X')
            with self.assertRaises(c.EvidenceError): c.check_raw(path,contract)

    def test_eight_mode_mock_engine_end_to_end_no_molecular_input(self):
        contract=copy.deepcopy(json.loads((ROOT/gate.CONTRACT).read_text()))
        contract['id']='SYNTHETIC_OFF_DOMAIN_ONLY'
        contract['grid'].update(times_atomic=[.01,.02],steps=[1,2,4])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            context=gate.AuthorizedContext(ROOT,output,contract,{'type':'SYNTHETIC_OFF_DOMAIN_ONLY'})
            def synthetic_input(context,budget):
                h,eri,constant=fixture(4)
                np.savez(output/'input.npz',h=h,eri=eri,constant=constant)
                pipeline.write_json(output/'input_seal.json',{'status':'SYNTHETIC_ONLY','input_SHA256':pipeline.file_sha(output/'input.npz')})
                return h,eri,constant
            with patch.object(pipeline,'generate_input',side_effect=synthetic_input):
                result=pipeline.run_registered(context)
            self.assertTrue(result['mandatory_stop'])
            self.assertEqual(result['counts']['reference_matrices'],4)
            self.assertEqual(result['counts']['PF_error_queries'],42)
            self.assertEqual(result['counts']['norm_SVD_including_checks'],43)
            self.assertNotIn('molecular_inputs',result['counts'])
            self.assertEqual(json.loads((output/'independent_audit.json').read_text())['status'],'PASS')
            self.assertLess(pipeline.directory_bytes(output),2**26)


class LaunchBarrierTests(unittest.TestCase):
    def test_no_approval_and_draft_rejected_before_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(pipeline,'generate_input',side_effect=AssertionError('engine forbidden')):
                with self.assertRaises(gate.LaunchError): gate.authorize(root,None)
                draft=root/'draft.json'; draft.write_text(json.dumps({'type':'DRAFT_NOT_AUTHORIZED'}))
                with self.assertRaises(gate.LaunchError): gate.authorize(root,draft)
            with self.assertRaises(gate.LaunchError): pipeline.generate_input(None,m.Budget())
            self.assertEqual(set(p.name for p in root.iterdir()),{'draft.json'})

    def test_source_hash_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'x').write_text('fixed')
            manifest={'files':{'x':gate.digest(root/'x')}}
            self.assertTrue(gate.verify_sources(root,manifest))
            (root/'x').write_text('changed')
            with self.assertRaises(gate.LaunchError): gate.verify_sources(root,manifest)
            with self.assertRaises(gate.LaunchError): gate.verify_sources(root,{'files':{'../escape':'x'}})

    def test_scope_retry_or_path_override_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            base={'type':'USER_EXPLICIT_SCIENCE_APPROVAL','user_approval_reference':'synthetic rejection test',
                  'scope_id':'QC_MIXED_ACCESS_FEASIBILITY_V1_20261013','output':gate.RUN_OUTPUT,'attempts':1,'retries':0}
            for changes in ({'scope_id':'H2'},{'output':'other'},{'attempts':2},{'retries':1}):
                file=root/'bad.json'; file.write_text(json.dumps({**base,**changes}))
                with self.assertRaises(gate.LaunchError): gate.authorize(root,file)
            self.assertFalse((root/gate.RUN_OUTPUT).exists())

    def test_bound_approval_hashes_and_single_attempt_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder=root/gate.ARTIFACT; folder.mkdir(parents=True)
            (root/'fixed_source').write_bytes(b'fixed')
            pipeline.write_json(folder/'source_manifest.json',{'files':{'fixed_source':gate.digest(root/'fixed_source')}})
            (folder/'requirements.lock').write_text('pinned')
            contract=root/gate.CONTRACT; contract.parent.mkdir(parents=True)
            contract.write_text('{}')
            report={'contract_SHA256':gate.digest(contract),'source_manifest_SHA256':gate.digest(folder/'source_manifest.json')}
            approval={'type':'USER_EXPLICIT_SCIENCE_APPROVAL','user_approval_reference':'non-molecular gate unit test',
                'scope_id':'QC_MIXED_ACCESS_FEASIBILITY_V1_20261013','output':gate.RUN_OUTPUT,'attempts':1,'retries':0,
                **report,'dependency_lock_SHA256':gate.digest(folder/'requirements.lock'),'source_payload_commit':'1'*40}
            file=root/'test_approval.json'
            def pf(root): return {**report,'attempt_already_exists':(root/gate.RUN_OUTPUT).exists()}
            def git(cmd,**kwargs):
                return SimpleNamespace(stdout='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git\n' if cmd[1]=='remote' else b'fixed')
            with patch.object(gate,'preflight',side_effect=pf),patch.object(gate.subprocess,'run',side_effect=git):
                file.write_text(json.dumps({**approval,'contract_SHA256':'0'*64}))
                with self.assertRaises(gate.LaunchError): gate.authorize(root,file)
                self.assertFalse((root/gate.RUN_OUTPUT).exists())
                file.write_text(json.dumps(approval)); context=gate.authorize(root,file)
                self.assertTrue((context.output/'launch_receipt.json').exists())
                with self.assertRaises(gate.LaunchError): gate.authorize(root,file)


if __name__=='__main__': unittest.main()
