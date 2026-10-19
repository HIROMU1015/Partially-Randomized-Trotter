"""Artificial contracts, scheduling and 1-system-qubit native toys only."""
import copy
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h46_m5_contract as c
import h46_m5_native as n
import launch_h46_m5_pilot as launch


def fake_reference(lane):return {'status':'PASS'}


def fake_worker(task):
    config=launch.CONFIGS[task['lane']]
    if config.get('simulate_failure'):
        return dict(status='STOP',lane=task['lane'],task=task['id'],reason='ARTIFICIAL_NUMERIC_MISMATCH')
    for axis in ('cosine','sine'):
        path=Path(config['ledger'])/(task['id']+'__'+axis+'.json');path.write_text('{}')
    return dict(status='PASS',lane=task['lane'],task=task['id'])


class PoolTests(unittest.TestCase):
    def run_artificial_pool(self,failed=None):
        with tempfile.TemporaryDirectory() as tmp:
            configs={}
            for lane in c.LANES:
                out=Path(tmp)/(lane+'_out');out.mkdir();ledger=Path(tmp)/(lane+'_ledger');ledger.mkdir()
                configs[lane]=dict(lane=lane,system=lane[:2],SOURCE='a'*40,plan={'scope':c.scope(lane)},
                    run_root=tmp,output=str(out),ledger=str(ledger),RSS_proxy_bytes=1,simulate_failure=lane==failed)
            with patch.object(c,'verify_seals'),patch.object(launch,'observe',return_value={'PSI':'high_observation_only'}):
                return launch.run_pool(configs,{'workers':4},worker=fake_worker,reference=fake_reference)

    def test_actual_spawn_pool_all_lanes_reaped_and_budget(self):
        r=self.run_artificial_pool()
        self.assertTrue(r['all_reaped']);self.assertFalse(r['next_phase_started'])
        self.assertEqual(sum(x['compile_entries'] for x in r['lanes'].values()),60)
        self.assertTrue(all(x['status']=='COMPLETE_MANDATORY_STOP' for x in r['lanes'].values()))

    def test_actual_spawn_lane_failure_does_not_stop_other_lanes(self):
        r=self.run_artificial_pool('H4_P2_DET')
        self.assertEqual(r['lanes']['H4_P2_DET']['status'],'STOP')
        self.assertTrue(all(x['status']=='COMPLETE_MANDATORY_STOP' for k,x in r['lanes'].items() if k!='H4_P2_DET'))
        self.assertTrue(r['all_reaped'])


class Contracts(unittest.TestCase):
    def test_exact_four_lane_budgets(self):
        self.assertEqual(sum(c.scope(x)['compile_axes'] for x in c.LANES),60)
        self.assertEqual(sum(len(c.scope(x)['tasks']) for x in c.LANES),30)
        self.assertEqual(sum(c.scope(x)['reference_jobs'] for x in c.LANES),2)
        self.assertTrue(all(c.scope(x)['phase3_authorized'] is False for x in c.LANES))

    def test_fresh_seed_reproducibility_and_collision_rejection(self):
        a=c.registry('a'*40,{3,7});self.assertEqual(a,c.registry('a'*40,{3,7}))
        values=[v for lane in a.values() for t in lane.values() for v in t.values()]
        self.assertEqual(len(values),120);self.assertEqual(len(values),len(set(values)))
        self.assertNotEqual(a,c.registry('b'*40,{3,7}))
        with self.assertRaisesRegex(ValueError,'COLLISION'):c.registry('a'*40,{values[0]})

    def test_invalid_grant_refused_before_seal_or_import(self):
        for bad in ({},{'authorized':True,'grant':{'valid':False}},
                    {'authorized':False,'grant':{'valid':True,'approval_reference':'x'}}):
            with patch.object(c,'verify_seals',side_effect=AssertionError('must not hash')):
                with self.assertRaisesRegex(ValueError,'EXPLICIT_PILOT'):c.authorize(bad)

    def test_grant_scope_mutations_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(authorized=True,lane=c.LANES[0],system='H4',SOURCE='a'*40,plan_sha256='b'*64,
                output=tmp+'/H4_out',ledger=tmp+'/H4_ledger',marker=tmp+'/H4_marker',
                run_root=tmp,reference_root=tmp+'/H4_reference')
            cfg['grant']=dict(valid=True,approval_reference='user approval',grant_id='unique',lane=cfg['lane'],
                SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],maximum_axes=18,maximum_trajectories=0,
                maximum_reference_jobs=1,retry=False,phase3_authorized=False,
                **{k:cfg[k] for k in ('output','ledger','marker')})
            with patch.object(c,'verify_seals') as verify:
                c.authorize(cfg);verify.assert_called_once()
                for key,value in [('maximum_axes',20),('maximum_reference_jobs',2),('SOURCE','c'*40),
                                  ('retry',True),('phase3_authorized',True),('lane',c.LANES[1])]:
                    bad=copy.deepcopy(cfg);bad['grant'][key]=value
                    with self.assertRaises(ValueError,msg=key):c.authorize(bad)
                bad=copy.deepcopy(cfg);bad['reference_root']=tmp+'/other'
                with self.assertRaisesRegex(ValueError,'REFERENCE_PATH'):c.authorize(bad)

    def test_compile_ledger_exclusive_and_budget_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(ledger=tmp,plan={'scope':c.scope(c.LANES[0])},SOURCE='a',plan_sha256='b')
            task=cfg['plan']['scope']['tasks'][0]
            p=c.reserve(cfg,task,'cosine','f');self.assertTrue(p.exists())
            with self.assertRaises(FileExistsError):c.reserve(cfg,task,'cosine','f')
            with self.assertRaises(ValueError):c.reserve(cfg,task,'extra','f')
            with self.assertRaises(ValueError):c.reserve(cfg,{**task,'sample':99},'sine','f')

    def test_ready_representatives_and_failure_isolation(self):
        groups=[c.scope(x) for x in c.LANES]
        tasks=[dict(**t,lane=g['lane']) for i in range(9) for g in groups
               for t in g['tasks'][i:i+1]]
        first=launch.decisions(tasks,[],[],[],4)
        self.assertEqual({t['lane'] for t in first},set(c.LANES))
        self.assertTrue(all(t['sample']==0 for t in first))
        pr=next(t for t in tasks if t['lane']=='H4_P2_PR' and t['sample']==1)
        self.assertNotIn(pr,launch.decisions(tasks,[],[],[],100))
        self.assertIn(pr,launch.decisions(tasks,[pr['group']['id']+'__0'],[],[],100))
        self.assertEqual({t['lane'] for t in launch.decisions(tasks,[],['H4_P2_DET'],[],100)},set(c.LANES)-{'H4_P2_DET'})

    def test_cli_draft_rejection_no_science_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';config.write_text(json.dumps({'authorized':False,'grant':{'valid':False}}))
            bundle=root/'bundle.json';bundle.write_text(json.dumps({'configs':{x:str(config) for x in c.LANES}}))
            before=set(root.iterdir())
            r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/launch_h46_m5_pilot.py'),
                '--execute','--bundle',str(bundle)],capture_output=True,text=True)
            self.assertEqual(r.returncode,2);self.assertEqual(set(root.iterdir()),before)
            self.assertIn('REFUSED',r.stdout)

    def test_scheduler_no_replica_after_own_failure(self):
        tasks=[dict(**t,lane='H6_P2_PR') for t in c.scope('H6_P2_PR')['tasks']]
        done=[t['id'] for t in tasks if t['sample']==0]
        self.assertTrue(launch.decisions(tasks,done,[],[],10))
        self.assertEqual(launch.decisions(tasks,done,['H6_P2_PR'],[],10),[])


class FakeCircuit:
    def __init__(self,n):self.n=n;self.data=[]
    def compose(self,other,inplace):self.data.extend(other.data)
    def p(self,angle,qubit):self.data.append(('phase',angle,qubit))


class Composition(unittest.TestCase):
    def test_negative_signed_pieces_scalar_once_all_formulas(self):
        seen=[]
        def build(blocks,**kw):
            seen.append(kw);qc=FakeCircuit(2);qc.data=[('piece',kw['T'])]
            return NS(circuit=qc,instruction_upper_bound=4)
        api=NS(block_instruction_bound=lambda b,m:2,build_deterministic_native=build,NativeEvolution=lambda *x:NS(args=x,circuit=x[0]))
        prep=NS(num_system_qubits=1,deterministic_blocks=[object()])
        for label in c.core.FORMULAS:
            seen.clear();g=c.core.candidate('H4','B0I',label,3,2)
            evo=n.compose(g,prep,[],.7,api,FakeCircuit,lambda q:'fingerprint')
            self.assertEqual(len(seen),2*len(c.core.weights(label)))
            self.assertEqual([p['T'] for p in seen],[s['signed_time'] for s in c.core.schedule(label,2,'PER_PIECE',1)])
            self.assertTrue(all(p['scalar']==0 and p['control_policy']=='symmetric_directional' for p in seen))
            self.assertEqual(evo.circuit.data[-1],('phase',-.7*.8,1))
            self.assertEqual(sum(x[0]=='phase' for x in evo.circuit.data),1)

    def test_pr_schedule_count_and_identity_rejected(self):
        g=c.core.pilot('H4')['PR'][-1];prep=NS(num_system_qubits=1)
        with self.assertRaisesRegex(ValueError,'REQUEST_SCHEDULE_COUNT'):n.compose(g,prep,[],.1,None,FakeCircuit,None)


@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'requires explicitly supplied frozen test provider')
class RealNativeToy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))
        class NoGPU:
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split('.')[0] in ('cupy','cupyx','cuquantum'):raise ImportError('CPU artificial test')
        sys.meta_path.insert(0,NoGPU())

    def test_actual_native_directional_scalar_and_compiled_vector(self):
        import numpy as np
        from qiskit import QuantumCircuit,transpile
        from qiskit.circuit.library import HGate
        from qiskit.quantum_info import Operator
        from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec as OB,DFDeterministicFragmentSpec as FB
        from trottertracks.resource_applicability import ax2b_native_df_v5 as api
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fp
        b0=OB('a','one_body',None,None,'a','h',(),(.6,),1,0,())
        b1=FB('b','df_fragment',0,0,'b','h',(),(.7,),.4,1,1,((HGate(),(0,)),))
        prep=NS(num_system_qubits=1,deterministic_blocks=(b0,b1))
        A=np.diag([0,.6]);H=np.array([[1,1],[1,-1]])/math.sqrt(2);B=H@np.diag([0,.4*.7**2])@H
        def exp(M,t):
            w,v=np.linalg.eigh(M);return(v*np.exp(-1j*t*w))@v.conj().T
        for label in c.core.FORMULAS:
            g=c.core.candidate('H4','B0I',label,3,2);e=n.compose(g,prep,[],.13,api,QuantumCircuit,fp)
            target=np.eye(2,dtype=complex)
            for s in c.core.schedule(label,2,'PER_PIECE',1):
                t=s['signed_time'];target=exp(A,t/2)@exp(B,t)@exp(A,t/2)@target
            target*=np.exp(-.13j*.8)
            expected=np.zeros((4,4),complex);expected[:2,:2]=np.eye(2);expected[2:,2:]=target
            self.assertLess(np.linalg.norm(Operator(e.circuit).data-expected),1e-12)
            for axis in ('cosine','sine'):
                wrapper=api.build_native_hadamard_wrapper(e,axis=axis,include_measurement=False,max_instructions=len(e.circuit.data)+4)
                compiled=transpile(wrapper,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
                self.assertLess(np.linalg.norm(Operator(compiled).data-Operator(wrapper).data),1e-11)

    def toy(self,fragments=3,prefix=1):
        import numpy as np
        from h46_new2_native import prepare,OccupationOracle
        from trotterlib.df_hamiltonian import DFHamiltonian
        rng=np.random.default_rng(10817)
        def matrix():
            x=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));return .1*(x+x.conj().T)
        ham=DFHamiltonian(.37,matrix(),np.array([.2,-.3,.1,-.14][:fragments]),tuple(matrix() for _ in range(fragments)),{})
        state=np.zeros(16,complex);state[3]=1/np.sqrt(2);state[12]=1j/np.sqrt(2)
        return ham,prepare(ham,prefix),OccupationOracle(ham,state,prefix=fragments)

    def test_actual_PR_m5_signed_sampler_directional_oracle_and_pair_compile(self):
        import numpy as np
        from qiskit import QuantumCircuit,transpile
        from trotterlib.df_partial_s2 import make_df_partial_s2_step_request
        from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
        from trotterlib.df_trotter.circuit import simulate_statevector
        from trottertracks.resource_applicability import ax2b_native_df_v5 as api
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fp
        ham,prep,oracle=self.toy();g=c.core.candidate('H4','PR','4th(m5_best)',1,2,4,'PER_PIECE',2)
        steps=c.core.schedule(g['formula'],2,g['policy'],2)
        seeds={f"{s['outer']}:{s['piece']}":1120+i for i,s in enumerate(steps)}
        kwargs=dict(factory=make_df_partial_s2_step_request,tolerance=_explicit_cutoff_tolerance)
        requests=n.make_requests(g,prep,seeds,prep.exact_rte_lambda_r,**kwargs)
        again=n.make_requests(g,prep,seeds,prep.exact_rte_lambda_r,**kwargs)
        events=lambda rs:[[e.to_dict() for e in r.rte_occurrence.events] for r in rs]
        self.assertEqual(events(requests),events(again));self.assertEqual(sum(len(r.rte_occurrence.events) for r in requests),44)
        self.assertEqual(sum(r.step_time<0 for r in requests),2)
        qstate=np.zeros(16,complex);qstate[oracle.qindices]=oracle.initial
        low=np.r_[qstate,np.zeros(16)];high=np.r_[np.zeros(16),qstate]
        scalar=ham.constant+prep.extracted_identity_coefficient
        evo=n.compose(g,prep,requests,scalar,api,QuantumCircuit,fp)
        self.assertLess(np.linalg.norm(simulate_statevector(evo.circuit,low)-low),1e-11)
        self.assertLess(np.linalg.norm(simulate_statevector(evo.circuit,high)[16:]-oracle.trajectory(requests)),1e-11)
        for axis in ('cosine','sine'):
            wrapper=api.build_native_hadamard_wrapper(evo,axis=axis,include_measurement=False,max_instructions=len(evo.circuit.data)+4)
            compiled=transpile(wrapper,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
            self.assertLess(np.linalg.norm(simulate_statevector(compiled,low)-simulate_statevector(wrapper,low)),1e-10)
        bad=list(requests)
        bad[0]=NS(preparation=prep,step_time=-bad[0].step_time,controlled=True,ancilla_qubit=4)
        with self.assertRaisesRegex(ValueError,'SIGNED_REQUEST_IDENTITY'):n.compose(g,prep,bad,scalar,api,QuantumCircuit,fp)

    def test_finite_signal_normalization_scalar_and_no_shot_adoption(self):
        import numpy as np
        from h46_new2_core import spectrum
        ham,prep,oracle=self.toy();matrices=[(s[1]*s[0])@s[2] for s in oracle.specs]
        scalar=prep.extracted_identity_coefficient;spec=spectrum(sum(matrices))
        reference=spec[1]@(np.exp(-.8j*spec[0])*(spec[2]@oracle.initial))*np.exp(-.8j*ham.constant)
        ctx=NS(np=np,oracle=oracle,reference=reference,scalar=ham.constant,tail_identity=scalar,
            lambda_R=prep.exact_rte_lambda_r,tail_spec=spectrum(sum(matrices[2:])-scalar*np.eye(len(oracle.initial))))
        for method in ('PR','B0','B0I','FULL'):
            g=c.core.candidate('H4',method,'4th(m5_best)',3 if method=='FULL' else 1,2,
                4 if method=='PR' else None,'PER_PIECE' if method=='PR' else None,2 if method=='PR' else None)
            row=n.signal(ctx,g)
            self.assertIsNone(row['allowance_evidence']['shot_allowance'])
            self.assertLess(row['allowance_evidence']['raw_corrected_vector_closure'],1e-11)
            self.assertTrue(math.isfinite(row['log_B']))
            if method!='PR':self.assertEqual(row['log_B'],0.)

    def test_end_to_end_artificial_PR_task_actual_cost_ledger_and_controls(self):
        import numpy as np
        import qiskit
        from functools import partial
        import threading
        from h46_new2_native import prepare
        from h46_new2_core import spectrum
        import trotterlib.rte_compiled_cost as costs
        ham,cut,oracle=self.toy(4,3);full=prepare(ham,4)
        matrices=[(s[1]*s[0])@s[2] for s in oracle.specs];sp=spectrum(sum(matrices))
        reference=sp[1]@(np.exp(-.8j*sp[0])*(sp[2]@oracle.initial))*np.exp(-.8j*ham.constant)
        qstate=np.zeros(16,complex);qstate[oracle.qindices]=oracle.initial
        scalar=cut.extracted_identity_coefficient
        ctx=NS(np=np,ham=ham,oracle=oracle,qstate=qstate,cut=cut,fullprep=full,reference=reference,
            scalar=ham.constant,tail_identity=scalar,lambda_R=cut.exact_rte_lambda_r,
            tail_spec=spectrum(sum(matrices[4:])-scalar*np.eye(len(oracle.initial))))
        lane='H4_P2_PR';task=next(t for t in c.scope(lane)['tasks'] if t['sample']==0 and t['group']['formula']=='4th(m5_best)')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'output').mkdir();(root/'ledger').mkdir();(root/'ref').mkdir()
            reference_file=root/'ref/reference.npz';np.savez(reference_file,reference=reference,qstate=qstate)
            c.save(root/'ref/receipt.json',dict(status='PASS',SOURCE='a'*40,snapshot='toy',
                full_preparation=full.preparation_hash,cut_preparation=cut.preparation_hash,
                reference_sha256=c.sha(reference_file)))
            c.save(root/'registry.json',dict(lanes=c.registry('a'*40,set())))
            cfg=dict(lane=lane,SOURCE='a'*40,system='H4',plan={'scope':c.scope(lane)},plan_sha256='b'*64,
                output=str(root/'output'),ledger=str(root/'ledger'),reference_root=str(root/'ref'),
                registry_path=str(root/'registry.json'),binding={'snapshot':'toy'})
            with patch.object(c,'authorize'),patch.object(c,'consumed'),patch.object(c,'verify_seals'),\
                 patch.object(n,'context',return_value=ctx),patch.object(n,'install'),\
                 patch.object(costs,'transpile',partial(qiskit.transpile,num_processes=1)):
                result=n.run_task(cfg,task,threading.Event())
            self.assertEqual(result['status'],'PASS',result.get('reason'))
            self.assertEqual(len(list((root/'ledger').glob('*.json'))),2)
            self.assertEqual(len(result['checks']),8)
            self.assertFalse(result['ranking']);self.assertFalse(result['shot_accounting'])
            self.assertEqual(len({r['event_digest'] for r in result['records']}),1)


if __name__=='__main__':unittest.main()
