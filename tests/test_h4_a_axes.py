"""Artificial saved-body/wrapper, authority, recovery and paired-cost checks."""
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_h4_a_gate import A, fixture
from d1_private import h4_a_axes as D
from d1_private import h4_d2_pauli_control as P
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Operator
import numpy as np
C=D.C


def active():
    return dict(kind=D.KIND,fixed=D.fixed(),execution_authorized=True,grant_valid=True,dispatch_enabled=True,
        new_targets=list(D.CASES),axis='sine',call_limits=D.LIMITS,per_target_compile_max=1,
        mandatory_STOP=True,automatic_retry=False,automatic_resume=False,automatic_next_stage=False,
        technical_recovery_authorized=True,recovery_requires_new_SOURCE_seal_grant=True,
        job_environment=D.JOB_ENV,worker_count=3,science_SOURCE='TOY',approval_sha256='TOY')


def body_fixture():
    m,j=fixture()
    return P.central_control_body(D.G.body_only(m,j).body).body


def synthetic_prices(root):
    base=root/'axes';ns=base/'run01';ns.mkdir(parents=True);oldh=root/'oldh';oldh.mkdir()
    C.save(oldh/'statistics.json',dict(central_cosine_RZ=dict(mean_exact='2001/2')))
    # The runtime expects H.BASE/run01/statistics.json.
    (oldh/'run01').mkdir();C.save(oldh/'run01/statistics.json',dict(central_cosine_RZ=dict(mean_exact='2001/2')))
    old=root/'old'
    for case in ('df_native_PF','df_native_PR'):
        d=old/case;d.mkdir(parents=True);C.save(d/'result.json',dict(circuit_cost=dict(axes={a:dict(mean=20000) for a in ('cosine','sine')})))
    rows=[]
    for method,index in [('PF',0)]+[('PR',i) for i in range(256)]:
        key='PF' if method=='PF' else f'PR{index:04d}'
        cf=root/'cos'/key;cf.mkdir(parents=True)
        cosRZ=15028 if method=='PF' else 1000+index%2
        cos=dict(RZ=cosRZ,CX=10,depth=20,rz_depth=7,cx_depth=11,SOURCE='ARTIFICIAL',compiler=D.H.fixed()['compiler'])
        C.save(cf/'metrics.json',cos)
        join=root/'joins'/(key+'.json');join.parent.mkdir(exist_ok=True)
        C.save(join,dict(compiled_RZ=dict(cosine=10000,sine=10001),events=[dict(taylor_order=2 if index%4==0 else 0)]))
        row=dict(key=key,method=method,trajectory_index=index,trajectory_id='TOY_'+key,event_record_hash='E'+key,
            trajectory_record_hash='T'+key,cosine_folder=str(cf.relative_to(root)),price_join=str(join.relative_to(root)))
        sf=root/'reused' if key=='PR0001' else ns/'trajectories'/key;sf.mkdir(parents=True)
        if key=='PR0001':row['reused_sine_folder']=str(sf.relative_to(root))
        else:
            native=ns/'native_tasks'/('compile_'+key);native.mkdir(parents=True)
            C.save(native/'native_exit.json',dict(exit_code=0,reaped=True,native_wait=True))
        (sf/'input.qpy').write_bytes(b'ARTIFICIAL_NOT_QPY');(sf/'output.qpy').write_bytes(b'ARTIFICIAL_NOT_QPY')
        sine=dict(cos,RZ=15030 if method=='PF' else cosRZ+index%3-1,axis='sine',trajectory_id=row['trajectory_id'],target=row,
            input_qpy_sha256=C.sha(sf/'input.qpy'),output_qpy_sha256=C.sha(sf/'output.qpy'))
        C.save(sf/'metrics.json',sine);rows.append(row)
    C.save(base/'case_inputs.json',dict(rows=rows))
    return base,ns,oldh,old


class AxesTests(unittest.TestCase):
    def test_a_unauthorized_has_no_quantum_import(self):
        code="""import sys,types,tempfile
from pathlib import Path
p=types.ModuleType('d1_private');p.__path__=[str(Path.cwd()/'src/trotterlib')];sys.modules[p.__name__]=p
from d1_private import h4_a_axes as D
with tempfile.TemporaryDirectory() as t:
 D.BASE=Path(t)
 try:D.authority(consume=True)
 except FileNotFoundError:pass
 else:raise AssertionError('authorized')
assert not any(k=='qiskit' or k.startswith('qiskit.') for k in sys.modules)
"""
        self.assertEqual(subprocess.run([sys.executable,'-B','-c',code],cwd=D.ROOT).returncode,0)

    def test_b_scope_only_256_new_sine_targets(self):
        D.scope(active());self.assertEqual(len(D.CASES),256);self.assertNotIn('PR0001',D.CASES)
        for key,value in [('axis','cosine'),('new_targets',list(D.CASES)+['PR0001']),('new_targets',list(D.CASES)[1:]),
            ('call_limits',{}),('automatic_retry',True),('recovery_requires_new_SOURCE_seal_grant',False)]:
            c=active();c[key]=value
            with self.assertRaises(ValueError):D.scope(c)

    def test_c_extract_and_direct_body_preserve_operators_phase_and_wires(self):
        body=body_fixture();body.global_phase=.173
        cos=A.api().hadamard_wrapper(body,'cosine',already_controlled=True)
        for direct in (None,body):
            got,proof=D.controlled_from_cosine(cos,direct)
            self.assertEqual(D.circuit_signature(got),D.circuit_signature(body))
            self.assertEqual(proof['global_phase'],.173)
            self.assertLess(np.max(np.abs(Operator(got).data-Operator(body).data)),2e-13)

    def test_d_QPY_roundtrip_sdg_position_both_control_branches(self):
        body=body_fixture();body.global_phase=.231
        cos=A.api().hadamard_wrapper(body,'cosine',already_controlled=True)
        stream=io.BytesIO();qpy.dump(cos,stream);stream.seek(0)
        got,proof=D.controlled_from_cosine(qpy.load(stream)[0])
        sine=A.api().hadamard_wrapper(got,'sine',measure=False,already_controlled=True)
        h=np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2),np.eye(4));sdg=np.kron(np.diag([1,-1j]),np.eye(4))
        self.assertLess(np.max(np.abs(Operator(sine).data-h@sdg@Operator(body).data@h)),3e-13)
        self.assertEqual(sine.data[-2].operation.name,'sdg')
        # The extracted control=0/1 blocks are the saved body's unchanged blocks.
        for sl in (slice(0,4),slice(4,8)):
            self.assertLess(np.max(np.abs(Operator(got).data[sl,sl]-Operator(body).data[sl,sl])),2e-13)

    def test_e_bad_wrapper_measurement_barrier_or_body_phase_rejected(self):
        body=body_fixture();cos=A.api().hadamard_wrapper(body,'cosine',already_controlled=True)
        changed=body.copy();changed.global_phase=.3
        with self.assertRaisesRegex(ValueError,'CORRESPONDENCE'):D.controlled_from_cosine(cos,changed)
        bad=QuantumCircuit(3,1);bad.h(1);bad.cx(2,0);bad.h(2);bad.measure(2,0)
        with self.assertRaisesRegex(ValueError,'WRAPPER'):D.controlled_from_cosine(bad)
        bad=QuantumCircuit(3,1);bad.h(2);bad.barrier();bad.h(2);bad.measure(2,0)
        with self.assertRaisesRegex(ValueError,'DOMAIN'):D.controlled_from_cosine(bad)

    def test_f_saved_QPY_pipeline_never_replays_body_or_control(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base=root/'axes';ns=base/'run01';(ns/'trajectories').mkdir(parents=True)
            reserve=ns/'native_tasks/compile_PF';reserve.mkdir(parents=True)
            C.save(reserve/'reservation.json',dict(kind='compile',case='PF',grant_id='TOY'))
            old=root/'old';old.mkdir();body=QuantumCircuit(9);body.cx(8,0);body.crz(.3,8,1);body.p(.27,8);body.global_phase=.17
            cos=A.api().hadamard_wrapper(body,'cosine',already_controlled=True)
            with (old/'input.qpy').open('wb') as f:qpy.dump(cos,f)
            row=dict(key='PF',method='PF',trajectory_index=0,trajectory_id='ARTIFICIAL',cosine_folder='old',body_qpy=None,
                cosine_input_fingerprint=A.api().canonical_qiskit_circuit_fingerprint(cos),input_files={'old/input.qpy':C.sha(old/'input.qpy')})
            C.save(base/'case_inputs.json',dict(rows=[row]));seal=dict(SOURCE='TOY',runtime_manifest_sha256='ARTIFICIAL',input_files={'axes/case_inputs.json':C.sha(base/'case_inputs.json')})
            def compiler(folder):
                inp=C.read(folder/'input.json');(folder/'output.qpy').write_bytes(b'ARTIFICIAL_COMPILER_STUB')
                return dict(input_fingerprint=inp['input_fingerprint'])
            with (patch.object(D,'ROOT',root),patch.object(D,'BASE',base),patch.object(D,'authority',return_value=(active(),seal,dict(grant_id='TOY'))),
                  patch.object(D.G,'body_only',side_effect=AssertionError('body replay')),patch.object(P,'central_control_body',side_effect=AssertionError('new control')),
                  patch.object(A,'build',side_effect=AssertionError('generic build')),patch.object(A,'compile_saved',side_effect=compiler) as compile_call):
                self.assertEqual(D.task_run('PF'),0);self.assertEqual(compile_call.call_count,1)
            used=C.read(ns/'call_ledger.json')['used'];self.assertEqual(used['saved_controlled_body_load'],1)
            self.assertEqual(used['sine_wrapper_build'],1);self.assertEqual(used['body_replay'],0);self.assertEqual(used['central_control_build'],0)

    def test_g_duplicate_claim_blocks_quantum_work(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)):
            ns=Path(td)/'run01';(ns/'trajectories/PF').mkdir(parents=True);r=ns/'native_tasks/compile_PF';r.mkdir(parents=True)
            C.save(r/'reservation.json',dict(kind='compile',case='PF',grant_id='TOY'))
            with patch.object(D,'authority',return_value=(active(),{},dict(grant_id='TOY'))),patch.object(A,'api') as api:
                with self.assertRaises(FileExistsError):D.task_run('PF')
                with self.assertRaisesRegex(ValueError,'SCOPE'):D.task_run('PR0001')
                api.assert_not_called()

    def test_h_rebound_grant_requires_new_namespace_and_consumes_once(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)),patch.object(D,'verify_seal'):
            base=Path(td);ns=base/'run01';ns.mkdir();C.save(base/'seal.json',dict(SOURCE='TOY',approval_sha256='TOY'))
            c=active();c['seal_sha256']=C.sha(base/'seal.json');C.save(base/'active_contract.json',c)
            g=dict(kind='H4_A_AXES_ONE_SHOT',SOURCE='TOY',namespace=str(ns),call_limits=D.LIMITS,contract_sha256=C.sha(base/'active_contract.json'),
                seal_sha256=c['seal_sha256'],approval_sha256='TOY',one_shot=True,automatic_retry=False)
            g['grant_id']=C.digest(g);C.save(ns/'grant.json',g);D.authority(consume=True)
            with self.assertRaisesRegex(ValueError,'CONSUMED'):D.authority(consume=True)
            C.save(ns/'ENGINE_STOP.json',dict(status='ARTIFICIAL_STOP'))
            with self.assertRaisesRegex(ValueError,'STOPPED'):D.authority()

    def test_i_reused_native_dispatch_reserves_before_start(self):
        with tempfile.TemporaryDirectory() as td:
            ns=Path(td);C.save(ns/'grant.json',dict(grant_id='TOY'));commands=[]
            class Child:
                pid=os.getpid()
                def __init__(self,command,**kw):
                    commands.append(command);assert C.read(ns/'native_tasks'/('compile_'+command[-1])/'reservation.json')['grant_id']=='TOY'
                def poll(self):return 0
                def wait(self):return 0
            with patch.object(D.D2,'subprocess',SimpleNamespace(Popen=Child)):D.dispatch(ns,('PF','PR0000'))
            self.assertEqual([x[-1] for x in commands],['PF','PR0000']);self.assertEqual(C.read(ns/'call_ledger.json')['used']['cost_transpile'],2)

    def test_j_failure_stops_future_waves_and_statistics(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)):
            ns=Path(td)/'run01';ns.mkdir()
            with patch.object(D,'authority',return_value=(active(),{},{})),patch.object(D,'dispatch',side_effect=ValueError('TOY')) as dispatcher,patch.object(D,'paired_statistics') as stats:
                with self.assertRaises(ValueError):D.worker()
                self.assertEqual(dispatcher.call_count,1);stats.assert_not_called()
            self.assertTrue(C.read(ns/'ENGINE_STOP.json')['technical_recovery_requires_diagnosis'])

    def test_k_science_budget_nonrefundable_zero_cosine_body_and_NG(self):
        with tempfile.TemporaryDirectory() as td:
            ns=Path(td);used=dict.fromkeys(D.LIMITS,0);used['cost_transpile']=255
            C.save(ns/'call_ledger.json',dict(limits=D.LIMITS,used=used,events=[]));budget=C.CallBudget(ns,D.LIMITS);budget.take('cost_transpile')
            for key in ('cost_transpile','body_replay','central_control_build','cosine_compile','sampling','N_G'):
                with self.assertRaises(ValueError):budget.take(key)

    def test_l_recovery_preserves_successes_calls_and_requires_diagnosis(self):
        diagnosis=dict(class_='TECHNICAL',cause='Known artificial I/O defect',scientific_conditions_changed=False,unresolved_semantic_mismatch=False)
        diagnosis['class']=diagnosis.pop('class_')
        result=D.recovery_plan(['PF','PR0000'],['PF'],diagnosis)
        self.assertEqual(result['targets'],['PR0000']);self.assertFalse(result['execution_enabled'])
        self.assertTrue(result['previous_consumed_calls_preserved']);self.assertTrue(result['new_SOURCE_seal_contract_namespace_grant_required'])
        for altered in [dict(diagnosis,cause=''),dict(diagnosis,unresolved_semantic_mismatch=True),dict(diagnosis,scientific_conditions_changed=True)]:
            with self.assertRaises(ValueError):D.recovery_plan(['PF'],[],altered)
        with self.assertRaises(ValueError):D.recovery_plan(['PR0001'],[],diagnosis)

    def test_m_artificial_paired_statistics_covariance_and_mixed_axes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base,ns,oldh,old=synthetic_prices(root)
            with patch.object(D,'ROOT',root),patch.object(D,'BASE',base),patch.object(D.H,'BASE',oldh),patch.object(D.H,'OLD_RUN',old),patch.object(D,'plot_paired'):
                D.paired_statistics(ns)
            result=C.read(ns/'statistics.json');pr=result['PR']
            self.assertEqual(pr['axes']['cosine']['mean_exact'],'2001/2')
            self.assertEqual(pr['paired_RZ_difference']['mean_exact'],'-1/256')
            self.assertEqual(pr['RZ_equal_count'],85);self.assertEqual(pr['RZ_unequal_count'],171)
            x=[1000+i%2 for i in range(256)];y=[x[i]+i%3-1 for i in range(256)]
            self.assertAlmostEqual(pr['sample_covariance_RZ'],statistics.covariance(x,y),places=12)
            self.assertIsNone(pr['order_groups']['contains_order4']['axes']['sine'])
            self.assertEqual(result['PF']['sine']['RZ'],15030);self.assertFalse(result['new_N_G'])

    def test_n_missing_native_receipt_cannot_complete_prices(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base,ns,oldh,old=synthetic_prices(root)
            C.save(ns/'native_tasks/compile_PF/native_exit.json',dict(exit_code=1,reaped=True,native_wait=True),exclusive=False)
            with patch.object(D,'ROOT',root),patch.object(D,'BASE',base),patch.object(D.H,'BASE',oldh),patch.object(D.H,'OLD_RUN',old),patch.object(D,'plot_paired'):
                with self.assertRaisesRegex(ValueError,'NATIVE'):D.paired_statistics(ns)
            self.assertFalse((ns/'statistics.json').exists())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AxesTests))
    if len(sys.argv)==2:
        import resource
        Path(sys.argv[1]).write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='ARTIFICIAL_ONLY_NO_REAL_H4'),indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
