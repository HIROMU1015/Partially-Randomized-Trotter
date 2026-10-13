"""Off-domain D1 semantic and launch tests. Registered provider never generated."""
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import resource
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from trottertracks.algorithm_codesign.rte_development.accounting import (
    error_budget,step_bounds,whole_budget,validate_events,whole_cost,affine_relation,validate_contract)
from trottertracks.algorithm_codesign.rte_development.numeric import finite_target,event_operator,controlled,strict_error
from trottertracks.algorithm_codesign.rte_development.evidence import (
    EvidenceWriter,verify_chunks,digest,atomic_json,safe_relative)
from trottertracks.algorithm_codesign.rte_development.launch import (
    validate_binding,verify_job,consume_marker,Guard,verify_launch,check_protected)
from trottertracks.algorithm_codesign.rte_development.pipeline import (
    freeze_inventory,inventory_key,validate_worker_result,acquire)
from trottertracks.algorithm_codesign.g7_generator import make_generator
from trottertracks.algorithm_codesign.g7_reference import reference_events
from trottertracks.algorithm_codesign.g9_p5 import P5Closed
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,strict_guard
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle

ROOT=Path(__file__).resolve().parents[3]
CP=ROOT/'artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json'
CONFIG=json.loads(CP.read_text())
P=(F(1,3),F(2,3)); X=F(1,7)
OPS=[np.array([[0,1],[1,0]],complex),np.diag([1.,-1.]).astype(complex)]


class Semantics(unittest.TestCase):
    def check_arm(self,arm):
        configure(100)
        g=P5Closed(P,X) if arm=='closed_P5_full' else make_generator(P,X,5,arm)
        es=list(g.reference_events() if arm=='closed_P5_full' else reference_events(g))
        b=step_bounds(generator=g,c=CONFIG);law=validate_events(es,b)
        mean=sum(float(e['coefficient'])*event_operator(e,OPS) for e in es)
        self.assertLess(strict_error(mean,finite_target(OPS,P,X)),1e-11)
        self.assertLessEqual(law['M2'],b['M2_upper'])
        self.assertLessEqual(law['range'],b['range_upper'])
        self.assertLess(strict_error(np.linalg.matrix_power(mean,3),
                                    np.linalg.matrix_power(finite_target(OPS,P,X),3)),1e-11)
        if arm=='full_return':self.assertGreater(law['zero_mass'],0)
    def test_ordinary_mean(self):self.check_arm('ordinary')
    def test_partial_mean(self):self.check_arm('partial_return_tail')
    def test_closed_p3_mean(self):self.check_arm('closed_P3_tail')
    def test_local_full_mean(self):self.check_arm('full_return')
    def test_closed_p5_mean(self):self.check_arm('closed_P5_full')
    def test_control_relative_phase(self):
        u=OPS[0];self.assertAlmostEqual(strict_error(controlled(u),controlled(-u)),2)
    def test_noncommuting_order(self):
        self.assertGreater(strict_error(OPS[0]@OPS[1],OPS[1]@OPS[0]),1)
    def test_signed_rotation_adjoint(self):
        e=dict(word=(),child=0,ratio=F(1,9),phase_i_power=0)
        a=event_operator(e,OPS);b=event_operator(dict(e,ratio=-F(1,9)),OPS)
        self.assertLess(strict_error(a.conj().T,b),1e-15)
    def test_empty_sequence_strict_zero(self):
        configure(100);self.assertLess(strict_guard('',Angle('atan',F(0))),F(1,10**20))
    def test_global_phase_guard(self):
        configure(100);self.assertGreater(strict_guard('T',Angle('pi',F(1,4))),F(1,10))
    def test_generic_native_control_lowering(self):
        # Two-system control fixture, distinct from the registered 3-system provider.
        from trottertracks.algorithm_codesign import g9_matrix as old
        theta=np.arctan(1/9);u=np.diag([np.exp(-1j*theta),np.exp(1j*theta)])
        q=np.kron(OPS[1],np.eye(2));wanted=controlled(np.cos(theta)*np.eye(4)-1j*np.sin(theta)*q)
        gates=[('R',0,'1/9',1),('CX',2,0),('R',0,'1/9',-1),('CX',2,0)]
        self.assertLess(strict_error(old.circuit(gates,None,n=3),wanted),1e-14)
    def test_synthesized_global_phase_cancels_with_actual_adjoint(self):
        from trottertracks.algorithm_codesign import g9_matrix as old
        # Exact T differs from Rz(pi/4) by phase; both signed factors share T/T-adjoint.
        cache={'1':{'sequence':'T'}}
        gates=[('R',0,'1',1),('CX',2,0),('R',0,'1',-1),('CX',2,0)]
        self.assertLess(strict_error(old.circuit(gates,cache,n=3),old.circuit(gates,None,n=3)),1e-14)
    def test_wrong_weight_rejected(self):
        with self.assertRaises(ArithmeticError):
            validate_events([dict(coefficient=F(1),proposal=F(1,2),weight=F(1))],
                            dict(acceptance_upper=1,M2_upper=10,range_upper=10))
    def test_excess_moment_rejected(self):
        with self.assertRaises(ArithmeticError):
            validate_events([dict(coefficient=F(1),proposal=F(1,2),weight=F(2))],
                            dict(acceptance_upper=1,M2_upper=F(1),range_upper=10))
    def test_zero_mass_product_cost(self):
        # Enumerate a made-up classical law, independently of the formula under test.
        q=[F(1,4),F(1,4)];prices=[2,6];z=sum(q);step=sum(a*b for a,b in zip(q,prices))
        direct=sum(a*b*(u+v) for a,u in zip(q,prices) for b,v in zip(q,prices))
        budget=dict(N_per_axis=7,accepted_tail_calls=9,hard_attempts=14)
        c=whole_cost(z,{'T':step},2,budget,{'T':6})
        self.assertEqual(c['per_whole_trial_native']['T'],direct)
        self.assertEqual(c['prep_readout_slope'],14*z*z)
        self.assertEqual(c['hard_native']['T'],168)
    def test_budget_is_signal_free(self):
        b=dict(M2_upper=F(2),range_upper=F(2),acceptance_upper=F(3,4))
        v=whole_budget(b,X,3,CONFIG)
        self.assertFalse(v['uses_signal']);self.assertEqual(v['M2_upper'],8)
        self.assertEqual(v['acceptance_upper'],F(27,64));self.assertGreater(v['remaining'],0)
    def test_exhaustion_no_rescue(self):
        with self.assertRaises(ArithmeticError):error_budget(1,3,CONFIG)
    def test_shot_cap_no_clipping(self):
        c=dict(CONFIG,caps=dict(CONFIG['caps'],shot_cap_per_axis=1))
        with self.assertRaises(ArithmeticError):whole_budget(dict(M2_upper=1,range_upper=1,acceptance_upper=1),X,1,c)
    def test_product_error_algebra(self):
        e=error_budget(X,3,CONFIG)
        self.assertEqual(e['axis_bias_upper'],2*((1+e['Taylor_step_upper']+e['rounded_native_step_upper'])**3-1))
    def test_affine_crossing(self):
        r=affine_relation(dict(intercept=10,slope=1),dict(intercept=4,slope=3))
        self.assertEqual(r['crossover'],3);self.assertEqual(r['classification'],'RESOURCE_TRADEOFF')
    def test_familywise_contract_metadata(self):
        self.assertIs(validate_contract(CONFIG),CONFIG)
        self.assertEqual(24*F(CONFIG['alpha_axis'])+12*F(CONFIG['resource_failure_per_row']),F(1,20))
    def test_primary_pair_cannot_change(self):
        with self.assertRaises(ValueError):validate_contract(dict(CONFIG,primary_pair=['ordinary','matched_CTS']))


class Integrity(unittest.TestCase):
    def test_utf8_lossless_chunks(self):
        with tempfile.TemporaryDirectory() as t:
            w=EvidenceWriter(t,100000,chunk_bytes=101)
            v=w.write('events.json',{'日本語':'漢🙂é'*200})
            self.assertTrue(verify_chunks(t,v))
            self.assertTrue(all(a['bytes']<=101 for a in v['chunks']))
            self.assertEqual(json.loads((Path(t)/'events.json').read_text()),{'日本語':'漢🙂é'*200})
    def test_chunk_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            v=EvidenceWriter(t,100000,chunk_bytes=32).write('a.json',{'a':'x'*100})
            (Path(t)/v['chunks'][0]['path']).write_bytes(b'bad')
            with self.assertRaises(ArithmeticError):verify_chunks(t,v)
    def test_default_512k_chunks(self):
        with tempfile.TemporaryDirectory() as t:
            v=EvidenceWriter(t,5000000).write('large.json',{'a':'🙂'*200000})
            self.assertTrue(verify_chunks(t,v));self.assertGreater(len(v['chunks']),1)
            self.assertTrue(all(a['bytes']<=524288 for a in v['chunks']))
    def test_exclusive_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.json';first=atomic_json(p,{'a':1})
            with self.assertRaises(FileExistsError):atomic_json(p,{'a':2})
            self.assertEqual(digest(p),first['sha256'])
    def test_partial_output_failure(self):
        with tempfile.TemporaryDirectory() as t:
            w=EvidenceWriter(t,100000,guard=lambda:(_ for _ in ()).throw(RuntimeError('synthetic cap')))
            with self.assertRaises(RuntimeError):w.write('a.json',{'a':1})
            self.assertFalse((Path(t)/'a.json').exists());self.assertTrue((Path(t)/'a.json.partial').exists())
    def test_aggregate_bytes_before_write(self):
        with tempfile.TemporaryDirectory() as t:
            w=EvidenceWriter(t,32770)
            with self.assertRaises(RuntimeError):w.write('a.json',{'a':'x'*300})
            self.assertLessEqual(w.total_bytes(),32770)
    def test_traversal(self):
        for s in ('../out','/tmp/out'):
            with self.assertRaises(ValueError):safe_relative(s)
    def test_marker_reuse(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'new';p,v=consume_marker(d,dict(retries=0))
            with self.assertRaises(FileExistsError):consume_marker(d,dict(retries=0))
            self.assertEqual(digest(p),v['sha256'])
    def test_protected_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);(d/'old').write_text('original');atomic_json(d/'ledger',{'sha256':{'old':digest(d/'old')}})
            self.assertTrue(check_protected(d,{'protected_ledger':'ledger'}))
            (d/'old').write_text('changed')
            with self.assertRaises(PermissionError):check_protected(d,{'protected_ledger':'ledger'})


class Protection(unittest.TestCase):
    def auth(self):return dict(status='APPROVED_FOR_ONE_RTE_DEVELOPMENT_D1_RUN',science_execution_authorized=True,
        source_commit='a'*40,contract_sha256='h',runs=1,retries=0,mandatory_STOP=True,
        explicit_execution_instruction='off-domain mocked authorization only')
    def bind(self,a=None,**kwargs):
        p=dict(contract_sha='h',requested_source='a'*40,head='b'*40,parents=['a'*40],changed=['auth'],dirty=False,allowed=['auth','receipt']);p.update(kwargs)
        return validate_binding(a or self.auth(),**p)
    def test_direct_child_binding(self):self.bind()
    def test_pending_binding(self):
        with self.assertRaises(PermissionError):self.bind(dict(self.auth(),science_execution_authorized=False))
    def test_source_parent_mismatch(self):
        with self.assertRaises(PermissionError):self.bind(parents=['c'*40])
    def test_extra_child_source_change(self):
        with self.assertRaises(PermissionError):self.bind(changed=['auth','code.py'])
    def test_contract_mismatch(self):
        with self.assertRaises(PermissionError):self.bind(contract_sha='bad')
    def test_boolean_runs_rejected(self):
        with self.assertRaises(PermissionError):self.bind(dict(self.auth(),runs=True))
    def test_dirty_rejected(self):
        with self.assertRaises(PermissionError):self.bind(dirty=True)
    def test_pending_before_git_or_science(self):
        with patch('trottertracks.algorithm_codesign.rte_development.launch.git',side_effect=AssertionError('must not query')):
            with self.assertRaises(PermissionError):verify_launch(ROOT,CP,'a'*40)
    def test_worker_binding_tamper(self):
        inv=dict(sha256='i',keys=[dict(key='k',ratio='1/9',acquisition='NEW_ONCE')])
        j=dict(inventory_sha256='i',marker_sha256='m',execution_HEAD='h',source_commit='s',authorization_sha256='a',contract_sha256='c',key='k',ratio='1/9')
        self.assertEqual(verify_job(j,inv,'m','h','s','a','c')['ratio'],'1/9')
        for field in ('inventory_sha256','marker_sha256','source_commit','authorization_sha256','contract_sha256','ratio'):
            with self.assertRaises(PermissionError):verify_job(dict(j,**{field:'wrong'}),inv,'m','h','s','a','c')
    def test_worker_result_tamper(self):
        key=dict(key='k',ratio='1/9');v=dict(status='KEY_COMPLETE',key='k',ratio='1/9',synthesis_calls=1,retries=0,saved={},inventory_sha256='i')
        self.assertEqual(validate_worker_result(v,key,{'inventory_sha256':'i'}),{})
        with self.assertRaises(PermissionError):validate_worker_result(v,key,{'inventory_sha256':'wrong'})
    def test_inventory_anchor_no_new_acquisition(self):
        rows=[dict(steps=1,events=[{'ratio':F(1,9)}]),dict(steps=2,events=[{'ratio':F(1,11)}])]
        with self.assertRaises(ArithmeticError):freeze_inventory(rows,{},CONFIG)
        v=freeze_inventory(rows,{'1/9':{}},CONFIG)
        self.assertEqual([r['acquisition'] for r in v['keys']].count('NEW_ONCE'),1)
    def test_inventory_key_binds_precision(self):
        self.assertNotEqual(inventory_key(F(1,9),CONFIG),inventory_key(F(1,9),dict(CONFIG,primitive_error='1/1000')))
    def test_native_transpile_static_absent(self):
        for p in (ROOT/'src/trottertracks/algorithm_codesign/rte_development').glob('*.py'):
            self.assertNotIn('transpile(',p.read_text());self.assertNotIn('.sample(',p.read_text())
    def test_main_pending_rejects_before_pipeline(self):
        spec=importlib.util.spec_from_file_location('future_d1',ROOT/'scripts/tracks/algorithm_codesign/run_rte_development_d1.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with patch('trottertracks.algorithm_codesign.rte_development.pipeline.run_pipeline',side_effect=AssertionError('must not launch')):
            with self.assertRaises(PermissionError):m.execute('a'*40)
        self.assertFalse((ROOT/CONFIG['result_directory']).exists())
    def test_observer_pending_rejects_before_process(self):
        spec=importlib.util.spec_from_file_location('future_observer',ROOT/'scripts/tracks/algorithm_codesign/observe_rte_development_d1.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not launch')):
            with self.assertRaises(PermissionError):m.execute('a'*40)
    def test_worker_pending_rejects_before_key(self):
        from trottertracks.algorithm_codesign.rte_development.worker import run_worker
        with self.assertRaises(PermissionError):run_worker(ROOT,ROOT/'not-a-job.json')
    def acquisition_mock(self,fail=False):
        with tempfile.TemporaryDirectory() as t:
            c=dict(CONFIG,result_directory='result');d=Path(t)/'result';d.mkdir()
            keys=[dict(key=str(i),ratio=str(F(1,i+8)),acquisition='NEW_ONCE') for i in range(3)]
            identity=dict(marker_sha256='m',execution_HEAD='h',source_commit='s',authorization_sha256='a',contract_sha256='c')
            state=dict(pid=10,launched=[],terminated=[]);guard=Guard(c['caps'])
            class Child:
                def __init__(self,pid):self.pid=pid;self.returncode=None
                def poll(self):return self.returncode
                def terminate(self):state['terminated'].append(self.pid)
                def kill(self):state['terminated'].append(self.pid)
                def wait(self,timeout=None):self.returncode=0;return 0
            def start(argv,**kwargs):
                state['pid']+=1;state['launched'].append(argv)
                self.assertTrue(all(kwargs['env'][k]=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')))
                path=Path(argv[-1]);j=json.loads(path.read_text())
                atomic_json(path.with_suffix('.result.json'),{'status':'KEY_COMPLETE','key':j['key'],'ratio':j['ratio'],
                    'synthesis_calls':1,'retries':0,'saved':{'mock_only':True},'inventory_sha256':'i',**identity})
                return Child(state['pid'])
            def wait4(pid,flags):return pid,(256 if fail else 0),SimpleNamespace(ru_utime=.1,ru_stime=.1,ru_maxrss=100)
            with patch('trottertracks.algorithm_codesign.rte_development.pipeline.subprocess.Popen',start),patch('os.wait4',wait4),patch('trottertracks.algorithm_codesign.rte_reallocation.numeric.validate_saved',lambda *a:None),patch.object(guard,'check',lambda:None):
                if fail:
                    with self.assertRaises(RuntimeError):acquire(Path(t),c,{'keys':keys},'i',{},identity,guard)
                    self.assertEqual(len(state['launched']),2);self.assertEqual(len(state['terminated']),1)
                else:
                    cache,receipts=acquire(Path(t),c,{'keys':keys},'i',{},identity,guard)
                    self.assertEqual(len(receipts),3);self.assertEqual(len(cache),3)
                    self.assertEqual(len(state['launched']),3);self.assertAlmostEqual(guard.reaped_cpu,.6)
                self.assertEqual(guard.children,set())
    def test_parallel_acquisition_mock_binding(self):self.acquisition_mock()
    def test_child_failure_no_retry_mock(self):self.acquisition_mock(fail=True)


class Resources(unittest.TestCase):
    def test_cpu_cap(self):
        g=Guard(CONFIG['caps'])
        with patch.object(g,'process_stats',return_value=(1801,1)):
            with self.assertRaisesRegex(RuntimeError,'CPU'):g.check()
    def test_worker_cpu_cap(self):
        g=Guard(CONFIG['caps'],worker=True)
        with patch.object(g,'process_stats',return_value=(21,1)):
            with self.assertRaisesRegex(RuntimeError,'CPU'):g.check()
    def test_rss_cap(self):
        g=Guard(CONFIG['caps'])
        with patch.object(g,'process_stats',return_value=(1,513*1048576)):
            with self.assertRaisesRegex(RuntimeError,'RSS'):g.check()
    def test_aggregate_children(self):
        g=Guard(CONFIG['caps']);g.children={99,100}
        with patch.object(g,'process_stats',return_value=(1,600*1048576)):
            with self.assertRaisesRegex(RuntimeError,'RSS'):g.check()
    def test_log_cap(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'key.log').write_bytes(b'x'*(CONFIG['caps']['per_key_log_bytes']+1));g=Guard(CONFIG['caps'],t)
            with self.assertRaisesRegex(RuntimeError,'log'):g.check()
    def test_alarm_conflict(self):
        g=Guard(CONFIG['caps'])
        with patch('signal.getitimer',return_value=(1,1)):
            with self.assertRaisesRegex(RuntimeError,'conflict'):g.__enter__()
    def test_restore_after_entry_failure(self):
        previous=resource.getrlimit(resource.RLIMIT_AS);handler=signal.getsignal(signal.SIGALRM)
        g=Guard(CONFIG['caps'])
        with patch.object(g,'check',side_effect=RuntimeError('mock entry fail')):
            with self.assertRaises(RuntimeError):g.__enter__()
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS),previous)
        self.assertEqual(signal.getsignal(signal.SIGALRM),handler);self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)
    def test_no_new_wall_cap(self):
        self.assertIsNone(CONFIG['caps']['total_wall_seconds']);self.assertIsNone(CONFIG['caps']['phase_wall_seconds'])


if __name__=='__main__':unittest.main()
