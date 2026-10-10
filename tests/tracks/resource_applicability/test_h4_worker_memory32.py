"""Memory-cap integration with native operations and every child mocked."""
import copy
import importlib.util
import io
import json
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from trottertracks.resource_applicability.h4_geometry import resources as res, observer as obs, launch_binding as bind, execution as ex, workers as wk, memory_budget as mem
from trottertracks.resource_applicability.h4_geometry.identity import Stop

spec=importlib.util.spec_from_file_location('memory32_fixtures',Path(__file__).with_name('test_h4_pressure_integration.py'))
fx=importlib.util.module_from_spec(spec);spec.loader.exec_module(fx)


def documents():
    p,a,r=fx.documents();p['caps']['worker_AS_RSS']=mem.WORKER_CAP
    a['memory_budget_amendment']=dict(approved=True,**{'from':mem.DRIVER_CAP,'to':mem.WORKER_CAP},
        profile={'path':'synthetic-memory.json','sha256':'3'*64},authority={'path':'synthetic-memory-authority.json','sha256':'4'*64})
    fx.rebound(p,a,r);return p,a,r


class WorkerMemory32Tests(unittest.TestCase):
    def test_legacy_and_amended_admission(self):
        self.assertEqual(mem.required_available(4,mem.DRIVER_CAP),int(120.25*2**30))
        self.assertEqual(mem.required_available(4,mem.WORKER_CAP),int(152.25*2**30))
        self.assertEqual(res.admission(72*2**30,1,6,range(6),range(6),now=1),6)
    def test_profile_and_worker_count_are_closed(self):
        mem.verify(dict(mem.PARAMETERS))
        for key,value in [('worker_AS_RSS',64*2**30),('driver_AS_RSS',32*2**30),('workers',12),('admission_bytes',0)]:
            p=dict(mem.PARAMETERS);p[key]=value
            with self.assertRaises(Stop):mem.verify(p)
        for cap,n in [(True,4),(16*2**30,4),(mem.WORKER_CAP,12)]:
            with self.assertRaises(Stop):mem.required_available(n,cap)
    def test_32_requires_exact_user_amendment(self):
        bind.authorize(*documents(),explicit_launch=True)
        for key,value in [('approved',False),('approved',1),('from',0),('to',64*2**30),('profile',{'path':'../bad','sha256':'3'*64})]:
            p,a,r=documents();a['memory_budget_amendment'][key]=value;fx.rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
        p,a,r=documents();a.pop('memory_budget_amendment');fx.rebound(p,a,r)
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_driver_observer_and_launch_permissions_do_not_expand(self):
        for key,value in [('driver_AS_RSS',mem.WORKER_CAP),('observer_AS',512*2**20),('headroom',0),('monitor_seconds',6)]:
            p,a,r=documents();p['caps'][key]=value;fx.rebound(p,a,r)
            with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
        p,a,r=documents()
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=False)
        a['approved']=False;fx.rebound(p,a,r)
        with self.assertRaises(Stop):bind.authorize(p,a,r,explicit_launch=True)
    def test_fresh_boundary_152_point25(self):
        p,a,r=documents();minimum=mem.required_available(4,mem.WORKER_CAP)
        for available,ok in [(minimum-1,False),(minimum,True)]:
            m=fx.fixture.memory(.18,available)
            if ok:bind.fresh_gate(p,fx.observation(m),now=1.01,pressure_profile=fx.pp.PARAMETERS)
            else:
                with self.assertRaises(Stop):bind.fresh_gate(p,fx.observation(m),now=1.01,pressure_profile=fx.pp.PARAMETERS)
    def test_inherited_hard8_rejected_without_setting(self):
        with patch.object(res.resource,'getrlimit',return_value=(8*2**30,8*2**30)),patch.object(res.resource,'setrlimit') as setter:
            with self.assertRaises(Stop):res.limit_owned_address_space(mem.WORKER_CAP)
            setter.assert_not_called()
    def test_soft8_preserves_hard32_then_worker32_and_driver8(self):
        state=[(mem.WORKER_CAP,mem.WORKER_CAP)]
        def setter(_kind,value):state[0]=value
        with patch.object(res.resource,'getrlimit',side_effect=lambda _kind:state[0]),patch.object(res.resource,'setrlimit',side_effect=setter):
            res.limit_owned_address_space(preserve_hard=True)
            self.assertEqual(state[0],(mem.DRIVER_CAP,mem.WORKER_CAP))
            res.limit_owned_address_space(mem.WORKER_CAP)
            self.assertEqual(state[0],(mem.WORKER_CAP,mem.WORKER_CAP))
            res.limit_owned_address_space()
            self.assertEqual(state[0],(mem.DRIVER_CAP,mem.DRIVER_CAP))
    def test_failed_limit_confirmation_is_not_ready(self):
        with patch.object(res.resource,'getrlimit',return_value=(-1,-1)),patch.object(res.resource,'setrlimit'):
            with self.assertRaises(Stop):res.limit_owned_address_space(mem.WORKER_CAP)
    def test_worker_decode_authorize_checkout_precede_raise(self):
        p,a,r=documents();permit=bind.authorize(p,a,r,explicit_launch=True);order=[];frames=[]
        def limit(cap=mem.DRIVER_CAP,**kw):order.append((cap,kw.get('preserve_hard',False)))
        incoming=Mock(buffer=io.BytesIO());outgoing=Mock(buffer=io.BytesIO())
        with patch.object(wk.sys,'stdin',incoming),patch.object(wk.sys,'stdout',outgoing),patch.object(wk.os,'getppid',return_value=permit.driver_pid),\
             patch.object(res,'limit_owned_address_space',side_effect=limit),patch.object(wk,'read_frame',side_effect=[permit,EOFError('mock job EOF')]),\
             patch.object(wk,'write_frame',side_effect=lambda stream,value:frames.append(value)),\
             patch.object(bind,'role_affinity',side_effect=lambda *x:order.append('authorize')),patch.object(bind.gates,'checkout_gate',side_effect=lambda p:(order.append('checkout') or ({},{}))):
            wk.owned_worker_main(permit.driver_pid,0)
        self.assertEqual(order,[(mem.DRIVER_CAP,True),'authorize','checkout',(mem.WORKER_CAP,False)])
        self.assertIn('ready',frames[0])
    def test_rejected_worker_permit_never_raises_or_readies(self):
        p,a,r=documents();permit=bind.authorize(p,a,r,explicit_launch=True);frames=[]
        with patch.object(wk.sys,'stdin',Mock(buffer=io.BytesIO())),patch.object(wk.sys,'stdout',Mock(buffer=io.BytesIO())),patch.object(wk.os,'getppid',return_value=permit.driver_pid),\
             patch.object(res,'limit_owned_address_space') as limit,patch.object(wk,'read_frame',return_value=permit),patch.object(wk,'write_frame',side_effect=lambda stream,v:frames.append(v)),\
             patch.object(bind,'role_affinity',side_effect=Stop('unapproved')),patch.object(bind.gates,'checkout_gate') as checkout:
            wk.owned_worker_main(permit.driver_pid,0)
            limit.assert_called_once_with(preserve_hard=True);checkout.assert_not_called()
        self.assertNotIn('ready',frames[0])
    def state(self):
        s=obs.ObservationState(fx.fixture.memory(),0.,workers=4,worker_cap=mem.WORKER_CAP,driver_pid=10)
        s.own_worker(11);return s
    def test_worker_over8_up_to32_allowed_in_any_sample_order(self):
        for value in (8*2**30+1,mem.WORKER_CAP):
            for reverse in (False,True):
                s=self.state();samples=[fx.fixture.role(10),fx.fixture.role(11)]
                samples[1]['rss']=samples[1]['address_space']=value
                if reverse:samples.reverse()
                self.assertIsNone(s.evaluate(fx.fixture.memory(),samples,fx.fixture.role(20),1.,1.01)['first_failure'])
    def test_driver_over8_and_worker_over32_stop(self):
        for pid,cap in [(10,mem.DRIVER_CAP),(11,mem.WORKER_CAP)]:
            for field in ('rss','address_space'):
                samples=[fx.fixture.role(10),fx.fixture.role(11)]
                next(x for x in samples if x['pid']==pid)[field]=cap+1
                self.assertEqual(self.state().evaluate(fx.fixture.memory(),samples,fx.fixture.role(20),1.,1.01)['first_failure']['reason'],'owned_role_rss_as')
    def test_unregistered_missing_or_spoofed_memory_role_stops(self):
        for samples in ([fx.fixture.role(11)],[fx.fixture.role(10),fx.fixture.role(12)],[fx.fixture.role(10),fx.fixture.role(10)]):
            self.assertIsNotNone(self.state().evaluate(fx.fixture.memory(),samples,fx.fixture.role(20),1.,1.01)['first_failure'])
        with self.assertRaises(Stop):obs.ObservationState(fx.fixture.memory(),0.,workers=4,worker_cap=mem.WORKER_CAP)
    def test_observer_cap_deadline_and_first_reason_preserved(self):
        s=self.state();small=fx.fixture.role(20);small['address_space']=obs.AS_CAP+1
        first=s.evaluate(fx.fixture.memory(),[fx.fixture.role(10),fx.fixture.role(11)],small,1.,1.01)['first_failure']
        self.assertEqual(first['reason'],'observer_rss_as')
        s.evaluate(None,[],small,2.,8.)
        self.assertEqual(s.first_failure,first)
    def test_bootstrap_boundaries_order_and_post_pool_failure_cleanup(self):
        minimum=mem.required_available(4,mem.WORKER_CAP)
        for mode in ('under','pass','pool_fail','driver_clamp_fail','cleanup_fail','hard8'):
            p,a,r=documents();m=fx.fixture.memory(0,minimum-(mode=='under'));m['observed_at']=time.monotonic()
            permit=replace(bind.authorize(p,a,r,explicit_launch=True),launch_observation=fx.observation(m))
            monitor,pool,budget=Mock(),Mock(),Mock(root=Path(p['output_root']))
            if mode=='cleanup_fail':
                monitor.stop_children.side_effect=PermissionError('synthetic pidfd EPERM')
                pool.shutdown.side_effect=PermissionError('synthetic second cleanup EPERM')
            state=[(-1,8*2**30 if mode=='hard8' else -1)];events=[]
            def setter(_kind,value):
                events.append(('limit',value))
                if mode in ('driver_clamp_fail','cleanup_fail') and value==(mem.DRIVER_CAP,mem.DRIVER_CAP):raise OSError('synthetic clamp failure')
                state[0]=value
            def spawn(*x):
                events.append(('pool',state[0]));self.assertEqual(state[0],(mem.DRIVER_CAP,-1))
                if mode=='pool_fail':raise OSError('synthetic partial pool cleanup already executed')
                return pool
            with patch.object(ex,'observe_memory',return_value=m),patch.object(bind,'read_pressure_profile',return_value=fx.pp.PARAMETERS),patch.object(ex.os,'sched_getaffinity',return_value={16}),\
                 patch.object(res.resource,'getrlimit',side_effect=lambda k:state[0]),patch.object(res.resource,'setrlimit',side_effect=setter),patch.object(ex.threading,'Thread'),\
                 patch.object(obs,'IndependentObserver',return_value=monitor) as observer,patch.object(wk,'OwnedPool',side_effect=spawn) as factory:
                if mode=='pass':
                    ex.OwnedRun(permit,a,prepared_budget=budget)
                    self.assertEqual(state[0],(mem.DRIVER_CAP,mem.DRIVER_CAP))
                    self.assertEqual(observer.call_args.kwargs['worker_cap'],mem.WORKER_CAP)
                else:
                    with self.assertRaises((Stop,OSError)) as error:ex.OwnedRun(permit,a,prepared_budget=budget)
                    if mode=='cleanup_fail':self.assertIn('synthetic clamp failure',str(error.exception))
                    if mode in ('under','hard8'):observer.assert_not_called();factory.assert_not_called()
                    else:monitor.close.assert_called_once_with(abort=True);budget.close.assert_called_once()
                    if mode in ('driver_clamp_fail','cleanup_fail'):pool.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
    def test_worker_foreign_parent_and_invalid_index_before_affinity(self):
        p,a,r=documents();permit=bind.authorize(p,a,r,explicit_launch=True)
        for parent,index in [(permit.driver_pid+1,0),(permit.driver_pid,-1),(permit.driver_pid,4)]:
            with patch.object(bind.os,'getppid',return_value=parent),patch.object(bind.os,'sched_setaffinity') as affinity:
                with self.assertRaises(Stop):bind.role_affinity(permit,'worker',index)
                affinity.assert_not_called()
    def test_partial_spawn_and_bad_ready_reap_and_close_mock_children(self):
        p,a,r=documents();permit=bind.authorize(p,a,r,explicit_launch=True)
        for mode in ('spawn','ready'):
            child=Mock(pid=110,stdin=io.BytesIO(),stdout=io.BytesIO());child.wait.return_value=0
            executor=Mock();future=Mock();future.result.side_effect=Stop('synthetic bad ready');executor.submit.return_value=future
            monitor=Mock();budget=Mock()
            with patch.object(wk,'ThreadPoolExecutor',create=True),patch('concurrent.futures.ThreadPoolExecutor',return_value=executor),patch.object(bind,'reference',return_value={'python':'synthetic-python'}),\
                 patch.object(wk.subprocess,'Popen',side_effect=[child,OSError('synthetic Popen failure')] if mode=='spawn' else [child]*4):
                with self.assertRaises((OSError,Stop)):wk.OwnedPool(4,permit,monitor,budget)
            self.assertTrue(child.stdin.closed and child.stdout.closed)
            self.assertTrue(child.wait.called);executor.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
    def test_schema_declares_memory_amendment_and_fixed_driver(self):
        schema=json.loads((fx.ROOT/'schemas/h4_newhost_launch_v2.json').read_text())
        self.assertIn('memory_budget_amendment',schema['properties']['authorization']['properties'])
        self.assertEqual(schema['properties']['plan']['properties']['caps']['properties']['driver_AS_RSS']['const'],mem.DRIVER_CAP)

    def test_signal_error_keeps_wait_pipe_and_executor_cleanup(self):
        pool=wk.OwnedPool.__new__(wk.OwnedPool);pool.monitor=Mock();pool.io=Mock()
        initial=PermissionError('synthetic pidfd EPERM');pool.monitor.stop_children.side_effect=initial
        pool.processes=[Mock(stdin=io.BytesIO(),stdout=io.BytesIO()) for _ in range(4)]
        with self.assertRaises(PermissionError) as error:pool.shutdown()
        self.assertIs(error.exception,initial)
        for child in pool.processes:
            child.wait.assert_called_once_with(timeout=2)
            self.assertTrue(child.stdin.closed and child.stdout.closed)
        pool.io.shutdown.assert_called_once_with(wait=True,cancel_futures=True)

    def test_wait_and_pipe_errors_do_not_skip_later_children(self):
        pool=wk.OwnedPool.__new__(wk.OwnedPool);pool.monitor=Mock();pool.io=Mock()
        initial=OSError('synthetic wait error');children=[Mock(stdin=Mock(),stdout=Mock()) for _ in range(4)]
        children[0].wait.side_effect=initial;children[0].stdin.close.side_effect=OSError('synthetic FD error')
        pool.processes=children
        with self.assertRaises(OSError) as error:pool.shutdown()
        self.assertIs(error.exception,initial)
        for child in children:
            child.wait.assert_called_once_with(timeout=2);child.stdin.close.assert_called_once();child.stdout.close.assert_called_once()
        pool.io.shutdown.assert_called_once_with(wait=True,cancel_futures=True)
