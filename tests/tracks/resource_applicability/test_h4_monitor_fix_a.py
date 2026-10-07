"""Bounded artificial-only tests: no transpile, molecular input or real workers."""
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import unittest
from unittest.mock import patch

from trottertracks.resource_applicability.h4_geometry import circuits, identity as ident, observer as obs
from trottertracks.resource_applicability.h4_geometry.streaming import LazyList

ROOT = Path(__file__).absolute().parents[3]
EVIDENCE = Path(os.environ['H4_ARTIFICIAL_EVIDENCE'])
spec = importlib.util.spec_from_file_location('h4_run05_oracle', Path(__file__).with_name('h4_run05_serializer_reference.py'))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
MEASUREMENTS = {}


def canonical_new(value):
    return b''.join(ident.canonical_chunks(value))


class SerializationTests(unittest.TestCase):
    def test_matrix_256_and_nine_qubit_circuit_old_new_bytes_digest(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        matrix = (np.arange(65536, dtype=np.float64).reshape(256,256)/65536).astype(np.complex128)
        matrix.imag = -matrix.real
        matrix[0,0] = complex(-0., -0.)
        circuit = QuantumCircuit(9,1)
        circuit.global_phase = -0.125
        circuit.h(8)
        circuit.append(UnitaryGate(matrix, check_input=False, label='人工matrix'), range(8))
        circuit.cx(8,0); circuit.rz(-0.,2); circuit.measure(8,0)
        results = []
        for axis in ('cosine','sine'):
            before = ident.canonical({'domain':'h4-numerical-circuit-v1', **old.serialize(circuit,axis)})
            after = canonical_new({'domain':'h4-numerical-circuit-v1', **circuits.serialize(circuit,axis)})
            self.assertEqual(before, after)
            digest = hashlib.sha256(before).hexdigest()
            self.assertEqual(circuits.numerical_fingerprint(circuit,axis), digest)
            results.append(dict(axis=axis, bytes=len(before), old_digest=digest, new_digest=digest))
        MEASUREMENTS['matrix_circuit'] = results

    def test_no_tolist_and_limited_metadata_retention(self):
        import numpy as np
        class NoList(np.ndarray):
            def tolist(self): raise AssertionError('tolist forbidden')
        array = np.zeros((256,256), dtype=np.complex128).view(NoList)
        memo = {}
        first = circuits.number(array,memo)
        self.assertIs(first,circuits.number(array,memo))
        self.assertIsInstance(first['array'],LazyList)
        chunks = list(ident.canonical_chunks(first))
        self.assertTrue(all(0 < len(c) <= 65536 for c in chunks))
        for _ in range(70): circuits.number(np.zeros((1,1)),memo)
        self.assertLessEqual(len(memo),64)

    def test_dtype_shape_order_zero_dimensions_empty_and_signed_zero(self):
        import numpy as np
        arrays = [np.array(-0.), np.array(complex(-0.,-0.)), np.zeros((0,2)),
                  np.array([True,False]), np.array([[1,-2]],dtype='>i4'),
                  np.array([[0.,-0.,1.25]],dtype='>f4'),
                  np.asfortranarray(np.array([[1+2j,3-4j],[5+6j,complex(-0.,-0.)]])),
                  np.arange(12,dtype=np.uint64).reshape(3,4)[:,::-1]]
        for array in arrays:
            self.assertEqual(canonical_new(circuits.number(array)), ident.canonical(old.number(array)))

    def test_phase_control_condition_custom_definition_roundtrip(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit import ControlledGate, Gate
        qc = QuantumCircuit(9,1)
        base = Gate('artificial_base',1,[0.25])
        base.definition=QuantumCircuit(1); base.definition.rz(0.25,0)
        definition=QuantumCircuit(2); definition.cx(0,1); definition.global_phase=-0.5
        ctrl=ControlledGate('artificial_ctrl',2,[0.25],num_ctrl_qubits=1,
                            definition=definition,ctrl_state=0,base_gate=base)
        qc.append(ctrl,[8,0]);qc.x(1).c_if(qc.cregs[0],1);qc.global_phase=0.375
        for axis in ('cosine','sine'):
            before=ident.canonical(old.serialize(qc,axis))
            self.assertEqual(canonical_new(circuits.serialize(qc,axis)),before)
            rebuilt=circuits.deserialize(circuits.serialize(qc,axis))
            self.assertEqual(canonical_new(circuits.serialize(rebuilt,axis)),before)

    def test_instruction_walk_is_deferred_and_no_full_exact_tree(self):
        from qiskit import QuantumCircuit
        qc=QuantumCircuit(9); qc.rz(0.1,0)
        with patch.object(circuits,'number',wraps=circuits.number) as number:
            value=circuits.serialize(qc,'cosine')
            self.assertIsInstance(value['instructions'],LazyList)
            self.assertEqual(number.call_count,1)  # one scalar global phase only
        self.assertEqual(canonical_new(value), ident.canonical(old.serialize(qc,'cosine')))

    def test_nonfinite_and_unsupported_stop_at_stream_boundary(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        for value in (np.array([float('nan')]), np.array([complex(1,float('inf'))]),
                      np.array(['bad']), Parameter('symbolic')):
            with self.assertRaises(ident.Stop): canonical_new(circuits.number(value))
        q=QuantumCircuit(1); q.rz(Parameter('p'),0)
        with self.assertRaises(ident.Stop):circuits.numerical_fingerprint(q,'cosine')

    def test_unicode_large_strings_chunk_bound(self):
        payload={'message':'日本語🙂'*10000, 'zero':complex(-0.,-0.)}
        chunks=list(ident.canonical_chunks(payload,1024))
        self.assertTrue(all(len(c)<=1024 for c in chunks))
        self.assertEqual(b''.join(chunks),ident.canonical(payload))

    def test_all_exact_identity_scalar_types_old_digest(self):
        values=[None,True,False,0,-123,2**100,'日本語/🙂\n"\\',-0.,0.,1e-300,
                float.fromhex('0x1.fffffffffffffp+1023'),complex(-0.,1.),
                ('a',1,1j),{'z':[1.,{'a':-1j}],'a':3}]
        for value in values:
            payload={'value':value}
            self.assertEqual(ident.fingerprint('ARTIFICIAL',payload),
                hashlib.sha256(ident.canonical({'domain':'ARTIFICIAL',**payload})).hexdigest())


def observation(at=0):
    return dict(available=32*2**30, observed_at=at, psi_full_avg10=0, oom_events={'fixture':0})


def sample(pid=10):
    return dict(pid=pid, start=str(pid), parent=1, uid=os.getuid(),rss=1024,address_space=4096)


class PolicyTests(unittest.TestCase):
    def state(self):return obs.ObservationState(observation(),0)
    def evaluate(self, state=None, observation_=None, begun=1, ended=1.1, samples=None, observer=None):
        return (state or self.state()).evaluate(observation_ or observation(begun),
                  samples or [sample()],observer or sample(100),begun,ended)

    def test_twelve_workers_roles_and_aggregate_fixture(self):
        owners=[sample(i) for i in range(13)]
        for owner in owners:owner['rss']=owner['address_space']=obs.ROLE_CAP
        record=self.evaluate(samples=owners)
        self.assertIsNone(record['first_failure'])
        self.assertEqual(sum(s['rss'] for s in record['processes']),104*2**30)

    def test_worker_rss_over_cap(self):
        owner=sample();owner['rss']=obs.ROLE_CAP+1
        self.assertEqual(self.evaluate(samples=[owner])['first_failure']['reason'],'owned_role_rss_as')

    def test_worker_as_over_cap(self):
        owner=sample();owner['address_space']=obs.ROLE_CAP+1
        self.assertEqual(self.evaluate(samples=[owner])['first_failure']['reason'],'owned_role_rss_as')

    def test_observer_rss_and_as(self):
        for field,cap in [('rss',obs.RSS_CAP),('address_space',obs.AS_CAP)]:
            owner=sample(100);owner[field]=cap+1
            self.assertEqual(self.evaluate(observer=owner)['first_failure']['reason'],'observer_rss_as')

    def test_missing_observation(self):
        record=self.state().evaluate(None,[sample()],sample(100),1,1.1)
        self.assertEqual(record['first_failure']['reason'],'missing_observation')

    def test_pressure_oom_headroom(self):
        for change,reason in [({'psi_full_avg10':0.01},'memory_pressure'),
                              ({'oom_events':{'fixture':1}},'oom_or_changed_cgroup'),
                              ({'available':16*2**30-1},'memory_headroom')]:
            self.assertEqual(self.evaluate(observation_={**observation(1),**change})['first_failure']['reason'],reason)

    def test_interval_duration_staleness_are_distinct_and_five_seconds(self):
        for begun,ended,at,reason in [(6,6.1,6,'observation_interval'),
                                    (1,6.1,1,'observation_duration'),
                                    (1,1.1,-5,'observation_staleness')]:
            record=self.evaluate(observation_=observation(at),begun=begun,ended=ended)
            self.assertEqual(record['first_failure']['reason'],reason)
            self.assertIn('observation_duration_seconds',record)
            self.assertIn('interval_seconds',record)
            self.assertIn('staleness_seconds',record)

    def test_first_reason_preserved_and_phase_sequence(self):
        state=self.state();state.set_phase(dict(sequence=1,name='synthetic_serialize',monotonic=0.5),1)
        first=self.evaluate(state,observation_={**observation(1),'psi_full_avg10':1})['first_failure']
        self.assertEqual(first['driver_phase']['name'],'synthetic_serialize')
        second=self.evaluate(state,begun=8,ended=8.1)['first_failure']
        self.assertIs(first,second)
        with self.assertRaises(ident.Stop):state.set_phase(dict(sequence=1,name='reuse',monotonic=1),1)

    def test_wall_carry_not_reset(self):
        state=obs.ObservationState(observation(),0,prior_wall=5466.188392877579)
        self.assertEqual(self.evaluate(state)['wall_seconds'],5466.188392877579+1.1)
        state.prior_wall=obs.WALL_CAP
        self.assertEqual(self.evaluate(state,begun=2,ended=2.1)['first_failure']['reason'],'cumulative_wall')

    def test_worker_count_and_duplicate_rejected(self):
        self.assertEqual(self.evaluate(samples=[sample(i) for i in range(14)])['first_failure']['reason'],'owned_process_count')
        self.assertEqual(self.evaluate(samples=[sample(),sample()])['first_failure']['reason'],'duplicate_owned_process')

    def test_foreign_or_reused_or_exited_identity_never_signalled(self):
        expected=obs.identity(sample())
        for changed in ({'uid':os.getuid()+1},{'start':'new'},{'parent':999}):
            owner=obs.OwnedIdentity.__new__(obs.OwnedIdentity)
            owner.expected=expected;owner.fd=100;owner.sampler=lambda _,c=changed:{**sample(),**c}
            with patch.object(obs.select,'select',return_value=([],[],[])),patch.object(obs.signal,'pidfd_send_signal') as kill:
                self.assertFalse(owner.terminate());kill.assert_not_called()
        owner.sampler=lambda _:sample()
        with patch.object(obs.select,'select',return_value=([100],[],[])),patch.object(obs.signal,'pidfd_send_signal') as kill:
            self.assertFalse(owner.terminate());kill.assert_not_called()

    def test_verified_pidfd_stop(self):
        owner=obs.OwnedIdentity.__new__(obs.OwnedIdentity)
        owner.expected=obs.identity(sample());owner.fd=100;owner.sampler=lambda _:sample()
        with patch.object(obs.select,'select',return_value=([],[],[])),patch.object(obs.signal,'pidfd_send_signal') as kill:
            self.assertTrue(owner.terminate());kill.assert_called_once_with(100,signal.SIGTERM)

    def test_failure_durable_before_own_stop_and_no_foreign_stop(self):
        # Runtime child setup is mocked; no production process is spawned.
        events=[]
        config=dict(scope='PRODUCTION',runtime_authorization=True,output_cap=100000,
                    driver=obs.identity(sample(os.getppid())),prior_wall=0,wall_started=0,workers=12)
        fake_owner=type('Owner',(),{'expected':config['driver'], 'sample':lambda _:sample(),
             'terminate':lambda _:events.append('signal'), 'close':lambda _:None})()
        with patch.object(obs,'receive',return_value=config),patch.object(obs.socket,'socket'),\
             patch.object(obs,'OwnedIdentity',return_value=fake_owner),patch.object(obs,'Trace') as trace,\
             patch.object(obs,'observe_memory',side_effect=OSError('synthetic missing I/O')),\
             patch.object(obs.resource,'setrlimit'),patch.object(obs.os,'close'):
            trace.return_value.write.side_effect=lambda record,**_:events.append('durable')
            obs.observer_main(100,101)
        self.assertEqual(events,['durable','signal'])

    def test_production_role_unapproved_stops_before_spawn(self):
        with patch.object(obs.subprocess,'Popen') as spawn:
            with self.assertRaises(ident.Stop):obs.IndependentObserver(sys.executable,EVIDENCE/'no-launch.jsonl',scope='PRODUCTION')
            spawn.assert_not_called()

    def test_twelve_worker_identities_via_mocked_pidfds(self):
        driver_pid=500
        owners=[]
        with patch.object(obs.os,'pidfd_open',side_effect=range(1000,1012)),\
             patch.object(obs.select,'select',return_value=([],[],[])),patch.object(obs.os,'close'):
            for pid in range(501,513):
                current={**sample(pid),'parent':driver_pid}
                owner=obs.OwnedIdentity(obs.identity(current),driver_pid,sampler=lambda _,v=current:v)
                self.assertEqual(owner.sample()['pid'],pid);owners.append(owner)
            with patch.object(obs.signal,'pidfd_send_signal') as kill:
                for owner in owners:self.assertTrue(owner.terminate())
                self.assertEqual(kill.call_count,12)
                self.assertEqual({call.args[0] for call in kill.call_args_list},set(range(1000,1012)))
            for owner in owners:owner.close()

    def test_private_frame_eof_and_truncation_rejected(self):
        for data,flags in [(b'',0),(b'{}',obs.socket.MSG_TRUNC)]:
            fake=type('Socket',(),{'recvmsg':lambda _,cap,d=data,f=flags:(d,[],f,None)})()
            with self.assertRaises(ident.Stop):obs.receive(fake)

    def test_trace_terminal_space_reservation(self):
        trace=obs.Trace(100,obs.TERMINAL_RESERVE+64)
        with patch.object(obs.os,'write',side_effect=lambda fd,v:len(v)),patch.object(obs.os,'fsync') as sync:
            trace.write({'kind':'observation'})
            with self.assertRaises(ident.Stop):trace.write({'padding':'x'*64})
            trace.write({'kind':'first_stop','reason':'budget'},terminal=True)
            self.assertEqual(sync.call_count,2)

    def test_ownedrun_unapproved_role_stops_before_resources_or_spawn(self):
        from trottertracks.resource_applicability.h4_geometry.execution import OwnedRun
        with patch('trottertracks.resource_applicability.h4_geometry.execution.observe_memory') as memory,\
             patch.object(obs.subprocess,'Popen') as spawn:
            with self.assertRaises(ident.Stop):OwnedRun(None,{'allowed_cpus':[]})
            memory.assert_not_called();spawn.assert_not_called()

    def test_shutdown_startup_included_in_wall(self):
        state=obs.ObservationState(observation(2),2,wall_started=0,prior_wall=100)
        record=self.evaluate(state,begun=2.5,ended=2.6)
        self.assertEqual(record['wall_seconds'],102.6)


class LiveObserverTests(unittest.TestCase):
    def start(self,name):
        return obs.IndependentObserver(sys.executable,EVIDENCE/(name+'.jsonl'),scope='SYNTHETIC_ONLY',workers=12)

    def test_independent_observation_during_gil_gc_and_serialization(self):
        observer=self.start('live-gil-gc')
        try:
            observer.phase('synthetic_gil_gc_serialize')
            observer.request('synthetic_fault',fault='fast_period',value=0.1)
            old_interval=sys.getswitchinterval()
            try:
                sys.setswitchinterval(10)
                until=time.monotonic()+1.2
                while time.monotonic()<until:pass
                cycles=[]
                for _ in range(10000):
                    row=[];row.append(row);cycles.append(row)
                del cycles; gc.collect()
            finally:sys.setswitchinterval(old_interval)
            import numpy as np
            ident.fingerprint('ARTIFICIAL',circuits.number(np.ones((256,256),dtype=np.complex128)))
            observer.poll()
        finally:observer.close()
        rows=[json.loads(line) for line in (EVIDENCE/'live-gil-gc.jsonl').read_text().splitlines()]
        polls=[r for r in rows if r['kind']=='observation']
        self.assertGreaterEqual(len(polls),4)
        self.assertTrue(all(r['first_failure'] is None for r in polls))
        self.assertTrue(all(r['interval_seconds']<=5 for r in polls))
        MEASUREMENTS['live_observer']=dict(observations=len(polls),max_rss=max(r['observer']['rss'] for r in polls),
            max_AS=max(r['observer']['address_space'] for r in polls),max_interval=max(r['interval_seconds'] for r in polls),
            driver_max_rss=max(r['processes'][0]['rss'] for r in polls),trace_bytes=(EVIDENCE/'live-gil-gc.jsonl').stat().st_size,
            shutdown_reaped=observer.process.poll() is not None)

    def test_delayed_observation_first_duration_stop(self):
        observer=self.start('live-io-delay')
        try:
            observer.phase('synthetic_io_delay')
            observer.request('synthetic_fault',fault='fast_period',value=0.05)
            observer.request('synthetic_fault',fault='io_delay',value=5.1)
            time.sleep(5.4)
            with self.assertRaises(ident.Stop):observer.poll()
        finally:observer.close(abort=True)
        rows=[json.loads(line) for line in (EVIDENCE/'live-io-delay.jsonl').read_text().splitlines()]
        stop=next(r for r in rows if r['kind']=='first_stop')
        self.assertEqual(stop['first_failure']['reason'],'observation_duration')
        self.assertEqual(observer.first_failure['reason'],'observation_duration')
        parent_stop=json.loads((EVIDENCE/'live-io-delay.jsonl.driver-first-stop.json').read_text())
        self.assertEqual(parent_stop['first_failure'],stop['first_failure'])
        MEASUREMENTS['io_delay_first_stop']=stop['first_failure']

    def test_observer_exit_is_fail_closed_and_reaped(self):
        observer=self.start('live-exit')
        try:
            self.assertTrue(observer.owner.terminate())
            observer.process.wait(timeout=2)
            with self.assertRaises(ident.Stop):observer.poll()
            first=observer.first_failure
            self.assertTrue((EVIDENCE/'live-exit.jsonl.driver-first-stop.json').is_file())
            with self.assertRaises(ident.Stop):observer.poll()
            self.assertIs(observer.first_failure,first)
        finally:observer.close(abort=True)
        self.assertIsNotNone(observer.process.poll())


if __name__=='__main__':unittest.main()
