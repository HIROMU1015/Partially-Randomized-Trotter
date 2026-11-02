"""Bounded mocked pidfd races. No real signals, workers or scientific data."""
import errno
import json
import os
from pathlib import Path
import signal
import threading
import unittest
from unittest.mock import Mock,patch
from trottertracks.resource_applicability.h4_geometry import observer as obs,workers
from trottertracks.resource_applicability.h4_geometry.identity import Stop

EVIDENCE=Path(os.environ['H4_PRELAUNCH_TEST_EVIDENCE'])


def owner(pid,fd,parent=500):
    value=obs.OwnedIdentity.__new__(obs.OwnedIdentity)
    value.fd=fd;value.expected=dict(pid=pid,start=str(pid),parent=parent,uid=os.getuid())
    value.sampler=lambda _:dict(value.expected,rss=1024,address_space=4096)
    return value


def race():return ProcessLookupError(errno.ESRCH,'ARTIFICIAL already-exited owned process')


class ESRCHTests(unittest.TestCase):
    def patterns(self,after_parent_exit=False):
        for missing in ({0},{1},{0,1,2}):
            with self.subTest(route='reparented' if after_parent_exit else 'direct',missing=sorted(missing)):
                values=[owner(501+i,101+i) for i in range(3)];parent=owner(500,100,parent=1)
                def ready(read,*_):return (read if after_parent_exit and read==[100] else [],[],[])
                effects=[race() if i in missing else None for i in range(3)]
                with patch.object(obs.select,'select',side_effect=ready),patch.object(obs.signal,'pidfd_send_signal',side_effect=effects) as send:
                    results=[v.terminate_after_parent_exit(parent) if after_parent_exit else v.terminate() for v in values]
                self.assertEqual(results,[i not in missing for i in range(3)])
                self.assertEqual([c.args[0] for c in send.call_args_list],[101,102,103])

    def test_direct_first_middle_all_esrch(self):self.patterns()
    def test_reparented_first_middle_all_esrch(self):self.patterns(True)

    def errors(self,after_parent_exit=False):
        value=owner(501,101);parent=owner(500,100,parent=1)
        def ready(read,*_):return (read if after_parent_exit and read==[100] else [],[],[])
        for code in (errno.EPERM,errno.EBADF,errno.EINVAL):
            with self.subTest(error=code),patch.object(obs.select,'select',side_effect=ready),\
                 patch.object(obs.signal,'pidfd_send_signal',side_effect=OSError(code,'ARTIFICIAL non-ESRCH')):
                with self.assertRaises(OSError) as caught:
                    value.terminate_after_parent_exit(parent) if after_parent_exit else value.terminate()
                self.assertEqual(caught.exception.errno,code)

    def test_direct_non_esrch_is_not_success(self):self.errors()
    def test_reparented_non_esrch_is_not_success(self):self.errors(True)

    def test_ownership_mismatch_both_routes_never_signalled(self):
        parent=owner(500,100,parent=1)
        for after in (False,True):
            for field in ('uid','pid','start'):
                value=owner(501,101)
                changed=dict(value.expected);changed[field]='wrong' if field=='start' else changed[field]+1
                value.sampler=lambda _:dict(changed,rss=1,address_space=1)
                def ready(read,*_):return (read if after and read==[100] else [],[],[])
                with patch.object(obs.select,'select',side_effect=ready),patch.object(obs.signal,'pidfd_send_signal') as send:
                    self.assertFalse(value.terminate_after_parent_exit(parent) if after else value.terminate())
                    send.assert_not_called()

    def test_parent_mismatch_while_original_parent_alive_never_signalled(self):
        value=owner(501,101);parent=owner(500,100,parent=1)
        value.sampler=lambda _:dict(value.expected,parent=999,rss=1,address_space=1)
        with patch.object(obs.select,'select',return_value=([],[],[])),patch.object(obs.signal,'pidfd_send_signal') as send:
            self.assertFalse(value.terminate());self.assertFalse(value.terminate_after_parent_exit(parent));send.assert_not_called()

    def test_double_termination_after_exit_and_close_is_idempotent(self):
        value=owner(501,101)
        with patch.object(obs.select,'select',side_effect=[([],[],[]),([101],[],[])]),\
             patch.object(obs.signal,'pidfd_send_signal',side_effect=race()) as send,patch.object(obs.os,'close') as close:
            self.assertFalse(value.terminate());self.assertFalse(value.terminate())
            value.close();value.close();self.assertFalse(value.terminate())
            send.assert_called_once_with(101,signal.SIGTERM);close.assert_called_once_with(101)

    def test_success_still_uses_requested_signal_and_verified_pidfd(self):
        value=owner(501,101)
        with patch.object(obs.select,'select',return_value=([],[],[])),patch.object(obs.signal,'pidfd_send_signal') as send:
            self.assertTrue(value.terminate(signal.SIGKILL));send.assert_called_once_with(101,signal.SIGKILL)


class CallerCleanupTests(unittest.TestCase):
    def client(self):
        c=obs.IndependentObserver.__new__(obs.IndependentObserver)
        c.children={501:owner(501,101),502:owner(502,102),503:owner(503,103)}
        c.owner=owner(504,104);c.sock=Mock();c.process=Mock();c.process.poll.return_value=0
        c.lock=threading.Lock();c.first_failure=None;c.last_phase=dict(sequence=1,name='ARTIFICIAL_cleanup',monotonic=1)
        return c

    def test_stop_children_visits_all_three_after_first_middle_all_esrch(self):
        for missing in ({0},{1},{0,1,2}):
            c=self.client()
            with patch.object(obs.select,'select',return_value=([],[],[])),\
                 patch.object(obs.signal,'pidfd_send_signal',side_effect=[race() if i in missing else None for i in range(3)]) as send:
                c.stop_children()
            self.assertEqual([call.args[0] for call in send.call_args_list],[101,102,103])

    def test_observer_close_reaps_and_closes_all_fds_then_double_close(self):
        c=self.client()
        with patch.object(obs.select,'select',return_value=([],[],[])),\
             patch.object(obs.signal,'pidfd_send_signal',side_effect=lambda *_:(_ for _ in ()).throw(race())) as send,\
             patch.object(obs.os,'close') as close:
            c.close(abort=True);c.close(abort=True)
        self.assertEqual([call.args[0] for call in send.call_args_list],[101,102,103,104])
        self.assertEqual([call.args[0] for call in close.call_args_list],[101,102,103,104])
        self.assertEqual(c.children,{});self.assertIsNone(c.owner.fd)
        self.assertEqual(c.process.wait.call_count,2);c.sock.close.assert_called()

    def test_pool_wait_pipe_executor_cleanup_after_esrch(self):
        for missing in ({0},{1},{0,1,2}):
            c=self.client();pool=workers.OwnedPool.__new__(workers.OwnedPool)
            pool.monitor=c;pool.processes=[Mock() for _ in range(3)];pool.io=Mock()
            for process in pool.processes:process.wait.return_value=0
            with patch.object(obs.select,'select',return_value=([],[],[])),\
                 patch.object(obs.signal,'pidfd_send_signal',side_effect=[race() if i in missing else None for i in range(3)]):
                pool.shutdown()
            for process in pool.processes:
                process.wait.assert_called_once_with(timeout=2);process.kill.assert_not_called()
                process.stdin.close.assert_called_once();process.stdout.close.assert_called_once()
            pool.io.shutdown.assert_called_once_with(wait=True,cancel_futures=True)

    def test_first_stop_reason_preserved_when_all_cleanup_sends_esrch(self):
        c=self.client();c.trace_path=EVIDENCE/'p1-first-reason.jsonl';c.trace_path.write_bytes(b'')
        c.sock.send.side_effect=Stop('ARTIFICIAL original observation failure')
        with patch.object(obs.select,'select',return_value=([],[],[])),\
             patch.object(obs.signal,'pidfd_send_signal',side_effect=lambda *_:(_ for _ in ()).throw(race())):
            with self.assertRaises(Stop):c.request('ping')
            first=c.first_failure
            self.assertIn('original observation failure',first['reason'])
            c.sock.send.side_effect=BrokenPipeError(errno.EPIPE,'ARTIFICIAL secondary pipe error')
            with self.assertRaises(Stop):c.request('ping')
            self.assertIs(c.first_failure,first)
        saved=json.loads(Path(str(c.trace_path)+'.driver-first-stop.json').read_text())
        self.assertEqual(saved['first_failure'],first)

    def test_non_esrch_stop_children_still_raises(self):
        c=self.client()
        with patch.object(obs.select,'select',return_value=([],[],[])),\
             patch.object(obs.signal,'pidfd_send_signal',side_effect=PermissionError(errno.EPERM,'ARTIFICIAL permission failure')):
            with self.assertRaises(PermissionError):c.stop_children()

    def test_observer_terminal_cleanup_continues_after_reparented_esrch(self):
        c=self.client();driver=owner(os.getppid(),100,parent=1);children=list(c.children.values())
        for child in children:child.expected['parent']=driver.expected['pid']
        config=dict(scope='SYNTHETIC_ONLY',runtime_authorization=False,driver=driver.expected,
                    output_cap=65536,prior_wall=0,wall_started=100,workers=3)
        messages=[config,*[dict(command='own',identity=v.expected) for v in children],Stop('ARTIFICIAL first observer failure')]
        baseline=dict(oom_events={},observed_at=100,available=32*2**30,psi_full_avg10=0)
        def ready(read,*_):
            if read==[c.sock]:return (read,[],[])
            return (read if read==[100] else [],[],[])
        with patch.object(obs.socket,'socket',return_value=c.sock),patch.object(obs,'receive',side_effect=messages),\
             patch.object(obs,'OwnedIdentity',side_effect=[driver,*children]),patch.object(driver,'sample',return_value=dict(driver.expected,rss=1,address_space=1)),\
             patch.object(obs,'process_sample',return_value=dict(driver.expected,rss=1,address_space=1)),\
             patch.object(obs,'observe_memory',return_value=baseline),patch.object(obs.time,'monotonic',return_value=100),\
             patch.object(obs.select,'select',side_effect=ready),patch.object(obs,'send'),patch.object(obs,'Trace') as trace,\
             patch.object(obs.resource,'setrlimit'),patch.object(obs.os,'close') as close,\
             patch.object(obs.signal,'pidfd_send_signal',side_effect=[race(),None,race()]) as send:
            obs.observer_main(1000,1001)
        self.assertEqual([call.args[0] for call in send.call_args_list],[101,102,103])
        self.assertIn('first observer failure',trace.return_value.write.call_args.args[0]['first_failure']['reason'])
        self.assertEqual({call.args[0] for call in close.call_args_list},{100,101,102,103,1001})
        c.sock.close.assert_called_once()


if __name__=='__main__':unittest.main()
