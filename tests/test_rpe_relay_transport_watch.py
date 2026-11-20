"""Per-invocation relay watching against artificial process generations only."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_rpe_relay_transport_selection as Fixture
S=Fixture.S
write=Fixture.write


class RelayTransportWatchTests(unittest.TestCase):
    def setUp(self):
        # Reuse the selection test's existing owned run/process fixture without
        # inheriting its tests or introducing any scientific process.
        fixture=Fixture.TransportSelectionTests('test_unconfigured_run_is_not_touched')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture=fixture
        self.configuration=fixture.state/'TRANSPORT.json'
        write(self.configuration,fixture.config)

    def test_two_batch_generations_are_replaced_but_no_between_batch_relay_is_started(self):
        f=self.fixture
        replaced=[]
        steps=[0]
        def select(config,result):
            replaced.append(json.loads((f.directory/'STATUS.json').read_text())['pid'])
            pid=replaced[-1]
            f.processes[pid]=f.arguments(f.script)
            return dict(status='RELAY_TRANSPORT_CONFIRMED',pid=pid)
        def advance(seconds):
            self.assertEqual(seconds,1)
            steps[0]+=1
            if steps[0]==1:
                f.processes.pop(101)
                write(f.directory/'STATUS.json',dict(pid=101,stopped=True))
            elif steps[0]==2:
                f.processes[201]=f.arguments('original.py')
                f.status(201)
            else:
                write(f.run/'STOP_REQUESTED.json',dict(reason='USER_REQUESTED_STOP'))
        with patch.object(S,'process_arguments',side_effect=lambda pid:f.processes.get(pid,[])), \
                patch.object(S,'ensure',side_effect=select),patch.object(S.time,'sleep',side_effect=advance):
            result=S.watch(self.configuration,'ARTIFICIAL','invocation')
        self.assertEqual(replaced,[101,201])
        self.assertEqual(result['reason'],'USER_STOP_REQUESTED')
        self.assertEqual(result['status'],'STOPPED')

    def test_stop_invocation_change_and_driver_exit_each_prevent_replacement(self):
        f=self.fixture
        for reason in ['USER_STOP_REQUESTED','INVOCATION_CHANGED','DRIVER_NOT_ACTIVE']:
            with self.subTest(reason=reason):
                (f.run/'STOP_REQUESTED.json').unlink(missing_ok=True)
                write(f.run/'LAUNCH.json',dict(invocation_id='invocation'))
                f.processes[100]=['ARTIFICIAL_CONTROLLER']
                if reason=='USER_STOP_REQUESTED':write(f.run/'STOP_REQUESTED.json',{})
                elif reason=='INVOCATION_CHANGED':write(f.run/'LAUNCH.json',dict(invocation_id='next'))
                else:f.processes.pop(100)
                with patch.object(S,'process_arguments',side_effect=lambda pid:f.processes.get(pid,[])), \
                        patch.object(S,'ensure')as ensure:
                    result=S.watch(self.configuration,'ARTIFICIAL','invocation')
                ensure.assert_not_called()
                self.assertEqual(result['reason'],reason)

    def test_ensure_watch_uses_one_owned_process_and_stop_prevents_new_start(self):
        f=self.fixture
        selection=dict(status='RELAY_TRANSPORT_CONFIRMED',invocation_id='invocation')
        commands=[]
        def launch(command,**options):
            commands.append(command)
            f.processes[300]=command
            return SimpleNamespace(pid=300)
        with patch.object(S,'process_arguments',side_effect=lambda pid:f.processes.get(pid,[])), \
                patch.object(S.subprocess,'Popen',side_effect=launch):
            first=S.ensure_watch(self.configuration,f.result,selection)
            second=S.ensure_watch(self.configuration,f.result,selection)
            self.assertEqual(len(commands),1)
            self.assertEqual(first['pid'],second['pid'])
            self.assertEqual(second['status'],'ALREADY_CURRENT')
            f.processes.pop(300)
            write(f.run/'STOP_REQUESTED.json',{})
            with self.assertRaisesRegex(ValueError,'USER_STOP_REQUESTED'):
                S.ensure_watch(self.configuration,f.result,selection)
            self.assertEqual(len(commands),1)
        self.assertIsNone(S.ensure_watch(self.configuration,f.result,dict(status='NOT_CONFIGURED')))

    def test_replacement_failure_stays_pending_and_stop_exits_without_forcing_retry(self):
        f=self.fixture
        def stop(seconds):write(f.run/'STOP_REQUESTED.json',{})
        with patch.object(S,'process_arguments',side_effect=lambda pid:f.processes.get(pid,[])), \
                patch.object(S,'ensure',side_effect=TimeoutError('relay selection pending'))as ensure, \
                patch.object(S.time,'sleep',side_effect=stop):
            result=S.watch(self.configuration,'ARTIFICIAL','invocation')
        self.assertEqual(ensure.call_count,1)
        self.assertEqual(result['reason'],'USER_STOP_REQUESTED')
        self.assertEqual(result['error'],'relay selection pending')

    def test_child_waits_for_parent_to_publish_then_observes_stop_without_starting_relay(self):
        f=self.fixture
        with (f.directory/'WATCH.lock').open('a')as lock:
            S.fcntl.flock(lock,S.fcntl.LOCK_EX)
            child=subprocess.Popen([sys.executable,'-B',str(Path(S.__file__).resolve()),
                '--config',str(self.configuration),'--watch','--run-id','ARTIFICIAL',
                '--invocation','invocation'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                # A nonblocking child would exit here before the parent can
                # publish WATCH.json; a blocking child retains the same PID.
                with self.assertRaises(subprocess.TimeoutExpired):child.wait(timeout=.2)
                write(f.run/'STOP_REQUESTED.json',{})
                S.fcntl.flock(lock,S.fcntl.LOCK_UN)
                output,error=child.communicate(timeout=5)
                self.assertEqual(child.returncode,0,error)
                self.assertEqual(json.loads(output)['reason'],'USER_STOP_REQUESTED')
            finally:
                if child.poll()is None:child.kill()
                child.communicate()


if __name__=='__main__':unittest.main()
