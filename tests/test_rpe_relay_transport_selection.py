"""Artificial transport processes; no providers or scientific calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('relay_selection', ROOT/'scripts/ensure_rpe_relay_transport.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TransportSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state = Path(self.temporary.name)
        self.run = self.state/'runs/ARTIFICIAL'
        self.directory = self.run/'distributed/relay'
        self.script = self.state/'selected.py'
        self.script.touch()
        self.config_path = self.directory/'CONFIG.json'
        write(self.config_path, {})
        write(self.state/'RUNS.json', {'ARTIFICIAL':dict(run_directory=str(self.run))})
        write(self.run/'LAUNCH.json', dict(invocation_id='invocation'))
        write(self.run/'NORMAL_DRIVER.json', dict(invocation_id='invocation', pid=100))
        self.config = dict(state_directory=str(self.state), python='ARTIFICIAL_PYTHON',
                           runs={'ARTIFICIAL':dict(entry=str(self.script))}, wait_seconds=1)
        self.result = dict(run_id='ARTIFICIAL', run_directory=str(self.run))
        self.processes = {100:['ARTIFICIAL_CONTROLLER'], 101:self.arguments('original.py')}
        self.status(101)

    def arguments(self, script):
        return ['python', '-B', str(script), '--config', str(self.config_path), '--run']

    def status(self, pid):
        write(self.directory/'STATUS.json', dict(pid=pid, configuration=str(self.config_path), stopped=False))

    def command(self, args, **kwargs):
        self.assertEqual(args[:3], ['ARTIFICIAL_PYTHON','-B',str(self.script)])
        if args[-1] == '--stop':
            self.processes.pop(101)
        elif args[-1] == '--start':
            self.processes[102] = self.arguments(self.script)
            self.status(102)
        else:
            self.fail('Unexpected operation: '+str(args))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    def test_replaces_only_transport_and_second_call_does_not_restart(self):
        source = self.run/'FROZEN_SOURCE'
        source.write_text('retain original science')
        unknown = self.run/'UNKNOWN'
        unknown.write_text('retain attempt')
        with patch.object(S, 'process_arguments', side_effect=lambda p:self.processes.get(p, [])), \
                patch.object(S.subprocess, 'run', side_effect=self.command) as command:
            changed = S.ensure(self.config, self.result)
            same = S.ensure(self.config, self.result)
        self.assertEqual(changed['action'], 'REPLACED')
        self.assertEqual(same['action'], 'ALREADY_CURRENT')
        self.assertEqual(same['pid'],102)
        self.assertEqual(command.call_count,2)
        self.assertEqual(source.read_text(),'retain original science')
        self.assertEqual(unknown.read_text(),'retain attempt')

    def test_user_stop_prevents_any_transport_restart(self):
        write(self.run/'STOP_REQUESTED.json', dict(reason='USER_REQUESTED_STOP'))
        with patch.object(S.subprocess,'run') as command:
            with self.assertRaisesRegex(ValueError,'Stop is pending'):
                S.ensure(self.config,self.result)
        command.assert_not_called()

    def test_foreign_process_configuration_is_never_stopped(self):
        self.processes[101] = ['python','original.py','--config','OTHER_RUN','--run']
        with patch.object(S,'process_arguments',side_effect=lambda p:self.processes.get(p,[])), \
                patch.object(S.subprocess,'run') as command:
            with self.assertRaisesRegex(ValueError,'does not own'):
                S.ensure(self.config,self.result)
        command.assert_not_called()

    def test_concurrent_normal_stop_lock_prevents_replacement(self):
        with (self.state/'REGISTRY.lock').open('a') as lock:
            S.fcntl.flock(lock,S.fcntl.LOCK_EX|S.fcntl.LOCK_NB)
            with patch.object(S.subprocess,'run') as command:
                with self.assertRaises(BlockingIOError):
                    S.ensure(self.config,self.result)
        command.assert_not_called()

    def test_unconfigured_run_is_not_touched(self):
        self.assertEqual(S.ensure({},dict(run_id='OTHER'))['status'],'NOT_CONFIGURED')


if __name__ == '__main__':
    unittest.main()
