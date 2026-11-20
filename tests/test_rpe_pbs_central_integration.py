"""Combined RomuTail/PBS ownership and release pinning, synthetic transports only.

These checks never invoke SSH, PBS, a compiler, or a scientific adapter.  The
ordinary executor/factory/registry run unchanged; process and transport edges
are stubs so ownership failures can be checked deterministically.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
import run_rpe_method_optimizer as CLI
from trottertracks import rpe_optimizer_concurrent as Concurrent
from trottertracks import rpe_pbs as Pbs
from trottertracks import rpe_price_validation as Validation
from trottertracks import rpe_q3_coordinator as Coordinator
from trottertracks.rpe_local0_search import Gateway, load, write
from trottertracks.rpe_release_registry import register
from test_rpe_local0_search import fixture_config, request


class OwnershipTests(unittest.TestCase):
    def create(self, directory, *, relay=False, fail_local_start=False):
        root = Path(directory)
        run = root/'run'
        shared = root/'shared-central-relay'
        shared.mkdir()
        (shared/'RELAY.json').write_text('SHARED_RELAY_MUST_REMAIN_UNCHANGED\n')
        manager = Mock(retain_between_batches=False)
        manager.metrics.return_value = dict(artificial_only=True)
        service = Mock()
        server = SimpleNamespace(server_address=('127.0.0.1',18763),
                                 shutdown=Mock(), server_close=Mock())
        server_thread = SimpleNamespace(join=Mock())
        process = Mock(pid=123456)
        process.wait.return_value = 0
        distributed = dict(bind='127.0.0.1', port=0,
                           pbs=dict(workers=[], remote_root='/ARTIFICIAL_ONLY'))
        if relay:
            distributed['relay'] = dict(state_directory=str(shared), workers=[],
                                        request_timeout_seconds=1)
        if fail_local_start:
            distributed['local_worker'] = dict(host_id='local', slots=1,
                root=str(ROOT), interpreter=sys.executable, route_views={})
        spec = dict(methods=['PR_PAULI','PR_DF'], science_approved=False,
                    max_new_q3_per_method=dict(PR_PAULI=0, PR_DF=0),
                    executor=dict(scheduler='concurrent', max_concurrent_jobs=2,
                                  distributed=distributed))
        problem = dict(accuracy=dict(fixture='ARTIFICIAL'),
                       target=dict(energy_ha=0), rpe=dict(M=1))
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(CLI, 'head', return_value='ARTIFICIAL_MERGED_SOURCE'))
        stack.enter_context(patch.object(Coordinator, 'source_manifest', return_value=dict(
            commit='ARTIFICIAL_MERGED_SOURCE', files={})))
        stack.enter_context(patch.object(Coordinator, 'serve', return_value=(server,server_thread)))
        stack.enter_context(patch.object(Validation, 'Service', return_value=service))
        factory = stack.enter_context(patch.object(Pbs, 'PbsManager', return_value=manager))
        popen = stack.enter_context(patch.object(Concurrent.subprocess, 'Popen',
            side_effect=OSError('ARTIFICIAL local startup failure') if fail_local_start else None,
            return_value=process))
        values = dict(manager=manager, service=service, server=server,
                      factory=factory, popen=popen, process=process,
                      run=run, shared=shared, spec=spec, problem=problem)
        return values

    def exercise(self, relay):
        with tempfile.TemporaryDirectory() as directory:
            values = self.create(directory, relay=relay)
            executor, _ = CLI.make_executor(ROOT, values['run'],
                dict(PR_PAULI=[], PR_DF=[]), {}, values['problem'], values['spec'])
            try:
                values['factory'].assert_called_once()
                values['manager'].start.assert_called_once_with()
                if relay:
                    values['popen'].assert_called_once()
                    owned = values['run']/'distributed/relay'
                    configured = load(owned/'CONFIG.json')
                    self.assertEqual(configured['state_directory'], str(owned))
                    self.assertEqual(configured['controller']['local_port'],18763)
                else:
                    values['popen'].assert_not_called()
                    self.assertFalse((values['run']/'distributed/relay').exists())
                views = [executor.for_method(method) for method in ('PR_PAULI','PR_DF','PR_PAULI')]
                with ThreadPoolExecutor(max_workers=3) as pool:
                    list(pool.map(lambda view:view.finish(),views))
                self.assertFalse(executor.state.closed)
                self.assertFalse(executor.state.driver.closed)
                self.assertFalse(executor.state.monitor_stop.is_set())
                values['manager'].close.assert_not_called()
                values['service'].close.assert_not_called()
                values['server'].shutdown.assert_not_called()
                values['process'].wait.assert_not_called()
                if relay:
                    self.assertFalse((values['run']/'distributed/relay/STOP').exists())
                executor.finish()
                executor.finish()
                values['manager'].close.assert_called_once_with(retain_allocation=False)
                values['service'].close.assert_called_once_with()
                values['server'].shutdown.assert_called_once_with()
                values['server'].server_close.assert_called_once_with()
                self.assertTrue(executor.state.driver.closed)
                if relay:
                    self.assertTrue((values['run']/'distributed/relay/STOP').exists())
                    values['process'].wait.assert_called_once()
                self.assertFalse((values['shared']/'STOP').exists())
                self.assertEqual((values['shared']/'RELAY.json').read_text(),
                                 'SHARED_RELAY_MUST_REMAIN_UNCHANGED\n')
                self.assertEqual(executor.state.reserved, {})
            finally:
                executor.finish()

    def test_central_default_keeps_external_relay_and_closes_owned_pbs_once(self):
        self.exercise(False)

    def test_explicit_run_relay_isolated_and_closed_once_with_pbs(self):
        self.exercise(True)

    def test_startup_failure_after_pbs_start_closes_manager_and_rpc_once(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.create(directory, fail_local_start=True)
            with self.assertRaisesRegex(OSError, 'ARTIFICIAL local startup failure'):
                CLI.make_executor(ROOT, values['run'],dict(PR_PAULI=[],PR_DF=[]),{},
                                  values['problem'],values['spec'])
            values['manager'].start.assert_called_once_with()
            values['manager'].close.assert_called_once_with()
            values['server'].shutdown.assert_called_once_with()
            values['server'].server_close.assert_called_once_with()
            owner = values['factory'].call_args.args[0]
            self.assertTrue(owner.state.closed)
            self.assertTrue(owner.state.driver.closed)
            self.assertFalse((values['shared']/'STOP').exists())
            values['service'].close.assert_not_called()


class InactiveReleaseTests(unittest.TestCase):
    def test_merged_pbs_release_registration_keeps_current_and_old_run_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            services={}
            for name in ('central','merged-pbs'):
                home=root/name;home.mkdir()
                service=fixture_config(home)
                service.update(code_source='ARTIFICIAL_'+name,
                               computation_source='UNCHANGED_ARTIFICIAL_SCIENCE')
                if name=='merged-pbs':
                    spec=load(service['round'])
                    spec['executor'].update(scheduler='concurrent',distributed=dict(
                        pbs=dict(transport_source='ARTIFICIAL_MERGED_PBS',
                                 remote_root='/ARTIFICIAL_PBS_RUNS',workers=[])))
                    write(service['round'],spec)
                path=home/'SERVICE.json';write(path,service);services[name]=path
            registry=root/'RELEASES.json';state=root/'state'
            register(registry,'central',services['central'],activate=True,evidence='ARTIFICIAL_TEST')
            bootstrap=dict(release_registry=str(registry),state_directory=str(state))
            gateway=Gateway(bootstrap,preflight=lambda *args:dict(status='PASS',scientific_calls=0),
                            launch=lambda *args:(_ for _ in ()).throw(AssertionError('No launch allowed')))
            query=request(1,2,0);query['action']='prepare'
            old=gateway.handle(query);old_run=Path(old['run_directory'])
            original_pin=(old_run/'CONTROLLER_CONFIG.json').read_bytes()
            register(registry,'merged-pbs',services['merged-pbs'],activate=False,evidence='ARTIFICIAL_COMBINED_TEST')
            self.assertEqual(load(registry)['current'],'central')
            self.assertEqual(load(registry)['releases']['merged-pbs']['status'],'VALIDATED')
            selected=gateway.handle(dict(query,release='merged-pbs'))
            new_run=Path(selected['run_directory'])
            self.assertEqual(load(new_run/'RELEASE.json')['id'],'merged-pbs')
            pinned=load(new_run/'ROUND.requested.json')
            self.assertEqual(pinned['executor']['distributed']['pbs']['transport_source'],
                             'ARTIFICIAL_MERGED_PBS')
            self.assertEqual(pinned['max_new_q3_per_method'],dict(PR_PAULI=0))
            self.assertFalse((new_run/'LAUNCH.json').exists())
            self.assertEqual((old_run/'CONTROLLER_CONFIG.json').read_bytes(),original_pin)
            status=gateway.handle(dict(action='status',run_id=old['run_id']))
            self.assertEqual(status['release']['id'],'central')
            self.assertEqual(load(registry)['current'],'central')


if __name__=='__main__':
    unittest.main()
