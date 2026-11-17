"""Release switching and run pinning with metadata/synthetic launch stubs only."""
from copy import deepcopy
from pathlib import Path
import json
import base64
import contextlib
import io
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks.rpe_release_registry import register, select, load_service
from trottertracks.rpe_local0_search import Gateway, RequestError, load, write
from test_rpe_local0_search import fixture_config, request


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        (self.root/'a').mkdir();(self.root/'b').mkdir()
        self.a=fixture_config(self.root/'a');self.b=fixture_config(self.root/'b')
        self.a.update(code_source='ARTIFICIAL_A',computation_source='FROZEN_SCIENCE_A')
        self.b.update(code_source='ARTIFICIAL_B',computation_source='FROZEN_SCIENCE_B')
        self.registry=self.root/'RELEASES.json';self.state=self.root/'state'
        for label,config in [('a',self.a),('b',self.b)]:
            path=self.root/label/'SERVICE.json';write(path,config)
            register(self.registry,label,path,activate=label=='a',evidence='ARTIFICIAL_UNIT_TESTS')
        self.bootstrap=dict(release_registry=str(self.registry),state_directory=str(self.state))
        self.launched=[]

    def gateway(self):
        def preflight(run,problem,spec):return dict(status='PASS',scientific_calls=0)
        def launch(run,problem,spec):
            pinned=load(run/'CONTROLLER_CONFIG.json')
            self.launched.append(pinned['code_source'])
            write(run/'LAUNCH.json',dict(pid=0,invocation_id=str(len(self.launched))))
            write(run/'EXIT.json',dict(invocation_id=str(len(self.launched)),returncode=0))
        return Gateway(self.bootstrap,preflight=preflight,launch=launch)

    def switch(self):
        register(self.registry,'b',self.root/'b/SERVICE.json',activate=True,evidence='ARTIFICIAL_UNIT_TESTS')

    def test_new_preview_prepare_start_pin_the_same_release_and_inputs(self):
        gateway=self.gateway();q=request(4,6);q['action']='preview'
        preview=gateway.handle(q)
        q['action']='prepare';prepared=gateway.handle(q)
        q['action']='start';started=gateway.handle(q)
        self.assertEqual(preview['run_id'],prepared['run_id'])
        self.assertEqual(preview['run_id'],started['run_id'])
        run=Path(started['run_directory'])
        self.assertEqual(load(run/'RELEASE.json')['id'],'a')
        self.assertEqual(load(run/'CONTROLLER_CONFIG.json')['computation_source'],'FROZEN_SCIENCE_A')
        self.assertEqual(self.launched,['ARTIFICIAL_A'])

    def test_resume_and_status_keep_run_source_after_current_switch(self):
        old=self.gateway().handle(request(4,6));run=Path(old['run_directory'])
        before=(run/'ROUND.requested.json').read_bytes()
        self.switch()
        status=self.gateway().handle(dict(action='status',run_id=old['run_id']))
        self.assertEqual(status['release']['id'],'a')
        resumed=self.gateway().handle(dict(action='resume',run_id=old['run_id'],instruction='Artificial resume'))
        self.assertEqual(resumed['controller_source'],'ARTIFICIAL_A')
        self.assertEqual(resumed['computation_source'],'FROZEN_SCIENCE_A')
        self.assertEqual(before,(run/'ROUND.requested.json').read_bytes())
        self.assertEqual(self.launched,['ARTIFICIAL_A','ARTIFICIAL_A'])
        q=request(4,6);q['action']='preview'
        self.assertEqual(self.gateway().handle(q)['release']['id'],'b')

    def test_explicit_release_and_registered_input_change_rejection(self):
        self.switch();q=request(4,6);q.update(action='preview',release='a')
        self.assertEqual(self.gateway().handle(q)['release']['id'],'a')
        p=Path(self.a['problem']);value=load(p);value['target']['energy_ha']=1;write(p,value)
        with self.assertRaisesRegex(RequestError,'release input changed'):
            self.gateway().handle(q)

    def test_repeated_completed_start_has_no_extra_launch(self):
        first=self.gateway().handle(request(4,6));run=Path(first['run_directory'])
        write(run/'ROUND_RESULT.json',dict(methods=dict(PR_PAULI=dict(stop_reason='NO_IMPROVING_PROPOSAL'))))
        before=len(self.launched)
        second=self.gateway().handle(request(4,6))
        self.assertEqual(second['status'],'REUSED_COMPLETED')
        self.assertEqual(len(self.launched),before)

    def test_stop_marks_run_without_signalling_unknown_pid(self):
        started=self.gateway().handle(request(4,6));run=Path(started['run_directory'])
        with patch('trottertracks.rpe_local0_search.os.kill')as kill:
            result=self.gateway().handle(dict(action='stop',run_id=started['run_id']))
        kill.assert_not_called()
        self.assertEqual(result['status'],'STOPPED_NO_ACTIVE_DRIVER')
        self.assertEqual(load(run/'STOP_REQUESTED.json')['reason'],'USER_REQUESTED_STOP')

    def test_migrated_controller_cannot_start_new_work(self):
        self.state.mkdir(parents=True,exist_ok=True)
        write(self.state/'MIGRATED_TO.json',dict(controller='RomuTail'))
        with self.assertRaisesRegex(RequestError,'state moved'):
            self.gateway().handle(request(4,6))
        q=request(4,6);q['action']='preview'
        self.assertEqual(self.gateway().handle(q)['status'],'PREVIEW')

    def test_planonly_uses_pinned_normal_entry_same_version(self):
        gateway=self.gateway();q=request(4,6);q['action']='preview'
        preview=gateway.handle(q);q['action']='planonly'
        with patch('trottertracks.rpe_local0_search.subprocess.run')as run:
            run.return_value.returncode=0
            planned=gateway.handle(q)
        self.assertEqual(planned['run_id'],preview['run_id'])
        command=run.call_args.args[0]
        self.assertIn('--plan-only',command)
        self.assertEqual(command[2],str(Path(self.a['code_root'])/'scripts/run_rpe_method_optimizer.py'))
        self.assertEqual(planned['release']['id'],'a')
        self.assertFalse((Path(planned['run_directory'])/'LAUNCH.json').exists())

    def test_client_bootstrap_ignores_stale_code_root_and_planonly_pointer(self):
        service=self.root/'SERVICE.json';write(service,self.bootstrap)
        q=request(4,6);q['action']='preview'
        transport=dict(server_config=str(service),code_root='/OLD_FIXED_CODE_ROOT_MUST_NOT_BE_IMPORTED',
            plan_config='/OLD_PLAN_ONLY_SOURCE_MUST_NOT_BE_USED',action='preview',
            request_base64=base64.b64encode(json.dumps(q).encode()).decode())
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(ROOT/'tools/local0-search/remote.py'),init_globals={'CONFIG':transport})
        result=json.loads(output.getvalue())
        self.assertTrue(result['ok'])
        self.assertEqual(result['result']['release']['id'],'a')
        self.assertEqual(result['result']['controller_source'],'ARTIFICIAL_A')

    def test_explicit_resume_archives_transient_stop_marker(self):
        started=self.gateway().handle(request(4,6));run=Path(started['run_directory'])
        write(run/'STOP_REQUESTED.json',dict(reason='USER_REQUESTED_STOP'))
        self.gateway().handle(dict(action='resume',run_id=started['run_id'],instruction='Artificial resume'))
        self.assertFalse((run/'STOP_REQUESTED.json').exists())
        self.assertEqual(len(list((run/'stop-history').glob('*.json'))),1)

    def test_unknown_action_is_rejected_before_run_creation(self):
        q=request(4,6);q['action']='launch_everything'
        with self.assertRaisesRegex(RequestError,'preview, planonly'):
            self.gateway().handle(q)
        self.assertFalse((self.state/'runs').exists())

    def test_artificial_planonly_and_real_input_rejection(self):
        fixture=self.root/'synthetic-fixture';fixture.mkdir()
        write(fixture/'ROUND.json',dict(artificial_fixture=dict(only=True)))
        config=deepcopy(self.a);config['artificial_controller_directory']=str(fixture)
        gateway=Gateway(config,preflight=lambda *args:dict(status='PASS'),launch=lambda *args:None)
        q=request(4,6);q['action']='planonly'
        with patch('trottertracks.rpe_local0_search.subprocess.run')as child:
            child.return_value.returncode=0
            planned=gateway.handle(q)
        command=child.call_args.args[0]
        self.assertIn('--plan-only',command)
        self.assertIn('--control-run',command)
        self.assertTrue(command[2].endswith('check_rpe_romutail_distributed.py'))
        with self.assertRaisesRegex(RequestError,'artificial problem'):
            gateway.normal_preflight(Path(planned['run_directory']),dict(problem=dict(system='H4_REAL')), {})


if __name__=='__main__':unittest.main(verbosity=2)
