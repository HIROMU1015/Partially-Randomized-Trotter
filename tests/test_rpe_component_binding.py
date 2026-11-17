"""Selected component versions and metadata-only resume; no scientific work."""
from pathlib import Path
import base64
import contextlib
import hashlib
import io
import json
import runpy
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'scripts')]
import test_rpe_release_registry as ReleaseFixture
from test_rpe_local0_price_validation_deployment import fixture
from trottertracks.rpe_release_registry import register,select,write,read
from trottertracks.rpe_local0_search import fixed_implementation,RequestError


class ComponentBinding(unittest.TestCase):
    def test_resolve_carries_manifest_and_selected_host_layout_without_launch(self):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary);root,_,_,config=fixture(base)
            # The bootstrap imports the actual module, while the test release
            # identity intentionally labels synthetic code only.
            config['code_root']=str(ROOT)
            manifest=dict(schema='RPE_RELEASE_COMPONENTS_V1',release_id='selected',
                components={'optimizer':{'source':config['code_source']}},stage_components={})
            layout={'host_id':'server2','component_roots':{'optimizer':'/OWNED/FIXED'}}
            write(base/'COMPONENTS.json',manifest);write(base/'SERVER2.json',layout)
            config.update(release_manifest=str(base/'COMPONENTS.json'),host_layouts={'server2':str(base/'SERVER2.json')})
            write(base/'SERVICE.json',config);register(base/'REGISTRY.json','selected',base/'SERVICE.json',activate=True)
            write(base/'BOOTSTRAP.json',dict(release_registry=str(base/'REGISTRY.json'),state_directory=config['state_directory']))
            envelope=dict(server_config=str(base/'BOOTSTRAP.json'),operation='resolve',host_id='server2',
                action='preview',request_base64=base64.b64encode(b'{"scope":{"methods":["PR_PAULI"]}}').decode())
            output=io.StringIO()
            with contextlib.redirect_stdout(output):runpy.run_path(str(ROOT/'tools/local0-search/remote.py'),init_globals={'CONFIG':envelope})
            resolved=json.loads(output.getvalue())['result']
            self.assertEqual(resolved['status'],'RELEASE_RESOLVED')
            self.assertEqual(resolved['components'],manifest);self.assertEqual(resolved['host_layout'],layout)
            self.assertEqual(resolved['release']['components']['id'],'selected')
            self.assertFalse((Path(config['state_directory'])/'runs').exists())
            manifest['components']['optimizer']['source']='OTHER'
            write(base/'COMPONENTS.json',manifest)
            with self.assertRaisesRegex(ValueError,'component versions changed'):
                select(read(base/'BOOTSTRAP.json'),dict(action='preview',scope={'methods':['PR_PAULI']}))

    def test_metadata_commit_does_not_hide_external_fixed_setting_change(self):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary);root,_,_,config=fixture(base)
            git=lambda *args:subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.name','Synthetic');git('config','user.email','fixture@example.invalid')
            git('add','src','scripts');git('commit','-qm','fixed synthetic implementation')
            config['code_source']=git('rev-parse','HEAD')
            setting=base/'FROZEN_PROFILE.json';setting.write_text('{"source":"FIXED"}\n')
            config['fixed_settings']={str(setting):hashlib.sha256(setting.read_bytes()).hexdigest()}
            (root/'docs').mkdir();(root/'docs/RESULT.json').write_text('{}')
            git('add','docs');git('commit','-qm','saved metadata')
            fixed_implementation(config)
            setting.write_text('{"source":"WRONG"}\n')
            with self.assertRaisesRegex(RequestError,'fixed deployment setting changed'):fixed_implementation(config)

    def test_adoption_refuses_changed_current_without_replacing_it(self):
        case=ReleaseFixture.ReleaseTests('test_unknown_action_is_rejected_before_run_creation');case.setUp()
        try:
            with self.assertRaisesRegex(ValueError,'Current release changed'):
                register(case.registry,'b',case.root/'b/SERVICE.json',activate=True,expected_current='stale')
            self.assertEqual(read(case.registry)['current'],'a')
        finally:case.doCleanups()


if __name__=='__main__':unittest.main()
