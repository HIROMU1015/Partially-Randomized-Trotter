"""Current host metadata must preserve every declared SOURCE/path/input role."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_rpe_preparation_transport as Fixture
from trottertracks import rpe_preparation_worker as P


class PreparationFingerprintContractTests(unittest.TestCase):
    setUp=Fixture.PreparationTransportTests.setUp
    tearDown=Fixture.PreparationTransportTests.tearDown
    profile=Fixture.PreparationTransportTests.profile
    task=Fixture.PreparationTransportTests.task

    def test_installed_metadata_does_not_substitute_for_required_import(self):
        with self.assertRaisesRegex(ValueError, 'Preparation runtime/package probe failed'):
            P._runtime({'interpreter':sys.executable},
                       {'packages':{'pip':None}, 'imports':['rpe_missing_artificial_provider']})

    def example(self):
        inner=self.root/'inner';inner_source=Fixture.repo(inner)
        config=self.root/'config.json';config.write_text('{"fixed":1}')
        provider=self.root/'provider';provider.mkdir();(provider/'provider.py').write_text('FIXED=1\n')
        data_sha=hashlib.sha256(b'ARTIFICIAL_WORKING_INPUT').hexdigest()
        required_paths={'config':dict(controller_path='/old/config.json',kind='file',sha256=hashlib.sha256(config.read_bytes()).hexdigest()),
            'provider':dict(controller_path='/old/provider',kind='directory',files={'provider.py':hashlib.sha256((provider/'provider.py').read_bytes()).hexdigest()})}
        task=self.task('audit','pass',inputs={'working/request.json':data_sha},
            working_input_hashes=[dict(path='working/request.json',sha256=data_sha)],
            required_views={'inner':dict(source=inner_source,controller_path='/old/inner')},required_paths=required_paths)
        profile=self.profile('server1')
        profile['preparation_sources']={self.source['commit']:str(self.root/'source'),inner_source['commit']:str(inner)}
        profile['preparation_paths']={'config':str(config),'/old/provider':{'path':str(provider)}}
        return task,profile,P.current_preparation_fingerprint(task,profile)

    def test_current_view_working_input_and_private_path_roles_match(self):
        task,profile,fp=self.example()
        self.assertEqual(P.validate_preparation_fingerprint(task,profile,fp),P.preparation_profile_core(P.select_preparation_profile(profile,task)))
        changes=[('required_views',lambda value:value['required_views']['inner']['source']['files'].__setitem__('leaf.py','0'*64)),
            ('working_input_hashes',lambda value:value['working_input_hashes'][0].__setitem__('sha256','0'*64)),
            ('required_paths',lambda value:value['required_paths']['config'].__setitem__('sha256','0'*64)),
            ('runtime_paths file',lambda value:value['runtime_paths']['config'].__setitem__('sha256','0'*64)),
            ('runtime_paths directory',lambda value:value['runtime_paths']['provider']['files'].__setitem__('provider.py','0'*64)),
            ('runtime_paths mapping',lambda value:value['runtime_paths']['provider'].__setitem__('path','/different/provider')),
            ('required source view',lambda value:value['views'].__setitem__('view:inner','/different/source')),
            ('profile role map',lambda value:value['profile']['preparation_paths'].__setitem__('config','/different/config.json'))]
        for label,change in changes:
            with self.subTest(label=label):
                altered=deepcopy(fp);change(altered)
                with self.assertRaises(ValueError):P.validate_preparation_fingerprint(task,profile,altered)
        for key in ['working_input_hashes','required_views','required_paths','runtime_paths']:
            with self.subTest(missing_key=key):
                old=deepcopy(fp);old.pop(key)
                with self.assertRaises(ValueError):P.validate_preparation_fingerprint(task,profile,old)
        self.assertFalse(Path(profile['work_dir']).exists())

    def test_working_input_declaration_cannot_differ_from_task_artifact(self):
        task,profile,fp=self.example()
        task['request']['preparation']['working_input_hashes'][0]['sha256']='0'*64
        task['request_sha256']=P.digest(task['request'])
        fp['request_sha256']=task['request_sha256'];fp['working_input_hashes']=deepcopy(task['request']['preparation']['working_input_hashes'])
        with self.assertRaisesRegex(ValueError,'WORKING input is not bound'):
            P.validate_preparation_fingerprint(task,profile,fp)
        self.assertFalse(Path(profile['work_dir']).exists())

if __name__=='__main__':unittest.main()
