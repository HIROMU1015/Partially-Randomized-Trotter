"""Release installation and existing scientific guards, using tiny Git fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_release_deployment import ensure_components,manifest_reference,validate_assignment_release
from trottertracks.rpe_q3_coordinator import digest


class ReleaseDeployment(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name);self.repository=self.directory/'repository';self.repository.mkdir()
        for args in [('init',),('config','user.name','Synthetic'),('config','user.email','fixture@example.invalid')]:self.git(*args)
        (self.repository/'scripts').mkdir();(self.repository/'scripts/leaf.py').write_text('# artificial metadata only\n')
        (self.repository/'src').mkdir();(self.repository/'src/provider.py').write_text('# never executed\n')
        self.git('add','.');self.git('commit','-qm','fixed artificial source');self.source=self.git('rev-parse','HEAD')
        self.manifest=dict(schema='RPE_RELEASE_COMPONENTS_V1',release_id='ARTIFICIAL_SELECTED',hosts=['local'],
            components=dict(optimizer=dict(source=self.source,required_paths=['scripts/leaf.py'],allow_metadata_commits=True),
                            q3=dict(source=self.source,required_paths=['scripts/leaf.py'])),
            stage_components={'q3':['q3']})
        self.layout=dict(host_id='local',repository=str(self.repository),
            component_roots={'optimizer':str(self.directory/'installed'),'q3':str(self.repository)},
            runtimes={'control':dict(python=sys.executable,modules={'json':None},components=['optimizer'])})

    def git(self,*args):
        return subprocess.check_output(['git',*args],cwd=self.repository,text=True,stderr=subprocess.DEVNULL).strip()

    def test_missing_component_installs_once_and_second_join_reuses_without_fetch(self):
        before=deepcopy((self.manifest,self.layout))
        first=ensure_components(self.manifest,self.layout,install=True)
        self.assertEqual(first['status'],'PASS');self.assertEqual(first['components']['optimizer']['status'],'INSTALLED')
        with patch('trottertracks.rpe_release_deployment._ensure_source',side_effect=AssertionError('Already installed')):
            second=ensure_components(self.manifest,self.layout,install=True)
        self.assertEqual(second['status'],'PASS');self.assertEqual(second['components']['optimizer']['status'],'REUSED')
        self.assertEqual(before,(self.manifest,self.layout));self.assertFalse(first['participation_modified'])

    def test_wrong_existing_checkout_and_input_are_held_without_overwriting(self):
        self.layout['component_roots']['optimizer']=str(self.repository)
        original=(self.repository/'scripts/leaf.py').read_bytes()
        (self.repository/'scripts/leaf.py').write_text('# changed executable code\n')
        saved=self.directory/'saved.json';saved.write_text('CORRECT_SAVED_INPUT')
        target=self.directory/'target.json';target.write_text('WRONG_EXISTING_INPUT')
        self.layout['input_files']={'fixed':dict(path=str(target),copy_from=str(saved),
            sha256=hashlib.sha256(saved.read_bytes()).hexdigest())}
        result=ensure_components(self.manifest,self.layout,install=True,components=['optimizer'])
        self.assertEqual(result['status'],'HOLD');self.assertEqual(len(result['holds']),2)
        self.assertEqual(target.read_text(),'WRONG_EXISTING_INPUT')
        self.assertNotEqual((self.repository/'scripts/leaf.py').read_bytes(),original)

    def test_metadata_commit_is_accepted_for_optimizer_but_not_fixed_scientific_checkout(self):
        self.layout['component_roots']['optimizer']=str(self.repository)
        (self.repository/'docs').mkdir();(self.repository/'docs/result.json').write_text('{}')
        self.git('add','docs');self.git('commit','-qm','metadata publication')
        control=ensure_components(self.manifest,self.layout,components=['optimizer'])
        science=ensure_components(self.manifest,self.layout,components=['q3'])
        self.assertEqual(control['status'],'PASS');self.assertEqual(science['status'],'HOLD')

    def test_missing_runtime_is_reasoned_hold_and_saved_input_only_copies_when_missing(self):
        self.layout['component_roots']['optimizer']=str(self.repository)
        self.layout['runtimes']['control']['modules']={'rpe_nonexistent_artificial_package':None}
        saved=self.directory/'saved.json';saved.write_text('UNCHANGED_SAVED_BYTES');target=self.directory/'new'/'input.json'
        self.layout['input_files']={'fixed':dict(path=str(target),copy_from=str(saved),
            sha256=hashlib.sha256(saved.read_bytes()).hexdigest())}
        result=ensure_components(self.manifest,self.layout,install=True,components=['optimizer'])
        self.assertEqual(result['status'],'HOLD');self.assertIn('RUNTIME_MODULES_MISSING',result['holds'][0]['reason'])
        self.assertEqual(target.read_bytes(),saved.read_bytes());self.assertEqual(result['inputs']['fixed']['status'],'INSTALLED')

    def test_failed_install_does_not_publish_a_partial_checkout(self):
        import trottertracks.rpe_release_deployment as api
        run=api.subprocess.run
        def fail_checkout(command,**kwargs):
            if command[:3]==['git','checkout','--detach']:raise subprocess.CalledProcessError(1,command)
            return run(command,**kwargs)
        with patch.object(api.subprocess,'run',side_effect=fail_checkout):
            result=ensure_components(self.manifest,self.layout,install=True,components=['optimizer'])
        self.assertEqual(result['status'],'HOLD');self.assertFalse(Path(self.layout['component_roots']['optimizer']).exists())
        self.assertFalse(list(self.directory.glob('.rpe-install-*')))

    def test_task_release_and_fixed_source_are_checked_without_running_leaf(self):
        request=dict(request_id='ARTIFICIAL_JOB',method='PR_PAULI',route={'kind':'artificial'})
        assignment=dict(request=request,request_sha256=digest(request),job_id='ARTIFICIAL_JOB',method='PR_PAULI',
            source_manifest=dict(commit=self.source,files={'scripts/leaf.py':hashlib.sha256((self.repository/'scripts/leaf.py').read_bytes()).hexdigest()}),
            input_hashes=[],required_release=manifest_reference(self.manifest))
        profile=dict(root=str(self.repository),interpreter=sys.executable,route_views={},host_id='local')
        checked=validate_assignment_release(self.manifest,self.layout,assignment,profile)
        self.assertEqual(checked['status'],'PASS');self.assertEqual(checked['scientific_calls'],0)
        for name,value in [('MANIFEST.json',self.manifest),('HOST.json',self.layout),('PROFILE.json',profile)]:
            (self.directory/name).write_text(json.dumps(value))
        command=[sys.executable,'-B',str(ROOT/'scripts/manage_rpe_release_deployment.py'),
            '--manifest',str(self.directory/'MANIFEST.json'),'--layout',str(self.directory/'HOST.json'),
            '--components','q3','--assignment-stdin','--profile',str(self.directory/'PROFILE.json')]
        process=subprocess.run(command,input=json.dumps(assignment),text=True,capture_output=True)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertEqual(json.loads(process.stdout)['assignment']['scientific_source'],self.source)
        wrong=deepcopy(assignment);wrong['required_release']['id']='ANOTHER_RELEASE'
        with self.assertRaisesRegex(ValueError,'Assigned release'):
            validate_assignment_release(self.manifest,self.layout,wrong,profile)
        rejected=subprocess.run(command,input=json.dumps(wrong),text=True,capture_output=True)
        self.assertEqual(rejected.returncode,2);self.assertEqual(json.loads(rejected.stdout)['status'],'HOLD')
        (self.repository/'scripts/leaf.py').write_text('# wrong scientific checkout\n')
        with self.assertRaisesRegex(ValueError,'component unavailable'):
            validate_assignment_release(self.manifest,self.layout,assignment,profile)


if __name__=='__main__':unittest.main()
