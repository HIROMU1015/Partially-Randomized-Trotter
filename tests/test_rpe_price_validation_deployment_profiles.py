"""Metadata-only deployment package regression for the optional compile route.

Create the real package/profile outputs using synthetic host metadata. No host
connection, worker, numerical body or compiler may be started by this test.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import prepare_rpe_preparation_deployment as D
from trottertracks.rpe_preparation_worker import select_preparation_profile

SOURCE='a'*40
PAULI='b'*40
DF='c'*40
OLDPF='d8780090fbb7a66d914fdd45ee9c8b8346686493'
PF='36ad751d4ebf0ec160532615da6425b268a0c18a'


def task(method,provider='PRICE_MODEL_VALIDATION',legacy=PAULI):
    return dict(method=method,route=dict(kind='preparation'),preparation=dict(
        stage='compile',provider=provider,source=dict(commit=SOURCE,files={'scripts/run_rpe_price_validation_job.py':'1'*64}),
        workspaces=dict(legacy=dict(source=dict(commit=legacy,files={'scripts/old.py':'2'*64}))),
        required_paths={},inputs={},input_hashes=[],outputs=['RESULT.json']))


class Profiles(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'repo';self.root.mkdir()
        old=self.root/'docs/research/rpe_q3_auto_allocation/deployment_20261117'
        (old/'local0_tools').mkdir(parents=True);(old/'profiles').mkdir();(old/'assignments').mkdir()
        (self.root/'scripts').mkdir()
        (self.root/'scripts/run_rpe_preparation_deployment.py').write_text('# metadata transport stub\n')
        (old/'local0_tools/run_auto_deployment.py').write_text('# deployment helper stub\n')
        hosts=[dict(host_id='server1',repository='/synthetic/server1/repo',transport_python='python3',ssh_alias='server1'),
            dict(host_id='local',repository='/synthetic/local/repo',transport_python='python3',ssh_alias='local'),
            dict(host_id='local0-wsl',repository='/synthetic/wsl/repo',transport_python='python3',transport='wsl',distribution='Ubuntu')]
        (old/'local0_tools/PACKAGE_CONFIG.json').write_text(json.dumps(dict(hosts=hosts,participation_file='synthetic-Pause.json')))
        self.profiles={}
        for host in ('server2','server1','local','local0-wsl'):
            profile=dict(host_id=host,root='/synthetic/fixed-q3-'+host,interpreter='/synthetic/python-'+host,
                route_views={PAULI:'/synthetic/all-pauli',DF:'/synthetic/all-df'},
                slots='auto',slot_policy=dict(mode='auto',synthetic_marker=True),
                work_dir='/synthetic/old-spool',route_profiles=[],adapter_root='/synthetic/old-adapter',
                adapter_manifest=dict(commit='old',files={}),runtime=dict(version='old-q3-specific'))
            (old/'profiles'/f'{host}.json').write_text(json.dumps(profile));self.profiles[host]=profile
        for method in ('PR_PAULI','PR_DF'):
            (old/'assignments'/f'{method}.json').write_text(json.dumps(dict(method=method)))

    def create(self,tasks):
        def source_manifest(source,names):
            return dict(commit=source,files={name:hashlib.sha256(name.encode()).hexdigest()for name in names})
        def rev_parse(args,**kwargs):
            if args[:2]!=['git','rev-parse']:raise AssertionError('unexpected process '+str(args))
            return (OLDPF if args[2]=='d8780090f'else args[2])+'\n'
        def raw(root,source,name):return ('fixed bytes '+name).encode()
        with patch.object(D,'ROOT',self.root),patch.object(D,'source_manifest',side_effect=source_manifest),\
             patch.object(D,'raw',side_effect=raw),patch.object(D.subprocess,'check_output',side_effect=rev_parse),\
             patch.object(D.subprocess,'run',side_effect=AssertionError('deployment must not execute a process')):
            manifest=D.create(tasks,SOURCE,self.root/'package')
        return manifest,self.root/'package'

    def test_pauli_and_df_share_one_unambiguous_compile_profile(self):
        with patch('trottertracks.rpe_host_profiles.select_host_profile',side_effect=AssertionError('optional validation must not select a Q3-only profile')):
            manifest,out=self.create([task('PR_PAULI',legacy=PAULI),task('PR_DF',legacy=DF)])
        for host,original in self.profiles.items():
            profile=json.loads((out/'profiles'/f'{host}.json').read_text())
            rules=[r for r in profile['preparation_profiles']if r['provider']=='PRICE_MODEL_VALIDATION']
            self.assertEqual(len(rules),1);rule=rules[0]
            self.assertEqual((rule['stage'],rule['source_commit']),('compile',SOURCE))
            for request in (task('PR_PAULI',legacy=PAULI),task('PR_DF',legacy=DF)):
                selected=select_preparation_profile(profile,request)
                self.assertEqual(selected['route_views'],original['route_views'])
                self.assertEqual(selected['slot_policy'],original['slot_policy'])
                self.assertEqual(selected['slots'],'auto')
                self.assertEqual(set(selected['preparation_sources']),{SOURCE,PAULI,DF,PF,OLDPF})
                self.assertNotIn('runtime',selected)
                self.assertNotIn('route_profiles',selected)
                self.assertEqual(selected['preparation_adapter_manifest']['commit'],SOURCE)
        self.assertEqual(manifest['compile_calls'],0);self.assertEqual(manifest['scientific_calls'],0)
        self.assertEqual(manifest['queue_operations'],0)

    def test_worker_manifest_and_archive_include_the_validation_entry(self):
        manifest,out=self.create([task('PR_PAULI')])
        profile=json.loads((out/'profiles/server2.json').read_text())
        rule=profile['preparation_profiles'][0]
        self.assertIn('scripts/run_rpe_price_validation_job.py',rule['profile']['preparation_adapter_manifest']['files'])
        self.assertIn('profiles/server2.json',manifest['files'])
        with zipfile.ZipFile(out/'PREPARATION_DEPLOYMENT.zip')as archive:
            self.assertEqual(json.loads(archive.read('profiles/server2.json')),profile)
        self.assertFalse(any(self.root.glob('.preparation-distributed/*/spool')))

    def test_ordinary_pauli_compile_provider_is_still_generated(self):
        def selected(base,request):
            result=deepcopy(base);result['route_views']={'selected-only':'/synthetic/narrow-q3'}
            return result
        with patch('trottertracks.rpe_host_profiles.select_host_profile',side_effect=selected):
            _,out=self.create([task('PR_PAULI',provider='PAULI_COMPILE')])
        profile=json.loads((out/'profiles/server2.json').read_text())
        providers=[r['provider']for r in profile['preparation_profiles']]
        self.assertIn('PAULI_COMPILE',providers)
        self.assertNotIn('PRICE_MODEL_VALIDATION',providers)


if __name__=='__main__':unittest.main()
