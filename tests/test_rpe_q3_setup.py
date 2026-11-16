"""Mailbox bootstrap via artificial SSH and temporary Git; no scientific leaf."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('_mailbox_setup',ROOT/'scripts/setup_rpe_q3_mailbox.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


def git(directory,*args):
    return subprocess.check_output(['git','-C',str(directory),*args],text=True,stderr=subprocess.DEVNULL).strip()


class MailboxSetup(unittest.TestCase):
    def test_fake_ssh_fixed_sparse_checkout_matching_profile_and_idempotence(self):
        with tempfile.TemporaryDirectory(prefix='q3-setup-artificial-')as temporary:
            base=Path(temporary);repo=base/'repo';origin=base/'origin.git';repo.mkdir()
            subprocess.run(['git','init','--bare',str(origin)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            git(repo,'init');git(repo,'config','user.name','Artificial Fixture');git(repo,'config','user.email','fixture@example.invalid')
            worker=repo/'scripts/run_rpe_q3_worker.py';worker.parent.mkdir()
            worker.write_text('''import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--profile');p.add_argument('--mailbox');a=p.parse_args()
profile=json.load(open(a.profile))
print(json.dumps(dict(host_id=profile['host_id'],slots=profile['slots'],attempts=[],cached_prefixes=[],
source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip())))
''')
            (repo/'src').mkdir();(repo/'src/artificial.py').write_text('ARTIFICIAL=True\n')
            (repo/'large-history').mkdir();(repo/'large-history/omitted').write_text('not needed\n')
            git(repo,'add','.');git(repo,'commit','-m','Artificial mailbox fixture')
            branch='codex/rpe-distributed-q3-coordinator';git(repo,'branch','-M',branch)
            git(repo,'remote','add','origin',str(origin));git(repo,'push','-u','origin',branch)
            source=git(repo,'rev-parse','HEAD');original_status=git(repo,'status','--porcelain')
            root=base/'fixed source';remote_profile=base/'remote profiles/server1.json'
            profile=dict(host_id='server1',root=str(root),work_dir=str(base/'worker-spool'),
                interpreter=sys.executable,slots=1,route_views={},prefix_cache_directory=str(base/'cache'))
            fake=base/'fake_ssh.py';fake.write_text('''import os,shlex,sys
assert sys.argv[-2]=='qg1'
command=shlex.split(sys.argv[-1]);os.execv(command[0],command)
''')
            config=dict(ssh_command=[sys.executable,str(fake)],workers=[dict(host_id='server1',ssh_alias='qg1',
                python=sys.executable,root=str(root),profile=str(remote_profile))])
            result=S.setup(config,profile,str(repo),source)
            self.assertTrue(result['ok']);self.assertEqual(result['started_jobs'],0)
            self.assertEqual(git(root,'rev-parse','HEAD'),source)
            self.assertTrue((root/'src/artificial.py').exists());self.assertFalse((root/'large-history').exists())
            self.assertEqual(json.loads(remote_profile.read_text()),profile)
            self.assertEqual(S.setup(config,profile,str(repo),source),result)
            config_file=base/'local relay.json';profile_file=base/'local profile.json'
            config_file.write_text(json.dumps(config),encoding='utf-8-sig');profile_file.write_text(json.dumps(profile),encoding='utf-8-sig')
            command=[sys.executable,'-B',str(ROOT/'scripts/setup_rpe_q3_mailbox.py'),'--config',str(config_file),
                '--profile',str(profile_file),'--repository',str(repo),'--source',source]
            completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
            self.assertEqual(json.loads(completed.stdout),result)
            self.assertEqual(git(repo,'rev-parse','HEAD'),source);self.assertEqual(git(repo,'status','--porcelain'),original_status)
            spool=Path(profile['work_dir'])/'owned-job/attempt';spool.mkdir(parents=True)
            (spool/'ASSIGNMENT.json').write_text('{}')
            (spool/'WORKER_STATE.json').write_text(json.dumps(dict(state='ASSIGNED')))
            with self.assertRaisesRegex(RuntimeError,'unstarted assignment'):S.setup(config,profile,str(repo),source)
            (spool/'WORKER_STATE.json').unlink();(spool/'ASSIGNMENT.json').unlink()
            changed=dict(profile,slots=2)
            with self.assertRaisesRegex(RuntimeError,'refusing overwrite'):S.setup(config,changed,str(repo),source)
            self.assertEqual(json.loads(remote_profile.read_text()),profile)
            (root/'src/artificial.py').write_text('ARTIFICIAL_CHANGED=True\n')
            with self.assertRaisesRegex(RuntimeError,'source root has changes'):S.setup(config,profile,str(repo),source)
            self.assertEqual((root/'src/artificial.py').read_text(),'ARTIFICIAL_CHANGED=True\n')

    def test_profile_host_source_and_remote_path_validation_before_ssh(self):
        profile=dict(host_id='server1',root='/owned/source',work_dir='/owned/spool',interpreter='/usr/bin/python3')
        config=dict(workers=[dict(host_id='different',root='/owned/source',profile='/owned/profile',ssh_alias='qg1')])
        with self.assertRaisesRegex(ValueError,'host_id'):S.setup(config,profile,source=S.DEFAULT_SOURCE)
        with self.assertRaisesRegex(ValueError,'full fixed'):S.setup(config,profile,source='main')
        config['workers'][0]['host_id']='server1';config['workers'][0]['root']='/other/source'
        with self.assertRaisesRegex(ValueError,'root differ'):S.setup(config,profile)
        config['workers'][0]['root']='/owned/source';config['workers'][0]['profile']='relative.json'
        with self.assertRaisesRegex(ValueError,'absolute Linux'):S.setup(config,profile)

    def test_windows_ssh_child_is_hidden(self):
        self.assertTrue(S.child_options('nt')['creationflags']&0x08000000)
        self.assertEqual(S.child_options('posix'),{})


if __name__=='__main__':unittest.main()
