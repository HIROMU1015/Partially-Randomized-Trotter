"""Read-only mailbox inspection with artificial SSH and metadata-only fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('_mailbox_inspect',ROOT/'scripts/inspect_rpe_q3_mailbox.py')
I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I)


def snapshot(root):
    return {str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns)for p in root.rglob('*')if p.is_file()}


class ReadOnlyInspection(unittest.TestCase):
    def test_assigned_and_completed_attempts_never_execute_or_change_files(self):
        with tempfile.TemporaryDirectory(prefix='q3-inspect-artificial-')as temporary:
            base=Path(temporary);source=base/'fixed source';source.mkdir()
            subprocess.run(['git','init',str(source)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True,stderr=subprocess.DEVNULL).strip()
            git('config','user.name','Artificial Fixture');git('config','user.email','fixture@example.invalid')
            script=source/'scripts/run_rpe_q3_worker.py';script.parent.mkdir()
            marker=base/'FORBIDDEN_EXECUTION'
            script.write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise SystemExit(9)\n')
            git('add','.');git('commit','-m','Artificial read-only mailbox SOURCE');revision=git('rev-parse','HEAD')
            spool=base/'spool with spaces'
            for job,state,kind in [('first','ASSIGNED','artificial'),('second','COMPLETED_PENDING_TRANSFER','artificial'),('third','RUNNING','pauli_b_common_entry')]:
                path=spool/job/'attempt1';path.mkdir(parents=True)
                assignment=dict(job_id=job,attempt_id='attempt1',request=dict(route=dict(kind=kind,exact_fixture_integer='HUGE_INTEGER')))
                raw=json.dumps(assignment).replace('"HUGE_INTEGER"','1'+'0'*5000)
                (path/'ASSIGNMENT.json').write_text(raw)
                (path/'WORKER_STATE.json').write_text(json.dumps(dict(state=state,epoch=123.5)))
                stages=[dict(stage='OTHER_ARTIFICIAL_STAGE')]
                if job!='first':stages.insert(0,dict(stage='ARTIFICIAL_EVALUATION_START'))
                (path/'JOB_STAGES.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in stages))
            complete=spool/'second/attempt1'
            receipt=dict(returncode=0,signal=None,wall_seconds=.25,automatic_retry=False)
            (complete/'PROCESS_RETURN.json').write_text(json.dumps(receipt))
            # An opaque payload demonstrates that inspection hashes bytes and
            # does not interpret/reanalyze numerical OUTCOME contents.
            outcome=b'ARTIFICIAL_OPAQUE_OUTCOME_BYTES\n';(complete/'OUTCOME.json').write_bytes(outcome)
            profile=dict(host_id='server1',root=str(source),work_dir=str(spool))
            fake=base/'fake_ssh.py';fake.write_text('import os,shlex,sys\nassert sys.argv[-2]=="qg1"\ncommand=shlex.split(sys.argv[-1]);os.execv(command[0],command)\n')
            config=dict(ssh_command=[sys.executable,str(fake)],workers=[dict(host_id='server1',root=str(source),ssh_alias='qg1',python=sys.executable)])
            config_file=base/'relay.json';profile_file=base/'profile.json'
            config_file.write_text(json.dumps(config),encoding='utf-8-sig');profile_file.write_text(json.dumps(profile),encoding='utf-8-sig')
            before=snapshot(base)
            result=I.inspect(config,profile)
            self.assertEqual(result['source_sha'],revision);self.assertEqual(result['host_id'],'server1')
            self.assertEqual(len(result['attempts']),3)
            assigned,finished,real=result['attempts']
            self.assertEqual(assigned['state'],'ASSIGNED');self.assertEqual(assigned['state_epoch'],123.5)
            self.assertIsNone(assigned['process_return']);self.assertIsNone(assigned['outcome_sha256'])
            self.assertEqual(assigned['artificial_execute_count'],0)
            self.assertEqual(finished['process_return'],receipt)
            self.assertEqual(finished['outcome_sha256'],hashlib.sha256(outcome).hexdigest())
            self.assertEqual(finished['artificial_execute_count'],1)
            self.assertIsNone(real['artificial_execute_count'])
            self.assertFalse(marker.exists());self.assertEqual(before,snapshot(base))
            command=[sys.executable,'-B',str(ROOT/'scripts/inspect_rpe_q3_mailbox.py'),'--config',str(config_file),'--profile',str(profile_file)]
            output=subprocess.check_output(command,text=True,encoding='utf-8')
            self.assertEqual(len(output.splitlines()),1);self.assertEqual(json.loads(output),result)
            self.assertFalse(marker.exists());self.assertEqual(before,snapshot(base))

    def test_host_root_and_alias_validation_before_ssh(self):
        profile=dict(host_id='server1',root='/owned/source',work_dir='/owned/spool')
        config=dict(workers=[dict(host_id='other',root='/owned/source',ssh_alias='qg1')])
        with self.assertRaisesRegex(ValueError,'host_id'):I.inspect(config,profile)
        config['workers'][0]['host_id']='server1';config['workers'][0]['root']='/other/source'
        with self.assertRaisesRegex(ValueError,'root differ'):I.inspect(config,profile)
        config['workers'][0]['root']='/owned/source';config['workers'][0]['ssh_alias']='-bad'
        with self.assertRaisesRegex(ValueError,'SSH alias'):I.inspect(config,profile)

    def test_windows_ssh_child_has_no_window(self):
        self.assertTrue(I.child_options('nt')['creationflags']&0x08000000)
        self.assertEqual(I.child_options('posix'),{})


if __name__=='__main__':unittest.main()
