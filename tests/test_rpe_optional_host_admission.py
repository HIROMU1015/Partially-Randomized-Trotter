"""Optional-host admission at the public entry and durable artificial RPC queue.

No scientific adapters or worker processes are launched. The existing readiness
checks are reused; this suite checks only the new participation boundary.
"""
import base64
from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import run_rpe_schedule_search as Entry
from trottertracks import pf_candidate_selection as PFSelect
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_optimizer_concurrent as C
from trottertracks import rpe_q3_coordinator as Q
import test_rpe_q3_host_eligibility_queue as QueueTests
from test_rpe_optimizer_concurrent import fixture

WSL = 'local0-wsl'
METHODS = ['FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF']
HOSTS = ['server2', 'server1', 'local', WSL]


def admission(accepting):
    return dict(schema='RPE_Q3_HOST_ADMISSION_V1',
                hosts={WSL:dict(optional=True, accepting=accepting, max_slots=1)})


def complete(ex, assignment, host):
    request = assignment['request']
    allocation = [dict(m=m, N_X=x, N_Y=y) for m, (x,y)
                  in enumerate(zip(request['N_X'], request['N_Y']))]
    outcome = dict(artificial=True, fixture='ARTIFICIAL_OPTIONAL_HOST_NO_ADAPTER_LEAF', status=O.PASS,
        MSE_lower=O.enc(Fraction(3)), MSE_upper=O.enc(Fraction(3)),
        identity=dict(request['expected'], allocation=allocation, contract_matches=True))
    identity = dict(host_id=host, job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])
    files = {'OUTCOME.json':outcome, 'PROCESS_RETURN.json':dict(returncode=0, signal=None, artificial=True)}
    return ex.rpc('/v1/complete', dict(identity, request_sha256=assignment['request_sha256'],
        files={name:base64.b64encode(json.dumps(value).encode()).decode() for name,value in files.items()}))


class OptionalHostAdmissionTests(unittest.TestCase):
    # Reuse the existing artificial queue setup without rerunning its tests.
    setUp = QueueTests.EligibilityQueue.setUp
    executor = QueueTests.EligibilityQueue.executor
    close = QueueTests.EligibilityQueue.close
    close_all = QueueTests.EligibilityQueue.close_all
    proposal = QueueTests.EligibilityQueue.proposal
    policy = QueueTests.EligibilityQueue.policy
    poll = QueueTests.EligibilityQueue.poll

    def test_optional_off_does_not_wait_or_block_existing_three_hosts(self):
        ex = self.executor(host_admission=admission(False), required_hosts=[WSL])
        view = ex.for_method('METHOD')
        proposals = [self.proposal('three-'+str(i)) for i in range(3)]
        facts = self.policy(ex, [(p,HOSTS) for p in proposals])
        for p in proposals: view._enqueue(p, demand=True)
        self.assertNotIn(WSL,ex.hosts)
        claims = []
        for host in HOSTS[:3]:
            reply = self.poll(ex,host,{job:hosts[host] for job,hosts in facts.items()})
            self.assertNotIn('waiting_for_hosts',reply)
            claims += [(host,row) for row in reply['assignments']]
        self.assertEqual([host for host,_ in claims],HOSTS[:3])
        self.assertEqual(self.poll(ex,WSL,slots=8)['assignments'], [])
        self.assertFalse(ex.metrics()['host_admission'][WSL]['accepting'])
        self.assertTrue(all(row['host_id'] != WSL for row in ex.store.rows()))

    def test_one_slot_includes_running_unknown_and_pending_transfer(self):
        ex = self.executor(host_admission=admission(True));view=ex.for_method('METHOD')
        proposals = [self.proposal('one-'+str(i)) for i in range(3)]
        facts = self.policy(ex, [(p,[WSL]) for p in proposals])
        for p in proposals:view._enqueue(p,demand=True)
        current={job:hosts[WSL]for job,hosts in facts.items()}
        first=self.poll(ex,WSL,current,slots=8)['assignments']
        self.assertEqual(len(first),1)
        owned=first[0]
        for state in ('RUNNING','UNKNOWN','COMPLETED_PENDING_TRANSFER'):
            ex.rpc('/v1/state',dict(host_id=WSL,job_id=owned['job_id'],attempt_id=owned['attempt_id'],state=state))
            self.assertEqual(self.poll(ex,WSL,current,slots=8)['assignments'],[])
        self.assertEqual(sum(row['host_id']==WSL for row in ex.store.rows()),1)
        self.assertEqual(sum(e['event']=='ASSIGN'for e in ex.store.events()),1)

    def test_pause_reload_and_controller_resume_keep_owned_attempt_and_accept_once(self):
        path=Path(self.tmp.name)/'ADMISSION.json';Q.atomic_json(path,admission(True))
        ex=self.executor(host_admission=str(path));view=ex.for_method('METHOD')
        p=self.proposal('owned');facts=self.policy(ex,[(p,[WSL])]);task=view._enqueue(p,demand=True)
        first=self.poll(ex,WSL,{task.job.name:facts[task.job.name][WSL]})['assignments'][0]
        marker=(task.job/'LAUNCH.json').read_bytes()
        Q.atomic_json(path,admission(False))
        self.assertFalse(ex.metrics()['host_admission'][WSL]['accepting'])
        self.close(ex)
        resumed=self.executor(host_admission=str(path))
        # No fresh facts or current readiness are needed to recover its owner.
        replay=self.poll(resumed,WSL,slots=8)['assignments']
        self.assertEqual(len(replay),1)
        self.assertEqual(replay[0]['attempt_id'],first['attempt_id'])
        self.assertEqual((task.job/'LAUNCH.json').read_bytes(),marker)
        self.assertFalse(complete(resumed,replay[0],WSL)['duplicate'])
        self.assertTrue(complete(resumed,replay[0],WSL)['duplicate'])
        self.assertEqual(len([e for e in resumed.store.events()if e['event']=='ACCEPT']),1)
        self.assertEqual(self.poll(resumed,WSL,slots=8)['assignments'],[])

    def test_missing_or_invalid_admission_file_cannot_create_new_claims(self):
        path=Path(self.tmp.name)/'ADMISSION.json';Q.atomic_json(path,admission(True))
        ex=self.executor(host_admission=str(path));view=ex.for_method('METHOD')
        proposals=[self.proposal('file-'+str(i))for i in range(2)]
        facts=self.policy(ex,[(p,HOSTS)for p in proposals])
        for p in proposals:view._enqueue(p,demand=True)
        current={job:hosts[WSL]for job,hosts in facts.items()}
        path.unlink()
        missing=self.poll(ex,WSL,current)
        self.assertFalse(missing['host_admission']['accepting'])
        self.assertEqual(missing['assignments'],[])
        path.write_text('{bad json')
        self.assertEqual(self.poll(ex,WSL,current)['assignments'],[])
        Q.atomic_json(path,admission(True))
        self.assertEqual(len(self.poll(ex,WSL,current)['assignments']),1)


class OptionalHostNormalEntryTests(unittest.TestCase):
    def test_public_entry_off_three_hosts_progress_join_one_slot_then_resume_off(self):
        problem,context,routes=fixture(methods=METHODS,configurations=1)
        problem['execution_routes']=routes
        cli=Entry.K.CLI;original_factory=cli.make_executor
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        created=threading.Event();executors=[];errors=[];results=[]
        with tempfile.TemporaryDirectory(prefix='rpe-optional-host-entry-')as temporary:
            base=Path(temporary);run=base/'run';admission_path=base/'HOST_ADMISSION.json'
            Q.atomic_json(admission_path,admission(False))
            problem_path=base/'PROBLEM.json';problem_path.write_text(json.dumps(problem))
            round_path=base/'ROUND.json'
            round_path.write_text(json.dumps(dict(round_id='ARTIFICIAL_OPTIONAL_HOST_ENTRY',methods=METHODS,initial='all',
                science_approved=True,max_new_q3_per_method={m:1 for m in METHODS},
                executor=dict(scheduler='concurrent',max_concurrent_jobs=4,dp_workers=1,
                    distributed=dict(bind='127.0.0.1',port=0,affinity_wait_seconds=0,
                        host_admission=str(admission_path),required_hosts=[WSL])))))

            def factory(*args,**kwargs):
                ex,optimize=original_factory(*args,**kwargs)
                self.assertIsInstance(ex,C.DistributedExecutor)
                executors.append(ex);created.set();return ex,optimize

            def invoke():
                try:results.append(Entry.main(['--problem',str(problem_path),'--round',str(round_path),'--run-dir',str(run)]))
                except BaseException as exc:errors.append(exc)

            def poll(ex,host,slots=1):
                return ex.rpc('/v1/poll',dict(host_id=host,source_sha=source,slots=slots,cached_prefixes=[]))

            def pf_prepare(root,problem,current,spec,directory,effective_routes):
                return current,effective_routes,dict(status='ARTIFICIAL_EXISTING_INPUTS',artificial_only=True)

            with patch.object(O,'prepare',return_value=context),patch.object(PFSelect,'prepare',side_effect=pf_prepare),\
                 patch.object(cli,'head',return_value=source),patch.object(cli,'make_executor',side_effect=factory),\
                 patch.object(O.Executor,'run_adapter',side_effect=AssertionError('No adapter leaf permitted')),\
                 redirect_stdout(io.StringIO()):
                thread=threading.Thread(target=invoke,daemon=True);thread.start()
                try:
                    self.assertTrue(created.wait(5),errors);ex=executors[0]
                    deadline=time.monotonic()+5
                    while len(ex.store.rows())<4 and time.monotonic()<deadline:
                        if errors:raise errors[0]
                        time.sleep(.005)
                    self.assertEqual(len(ex.store.rows()),4)
                    self.assertNotIn(WSL,ex.hosts)
                    claims=[]
                    for host in HOSTS[:3]:
                        rows=poll(ex,host)['assignments'];self.assertEqual(len(rows),1)
                        claims.append((host,rows[0]))
                    self.assertEqual(len(claims),3)
                    self.assertEqual(poll(ex,WSL,8)['assignments'],[])
                    Q.atomic_json(admission_path,admission(True))
                    joined=poll(ex,WSL,8)['assignments'];self.assertEqual(len(joined),1)
                    claims.append((WSL,joined[0]))
                    self.assertEqual({row['method']for _,row in claims},set(METHODS))
                    self.assertEqual(poll(ex,WSL,8)['assignments'],joined) # replay, no new claim
                    Q.atomic_json(admission_path,admission(False))
                    for host,row in claims:self.assertFalse(complete(ex,row,host)['duplicate'])
                    thread.join(8)
                    self.assertFalse(thread.is_alive(),errors)
                    if errors:raise errors[0]
                    markers={str(p.relative_to(run)):p.read_bytes()for p in run.glob('jobs/*/*/LAUNCH.json')}
                    self.assertEqual(len(markers),4)
                    invoke()
                    if errors:raise errors[0]
                    self.assertEqual(markers,{str(p.relative_to(run)):p.read_bytes()for p in run.glob('jobs/*/*/LAUNCH.json')})
                    self.assertEqual(len([e for e in executors[-1].store.events()if e['event']=='ACCEPT']),4)
                    self.assertFalse(executors[-1].metrics()['host_admission'][WSL]['accepting'])
                    report=dict(schema='RPE_OPTIONAL_HOST_ARTIFICIAL_ENTRY_V1',status='PASS',
                        entry='scripts/run_rpe_schedule_search.py',driver_factory_used=False,
                        scope='ARTIFICIAL_QUEUE_ONLY_NO_WORKER_OR_ADAPTER_PROCESS',
                        off_existing_hosts=[h for h,_ in claims[:3]],off_wsl_claims=0,
                        join_wsl_claims=1,wsl_max_slots=1,
                        claims=[dict(host_id=h,method=a['method'],job_id=a['job_id'])for h,a in claims],
                        completed_artificial_jobs=4,resume_new_launches=0,duplicate_adoptions=0,
                        final_wsl_accepting=False,scientific_adapter_calls=0,production_queue_submissions=0)
                    output=os.environ.get('RPE_OPTIONAL_HOST_ENTRY_ACCEPTANCE')
                    if output:Q.atomic_json(Path(output),report)
                finally:
                    if thread.is_alive()and executors:
                        ex=executors[0]
                        with ex.state.cv:
                            for task in ex.state.tasks.values():
                                if task.status=='RUNNING':task.status='DONE';task.error='ARTIFICIAL_CLEANUP';task.event.set()
                            ex.state.active=0;ex.state.cv.notify_all()
                        thread.join(5)


if __name__=='__main__':unittest.main()
