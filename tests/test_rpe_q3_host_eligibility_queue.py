"""Synthetic eligibility at durable queue/RPC boundaries; no adapters run."""
import base64
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_q3_coordinator as Q
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor


class EligibilityQueue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / 'run'
        self.executors = []
        self.addCleanup(self.close_all)
        self.source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        self.routes = {'METHOD': [dict(kind='artificial', interpreter=sys.executable, configuration_ids='*')]}
        self.problem = dict(accuracy=dict(epsilon_E_decimal='1'), target=dict(energy_ha=0), rpe=dict(M=1))

    def executor(self, **distributed):
        ex = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem, optimizer_source=self.source,
            max_concurrent=3, budgets={'METHOD':8}, distributed=dict(port=0, affinity_wait_seconds=0, **distributed))
        self.executors.append(ex)
        return ex

    def close(self, ex):
        if ex.state.driver.closed: return
        with ex.state.cv:
            for task in ex.state.tasks.values():
                if task.status == 'RUNNING': task.status = 'DONE'; task.event.set()
            ex.state.active = 0
        ex.finish()

    def close_all(self):
        for ex in self.executors: self.close(ex)

    def proposal(self, cid, chain=False):
        return dict(configuration_id=cid, vector=(1,1,1,1), cost=Fraction(4),
                    origin=dict(kind='CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES' if chain else 'SYNTHETIC_INDEPENDENT'))

    def policy(self, ex, definitions):
        policy = dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', jobs={})
        facts = {}
        for proposal, hosts in definitions:
            view = ex.for_method('METHOD');job = view.job_dir(proposal)
            request = view.request(proposal, job, view.route(proposal['configuration_id']))
            cells = {};facts[job.name] = {}
            for host in hosts:
                raw = dict(host=host, runtime='SYNTHETIC', request_sha256=Q.digest(request))
                cells[host] = dict(status='PASS', current_host_fingerprint=raw, current_host_fingerprint_sha256=Q.digest(raw))
                facts[job.name][host] = raw
            record = dict(method='METHOD', request_sha256=Q.digest(request), source_manifest=ex.source_manifest,
                          source_manifest_sha256=Q.digest(ex.source_manifest), input_hashes=[],
                          eligible_host_ids=hosts, hosts=cells)
            record['decision_sha256'] = Q.digest(record)
            policy['jobs'][job.name] = record
        ex.distributed['host_eligibility'] = policy
        return facts

    def poll(self, ex, host, facts=None, slots=1):
        return ex.rpc('/v1/poll', dict(host_id=host, slots=slots, source_sha=ex.optimizer_source,
                      cached_prefixes=[], host_fingerprints=facts or {}))

    def test_skips_unsupported_head_for_next_independent_without_chain_backfill(self):
        ex = self.executor();view = ex.for_method('METHOD')
        head, chain, later = self.proposal('head'), self.proposal('chain', True), self.proposal('later')
        facts = self.policy(ex, [(head,['a']), (chain,['b']), (later,['b'])])
        view._enqueue(head, demand=True);view._enqueue(chain);view._enqueue(later)
        raw = {job:hosts['b'] for job, hosts in facts.items() if 'b' in hosts}
        rows = self.poll(ex,'b',raw)['assignments']
        self.assertEqual([r['job_id'] for r in rows], [view.job_dir(later).name])
        self.assertEqual(ex.store.row(view.job_dir(head).name)['state'], 'QUEUED')
        self.assertEqual(ex.store.row(view.job_dir(chain).name)['state'], 'QUEUED')
        self.assertFalse((view.job_dir(head)/'LAUNCH.json').exists())

    def test_all_hosts_held_has_no_reservation_launch_or_scientific_verdict(self):
        ex = self.executor();view = ex.for_method('METHOD');p = self.proposal('held')
        self.policy(ex, [(p,[])])
        self.assertFalse(view._enqueue(p))
        answer = view.answer(p)
        self.assertEqual(answer['status'],'HOST_ELIGIBILITY_HOLD')
        self.assertEqual(answer['calls'],0)
        self.assertEqual(ex.store.counts('METHOD'),0)
        self.assertIsNone(ex.store.row(view.job_dir(p).name))
        self.assertFalse(list(self.run.glob('jobs/*/*/LAUNCH.json')))

    def test_challenge_then_current_fingerprint_and_runtime_change_hold(self):
        ex = self.executor();view = ex.for_method('METHOD');p = self.proposal('current')
        facts = self.policy(ex, [(p,['a'])]);task = view._enqueue(p, demand=True)
        pending = self.poll(ex,'a')
        self.assertEqual(pending['assignments'],[])
        self.assertNotIn('attempt_id',pending['eligibility_challenges'][0])
        self.assertFalse((task.job/'LAUNCH.json').exists())
        changed = dict(facts[task.job.name]['a'],runtime='CHANGED')
        self.assertEqual(self.poll(ex,'a',{task.job.name:changed})['assignments'],[])
        self.assertEqual(ex.store.counts('METHOD'),0)
        self.assertEqual(view.answer(p)['status'],'HOST_ELIGIBILITY_HOLD')

    def test_busy_only_pass_host_waits_without_assigning_unsupported_host(self):
        ex = self.executor();view = ex.for_method('METHOD');p = self.proposal('busy')
        facts = self.policy(ex, [(p,['a'])]);task = view._enqueue(p, demand=True)
        current = {task.job.name:facts[task.job.name]['a']}
        self.assertEqual(self.poll(ex,'a',current,slots=0)['assignments'],[])
        self.assertEqual(self.poll(ex,'b',slots=1)['assignments'],[])
        waiting = ex.store.row(task.job.name)
        self.assertEqual(waiting['state'],'QUEUED')
        self.assertIsNone(waiting['host_id'])
        self.assertEqual(ex.store.counts('METHOD'),1)
        self.assertFalse((task.job/'LAUNCH.json').exists())
        acquired = self.poll(ex,'a',current,slots=1)['assignments']
        self.assertEqual([row['job_id']for row in acquired],[task.job.name])
        self.assertEqual(acquired[0]['request_sha256'],waiting['request_sha256'])

    def test_assigned_recovery_bypasses_new_readiness_and_duplicate_acceptance_stays_once(self):
        ex = self.executor();view = ex.for_method('METHOD');p = self.proposal('owned')
        facts = self.policy(ex, [(p,['a'])]);task = view._enqueue(p, demand=True)
        assigned = self.poll(ex,'a',{task.job.name:facts[task.job.name]['a']})['assignments'][0]
        self.close(ex)
        resumed = self.executor(required_hosts=['a','offline-host'])
        replay = self.poll(resumed,'a',{task.job.name:dict(runtime='CHANGED')})['assignments'][0]
        self.assertEqual(replay['attempt_id'],assigned['attempt_id'])
        request = replay['request'];allocation=[dict(m=m,N_X=x,N_Y=y)for m,(x,y)in enumerate(zip(request['N_X'],request['N_Y']))]
        outcome=dict(status=O.PASS,MSE_lower=O.enc(Fraction(1,2)),MSE_upper=O.enc(Fraction(1,2)),
                     artificial=True,identity=dict(request['expected'],allocation=allocation,contract_matches=True))
        files={'OUTCOME.json':outcome,'PROCESS_RETURN.json':dict(returncode=0,signal=None)}
        complete=dict(host_id='a',job_id=replay['job_id'],attempt_id=replay['attempt_id'],request_sha256=replay['request_sha256'],
                      files={name:base64.b64encode(json.dumps(value).encode()).decode()for name,value in files.items()})
        self.assertFalse(resumed.rpc('/v1/complete',complete)['duplicate'])
        self.assertTrue(resumed.rpc('/v1/complete',complete)['duplicate'])
        self.assertEqual(resumed.store.counts('METHOD'),1)
        self.assertEqual(sum(e['event']=='ACCEPT'for e in resumed.store.events()),1)

    def test_separate_fixed_computation_source_keeps_request_source(self):
        fixed = ROOT.parent/'rpe-distributed-q3-source-28d976b99'
        if not fixed.exists(): self.skipTest('fixed computation checkout is not available in this environment')
        sha='28d976b99cdf8329adcc1b79327e329e947ef774'
        ex=self.executor(computation_root=str(fixed),computation_source=sha)
        p=self.proposal('separate');view=ex.for_method('METHOD')
        request=view.request(p,view.job_dir(p),view.route(p['configuration_id']))
        self.assertEqual(request['optimizer_source'],sha)
        self.assertEqual(ex.source_manifest,Q.source_manifest(fixed,sha))
        self.assertEqual(ex.worker_script,ROOT/'scripts/run_rpe_q3_worker.py')


if __name__ == '__main__':
    unittest.main()
