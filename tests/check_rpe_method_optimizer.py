"""Artificial checks for the per-method optimizer; plus one saved-evidence integration. No new science."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O

TAU = Fraction(1, 100)
SEARCH = dict(decrement_steps=[1, 2], large_step_headroom='9/10', seed_rules=['B_PRICE'], max_steps_per_method=30,
              max_new_q3_per_method=0)


def stub_generate(caps, prices, budget, rule):
    """Largest uniform scale-down of caps within budget (artificial generator)."""
    for k in range(100, 0, -1):
        shots = [max(1, (n * k) // 100) for n in caps]
        t = sum(Fraction(n) * c for n, c in zip(shots, prices))
        if t <= budget:
            return dict(shots=shots, conditional_cost=t)
    return None


def candidate(cid, prices, caps=None, ready=True):
    return dict(method='M', family='PR', key=dict(L_D=1), configuration_id=cid, name='M:' + cid, prices=prices,
                caps=caps or [4] * 13, ready=ready, needs=[] if ready else ['PRICE_26AXIS'], labels=[], price_status='X',
                signal_source=None, saved_points=[])


def flat(prices13):
    return [Fraction(p) for p in prices13 for _ in O.AXES]


def record(status, upper):
    return dict(status=status, MSE_lower=Fraction(upper), MSE_upper=Fraction(upper), source='SAVED', calls=0, label=None)


class DictBackend:
    def __init__(self, table, calls=0):
        self.table, self.calls = table, calls

    def answer(self, proposal, allow_new=False):
        hit = self.table.get((proposal['configuration_id'], proposal['vector']))
        return None if hit is None else dict(hit, calls=self.calls)


class Search(unittest.TestCase):
    prices = flat([1] * 12 + [100])

    def test_decrements_headroom_gate_strict_cheaper_and_order(self):
        c = candidate('A', self.prices);base = tuple([4] * 26)
        props, inc = O.proposals([c], {('A', base): record(O.PASS, '0.002')}, TAU, SEARCH, stub_generate)
        self.assertEqual(inc['vector'], base)
        self.assertEqual(props[0]['origin'], dict(props[0]['origin'], kind='DECREMENT_FROM_CONFIGURATION_INCUMBENT', m=12, d=2))
        self.assertTrue(all(p['cost'] < inc['cost'] for p in props))
        self.assertEqual([p['cost'] for p in props], sorted(p['cost'] for p in props))
        tight, _ = O.proposals([c], {('A', base): record(O.PASS, '0.0095')}, TAU, SEARCH, stub_generate)
        self.assertEqual({p['origin']['d'] for p in tight}, {1})

    def test_no_monotonicity_pruning_and_axis_split_when_prices_differ(self):
        prices = flat([1] * 13);prices[24] = Fraction(50);prices[25] = Fraction(70)
        c = candidate('A', prices);base = tuple([4] * 26);fail = list(base);fail[24] -= 1;fail[25] -= 1
        known = {('A', base): record(O.PASS, '0.001'), ('A', tuple(fail)): record(O.FAIL, '0.02')}
        props, _ = O.proposals([c], known, TAU, SEARCH, stub_generate)
        self.assertTrue(any(p['origin']['axes'] == ['Y'] for p in props))
        dominated = [p for p in props if all(a <= b for a, b in zip(p['vector'], fail))]
        self.assertTrue(dominated, 'proposals below a failed allocation must not be pruned (no monotonicity assumption)')
        self.assertNotIn(tuple(fail), {p['vector'] for p in props})

    def test_seed_for_configuration_without_feasible_point(self):
        a = candidate('A', self.prices);b = candidate('B', flat([1] * 13))
        props, inc = O.proposals([a, b], {('A', tuple([4] * 26)): record(O.PASS, '0.009')}, TAU, SEARCH, stub_generate)
        seeds = [p for p in props if p['configuration_id'] == 'B']
        self.assertTrue(seeds and all(p['origin']['kind'] == 'GENERATOR_SEED' and p['cost'] < inc['cost'] for p in seeds))

    def test_loop_continues_after_improvement_and_stops_by_rule(self):
        c = candidate('A', self.prices, caps=[8] * 13)
        def mse(cid, v):
            u = sum(Fraction(1, n) for n in v) / 400
            return u, u
        fx = O.FixtureBackend(mse, TAU)
        known = {}
        r = O.run_method('M', [c], known, fx, SEARCH, TAU, stub_generate, max_new=50)
        self.assertGreater(sum(t['improved'] for t in r['trace']), 1)
        self.assertEqual(r['stop_reason'], 'NO_IMPROVING_PROPOSAL')
        self.assertTrue(all(t['calls'] == 1 for t in r['trace']))
        budget = O.run_method('M', [c], {}, fx, SEARCH, TAU, stub_generate, max_new=2)
        self.assertEqual(budget['stop_reason'], 'Q3_BUDGET_EXHAUSTED');self.assertIsNotNone(budget['pending_request'])

    def test_saved_answers_cost_no_calls_and_missing_answer_becomes_request(self):
        c = candidate('A', self.prices);base = tuple([4] * 26);nxt = list(base);nxt[24] -= 2;nxt[25] -= 2
        backend = DictBackend({('A', tuple(nxt)): record(O.PASS, '0.0095')})
        free = lambda p: (p['configuration_id'], p['vector']) in backend.table
        r = O.run_method('M', [c], {('A', base): record(O.PASS, '0.002')}, backend, SEARCH, TAU, stub_generate, max_new=0, free=free)
        self.assertEqual([t['answer_source'] for t in r['trace']], ['SAVED']);self.assertEqual(r['new_calls'], 0)
        self.assertIsNone(O.run_method('M', [c], {('A', base): record(O.PASS, '0.002')}, backend, SEARCH, TAU, stub_generate,
                                       max_new=0)['trace'] or None, 'without free knowledge a budget-0 run must not call the backend')
        self.assertEqual(r['stop_reason'], 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET')
        self.assertEqual(r['pending_request']['origin']['base_vector'], nxt)

    def test_budget_checked_before_backend_call(self):
        c = candidate('A', self.prices, caps=[8] * 13)
        calls = []
        class Counting:
            def answer(self, p):
                calls.append(p['vector']);u = sum(Fraction(1, n) for n in p['vector']) / 400
                return dict(status=O.PASS if u <= TAU else O.FAIL, MSE_lower=u, MSE_upper=u, source='ARTIFICIAL', calls=1)
        r = O.run_method('M', [c], {}, Counting(), SEARCH, TAU, stub_generate, max_new=2)
        self.assertEqual(len(calls), 2);self.assertEqual(r['new_calls'], 2)
        self.assertEqual(r['stop_reason'], 'Q3_BUDGET_EXHAUSTED');self.assertIsNotNone(r['pending_request'])
        self.assertNotIn(r['pending_request']['vector'], calls)

    def test_resume_records_written_by_a_run_are_valid_ingest_input(self):
        c = candidate('A', self.prices, caps=[8] * 13)
        fx = O.FixtureBackend(lambda cid, v: (sum(Fraction(1, n) for n in v) / 400,) * 2, TAU)
        r = O.run_method('M', [c], {}, fx, SEARCH, TAU, stub_generate, max_new=3)
        summary = O.summarize_method('M', [c], r, {(t['configuration_id'], tuple(t['vector'])): dict(t, source='ARTIFICIAL_FIXTURE')
                                                   for t in r['trace']}, TAU)
        records = json.loads(json.dumps(O.jsonable(summary['resume_records'])))
        self.assertEqual(len(records), 3)
        resumed = O.ingest({}, [c], records)
        again = O.run_method('M', [c], resumed, fx, SEARCH, TAU, stub_generate, max_new=1)
        self.assertNotIn(tuple(again['trace'][0]['vector']), set(resumed))
        self.assertEqual(again['trace'][0]['vector'], list(r['pending_request']['vector']))

    def test_unknown_stops_and_inconclusive_is_held(self):
        c = candidate('A', self.prices);base = tuple([4] * 26)
        class Unknown:
            def answer(self, p, allow_new=False):return dict(status='UNKNOWN', MSE_lower=None, MSE_upper=None, source='X', calls=1)
        r = O.run_method('M', [c], {('A', base): record(O.PASS, '0.002')}, Unknown(), SEARCH, TAU, stub_generate, max_new=5)
        self.assertEqual(r['stop_reason'], 'UNKNOWN_RESULT')
        fx = O.FixtureBackend(lambda cid, v: (Fraction(1, 200), Fraction(1, 50)), TAU)
        h = O.run_method('M', [c], {('A', base): record(O.PASS, '0.002')}, fx, SEARCH, TAU, stub_generate, max_new=3)
        self.assertTrue(h['held'] and all(x['status'] == 'INCONCLUSIVE' for x in h['held']))

    def test_ingest_changes_next_request(self):
        c = candidate('A', self.prices);base = tuple([4] * 26)
        known = {('A', base): record(O.PASS, '0.002')}
        first = O.run_method('M', [c], dict(known), None, SEARCH, TAU, stub_generate, 0)['pending_request']
        nx, ny = O.split(first['vector'])
        after = O.ingest(dict(known), [c], [dict(configuration_id='A', N_X=nx, N_Y=ny, status=O.PASS,
                                                  MSE_lower='1/200', MSE_upper='1/200')])
        second = O.run_method('M', [c], after, None, SEARCH, TAU, stub_generate, 0)['pending_request']
        self.assertEqual(second['origin']['base_vector'], list(first['vector']));self.assertLess(second['cost'], first['cost'])

    def test_not_ready_candidates_are_never_searched(self):
        c = candidate('A', None, ready=False)
        r = O.run_method('M', [c], {}, None, SEARCH, TAU, stub_generate, 0)
        self.assertEqual(r['stop_reason'], 'NO_IMPROVING_PROPOSAL');self.assertIsNone(r['incumbent'])


ART = dict(kind='artificial', interpreter=sys.executable, configuration_ids='*', view_commit=None,
           artificial=dict(weights=[1] * 13, scale='1/1000', tau=str(TAU)))


class ExecutorJobs(unittest.TestCase):
    def make(self, d, route=ART):
        problem = dict(accuracy=dict(e=1), target=dict(energy_ha=0), rpe=dict(M=12))
        return O.Executor(ROOT, Path(d), dict(M=[route]), dict(A=dict(total_configuration_id='T', binding_id='B')), problem).for_method('M')

    def proposal(self, n=4):
        return dict(configuration_id='A', vector=tuple([n] * 26), cost=Fraction(1), origin=dict(kind='TEST'))

    def test_budget_launch_reuse_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            ex = self.make(d);p = self.proposal()
            self.assertIsNone(ex.answer(p, allow_new=False));self.assertEqual(ex.launched('M'), 0)
            first = ex.answer(p, allow_new=True)
            self.assertEqual((first['status'], first['calls'], first['source']), (O.PASS, 1, 'ARTIFICIAL_JOB'))
            self.assertEqual(ex.launched('M'), 1)
            again = ex.answer(p, allow_new=True);self.assertEqual((again['status'], again['calls']), (O.PASS, 0))
            self.assertEqual(ex.launched('M'), 1)
            bad = self.make(d);bad.expected = dict(A=dict(total_configuration_id='OTHER'))
            self.assertTrue(bad.for_method('M').answer(p)['reason'].startswith('IDENTITY_MISMATCH'))

    def test_unknown_is_distinct_from_model_failure(self):
        with tempfile.TemporaryDirectory() as d:
            unknown = dict(ART, artificial=dict(ART['artificial'], force_unknown=True))
            self.assertEqual(self.make(d, unknown).answer(self.proposal(), allow_new=True)['status'], 'UNKNOWN')
        with tempfile.TemporaryDirectory() as d:
            fail = self.make(d).answer(self.proposal(1), allow_new=True)
            self.assertEqual(fail['status'], O.FAIL)

    def test_running_job_is_not_launched_twice_and_dead_job_is_reported(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            ex = self.make(d);p = self.proposal();job = ex.job_dir(p);job.mkdir(parents=True)
            sleeper = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)', str(job)])
            try:
                (job / 'LAUNCH.json').write_text(json.dumps(dict(pid=sleeper.pid)))
                self.assertEqual(ex.answer(p, allow_new=True)['status'], 'IN_PROGRESS')
            finally:
                sleeper.kill();sleeper.wait()
            (job / 'REQUEST.json').write_text(json.dumps(ex.request(p, job, ART), default=str))
            self.assertEqual(ex.answer(p, allow_new=True)['status'], 'STALE_JOB')
            self.assertFalse((job / 'OUTCOME.json').exists())
            (job / 'ARTIFICIAL_RESULT.json').write_text(json.dumps(dict(status=O.PASS, MSE_lower='1/1000', MSE_upper='1/1000',
                                                                       scientific_evaluation_calls=1)))
            collected = ex.answer(p, allow_new=False)
            self.assertEqual((collected['status'], collected['calls']), (O.PASS, 0))


class Inputs(unittest.TestCase):
    def test_hamiltonian_facts_from_receipt_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            snap = Path(d) / 's.npz';snap.write_bytes(b'artificial')
            sh = hashlib.sha256(b'artificial').hexdigest()
            receipt = dict(sha256=sh, rank=3, hamiltonian_hash='H', state_hash='S',
                           arrays=dict(g_matrices=dict(shape=[3, 4, 4]), state_vector=dict(shape=[16]), sector_state_vector=dict(shape=[6])))
            rp = Path(d) / 'r.json';rp.write_text(json.dumps(receipt))
            problem = dict(problem=dict(hamiltonian_data=dict(snapshot=dict(path=str(snap), sha256=sh),
                receipt=dict(path=str(rp), sha256=hashlib.sha256(rp.read_bytes()).hexdigest()), expected=dict(fragment_count=3, n_qubits=4))))
            facts = O.hamiltonian_facts(problem)
            self.assertEqual((facts['fragment_count'], facts['n_qubits'], facts['sector_dimension']), (3, 4, 6))
            problem['problem']['hamiltonian_data']['expected']['n_qubits'] = 8
            with self.assertRaises(ValueError):O.hamiltonian_facts(problem)

    def test_prices_must_reproduce_saved_totals(self):
        axes = [dict(m=m, axis=a, C=1 + m) for m in range(13) for a in O.AXES]
        point = dict(price_axes=axes, conditional_cost=dict(numerator=sum(2 * (1 + m) for m in range(13)), denominator=1),
                     N_X=[1] * 13, N_Y=[1] * 13)
        prices, status = O.configuration_prices(None, [point])
        self.assertEqual(status, 'SAVED_UNIT_PRICES_REPRODUCE_ALL_SAVED_TOTALS');self.assertEqual(prices[0], 1)
        bad = dict(point, conditional_cost=dict(numerator=1, denominator=1))
        self.assertEqual(O.configuration_prices(None, [bad])[1], 'UNIT_PRICES_DO_NOT_REPRODUCE_SAVED_TOTAL')


class SavedEvidence(unittest.TestCase):
    """Reads fixed Git objects only (C's catalog inputs); no new science."""

    @classmethod
    def setUpClass(cls):
        cls.problem = O.load_json(ROOT / 'configs/rpe_method_optimizer/h4_df10_problem.json')
        cls.context = O.prepare(ROOT, cls.problem)

    def test_minima_ld_enumeration_and_replay(self):
        full = O.optimize(ROOT, self.problem, context=self.context)
        minima = self.context['catalog']['method_minima']
        for m in full:
            self.assertEqual(full[m]['best']['cost'], O.frac(minima[m]['conditional_cost']))
        lds = sorted({c['key']['L_D'] for c in self.context['candidates'] if c['method'] == 'PR_DF'})
        self.assertEqual(lds, list(range(1, self.context['facts']['fragment_count'])))
        replay = O.optimize(ROOT, self.problem, context=self.context, methods=['TRUNCATED_PF'],
                            initial=self.problem['verification']['replay_seeds'])['TRUNCATED_PF']
        self.assertLess(replay['best']['cost'], replay['trace'][0]['incumbent_before'])
        self.assertEqual(replay['new_calls'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
