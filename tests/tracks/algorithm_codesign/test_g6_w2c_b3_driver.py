"""Synthetic endpoints/fake clocks only. Never import or execute frozen science."""
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
from itertools import cycle
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
M = importlib.import_module('trottertracks.algorithm_codesign.g6_w2c_b3_driver')
A = M.audit


class FakeClock:
    def __init__(self):self.n = 0
    def __call__(self):
        self.n += 10
        return self.n, 2*self.n


def event(canonical=False):
    raw = (0, 0) if canonical else (0, 1)
    word = () if canonical else raw
    return {'raw_word': raw, 'word': word, 'child': 2, 'phase_i_power': 2,
            'circuit_time_word': tuple(reversed(word)), 'ratio': F(1, 9),
            'proposal': F(1, 2), 'coefficient': F(1, 4), 'weight': F(1, 2),
            'provider_calls': {2: 2} if canonical else {0: 1, 1: 1, 2: 2}}


class Record:
    def __init__(self, data):
        self.data = data
        self.document = A.canonical(M.exact(data));self.digest = sha256(self.document.encode()).hexdigest()
    def read(self):return deepcopy(self.data)


class FakeGenerator:
    """Ignores supplied science parameters, always an artificial witness fixture."""
    def __init__(self, arm):
        self.arm = arm;self.p = (F(1, 2), F(1, 2));self.x = F(1, 4);self.m = 7
        self.H = self.K = 8;self.eta = self.rho = F(1, 8);self.chi = F(1, 2)
        self.groups = (NS(mode='iid', degree=0, square=F(1), ratio=F(1, 9), first=-1),)
        self.roots = (NS(lo=F(1), hi=F(1)),);self.B = NS(lo=F(1), hi=F(1))
        self.label_law = self.p;self.group_law = (F(1),)
        self.event_queries, self.packet_queries = 0, 0
    def packet(self, word):
        self.packet_queries += 1
        return {'artificial_packet': True}
    def event(self, *args):
        self.event_queries += 1
        return event(canonical=self.arm != 'full_return')
    def sample(self, bits):
        choice = bits.bits(8)
        if self.arm == 'full_return':
            if choice == 0:return None
            self.packet((0, 1))
            if choice == 1:return None
            return event()
        return self.event(0, (0, 0), 2)


class FakeSession:
    def __init__(self):
        self._generator = FakeGenerator('full_return');self.preflight = Record({'artificial': True})
        self.angle_calls = 0
    def sample(self, bits, *, trial_id):
        result = self._generator.sample(bits)
        return Record({'kind': 'EVENT', 'event': result} if result is not None else
                      {'kind': 'ZERO', 'reason': 'RAW_NONREDUCED' if bits.last == 0 else 'COIN_REJECT'})
    def angle_for_event(self, record):
        self.angle_calls += 1;self._generator.event((0, 1), 2)
        return Record({'angle': 'ARTIFICIAL_ENCLOSURE'})


class Bits:
    def __init__(self, values=(0, 1, 2)):self.values = iter(values);self.last = None;self.consumed_bits = 0
    def bits(self, n):
        self.last = next(self.values);self.consumed_bits += n
        return self.last


def modules():
    def make(p, x, m, arm, **kw):return FakeGenerator(arm)
    def prepare(*a, **kw):return FakeSession()
    return {'synthetic_only': True, 'g7': NS(make_generator=make, _law=lambda *a: (F(1),)),
            'f3': NS(prepare=prepare, _state=lambda g: {'fixture_state': g.p}),
            'precision': NS(enclose_atan=lambda *a, **kw: Record({'fixture_angle': True}))}


def accepted_candidate():
    """Artificial adoption metadata; no real external admission port is present."""
    return {'schema': M.SCHEMA, 'input': deepcopy(M.INPUT), 'policy': deepcopy(M.POLICY),
            'cells': list(M.schedule()), 'adopted': True, 'contract_grants_execution': False,
            'output_relative': 'mock_output',
            'source_pins': {**{'src/trottertracks/algorithm_codesign/'+k: v for k, v in M.FROZEN_PINS.items()},
                            M.FASTPATH_SOURCE: sha256((ROOT/M.FASTPATH_SOURCE).read_bytes()).hexdigest()},
            'decisions': {'angle': {'tau': '1/16', 'K_phi': 8, 'J': 4,
                                  'mode': 'MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE'},
                          'seed': {'algorithm': 'FROZEN_G7_SHA256_COUNTER',
                                   'namespace': 'B3_FULL_N_NEW_ARTIFICIAL', 'IID_claim': False},
                          'schedule_and_output': {}, 'source_release': 'a'*40, 'external_admission': 'mock_only'}}


class FakePort:
    def __init__(self, **overrides):self.overrides = overrides;self.calls = 0
    def admit_once(self, digest):
        self.calls += 1
        return dict({'contract_digest': digest, 'approved': True, 'fresh_claim': True,
                     'previously_consumed': False, 'retry': 0, 'resume': False,
                     'claim_id': 'B3_NEW_ARTIFICIAL_ONLY',
                     'scope': 'B3_CORRECTNESS_AND_FIXED_FULL_N_TWO_PASS'}, **self.overrides)


class Guards(unittest.TestCase):
    def test_import_does_not_load_science(self):
        for name in ('g7_generator', 'g6_f3_generator', 'g6_f3_precision', 'return_aggregation'):
            self.assertNotIn('trottertracks.algorithm_codesign.'+name, sys.modules)
    def test_unapproved_does_not_import_construct_or_create_output(self):
        c = accepted_candidate();c['adopted'] = False
        with tempfile.TemporaryDirectory() as d, patch.object(M, 'load_modules') as loader:
            out = Path(d)/'should_not_exist'
            with self.assertRaises(M.Stop):M.execute(c, root=d, output=out)
            loader.assert_not_called();self.assertFalse(out.exists())
    def test_boolean_candidate_is_not_authority(self):
        with self.assertRaises(M.Stop):M.Boundary().enter(accepted_candidate())
    def test_input_policy_and_saved_n_axis_binding(self):
        for key in ('input', 'policy', 'cells'):
            c = accepted_candidate()
            if key == 'input':c[key]['m'] = 9
            elif key == 'policy':c[key]['root_bits'] = 32
            else:c[key][0]['axis'] = 'Y'
            with self.subTest(key=key), self.assertRaises(M.Stop):M.Boundary().enter(c, FakePort())
    def test_old_consumed_claim_refused(self):
        for fields in ({'previously_consumed': True}, {'claim_id': 'OLD_G7_CLAIM'},
                       {'fresh_claim': False}, {'resume': True}, {'retry': 1},
                       {'contract_digest': 'wrong'}):
            with self.subTest(fields=fields), self.assertRaises(M.Stop):
                M.Boundary().enter(accepted_candidate(), FakePort(**fields))
    def test_one_shot_and_stop_no_retry(self):
        b = M.Boundary();p = FakePort();b.enter(accepted_candidate(), p);b.stop()
        with self.assertRaises(M.Stop):b.enter(accepted_candidate(), p)
        self.assertEqual(p.calls, 1)
    def test_failed_admission_is_terminal(self):
        b = M.Boundary()
        with self.assertRaises(M.Stop):b.enter(accepted_candidate())
        with self.assertRaises(M.Stop):b.enter(accepted_candidate(), FakePort())
    def test_no_angle_default(self):
        for p in (None, {}, {'tau': '1/16', 'K_phi': 8, 'J': 4, 'mode': 'PHYSICAL_NEW_ALLOCATION'}):
            with self.subTest(p=p), self.assertRaises(M.Stop):M.validate_angle(p)
    def test_new_seed_common_A_B_distinct_axes_and_old_seed_rejected(self):
        c = accepted_candidate()
        self.assertNotEqual(M.seed_for(c, 'ordinary', 'X'), M.seed_for(c, 'ordinary', 'Y'))
        c['decisions']['seed']['namespace'] = 'G7-development-bitstream-v1'
        with self.assertRaises(M.Stop):M.seed_for(c, 'ordinary', 'X')
    def test_source_mismatch_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'artificial';p.write_text('mock')
            with self.assertRaises(A.InvalidEvidence):M.check_pins(d, {'artificial': 'a'*64})
            with self.assertRaises(A.InvalidEvidence):M.check_pins(d, {'../other': 'a'*64})
    def test_missing_or_changed_frozen_source_policy_rejected(self):
        c = accepted_candidate();c['source_pins'] = {}
        with self.assertRaises(M.Stop):M.Boundary().enter(c, FakePort())
    def test_saved_budget_loading_no_numeric_recomputation(self):
        raw = {'rows': [{'input': M.INPUT['id'], 'arm': arm, **{k: M.INPUT[k] for k in ('p', 'x', 'm')},
                        'budget': {'N_per_axis': n, 'quantum_call_hard_cap_per_axis': n,
                        'alpha_axis': '1/320', 'alpha_familywise_16_axes': '1/20',
                        'uses_reference_B_new_or_m2': False}} for arm, n in M.SAVED_N.items()]}
        data = json.dumps(raw).encode()
        self.assertEqual(len(M.load_saved(data, sha256(data).hexdigest())), 4)
        with self.assertRaises(M.Stop):M.load_saved(data, 'a'*64)
        raw['rows'][0]['budget']['N_per_axis'] += 1;data = json.dumps(raw).encode()
        with self.assertRaises(M.Stop):M.load_saved(data, sha256(data).hexdigest())


class Interfaces(unittest.TestCase):
    def test_real_module_endpoint_constructor_cannot_bypass_admission(self):
        with self.assertRaises(M.Stop):
            M.LegacyEndpoint({'g7': ModuleType('artificial_module_marker')},
                             'ordinary', 'A', None, lambda: 'source')
    def test_endpoint_failure_is_terminal_without_retry(self):
        e = M.LegacyEndpoint(modules(), 'full_return', 'A', None, lambda: 'source')
        with self.assertRaises(StopIteration):e.sample(Bits(()), 'fails')
        b = Bits((2,))
        with self.assertRaises(M.Stop):e.sample(b, 'retry')
        self.assertEqual(b.consumed_bits, 0)
    def test_full_A_all_three_paths_without_extra_scientific_queries(self):
        e = M.LegacyEndpoint(modules(), 'full_return', 'A', None, lambda: 'source')
        for i, kind in enumerate(('ZERO_R', 'ZERO_C', 'EVENT')):
            b = Bits((i,));r = e.sample(b, 'fixture');self.assertEqual(r.kind, kind)
            self.assertEqual(b.consumed_bits, 8)
        self.assertEqual(e.g.packet_queries, 2);self.assertEqual(e.g.event_queries, 0)
    def test_full_B_retains_F3_sample_and_angle_requery(self):
        e = M.LegacyEndpoint(modules(), 'full_return', 'B', accepted_candidate()['decisions']['angle'], lambda: 'source')
        r = e.sample(Bits((2,)), 'fixture');e.angle_for(r)
        self.assertEqual(e.session.angle_calls, 1);self.assertEqual(e.g.event_queries, 1)
    def test_canonical_B_audits_reduced_word_original_phase_and_requeries(self):
        e = M.LegacyEndpoint(modules(), 'ordinary', 'B', accepted_candidate()['decisions']['angle'], lambda: 'source')
        r = e.sample(Bits((2,)), 'fixture');e.angle_for(r)
        self.assertEqual(r.event['raw_word'], (0, 0));self.assertEqual(r.event['word'], ())
        self.assertEqual(r.event['phase_i_power'], 2);self.assertEqual(r.event['weight'], F(1, 2))
        self.assertEqual(e.g.event_queries, 2)
    def test_canonical_and_full_have_paid_nonempty_audit(self):
        for arm in M.ARMS:
            reads = []
            e = M.LegacyEndpoint(modules(), arm, 'B', accepted_candidate()['decisions']['angle'],
                                 lambda: reads.append(1) or 'source')
            e.sample(Bits((2,)), 'fixture')
            self.assertGreaterEqual(len(reads), 2)
    def test_held_state_drift_refused(self):
        e = M.LegacyEndpoint(modules(), 'ordinary', 'B', accepted_candidate()['decisions']['angle'], lambda: 'source')
        e.g.p = (F(1),)
        with self.assertRaises(A.InvalidEvidence):e.sample(Bits((2,)), 'fixture')
    def test_phase_order_and_weight_tamper_refused(self):
        for k, v in (('phase_i_power', 0), ('circuit_time_word', (0, 1)), ('weight', F(1))):
            ev = event();ev[k] = v
            with self.subTest(k=k), self.assertRaises(A.InvalidEvidence):
                A.event_witness(ev, full=True, degree_cap=5, labels=3)
    def test_failure_cannot_be_zero(self):
        with self.assertRaises(M.Stop):M.Outcome('TYPED_FAILURE').validate()


class CostsAndStreaming(unittest.TestCase):
    def test_nested_exclusive_costs_and_reconciliation(self):
        c = FakeClock();m = M.Meter(c)
        with m.span('parent'):
            with m.span('child'):pass
        self.assertEqual(m.rows['child']['cpu_ns'], 10)
        self.assertEqual(m.rows['parent']['cpu_ns'], 20)
        r = m.reconcile((50, 100));self.assertEqual(r['exclusive_sum_ns'], (30, 60))
        with self.assertRaises(M.Stop):m.reconcile((1, 2))
    def test_backwards_clock_stops(self):
        c = iter(((5, 5), (4, 6)));m = M.Meter(lambda: next(c))
        with self.assertRaises(M.Stop):
            with m.span('bad'):pass
        self.assertTrue(m.stopped)
    def test_cache_hit_miss_and_costs(self):
        m = M.Meter(FakeClock());c = M.ChargedCache('p', m)
        self.assertEqual(c.get('p', 'k', lambda: 7), 7)
        self.assertEqual(c.get('p', 'k', lambda: self.fail('fictional miss')), 7)
        self.assertEqual((c.hits, c.misses), (1, 1))
        self.assertEqual(m.rows['cache.lookup']['calls'], 2)
        self.assertGreater(m.rows['cache.acquire']['cpu_ns'], 0)
        with self.assertRaises(M.Stop):c.get('other', 'k', lambda: 7)
    def test_unknown_cache_value_not_zero(self):
        c = M.ChargedCache('p', M.Meter(FakeClock()))
        with self.assertRaises(M.Stop):c.get('p', 'k', lambda: None)
        self.assertNotIn('k', c.values)
    def test_output_hash_write_flush_and_no_resume(self):
        with tempfile.TemporaryDirectory() as d:
            m = M.Meter(FakeClock());p = Path(d)/'mock.jsonl'
            s = M.Stream(p, m, byte_limit=10000, flush_rows=1)
            s.emit({'event': event()});s.close()
            row = json.loads(p.read_text())
            self.assertEqual(sha256((row['previous']+row['payload']).encode()).hexdigest(), row['sha256'])
            self.assertEqual(s.file_hash.hexdigest(), sha256(p.read_bytes()).hexdigest())
            for stage in ('serialize', 'hash', 'write', 'file_hash', 'flush', 'finalize'):
                self.assertGreater(m.rows['save.'+stage]['cpu_ns'], 0)
            with self.assertRaises(FileExistsError):M.Stream(p, m, byte_limit=10000, flush_rows=1)
            with self.assertRaises(M.Stop):s.emit({})
    def test_output_failure_preserves_committed_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            s = M.Stream(Path(d)/'mock', M.Meter(FakeClock()), byte_limit=1000, flush_rows=1)
            s.emit({'mock': 1})
            with self.assertRaises(M.Stop):s.emit({'huge': 'x'*2000})
            self.assertEqual(s.rows, 1);self.assertEqual(s.failed['committed_rows'], 1)
            s.close();self.assertEqual(len(s.path.read_text().splitlines()), 1)
            log = Path(d)/'bounded_failure.json'
            self.assertFalse(M.write_control_metadata(log, {'large_mock_failure': 'x'*10000}, 400))
            self.assertLessEqual(log.stat().st_size, 400)
            self.assertEqual(json.loads(log.read_text())['kind'], 'LOG_BYTE_LIMIT')
    def test_function_observer_synthetic_nested_only(self):
        def inner():return 1
        def outer():return inner()+inner()
        obs = M.FunctionObserver([__file__], FakeClock())
        with obs:self.assertEqual(outer(), 2)
        rows = obs.meter.rows
        outer_row = next(v for k, v in rows.items() if k.endswith('outer'))
        inner_row = next(v for k, v in rows.items() if k.endswith('inner'))
        self.assertEqual(inner_row['calls'], 2);self.assertGreaterEqual(outer_row['cpu_ns'], 0)
        self.assertIsNone(sys.getprofile())
    def test_existing_profiler_not_overwritten(self):
        old = lambda *a: None
        sys.setprofile(old)
        try:
            with self.assertRaises(M.Stop):
                with M.FunctionObserver([]):pass
            self.assertIs(sys.getprofile(), old)
        finally:sys.setprofile(None)


class PairSchedule(unittest.TestCase):
    def cells(self, layer='A'):
        return [{'arm': 'full_return', 'layer': layer, 'axis': a, 'N': 3} for a in M.AXES]
    def test_six_mock_trials_and_angle_only_on_accepted_B(self):
        for layer in M.LAYERS:
            with self.subTest(layer=layer), tempfile.TemporaryDirectory() as d:
                m = M.Meter(FakeClock());s = M.Stream(Path(d)/'mock', m, byte_limit=100000, flush_rows=2)
                e = M.LegacyEndpoint(modules(), 'full_return', layer,
                    accepted_candidate()['decisions']['angle'] if layer == 'B' else None, lambda: 'source')
                r = M.run_pair(e, lambda a: Bits(), self.cells(layer), s, m,
                               phase='profile', pass_name='baseline', correctness_trials=16)
                s.close();self.assertEqual(r['counts'], {'ZERO_R': 2, 'ZERO_C': 2, 'EVENT': 2})
                self.assertEqual(sum(a['N_executed'] for a in r['axes']), 6)
                if layer == 'B':self.assertEqual(e.session.angle_calls, 2)
    def test_axis_or_n_mismatch_rejected(self):
        cells = self.cells();cells[1]['N'] = 2
        with self.assertRaises(M.Stop):
            M.run_pair(None, None, cells, None, M.Meter(FakeClock()), phase='profile',
                       pass_name='baseline', correctness_trials=16)
    def test_axis_sharing_drift_rejected_with_failure_prefix(self):
        class E:
            n = 0
            def shared_identity(self):self.n += 1;return str(self.n)
        with tempfile.TemporaryDirectory() as d:
            m = M.Meter(FakeClock());s = M.Stream(Path(d)/'mock', m, byte_limit=10000, flush_rows=1)
            with self.assertRaises(M.Stop):
                M.run_pair(E(), None, self.cells(), s, m, phase='profile',
                           pass_name='baseline', correctness_trials=16)
            s.close();self.assertIn('TYPED_FAILURE', s.path.read_text())
    def test_scientific_failure_prefix_is_not_zero_or_retry(self):
        class E:
            n = 0
            def shared_identity(self):return 'fixed'
            def sample(self, bits, trial_id):
                self.n += 1
                if self.n == 2:raise RuntimeError('mock typed failure')
                return M.Outcome('ZERO_R')
        with tempfile.TemporaryDirectory() as d:
            m = M.Meter(FakeClock());s = M.Stream(Path(d)/'mock', m, byte_limit=10000, flush_rows=1);e = E()
            with self.assertRaises(RuntimeError):
                M.run_pair(e, lambda a: Bits(), self.cells(), s, m, phase='profile',
                           pass_name='baseline', correctness_trials=16)
            s.close();self.assertEqual(e.n, 2)
            self.assertEqual(len(s.path.read_text().splitlines()), 2)
            self.assertIn('completed_prefix', s.path.read_text())
    def test_all_sixteen_cells_bound_verbatim_N(self):
        cells = M.schedule();self.assertEqual(len(cells), 16)
        self.assertEqual(len({(c['arm'], c['layer'], c['axis']) for c in cells}), 16)
        self.assertTrue(all(c['N'] == M.SAVED_N[c['arm']] for c in cells))
    def test_driver_observer_excludes_its_own_meter_and_preserves_mock_execution(self):
        with tempfile.TemporaryDirectory() as d:
            c = FakeClock();m = M.Meter(c);s = M.Stream(Path(d)/'mock', m, byte_limit=100000, flush_rows=2)
            e = M.LegacyEndpoint(modules(), 'full_return', 'A', None, lambda: 'source')
            with M.FunctionObserver([M.__file__], c) as obs:
                r = M.run_pair(e, lambda a: Bits(), self.cells(), s, m,
                               phase='profile', pass_name='instrumented', correctness_trials=16)
            s.close();self.assertEqual(r['counts']['EVENT'], 2)
            self.assertTrue(obs.meter.rows);self.assertIsNone(sys.getprofile())
    def test_entire_candidate_schedule_with_only_mock_modules_and_fake_clock(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(M.SAVED_N, {a: 17 for a in M.ARMS}, clear=True):
            root = Path(d);src = root/'artificial.txt';src.write_text('fixture only')
            c = accepted_candidate();c.update(passes=list(M.PASSES), correctness_trials_per_cell=16)
            c['source_pins'] = {'artificial.txt': sha256(src.read_bytes()).hexdigest()}
            c['schedule_proposal'] = {'result_bytes': 10000000, 'log_bytes': 1000000, 'flush_rows': 16}
            c['decisions']['schedule_and_output'] = c['schedule_proposal']
            rows = {'rows': [{'input': M.INPUT['id'], 'arm': a,
                     **{k: M.INPUT[k] for k in ('p', 'x', 'm')},
                     'budget': {'N_per_axis': n, 'quantum_call_hard_cap_per_axis': n,
                     'alpha_axis': '1/320', 'alpha_familywise_16_axes': '1/20',
                     'uses_reference_B_new_or_m2': False}} for a, n in M.SAVED_N.items()]}
            raw = json.dumps(rows).encode();(root/'saved.json').write_bytes(raw)
            c['saved_evidence'] = {'path': 'saved.json', 'sha256': sha256(raw).hexdigest()}
            mods = modules();mods['g7'].DeterministicBits = lambda seed: Bits(cycle((0, 1, 2)))
            # Admission pins here are a synthetic test double, never an actual claim.
            with patch.object(M, 'candidate_errors', return_value=()), \
                 patch.object(M, 'load_modules', return_value=mods) as loader, \
                 patch.object(M, 'clock', FakeClock()):
                result = M.execute(c, root=root, output=root/'mock_output', approval_port=FakePort())
            loader.assert_called_once();self.assertEqual(len(result['pairs']), 24)
            self.assertEqual(sum(sum(a['N_executed'] for a in p['axes']) for p in result['pairs']), 800)
            self.assertTrue((root/'mock_output/completion_receipt.json').is_file())
            self.assertFalse((root/'mock_output/failure_prefix.json').exists())
            self.assertTrue(result['mandatory_STOP'])
            with patch.object(M, 'candidate_errors', return_value=()), \
                 patch.object(M, 'load_modules', return_value=mods), patch.object(M, 'clock', FakeClock()):
                with self.assertRaises(M.Stop):
                    M.execute(c, root=root, output=root/'mock_output', approval_port=FakePort())
            self.assertFalse((root/'mock_output/failure_prefix.json').exists())


if __name__ == '__main__':unittest.main()
