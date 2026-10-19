"""R1 verifier/cost/acceptance regression, strictly synthetic endpoints/bytes."""
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import b3_offline_artificial_fixtures as A
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_cost as C
from trottertracks.algorithm_codesign import g6_w2c_b3_integration_acceptance as H
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_cli as CLI


class Verifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = tempfile.TemporaryDirectory(prefix='b3-offline-artificial-master-')
        cls.master = Path(cls.store.name)/'input'
        cls.manifest, cls.final = A.bundle(cls.master)

    @classmethod
    def tearDownClass(cls):
        cls.store.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='b3-offline-artificial-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root/'input'
        shutil.copytree(self.master, self.input)
        self.m = deepcopy(self.manifest)
        self.s = deepcopy(self.final)

    def manifest_write(self):
        (self.input/'manifest.json').write_text(json.dumps(self.m)+'\n')

    def summary_write(self):
        path = self.input/'summary.jsonl'
        path.unlink()
        meta = A.emit_file(path, [self.s])
        receipt_path = self.input/'completion_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['summary_chain_sha256'] = meta['output_chain_sha256']
        receipt_path.write_text(json.dumps(receipt)+'\n')

    def verify(self):
        return V.verify_bundle(self.input/'manifest.json')

    def pair(self, which=0, summary=True):
        return V.verify_pair(self.input/self.m['pairs'][which]['path'], self.m['pairs'][which],
                             self.s['pairs'][which] if summary else None)

    def change_payloads(self, fn, which=0):
        path = self.input/self.m['pairs'][which]['path']
        docs = [r.payload for r in V.Reader(path)]
        fn(docs)
        path.unlink()
        meta = A.emit_file(path, docs)
        self.s['pairs'][which].update(meta)

    def test_actual_stream_bundle_mixed_RCA_XY_AB_two_pass(self):
        r = self.verify()
        self.assertEqual(r['status'], 'VERIFIED_COMPLETE')
        self.assertEqual(len(r['pairs']), 24)
        self.assertTrue(all(all(v > 0 for v in p['counts'].values()) for p in r['pairs'] if p['pair'][2] == 'full_return'))
        self.assertTrue(all(c['status'] == 'PASS' for c in r['cross_pairs']))

    def test_retained_correctness_trace_independently_recomputed(self):
        r = self.pair()
        self.assertEqual(r['evidence']['full_trace'], V.RAW)
        self.assertEqual(r['trace_sha256'], self.s['pairs'][0]['trace_sha256'])

    def test_compact_trace_cannot_be_reconstructed(self):
        r = self.pair(8)
        self.assertEqual(r['status'], 'VERIFIED_COMPLETE')
        self.assertEqual(r['evidence']['full_trace'], V.UNKNOWN)
        self.assertIsNone(r['trace_sha256'])
        self.assertEqual(r['evidence']['summary_full_trace'], V.CLAIM)

    def test_receipt_missing_does_not_certify_completion(self):
        (self.input/'completion_receipt.json').unlink()
        self.assertEqual(self.verify()['status'], 'INSUFFICIENT_EVIDENCE')

    def test_receipt_alone_does_not_certify_N(self):
        (self.input/self.m['pairs'][0]['path']).unlink()
        self.assertEqual(self.verify()['status'], 'INSUFFICIENT_EVIDENCE')

    def test_valid_short_prefix_without_summary(self):
        path = self.input/self.m['pairs'][0]['path']
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b''.join(lines[:4]))
        r = self.pair(summary=False)
        self.assertEqual((r['status'], r['committed_trials']), ('VERIFIED_PARTIAL_PREFIX', 4))

    def test_partial_final_line_excluded_from_rows_and_trials(self):
        path = self.input/self.m['pairs'][0]['path']
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b''.join(lines[:4])+lines[4][:25])
        r = self.pair(summary=False)
        self.assertEqual(r['status'], 'VERIFIED_PARTIAL_PREFIX')
        self.assertEqual((r['committed_trials'], r['file']['rows'], r['file']['incomplete_tail_bytes']), (4, 4, 25))

    def test_nontrial_memory_not_counted(self):
        path = self.root/'memory.jsonl'
        spec = dict(self.m['pairs'][8], N=4096)
        def records():
            for axis in ('X', 'Y'):
                for i in range(4096):
                    yield self.zero(spec, axis, i)
                yield dict(memory_sample={'VmRSS_bytes': None}, completed_trial=4096, axis=axis)
        A.emit_file(path, records())
        r = V.verify_pair(path, spec)
        self.assertEqual((r['visible_trial_rows'], r['nontrial_rows']), (8192, 2))
        self.assertEqual(r['status'], 'INSUFFICIENT_EVIDENCE')

    @staticmethod
    def zero(spec, axis, index):
        base = dict(trial_id=V.trial(spec, axis, index), axis=axis, kind='ZERO_R', generator_exclusive_ns=(1, 2))
        if index < 16:
            return dict(base, event=None, evidence=None, angle=None)
        return dict(base, event_sha256=V.identity({'kind': 'ZERO_R', 'event': None}),
                    evidence_sha256=V.identity(None), angle_sha256=V.identity(None),
                    retention='HASHED_FULL_DOCUMENTS_NOT_RETAINED')

    def test_source_inputs_open_readonly_and_not_modified(self):
        before = {p.name: sha256(p.read_bytes()).hexdigest() for p in self.input.iterdir()}
        original = Path.open
        modes = []
        def opened(p, mode='r', *args, **kw):
            if p.is_relative_to(self.input):
                modes.append(mode)
                self.assertIn(mode, ('r', 'rb'))
            return original(p, mode, *args, **kw)
        with patch.object(Path, 'open', opened):
            self.assertEqual(self.verify()['status'], 'VERIFIED_COMPLETE')
        self.assertTrue(modes)
        after = {p.name: sha256(p.read_bytes()).hexdigest() for p in self.input.iterdir()}
        self.assertEqual(before, after)

    def test_cli_all_seven_outputs_new_outside_input(self):
        t = json.loads((Path(__file__).resolve().parents[3]/'docs/research/track_b_g6_w2c_b3_r1_f01_admission_integration_20261019/technical_contract_candidate.json').read_text())['formal_v2_template_not_executable']
        out = self.root/'reports'
        v, c, a = CLI.run(self.input/'manifest.json', out, source_root=Path(__file__).resolve().parents[3], acceptance_baseline=t)
        self.assertEqual(v['status'], 'VERIFIED_COMPLETE')
        self.assertEqual(len(list(out.iterdir())), 7)
        self.assertTrue(all(r['judgment'] != 'PASS' for r in a['checks'] if r['check_id'].startswith(('P0', 'P1', 'FUSED'))))
        with self.assertRaisesRegex(V.Invalid, 'NO_OVERWRITE'):
            CLI.run(self.input/'manifest.json', out, source_root=self.root, acceptance_baseline=t)
        with self.assertRaisesRegex(V.Invalid, 'OUTSIDE_INPUT'):
            CLI.run(self.input/'manifest.json', self.input/'reports', source_root=self.root, acceptance_baseline=t)

    def test_cross_pair_mismatch_on_compact_saved_hash(self):
        left = self.input/self.m['pairs'][8]['path']
        self.change_payloads(lambda docs: docs[16].update(event_sha256='f'*64), which=9)
        right = self.input/self.m['pairs'][9]['path']
        r = V.cross_pair(left, right, 'X')
        self.assertEqual(r['status'], 'FAIL')
        self.assertEqual(r['compared'], 16)

    def test_unknown_codec_tag_is_not_executed(self):
        with self.assertRaises(V.Invalid):
            V.decode(json.dumps({'int': '__import__("os")'}))

    def test_exact_decoder_negative_huge_rational_unicode(self):
        v = {'big': -(1 << 4096), 'r': F(7, 13), 'x': '\ud800\x7fé漢字', 'b': True}
        self.assertEqual(V.decode(V.canonical(v)), v)
        self.assertEqual(V.canonical(v), A.T.A.canonical(A.D.exact(v)))

    def test_parallel_adapter_refuses_to_infer_schema(self):
        r = V.PendingParallelAdapter().verify({'workers': 8})
        self.assertEqual(r['status'], 'PENDING_SERVER_CONTRACT')
        self.assertFalse(r['implemented'])

    def test_bounded_memory_does_not_scale_with_stream_size(self):
        peaks = []
        for n in (1000, 10000):
            path = self.root/f'bounded-{n}.jsonl'
            spec = dict(self.m['pairs'][8], N=n)
            A.emit_file(path, (self.zero(spec, axis, i) for axis in ('X', 'Y') for i in range(n)))
            tracemalloc.start()
            r = V.verify_pair(path, spec)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertEqual(r['committed_trials'], 2*n)
            peaks.append(dict(N=n, file_bytes=path.stat().st_size, peak_python_bytes=peak))
        self.assertLess(peaks[1]['peak_python_bytes'], peaks[0]['peak_python_bytes']*3 + 200000)
        self.assertGreater(peaks[1]['file_bytes'], peaks[0]['file_bytes']*8)
        if target := __import__('os').environ.get('B3_OFFLINE_MEMORY_REPORT'):
            Path(target).write_text(json.dumps(peaks, indent=2)+'\n')


PAIR_MUTATIONS = {
 'trial_gap': lambda d: d.pop(3),
 'trial_duplicate': lambda d: d.insert(3, deepcopy(d[2])),
 'trial_index': lambda d: d[0].update(trial_id=d[0]['trial_id'][:-1]+'2'),
 'phase': lambda d: d[0].update(trial_id=d[0]['trial_id'].replace('correctness', 'profile')),
 'pass': lambda d: d[0].update(trial_id=d[0]['trial_id'].replace('baseline', 'instrumented')),
 'axis': lambda d: d[0].update(axis='Y'),
 'kind': lambda d: d[0].update(kind='TYPED_FAILURE_IS_ZERO'),
 'event_binding': lambda d: d[0].update(kind='ZERO_R', event={'forged': True}),
 'angle_layer': lambda d: d[0].update(angle={'forged': True}),
 'unknown_record': lambda d: d.append({'unknown': True}),
 'unknown_codec_float': lambda d: d[0].update(generator_exclusive_ns=[-1, 2]),
}
for name, mutate in PAIR_MUTATIONS.items():
    def test(self, mutate=mutate, name=name):
        self.change_payloads(mutate)
        r = self.pair()
        self.assertEqual(r['status'], 'UNSUPPORTED_SCHEMA' if name == 'unknown_record' else 'INTEGRITY_MISMATCH', r)
    setattr(Verifier, 'test_rechained_' + name + '_refused', test)

SUMMARY_MUTATIONS = {
 'file_sha': lambda s: s['pairs'][0].update(file_sha256='f'*64),
 'chain_sha': lambda s: s['pairs'][0].update(output_chain_sha256='f'*64),
 'count': lambda s: s['pairs'][0]['counts'].update(EVENT=100),
 'rows': lambda s: s['pairs'][0].update(output_rows=0),
 'bytes': lambda s: s['pairs'][0].update(output_bytes=0),
 'axis_count': lambda s: s['pairs'][0]['axes'][0].update(N_executed=1),
 'trace': lambda s: s['pairs'][0].update(trace_sha256='f'*64),
 'prefix': lambda s: s['pairs'][0]['axes'][0].update(first_correctness_prefix_sha256='f'*64),
 'contract': lambda s: s.update(contract_digest='f'*64),
 'schedule_missing': lambda s: s['pairs'].pop(),
 'schema': lambda s: s.update(schema='future_parallel_v9'),
}
for name, mutate in SUMMARY_MUTATIONS.items():
    def test(self, mutate=mutate, name=name):
        mutate(self.s)
        self.summary_write()
        r = self.verify()
        self.assertEqual(r['status'], 'UNSUPPORTED_SCHEMA' if name == 'schema' else 'INTEGRITY_MISMATCH')
    setattr(Verifier, 'test_summary_' + name + '_refused', test)


class Failure(unittest.TestCase):
    def exercise(self, mode, change=None):
        with tempfile.TemporaryDirectory(prefix='b3-offline-f01-') as t:
            root = Path(t)/'input'
            spec, f = A.failure(root, mode)
            if change:
                change(f)
                # Force control-only provenance to avoid a trivial external/embedded mismatch.
                path = root/'pair.jsonl'
                docs = [r.payload for r in V.Reader(path) if r.payload.get('kind') != 'TYPED_FAILURE']
                path.unlink()
                A.emit_file(path, docs)
                (root/'pair.failure.json').write_text(json.dumps(dict(kind='TYPED_FAILURE', trial_failure_provenance=V.canonical(f))))
                # Rebind writer snapshot to unchanged prefix, leaving target mutation intact.
            return V.verify_pair(root/'pair.jsonl', spec, failure_path=root/'pair.failure.json')

    def test_sample_failure_no_stale_event(self):
        r = self.exercise('sample')
        self.assertEqual((r['status'], r['committed_trials'], r['visible_trial_rows']), ('TECHNICAL_FAILURE', 1, 1))
        self.assertFalse(r['failure']['current_attempt']['result_acquired'])

    def test_partial_write_tail_not_trial(self):
        r = self.exercise('partial')
        self.assertEqual(r['status'], 'TECHNICAL_FAILURE', r)
        self.assertEqual(r['committed_trials'], 1)
        self.assertEqual(r['file']['incomplete_tail_bytes'], 13)

    def test_flush_complete_row_not_trial_commit(self):
        r = self.exercise('flush')
        self.assertEqual(r['status'], 'TECHNICAL_FAILURE', r)
        self.assertEqual((r['committed_trials'], r['visible_trial_rows']), (1, 2))
        self.assertEqual(r['evidence']['candidate_trace'], V.RAW)
        self.assertFalse(r['evidence']['failure_prefix']['producer_fsync_durability_verified'])

    def test_file_hash_failure_keeps_unknown_writer_hash(self):
        r = self.exercise('file_hash')
        self.assertEqual(r['status'], 'TECHNICAL_FAILURE', r)
        self.assertEqual((r['committed_trials'], r['visible_trial_rows']), (1, 2))
        self.assertIsNone(r['failure']['stream_prefix_before_failure_record']['written_bytes_sha256'])
        self.assertFalse(r['failure']['stream_prefix_before_failure_record']['file_hash_covers_written_bytes'])

    def test_stale_inflight_rejected(self):
        r = self.exercise('sample', lambda f: f.update(inflight_event={'old_event': True}))
        self.assertEqual(r['status'], 'INTEGRITY_MISMATCH')
        self.assertIn('STALE_INFLIGHT_EVENT', r['issues'])

    def test_uncompleted_trial_miscount_rejected(self):
        r = self.exercise('sample', lambda f: f['completed_prefix'].update(EVENT=2))
        self.assertEqual(r['status'], 'INTEGRITY_MISMATCH')

    def test_candidate_ready_without_record_rejected(self):
        r = self.exercise('sample', lambda f: f['current_attempt'].update(candidate_ready=True))
        self.assertEqual(r['status'], 'INTEGRITY_MISMATCH')

    def test_retry_resume_refused(self):
        for field, value in (('retry', 1), ('resume', True)):
            with self.subTest(field=field):
                self.assertEqual(self.exercise('sample', lambda f: f.update({field: value}))['status'], 'INTEGRITY_MISMATCH')

    def test_parent_failure_prefix_not_complete_result(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'input'
            m,s=A.bundle(root)
            (root/'parent_failure.json').write_text(json.dumps(dict(kind='TYPED_FAILURE', completed_pairs=s['pairs'][:1], mandatory_STOP=True, retry=0, resume=False)))
            m['failure_prefix']='parent_failure.json'
            (root/'manifest.json').write_text(json.dumps(m))
            r=V.verify_bundle(root/'manifest.json')
            self.assertEqual(r['status'],'TECHNICAL_FAILURE')
            self.assertEqual(C.aggregate(r)['comparison'],[])


class Costs(Verifier):
    # Separate only this class's methods are loaded by explicit suite below.
    def test_cost_coordinates_setup_once_and_separate_passes(self):
        c=C.aggregate(self.verify())
        self.assertEqual(c['status'],'COMPLETE_ARTIFICIAL_DESCRIPTIVE_COSTS')
        self.assertEqual(len(c['comparison']),16)
        self.assertEqual({r['pass_name'] for r in c['comparison']},{'baseline','instrumented'})
        for p in c['comparison']:
            self.assertTrue(p['shared_XY_setup_charged_once'])
            self.assertTrue(p['full_prepare_included_in_constructor_and_preflight'])
            self.assertEqual(p['setup_cpu_ns'],sum(v['cpu_ns'] for k,v in p['coarse_partition']['rows'].items() if k in C.SETUP))
        self.assertIsNone(c['receipt_self_write_cost_ns'])
        self.assertIsNone(c['ranking'])
        self.assertFalse(c['pure_observer_overhead_certified'])
        self.assertFalse(c['RSS_sum_is_unique_physical_memory'])
        self.assertFalse(c['parent_child_wall_added'])

    def test_failed_bundle_excluded(self):
        (self.input/'completion_receipt.json').unlink()
        self.assertFalse(C.aggregate(self.verify())['comparison'])

    def bad_cost(self, fn):
        r=self.verify();fn(r)
        c=C.aggregate(r)
        self.assertEqual(c['status'],'COST_INTEGRITY_MISMATCH',c)
        self.assertFalse(c['comparison'])

    def test_coarse_fine_added_double_total_refused(self):
        self.bad_cost(lambda r:r['summary']['pairs'][0]['coarse'].update(fine_added=dict(calls=1,cpu_ns=100,wall_ns=100)))

    def test_parent_wall_inserted_into_child_refused(self):
        self.bad_cost(lambda r:r['summary']['pairs'][0]['coarse'].update(shared_parent_wall=dict(calls=1,cpu_ns=0,wall_ns=100)))

    def test_unknown_receipt_zero_refused(self):
        self.bad_cost(lambda r:r['completion_receipt'].update(receipt_self_write_cost_ns=0))

    def test_full_prepare_extra_constructor_refused(self):
        self.bad_cost(lambda r:r['summary']['pairs'][0]['coarse'].update({'setup.full_prepare':dict(calls=1,cpu_ns=1,wall_ns=1)}))

    def test_fine_baseline_mixing_refused(self):
        self.bad_cost(lambda r:r['summary']['pairs'][0].update(functions={}))

    def test_missing_coordinate_preserved_null(self):
        c=C.aggregate(self.verify())
        a=next(r for r in c['comparison'] if r['layer']=='A')
        self.assertIsNone(a['angle_requery_cpu_wall_ns'])
        self.assertIsNone(a['periodic_max_observed_RSS_bytes'])

    def test_raw_generator_cost_summary_mismatch_refused(self):
        self.bad_cost(lambda r:r['summary']['pairs'][0]['coarse']['online.EVENT.generator_and_audit'].update(calls=0))

    def test_invalid_memory_value_refused(self):
        self.bad_cost(lambda r:r['summary']['pairs'][0]['memory'].update(VmRSS_bytes=-1))

    def test_partition_residual_and_negative_cost_refused(self):
        for row in ({'x':dict(calls=1,cpu_ns=-1,wall_ns=1)}, {'x':dict(calls=1,cpu_ns=100,wall_ns=100)}):
            with self.assertRaises(V.Invalid):
                C.partition(row,dict(exclusive_sum_ns=[0,0],outer_total_ns=[10,10],unattributed_and_observer_residual_ns=[10,10],unknown_bit_operations=None,includes_observer_effect=True))


# Avoid rerunning inherited verifier tests in Costs: meaningful IDs counted once.
def load_tests(loader, standard, pattern):
    suite=unittest.TestSuite()
    for cls in (Verifier,Failure,Costs,Acceptance):
        for name in sorted(cls.__dict__):
            if name.startswith('test_'):
                suite.addTest(cls(name))
    return suite


class Acceptance(unittest.TestCase):
    def test_canonical_acceptance_against_actual_reference(self):
        rows=H.canonical_checks(lambda v:A.T.A.canonical(A.D.exact(v)))
        self.assertTrue(all(r['judgment']=='PASS' for r in rows))

    def test_canonical_wrong_callback_fails(self):
        self.assertTrue(any(r['judgment']=='FAIL' for r in H.canonical_checks(lambda v:'null')))

    def test_full_source_digest_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'driver.py';p.write_text('artificial')
            base={k:None for k in ('input','policy','cells','passes','correctness_trials_per_cell','decisions','one_shot','call_budget','schedule_proposal','source_pins')}
            rows=H.identity_checks(t,base,base,{'driver.py':'f'*64},candidate_commit=V.BASE_COMMIT)
            self.assertEqual(rows[0]['judgment'],'FAIL')

    def test_contract_changes_require_new_contract_despite_trace_match(self):
        base={k:None for k in ('input','policy','cells','passes','correctness_trials_per_cell','decisions','one_shot','call_budget','schedule_proposal','source_pins')}
        candidate=dict(base,passes=['baseline'])
        rows=H.identity_checks('.',base,candidate,{},candidate_commit=V.BASE_COMMIT)
        self.assertEqual(next(r for r in rows if r['check_id']=='A_CONTRACT_passes')['judgment'],'REQUIRES_CONTRACT_CHANGE')

    def test_observer_path_function_change_not_pass(self):
        r=H.observer_check(['driver.py'],['driver.py','new.py'],['exact'],['fused'],V.BASE_COMMIT)
        self.assertEqual(r['judgment'],'REQUIRES_CONTRACT_CHANGE')

    def test_pending_parallel_and_unknown_evidence_not_pass(self):
        base={k:None for k in ('input','policy','cells','passes','correctness_trials_per_cell','decisions','one_shot','call_budget','schedule_proposal','source_pins')}
        r=H.build_report('.',base,{})
        pending=[x for x in r['checks'] if x['check_id'] in H.PARALLEL]
        self.assertEqual(len(pending),11)
        self.assertTrue(all(x['judgment']=='PENDING_EVIDENCE' and x['fixed_commit']is None for x in pending))

    def test_no_scientific_imports(self):
        for n in ('g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator'):
            self.assertNotIn('trottertracks.algorithm_codesign.'+n,sys.modules)


RAW_MUTATIONS = {
 'chain': lambda lines: [lines[0].replace(b'"previous":"0', b'"previous":"f', 1)] + lines[1:],
 'mid_file': lambda lines: lines[:3] + [b'{broken}\n'] + lines[4:],
 'duplicate_raw_line': lambda lines: lines[:3] + [lines[2]] + lines[3:],
 'missing_raw_line': lambda lines: lines[:3] + lines[4:],
 'payload_without_rehash': lambda lines: [lines[0].replace(b'generator_exclusive_ns', b'generator_exclusive_nt')] + lines[1:],
}
for name, mutate in RAW_MUTATIONS.items():
    def test(self, mutate=mutate):
        path=self.input/self.m['pairs'][0]['path']
        path.write_bytes(b''.join(mutate(path.read_bytes().splitlines(keepends=True))))
        self.assertEqual(self.pair()['status'],'INTEGRITY_MISMATCH')
    setattr(Verifier,'test_raw_'+name+'_refused',test)

MANIFEST_MUTATIONS = {
 'N':lambda m:m['pairs'][0].update(N=17),
 'producer_commit':lambda m:m.update(producer_commit='0'*40),
 'contract_digest':lambda m:m.update(contract_digest='f'*64),
 'not_artificial':lambda m:m.update(artificial_only=False),
 'pair_path_duplicate':lambda m:m['pairs'][1].update(path=m['pairs'][0]['path']),
 'path_escape':lambda m:m['pairs'][0].update(path='../outside.jsonl'),
 'unsupported_schema':lambda m:m.update(schema='future_parallel_v1'),
}
for name, mutate in MANIFEST_MUTATIONS.items():
    def test(self,mutate=mutate,name=name):
        mutate(self.m);self.manifest_write()
        self.assertEqual(self.verify()['status'],'UNSUPPORTED_SCHEMA' if name=='unsupported_schema' else 'INTEGRITY_MISMATCH')
    setattr(Verifier,'test_manifest_'+name+'_refused',test)

CODEC_BAD = ['{"int":"0x00"}', '{"int":"0xA"}', '{"rat":["0x2","0x4"]}',
             '{"rat":["0x1","0x0"]}', '{"x":1}', '{"x":1.0}', '{"x":NaN}',
             '{"x":null,"x":null}', '{ "x":null}']
for i,text in enumerate(CODEC_BAD):
    def test(self,text=text):
        with self.assertRaises((V.Invalid,ValueError)):
            V.decode(text)
    setattr(Acceptance,'test_bad_codec_'+str(i)+'_refused',test)
