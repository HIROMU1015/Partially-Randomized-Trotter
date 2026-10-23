"""Artificial evidence only, plus fixed public metadata coverage checks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h2_pr_identity_recovery as r
import prepare_h2_pr_identity_recovery as prepare


def event(order=0, component='COMPONENT_ROTATION'):
    ids = ['COMPONENT_PRODUCT_' + str(i) for i in range(order)] + [component]
    return dict(taylor_order=order, product_component_ids=ids[:-1],
                rotation_component_id=component, selected_component_ids=ids,
                phase=dict(real=1., imag=0.), application_sequence=[
                    dict(application_index=i, component_id=c, role='product' if i < order else 'rotation')
                    for i, c in enumerate(ids)])


def event_fixture():
    value = dict(events=[[event(), event(2)], [event(4)]], conditional=False, cohort='EXPLORATION')
    value['event_digest'] = r.digest(value['events'])
    anchor = dict(task=dict(id='ARTIFICIAL_TASK'), result=dict(saved=dict(event_digest=value['event_digest'])))
    return value, anchor, [2, 1]


class Events(unittest.TestCase):
    def refreshed(self, value, anchor):
        anchor['result']['saved']['event_digest'] = value['event_digest'] = r.digest(value['events'])

    def test_saved_full_event_body_is_accepted_as_metadata_only(self):
        result = r.check_events(*event_fixture())
        self.assertEqual(result['status'], 'SAVED_EVENT_CONTENT_MATCH')

    def test_event_digest_change_rejected(self):
        value, anchor, counts = event_fixture(); value['events'][0].reverse()
        with self.assertRaisesRegex(r.EvidenceError, 'DIGEST_JOIN'):r.check_events(value, anchor, counts)

    def test_conditional_or_other_cohort_rejected(self):
        for key, val in [('conditional', True), ('cohort', 'DESIGN_PILOT')]:
            with self.subTest(key=key):
                value, anchor, counts = event_fixture(); value[key] = val
                with self.assertRaisesRegex(r.EvidenceError, 'COHORT'):r.check_events(value, anchor, counts)

    def test_signed_piece_coverage_rejected(self):
        value, anchor, counts = event_fixture()
        with self.assertRaisesRegex(r.EvidenceError, 'SIGNED_PIECES'):r.check_events(value, anchor, counts[:1])

    def test_wrong_piece_event_count_rejected(self):
        value, anchor, _ = event_fixture()
        with self.assertRaisesRegex(r.EvidenceError, 'PIECE_COUNT'):r.check_events(value, anchor, [1, 2])

    def test_application_order_rejected_even_with_refreshed_digest(self):
        value, anchor, counts = event_fixture(); value['events'][0][1]['application_sequence'].reverse()
        self.refreshed(value, anchor)
        with self.assertRaisesRegex(r.EvidenceError, 'APPLICATION_ORDER'):r.check_events(value, anchor, counts)

    def test_boolean_order_rejected(self):
        value, anchor, counts = event_fixture(); value['events'][0][0]['taylor_order'] = False
        self.refreshed(value, anchor)
        with self.assertRaisesRegex(r.EvidenceError, 'EVENT_ORDER'):r.check_events(value, anchor, counts)

    def test_component_join_rejected(self):
        value, anchor, counts = event_fixture(); value['events'][0][0]['selected_component_ids'] = [True]
        self.refreshed(value, anchor)
        with self.assertRaisesRegex(r.EvidenceError, 'COMPONENTS'):r.check_events(value, anchor, counts)

    def test_boolean_phase_rejected(self):
        value, anchor, counts = event_fixture(); value['events'][0][0]['phase']['real'] = True
        self.refreshed(value, anchor)
        with self.assertRaisesRegex(r.EvidenceError, 'PHASE'):r.check_events(value, anchor, counts)


class TemporaryEvidence(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='h2-recovery-artificial-')
        self.addCleanup(temp.cleanup); self.root = Path(temp.name)


class Files(TemporaryEvidence):
    def make_archive(self, raw=b'ARTIFICIAL_QPY_OPAQUE_BYTES' * 500):
        path = self.root / 'compiled.qpy.zlib'; path.write_bytes(zlib.compress(raw))
        stored = dict(bytes=path.stat().st_size, sha256=r.file_digest(path))
        original = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return path, stored, original

    def test_archive_hash_and_raw_hash_without_deserialization_or_output(self):
        result = r.verify_archive(*self.make_archive())
        self.assertEqual(result['status'], 'ARCHIVE_AND_RAW_BYTES_MATCH')
        self.assertFalse(result['scientific_body_decoded']); self.assertFalse(result['raw_file_written'])
        self.assertEqual([p.name for p in self.root.iterdir()], ['compiled.qpy.zlib'])

    def test_corrupt_compressed_bytes_rejected(self):
        path, stored, original = self.make_archive(); path.write_bytes(b'X' * stored['bytes'])
        with self.assertRaisesRegex(r.EvidenceError, 'ARCHIVE_SHA_OR_SIZE'):r.verify_archive(path, stored, original)

    def test_wrong_original_raw_hash_rejected(self):
        path, stored, original = self.make_archive(); original['sha256'] = '0' * 64
        with self.assertRaisesRegex(r.EvidenceError, 'RAW_SHA_OR_SIZE'):r.verify_archive(path, stored, original)

    def test_trailing_archive_data_rejected_with_matching_compressed_hash(self):
        path, stored, original = self.make_archive(); path.write_bytes(path.read_bytes() + b'TRAILING')
        stored.update(bytes=path.stat().st_size, sha256=r.file_digest(path))
        with self.assertRaisesRegex(r.EvidenceError, 'TRAILING'):r.verify_archive(path, stored, original)

    def test_truncated_archive_rejected_with_matching_compressed_hash(self):
        path, stored, original = self.make_archive(); path.write_bytes(path.read_bytes()[:-1])
        stored.update(bytes=path.stat().st_size, sha256=r.file_digest(path))
        with self.assertRaisesRegex(r.EvidenceError, 'INCOMPLETE'):r.verify_archive(path, stored, original)

    def test_inflation_cannot_exceed_declared_raw_size(self):
        path, stored, original = self.make_archive(); original['bytes'] = 10
        with self.assertRaisesRegex(r.EvidenceError, 'EXCEEDS_DECLARED'):r.verify_archive(path, stored, original)

    def test_archive_change_during_read_rejected(self):
        path, stored, original = self.make_archive()
        with patch.object(r, 'file_digest', side_effect=[stored['sha256'], '0' * 64]):
            with self.assertRaisesRegex(r.EvidenceError, 'CHANGED_DURING_READ'):r.verify_archive(path, stored, original)

    def test_expected_raw_size_cannot_be_boolean(self):
        path, stored, original = self.make_archive(); original['bytes'] = True
        with self.assertRaisesRegex(r.EvidenceError, 'RAW_SIZE'):r.verify_archive(path, stored, original)

    def test_explicit_exporter_map(self):
        path = self.root / 'map.json'; path.write_text(json.dumps(dict(roots={str(self.root):'NEW_EXECUTION'}, original_sha256={})))
        self.assertEqual(r.root_from_map(path), self.root)

    def test_duplicate_or_relative_root_mapping_rejected(self):
        for roots in ({str(self.root):'NEW_EXECUTION',str(self.root/'other'):'NEW_EXECUTION'}, {'relative':'NEW_EXECUTION'}):
            with self.subTest(roots=roots):
                path = self.root / 'map.json'; path.write_text(json.dumps(dict(roots=roots)))
                with self.assertRaisesRegex(r.EvidenceError, 'UNIQUE_ABSOLUTE'):r.root_from_map(path)

    def test_locator_traversal_and_foreign_token_rejected(self):
        for suffix in ('../outside', '/absolute', 'a//b', 'a/./b'):
            with self.subTest(suffix=suffix):
                with self.assertRaises(r.EvidenceError):r.resolve(self.root, r.TOKEN + suffix)
        with self.assertRaisesRegex(r.EvidenceError, 'EXECUTION_TOKEN'):r.resolve(self.root, '${OTHER}/file')

    def test_symlink_escape_rejected(self):
        base = self.root/'original'; base.mkdir(); (base/'escape').symlink_to(self.root)
        with self.assertRaisesRegex(r.EvidenceError, 'ESCAPES_ROOT'):r.resolve(base, r.TOKEN + 'escape/file')

    def test_existing_report_is_never_overwritten(self):
        report = self.root/'report.json'; report.write_text('PRESERVE')
        with patch.object(r, 'Catalogue'), patch.object(sys, 'argv', ['tool', '--report', str(report)]):
            with self.assertRaisesRegex(r.EvidenceError, 'NEW_REPORT_REQUIRED'):r.main()
        self.assertEqual(report.read_text(), 'PRESERVE')

    def test_report_in_other_worktree_rejected(self):
        with patch.object(r, 'Catalogue'), patch.object(r, 'ROOT', self.root/'own'), \
             patch.object(sys, 'argv', ['tool', '--report', str(self.root/'other.json')]):
            with self.assertRaisesRegex(r.EvidenceError, 'RECOVERY_WORKTREE'):r.main()

    def test_unrelated_worktree_documentation_change_is_observation_only(self):
        path = self.root/'other_worktree_PROJECT_MAP.md'; path.write_text('CONCURRENT_DOC_UPDATE')
        cfg = dict(files=[],accepted_diagnostics=dict(run_root=str(self.root/'old_run')))
        rows = prepare.check_preservation([dict(path=str(path),sha256='0'*64)],cfg,self.root/'contract.json')
        self.assertFalse(rows[0]['matches']); self.assertFalse(rows[0]['blocking'])

    def test_actual_input_or_diagnostic_result_change_blocks(self):
        input_path = self.root/'scientific_input'; input_path.write_text('CHANGED')
        run = self.root/'old_run'; run.mkdir(); result = run/'result.json'; result.write_text('CHANGED')
        cfg = dict(files=[dict(path=str(input_path))],accepted_diagnostics=dict(run_root=str(run)))
        for path in (input_path,result):
            with self.subTest(path=path.name):
                with self.assertRaisesRegex(r.EvidenceError,'EXECUTION_BOUND_ORIGINAL_CHANGED'):
                    prepare.check_preservation([dict(path=str(path),sha256='0'*64)],cfg,self.root/'contract.json')


class Configuration(TemporaryEvidence):
    def fixture(self):
        catalogue = r.Catalogue.__new__(r.Catalogue)
        keys = dict(SOURCE=r.ACQUISITION, stage='H2_PR_EXPLORATION', binding={'ARTIFICIAL':True},
                    preparation_hash='a'*64, compiler_phase_policy='ARTIFICIAL')
        tasks = [dict(id='ARTIFICIAL_' + str(i), sample=i) for i in range(67)]
        catalogue.context = {k:dict(saved=v) for k,v in keys.items()}
        catalogue.tasks = [dict(task=t) for t in tasks]
        value = dict(keys, tasks=tasks); path = self.root/'config.json'; path.write_text(json.dumps(value))
        catalogue.origins = {'O00002':dict(bytes=path.stat().st_size, original_sha256=r.file_digest(path))}
        return catalogue, path, value

    def test_exact_original_config_and_task_join(self):
        catalogue, path, _ = self.fixture(); catalogue.verify_config(path)

    def test_original_config_sha_mismatch_rejected(self):
        catalogue, path, _ = self.fixture(); raw=path.read_bytes(); path.write_bytes(raw.replace(b'ARTIFICIAL_0', b'ARTIFICIAL_X', 1))
        with self.assertRaisesRegex(r.EvidenceError, 'SHA_MISMATCH'):catalogue.verify_config(path)

    def test_refreshed_hash_does_not_hide_missing_task(self):
        catalogue, path, value = self.fixture(); value['tasks'].pop(); path.write_text(json.dumps(value))
        catalogue.origins['O00002'].update(bytes=path.stat().st_size, original_sha256=r.file_digest(path))
        with self.assertRaisesRegex(r.EvidenceError, '67_TASK_JOIN'):catalogue.verify_config(path)

    def test_refreshed_hash_does_not_hide_wrong_policy(self):
        catalogue, path, value = self.fixture(); value['compiler_phase_policy']='OTHER'; path.write_text(json.dumps(value))
        catalogue.origins['O00002'].update(bytes=path.stat().st_size, original_sha256=r.file_digest(path))
        with self.assertRaisesRegex(r.EvidenceError, 'CONFIG_FIELD'):catalogue.verify_config(path)

    def test_partial_recovery_preserves_blocked_state(self):
        value, anchor, counts = event_fixture()
        path = self.root/'events.json'; path.write_text(json.dumps(value))
        catalogue = r.Catalogue.__new__(r.Catalogue); catalogue.tasks = [anchor]
        catalogue.requests = lambda:[dict(id='EVENT__ARTIFICIAL_TASK',task='ARTIFICIAL_TASK',
            locator=r.TOKEN+'events.json',piece_counts=counts),
            dict(id='MISSING',locator=r.TOKEN+'missing.json',bytes=1,sha256='0'*64)]
        with patch.object(catalogue, 'verify_config') as gate:
            result = catalogue.recover(self.root)
        gate.assert_called_once()
        self.assertEqual([x['status'] for x in result['recovered']],
                         ['SAVED_EVENT_CONTENT_MATCH','MISSING_ORIGINAL_FILE'])
        self.assertEqual(result['PR_reuse'],'BLOCKED'); self.assertFalse(result['execution_ready'])

    def test_config_failure_cannot_start_archive_reads(self):
        catalogue = r.Catalogue.__new__(r.Catalogue); catalogue.requests = lambda:[]
        with patch.object(catalogue,'verify_config',side_effect=r.EvidenceError('ORIGINAL_SHA_MISMATCH')), \
             patch.object(r,'verify_archive',side_effect=AssertionError('RAW_READ')):
            with self.assertRaisesRegex(r.EvidenceError,'SHA_MISMATCH'):catalogue.recover(self.root)

    def test_bad_original_is_rejected_without_hiding_later_missing_evidence(self):
        path = self.root/'bad.json'; path.write_text('BAD')
        catalogue = r.Catalogue.__new__(r.Catalogue); catalogue.tasks = []
        catalogue.requests = lambda:[dict(id='BAD',locator=r.TOKEN+'bad.json',bytes=3,sha256='0'*64),
            dict(id='MISSING',locator=r.TOKEN+'missing.json',bytes=1,sha256='0'*64)]
        with patch.object(catalogue,'verify_config'):
            result = catalogue.recover(self.root)
        self.assertEqual([x['status'] for x in result['recovered']],['EVIDENCE_REJECTED','MISSING_ORIGINAL_FILE'])
        self.assertFalse(result['identity_witness_created']); self.assertEqual(result['scientific_calls'],0)


class FixedMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.catalogue = r.Catalogue()

    def test_fixed_67_tasks_134_axes_402_archives_and_event_request_counts(self):
        requests = self.catalogue.requests(); events = [q for q in requests if q['id'].startswith('EVENT__')]
        self.assertEqual(len(events),67); self.assertEqual(sum('archive' in q for q in requests),402)
        self.assertEqual(sum(len(q['piece_counts']) for q in events),336)
        self.assertEqual(sum(sum(q['piece_counts']) for q in events),6400)
        self.assertEqual(len({q['locator'] for q in requests}),len(requests))

    def test_no_locator_stays_blocked_without_opening_private_originals(self):
        with patch.object(r, 'verified_bytes', side_effect=AssertionError('PRIVATE_READ')), \
             patch.object(r, 'verify_archive', side_effect=AssertionError('RAW_READ')):
            result = self.catalogue.recover()
        self.assertFalse(result['execution_ready']); self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['identity_witness_created']); self.assertEqual(result['live_grants'],0)
        self.assertTrue(all(x['status']=='MISSING_CANONICAL_ROOT' for x in result['recovered']))

    def test_nonfinite_and_duplicate_json_keys_rejected(self):
        for text in ('{"events":[],"events":[]}', '{"phase":NaN}'):
            with self.subTest(text=text):
                with self.assertRaises(r.EvidenceError):r.loads(text)

    def test_no_scientific_import_or_control_plane_import(self):
        self.assertFalse(any(k.split('.')[0] in ('numpy','scipy','qiskit','trotterlib','trottertracks') for k in sys.modules))
        self.assertNotIn('h2_accepted_compare_backend', sys.modules)


if __name__ == '__main__':unittest.main()
