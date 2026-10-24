"""Focused approval/environment/convention tests. No real issuance or science."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier,FORBIDDEN
guard=Barrier();sys.meta_path.insert(0,guard)
try:
    from trottertracks.b2d_evaluator import manual_authority as manual
    from trottertracks.b2d_evaluator.environment import verify_environment
    from trottertracks.b2d_evaluator import _published_reference as published
    from trottertracks.b2d_evaluator.scope import candidate_graph,registry
    from trottertracks.pr_dictionary_validation.authorization import _canonical,file_sha
finally:
    sys.meta_path.remove(guard)


class ExecutionReadyTests(unittest.TestCase):
    def setUp(self):
        guard=Barrier();sys.meta_path.insert(0,guard)
        self.addCleanup(sys.meta_path.remove,guard)

    def test_approval_draft_is_not_adopted_or_issued(self):
        draft=dict(proposal_status='UNAPPROVED_TEMPLATE',stage='B2D_STAGE1',value='MOCK')
        adopted=json.loads(manual.adoption_bytes(draft))
        self.assertEqual(adopted['status'],'USER_ADOPTED_STAGE1')
        self.assertEqual(draft['proposal_status'],'UNAPPROVED_TEMPLATE')
        for bad in ({},dict(proposal_status='APPROVED')):
            with self.assertRaises(PermissionError): manual.adoption_bytes(bad)

    def test_non_admin_cannot_publish_a_manual_grant(self):
        # Denied before any filesystem access; no issuance is performed.
        with patch('os.geteuid',return_value=1000),self.assertRaises(PermissionError):
            manual.issue_manual_packet('not read','not created','not read','MOCK','MOCK')

    def test_agent_owned_or_symlink_receipt_cannot_authorize(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'mock.json';file.write_text('{}')
            with self.assertRaises(PermissionError): manual.protected_receipt(file)
            link=Path(directory)/'link.json';link.symlink_to(file)
            with self.assertRaises(PermissionError): manual.protected_receipt(link)

    def test_manual_receipt_binds_source_scope_environment_output_and_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory)
            contract=dict(source_commit='MOCK_SOURCE',recipe_fingerprint='MOCK_RECIPE',graph_layout='MOCK_LAYOUT',
                          call_scope={'mock':'scope'},environment_binding={'mock':'env'})
            grant=dict(grant_id='b2d_MOCK_NOT_REAL',approval_reference='MOCK_USER_RECORD',
                       fresh_output_namespace='/tmp/MOCK_NOT_CREATED')
            cb,gb,approval=_canonical(contract),_canonical(grant),b'MOCK_TEXT_NOT_USER_APPROVAL'
            (packet/'contract.json').write_bytes(cb);(packet/'authorization.json').write_bytes(gb)
            receipt=manual.receipt_fields(packet,approval,gb,cb)
            # Only read-only receipt schema matching; trusted OS read is stubbed.
            with patch.object(manual,'protected_receipt',return_value=_canonical(receipt)):
                self.assertTrue(manual.MANUAL_AUTHORITY.verify_approval(packet,approval,file_sha(gb),file_sha(cb)))
                with self.assertRaises(PermissionError):
                    manual.MANUAL_AUTHORITY.verify_approval(packet,b'CHANGED',file_sha(gb),file_sha(cb))
                with self.assertRaises(PermissionError):
                    manual.MANUAL_AUTHORITY.verify_approval(packet,approval,'wrong',file_sha(cb))
            for field in ('source_commit','scope_sha256','environment_sha256','fresh_output_namespace','packet_dir'):
                bad={**receipt,field:'WRONG'}
                with self.subTest(field=field),patch.object(manual,'protected_receipt',return_value=_canonical(bad)),self.assertRaises(PermissionError):
                    manual.MANUAL_AUTHORITY.verify_approval(packet,approval,file_sha(gb),file_sha(cb))

    def test_missing_protected_record_never_authorizes(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory)
            (packet/'authorization.json').write_text('{"grant_id":"b2d_MOCK_NOT_ISSUED"}')
            (packet/'contract.json').write_text('{}')
            with self.assertRaises(PermissionError):
                manual.MANUAL_AUTHORITY.verify_approval(packet,b'MOCK',
                    file_sha((packet/'authorization.json').read_bytes()),file_sha(b'{}'))

    def test_environment_mismatch_fails_without_scientific_import(self):
        expected=dict(mock='PINNED_ENVIRONMENT')
        with patch('trottertracks.b2d_evaluator.environment.inspect_environment',return_value=expected):
            verify_environment(expected)
            with self.assertRaises(PermissionError): verify_environment(dict(mock='WRONG'))

    def test_frozen_real_environment_metadata_is_complete(self):
        env=json.loads((ROOT/'docs/research/b2d_stage1_execution_ready_20261025/environment.json').read_text())
        self.assertEqual(env['python_runtime']['version'],[3,11,9,'final',0])
        self.assertEqual(len(env['dependency_versions']),50)
        self.assertEqual(env['dependency_versions']['qiskit-nature'],'0.7.2')
        self.assertEqual(env['dependency_versions']['numpy'],'1.26.4')
        self.assertTrue(env['dependency_requirements_consistent'])
        self.assertEqual(set(env['dependency_versions']),set(env['distribution_records']))

    def test_upstream_reference_constructor_asts_are_unchanged(self):
        expected={
            '_get_kernel_s2_sequence':'bd6b132c220d7feda8870218256d557ef4043c9eb78112dec8121cf91ebec406',
            '_get_processor_s2_sequence':'656108db36a8a0b9b2439ea3e4bfb40fd6fdb7523becf6d3f5689d6da87fe813',
            '_get_s2_sequence':'6a1d3143d1cb487b23e20953c9307b2abd0cfafd4c46aceb17c2fb34e702fad9'}
        tree=ast.parse(Path(published.__file__).read_text())
        actual={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in expected}
        self.assertEqual(actual,expected)

    def test_published_reference_is_registered_without_changing_grid_or_cache_keys(self):
        s=registry(candidate_graph('MOCK'))
        self.assertEqual(s['rows'],36);self.assertEqual(s['dispatch_total'],941)
        self.assertEqual(s['boundaries']['backend.expm'],448)
        self.assertEqual(s['boundaries']['published.convention'],4)
        self.assertEqual(s['max_event_records'],3134)
        from test_b2d_stage1_launcher import MockBackend,MockPermit,verified,mock_loader,mock_save
        from trottertracks.b2d_evaluator.launcher import _run_claimed
        with tempfile.TemporaryDirectory() as directory:
            permit=MockPermit()
            status=_run_claimed(verified(directory),permit,mock_only=True,input_loader=mock_loader,
                backend_factory=lambda settings:MockBackend(permit.scope),input_saver=mock_save)
            self.assertEqual(status['calls']['completed']['published.convention'],4)
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
            pub=[i for i,v in enumerate(events) if v.get('event')=='complete' and v.get('boundary')=='published.convention']
            comparison=next(i for i,v in enumerate(events) if v.get('boundary')=='dag.operator_error')
            self.assertLess(max(pub),comparison)

    def test_no_scientific_dependency_was_imported(self):
        self.assertFalse(FORBIDDEN.intersection(n.split('.')[0] for n in sys.modules))

    def test_published_convention_failure_stops_before_performance_comparison(self):
        from test_b2d_stage1_launcher import MockBackend,MockPermit,verified,mock_loader,mock_save
        from trottertracks.b2d_evaluator.launcher import _run_claimed
        from trottertracks.b2d_evaluator.errors import Stage1Failure
        class WrongConvention(MockBackend):
            def perform(self,kind,args,params):
                if kind=='a4_reference_check':
                    def fail(): raise Stage1Failure('PUBLISHED_CONVENTION_CORRECTNESS_FAILURE')
                    return self.scope.call('published.convention',fail,identity=(str(params[0]),params[1]))
                return super().perform(kind,args,params)
        with tempfile.TemporaryDirectory() as directory:
            permit=MockPermit()
            result=_run_claimed(verified(directory),permit,mock_only=True,input_loader=mock_loader,
                backend_factory=lambda s:WrongConvention(permit.scope),input_saver=mock_save)
            self.assertEqual(result['reason'],'PUBLISHED_CONVENTION_CORRECTNESS_FAILURE')
            self.assertNotIn('dag.operator_error',result['calls']['attempted'])
            self.assertNotIn('dag.signals',result['calls']['attempted'])
            self.assertTrue((Path(directory)/'STOP.json').is_file())

    def test_existing_atomic_claim_refuses_an_existing_marker(self):
        import os
        from trottertracks.pr_dictionary_validation.authorization import claim_packet
        # Mock the existing OS exclusive-create boundary: no actual claim or
        # grant marker is written, and output creation cannot be reached.
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'MOCK_ALREADY_USED'
            marker.write_text('MOCK_NO_GRANT')
            output=NS(mkdir=lambda **kwargs:self.fail('Duplicate claim created output'))
            with patch('os.open',side_effect=FileExistsError) as create,self.assertRaises(FileExistsError):
                claim_packet(dict(consumption_path=marker,output=output,authorization_digest='MOCK'))
            self.assertTrue(create.call_args.args[1] & os.O_EXCL)
            self.assertEqual(marker.read_text(),'MOCK_NO_GRANT')


if __name__=='__main__': unittest.main()
