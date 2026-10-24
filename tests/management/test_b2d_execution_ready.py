"""Retained scientific reference/gate checks; stdlib mocks only."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier,FORBIDDEN
from trottertracks.b2d_evaluator import _published_reference as published
from trottertracks.b2d_evaluator.scope import candidate_graph,registry


class ExecutionReadyTests(unittest.TestCase):
    def setUp(self):
        guard=Barrier();sys.meta_path.insert(0,guard)
        self.addCleanup(sys.meta_path.remove,guard)

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
        from test_b2d_stage1_launcher import MockBackend,MockRun,mock_loader,mock_save
        from trottertracks.b2d_evaluator.launcher import _run
        with tempfile.TemporaryDirectory() as directory:
            permit=MockRun(directory)
            status=_run(permit,mock_only=True,input_loader=mock_loader,
                backend_factory=lambda settings:MockBackend(permit.scope),input_saver=mock_save)
            self.assertEqual(status['calls']['completed']['published.convention'],4)
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
            pub=[i for i,v in enumerate(events) if v.get('event')=='complete' and v.get('boundary')=='published.convention']
            comparison=next(i for i,v in enumerate(events) if v.get('boundary')=='dag.operator_error')
            self.assertLess(max(pub),comparison)

    def test_no_scientific_dependency_was_imported(self):
        self.assertFalse(FORBIDDEN.intersection(n.split('.')[0] for n in sys.modules))

    def test_published_convention_failure_stops_before_performance_comparison(self):
        from test_b2d_stage1_launcher import MockBackend,MockRun,mock_loader,mock_save
        from trottertracks.b2d_evaluator.launcher import _run
        from trottertracks.b2d_evaluator.errors import Stage1Failure
        class WrongConvention(MockBackend):
            def perform(self,kind,args,params):
                if kind=='a4_reference_check':
                    def fail(): raise Stage1Failure('PUBLISHED_CONVENTION_CORRECTNESS_FAILURE')
                    return self.scope.call('published.convention',fail,identity=(str(params[0]),params[1]))
                return super().perform(kind,args,params)
        with tempfile.TemporaryDirectory() as directory:
            permit=MockRun(directory)
            result=_run(permit,mock_only=True,input_loader=mock_loader,
                backend_factory=lambda s:WrongConvention(permit.scope),input_saver=mock_save)
            self.assertEqual(result['reason'],'PUBLISHED_CONVENTION_CORRECTNESS_FAILURE')
            self.assertNotIn('dag.operator_error',result['calls']['attempted'])
            self.assertNotIn('dag.signals',result['calls']['attempted'])
            self.assertTrue((Path(directory)/'STOP.json').is_file())


if __name__=='__main__': unittest.main()
