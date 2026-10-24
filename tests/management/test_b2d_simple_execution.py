"""CLI and scientific preflight only, with isolated stub data and no science."""
from dataclasses import replace
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier,FORBIDDEN
from test_b2d_stage1_launcher import MockBackend,mock_loader,mock_save
from trottertracks.b2d_evaluator.environment import ENVIRONMENT_PATH,verify_environment
from trottertracks.b2d_evaluator.input_package import InputRecipe
from trottertracks.b2d_evaluator.preflight import RunContext,verify_execution
from trottertracks.b2d_evaluator.scope import candidate_graph,registry,Stage1Failure
from trottertracks.b2d_evaluator.launcher import launch,_run

SOURCE='a'*40


class SimpleExecutionTests(unittest.TestCase):
    def setUp(self):
        barrier=Barrier();sys.meta_path.insert(0,barrier)
        self.addCleanup(sys.meta_path.remove,barrier)

    def cli(self,args):
        with patch.object(sys,'argv',['run_b2d_stage1.py',*args]),patch('sys.stdout',new_callable=io.StringIO) as output:
            with self.assertRaises(SystemExit) as stop:
                runpy.run_path(str(ROOT/'scripts/run_b2d_stage1.py'),run_name='__main__')
        return stop.exception.code,output.getvalue()

    def fixture_metadata(self,root):
        # Copy source bytes only; the fixture helper is NEVER imported/called.
        fixture=root/InputRecipe().fixture_path;fixture.parent.mkdir(parents=True)
        fixture.write_bytes((ROOT/InputRecipe().fixture_path).read_bytes())
        env=json.loads((ROOT/ENVIRONMENT_PATH).read_bytes())
        envpath=root/ENVIRONMENT_PATH;envpath.parent.mkdir(parents=True)
        envpath.write_text(json.dumps(env))
        return env

    def test_cli_requires_explicit_mode_and_execute_requires_scientific_arguments(self):
        for args in ([],['--output','MOCK'],['--execute'],['--execute','--source-commit',SOURCE]):
            with self.subTest(args=args),patch('sys.stderr'),patch('trottertracks.b2d_evaluator.launcher.launch') as run:
                code,_=self.cli(args)
                self.assertEqual(code,2);run.assert_not_called()

    def test_cli_inspect_runs_no_science(self):
        with patch('trottertracks.b2d_evaluator.launcher.launch') as run:
            code,out=self.cli(['--inspect'])
        self.assertEqual(code,0);run.assert_not_called()
        self.assertEqual(json.loads(out)['scope']['dispatch_total'],941)
        self.assertFalse(FORBIDDEN.intersection(n.split('.')[0] for n in sys.modules))

    def test_execute_cli_takes_no_approval_file_packet_or_grant(self):
        with patch('trottertracks.b2d_evaluator.launcher.launch',return_value={
                'status':'COMPLETED_DESCRIPTIVE_MANDATORY_STOP'}) as run:
            code,_=self.cli(['--execute','--source-commit',SOURCE,'--output','MOCK_ONLY_NOT_CREATED'])
        self.assertEqual(code,0)
        run.assert_called_once_with(ROOT,'MOCK_ONLY_NOT_CREATED',SOURCE)

    def test_scientific_preflight_is_read_only_and_needs_no_approval_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=self.fixture_metadata(root)
            output=root/'artifacts/research_b_b2d_stage1/MOCK'
            with patch('trottertracks.b2d_evaluator.preflight._check_source'),patch(
                    'trottertracks.b2d_evaluator.environment.inspect_environment',return_value=env):
                context=verify_execution(root,output,SOURCE)
            self.assertIsInstance(context,RunContext)
            self.assertEqual(context.call_scope['dispatch_total'],941)
            self.assertFalse(output.parent.exists())
            self.assertFalse(any(root.rglob('authorization.json')))

    def test_preflight_rejects_source_grid_scope_and_output_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);env=self.fixture_metadata(root)
            output=root/'artifacts/research_b_b2d_stage1/MOCK'
            graph=candidate_graph(SOURCE)
            bad_scope=registry(graph);bad_scope['boundaries']['backend.expm']=447
            cases=[dict(source_commit='WRONG'),dict(source_commit=SOURCE,graph=candidate_graph('b'*40)),
                dict(source_commit=SOURCE,graph=replace(graph,settings=replace(graph.settings,q_values=(2,)))),
                dict(source_commit=SOURCE,call_scope=bad_scope)]
            with patch('trottertracks.b2d_evaluator.preflight._check_source'),patch(
                    'trottertracks.b2d_evaluator.environment.inspect_environment',return_value=env):
                for kwargs in cases:
                    with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,PermissionError,Stage1Failure)):
                        verify_execution(root,output,**kwargs)
                for bad in (root/'outside',root/'artifacts/research_b_b2d_stage1'):
                    with self.subTest(output=bad),self.assertRaises(ValueError): verify_execution(root,bad,SOURCE)
                output.mkdir(parents=True)
                with self.assertRaises(ValueError): verify_execution(root,output,SOURCE)

    def test_environment_checks_versions_and_accepts_relocation_without_bitwise_identity(self):
        expected=json.loads((ROOT/ENVIRONMENT_PATH).read_bytes())
        actual=dict(python_runtime=expected['python_runtime'],dependency_versions=dict(expected['dependency_versions']),
            executable='/MOCK/RELOCATED/python',prefix='/MOCK/RELOCATED',package_origins={'numpy':'MOCK_NEW_PATH'})
        with patch('trottertracks.b2d_evaluator.environment.inspect_environment',return_value=actual):
            self.assertEqual(verify_environment(expected),actual)
            actual['dependency_versions']['numpy']='WRONG'
            with self.assertRaises(ValueError): verify_environment(expected)
            actual['dependency_versions']['numpy']=expected['dependency_versions']['numpy']
            actual['python_runtime']={'implementation':'cpython','version':[0,0,0]}
            with self.assertRaises(ValueError): verify_environment(expected)

    def test_full_launch_uses_exclusive_output_and_scientific_context_without_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'MOCK_OUTPUT'
            context=RunContext(ROOT,output,SOURCE,registry(candidate_graph(SOURCE)),{'mode':'MOCK_ONLY'})
            def mock_run(run):
                return _run(run,mock_only=True,input_loader=mock_loader,input_saver=mock_save,
                    backend_factory=lambda s:MockBackend(run.scope))
            with patch('trottertracks.b2d_evaluator.launcher.verify_execution',return_value=context),patch(
                    'trottertracks.b2d_evaluator.preflight._check_source'),patch(
                    'trottertracks.b2d_evaluator.launcher._run',side_effect=mock_run) as runner:
                result=launch(ROOT,output,SOURCE)
                self.assertEqual(result['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP')
                before={p.name:p.read_bytes() for p in output.iterdir()}
                second=launch(ROOT,output,SOURCE)
                self.assertEqual(second['status'],'FRESH_OUTPUT_FAILED_MANDATORY_STOP')
                self.assertEqual(before,{p.name:p.read_bytes() for p in output.iterdir()})
                runner.assert_called_once()
            from trottertracks.b2d_evaluator.evaluation import build_graph
            bound=context._input_binding
            graph=build_graph(replace(candidate_graph(SOURCE).settings,context=bound[0],
                hamiltonian_id=bound[1],sector_id=bound[2],state_id=bound[3],input_data_sha256=bound[4]))
            with patch('trottertracks.b2d_evaluator.preflight._check_source'),self.assertRaisesRegex(ValueError,'already attempted'):
                context.begin_evaluation(graph)
            with patch('trottertracks.b2d_evaluator.preflight._check_source'),self.assertRaises(PermissionError):
                context.begin_input(ROOT,InputRecipe())
            self.assertNotIn('approval_reference',json.loads((output/'result.json').read_bytes()))


if __name__=='__main__': unittest.main()
