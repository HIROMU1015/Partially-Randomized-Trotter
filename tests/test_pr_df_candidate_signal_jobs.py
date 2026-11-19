"""Fake DF numerical provider only; checks real fork orchestration and resume."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_df_signal_execution',ROOT/'scripts/run_pr_df_candidate_signal_jobs.py')
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)


class FakeProvider:
    def __init__(self,fail=None):self.fail=fail
    def numerical_recipe(self,source):return dict(source=source)
    def saved_h4_reference(self,root,job):return dict(source={'fixture':'reference'},m=job['m'])
    def load_h4_context(self,root,config):return SimpleNamespace(constant=0.,tails={2:dict(scalar=0.)})
    def evaluate(self,ctx,job,ref,root):
        return dict(**{k:job[k]for k in ('m','B_configuration_id','B_candidate_id','B_identity','B_law_binding','physical_context')},
                    status=E.LOCAL if self.fail!=job['m']else'NUMERICAL_FAILURE',
                    numerical=dict(gate_pass=self.fail!=job['m']),scalar_once=True,source=E.IMPLEMENTATION,
                    signal={'raw':'ARTIFICIAL'},input_source={'fixture':'input'},reference_source=ref['source'])


def fixture():
    spec=dict(L_D=2,binding={'fixture':'H'},profile={'fixture':'P'},split={'fixture':'LD2'},law={'fixture':'Q'})
    cfg=dict(id='ARTIFICIAL_CONFIG',spec=spec,rounds=[]);plan=dict(configuration_id=cfg['id'],provider=E.IMPLEMENTATION,rounds=[])
    for m in range(3):
        row=dict(m=m,candidate={'id':'ARTIFICIAL_'+str(m)},identity={'m':m},law_binding={'fixture':'law'},physical_candidate={'id':'ARTIFICIAL_'+str(m)},structure={'fixture':'S'})
        cfg['rounds'].append(row)
        job=dict(m=m,B_configuration_id=cfg['id'],B_candidate_id=row['candidate']['id'],B_identity=row['identity'],B_law_binding=row['law_binding'],
                 physical_context={**{k:spec[k]for k in ('binding','profile','split','law')},'physical_candidate':row['physical_candidate'],'structure':row['structure'],'numerical_recipe':{'source':E.IMPLEMENTATION}})
        plan['rounds'].append(dict(m=m,decision='REQUEST'if m<2 else'REUSE',job=job,request=dict(reference={'source':{'fixture':'reference'},'m':m})))
    metadata=SimpleNamespace(common=SimpleNamespace(same=lambda a,b:a==b));b=SimpleNamespace(validate=lambda c:None)
    return cfg,plan,metadata,b


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        # These orchestration fixtures have no real saved-input index. The
        # relocated frozen-loader contract is tested in descriptor_extension.
        context=patch.object(E,'saved_context',side_effect=lambda config,locator,provider:
                             provider.load_h4_context(ROOT,config))
        context.start();self.addCleanup(context.stop)

    def test_fixed_view_beats_preloaded_controller_module_and_survives_fork(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            view = Path(temp)
            directory = view / 'src/trottertracks'
            directory.mkdir(parents=True)
            sources = {
                'src/trottertracks/__init__.py': '',
                'src/trottertracks/pr_rpe_signal_plan.py':
                    'from types import SimpleNamespace\ncommon = SimpleNamespace(same=lambda a, b: a == b)\n',
                'src/trottertracks/pr_rpe_rmse_s0_v2.py':
                    'from types import SimpleNamespace\ndef b_metadata(root):\n'
                    '    return SimpleNamespace(validate=lambda config: None, release=B_RELEASE)\n',
                E.PROVIDER_PATH:
                    'def evaluate(*args, **kwargs):\n'
                    '    from .worker_fixture import record\n'
                    '    return record(fixture, *args, **kwargs)\n',
                'src/trottertracks/worker_fixture.py':
                    'def record(fixture, *args, **kwargs):\n'
                    '    return dict(fixture(*args, **kwargs), fixture_leaf=__file__)\n',
            }
            for path, content in sources.items():
                (view / path).write_text(content)
            original = importlib.import_module('trottertracks')
            original_path = list(original.__path__)
            foreign = SimpleNamespace(__file__=str(ROOT / E.PROVIDER_PATH))
            fake = FakeProvider()
            config, plan, _, _ = fixture()
            output = view / 'output'
            input_file = view / 'REQUEST.json'
            requested_view = '/fixture/controller/fixed-signal-view'
            E.write_new(input_file, dict(configuration=config, plan=plan,
                execution_source=E.PROVIDER_COMMIT, output_directory=str(output.relative_to(ROOT)),
                runtime={'signal_view': requested_view}))
            sources[E.CALLER] = (ROOT / E.CALLER).read_text()
            sources[str(input_file.relative_to(ROOT))] = input_file.read_text()
            aliases = {'RPE_PREPARATION_PATH_MAP': json.dumps({requested_view: str(view)})}
            argv = [E.CALLER, '--input', str(input_file), '--inputs-commit', 'fixture-inputs',
                    '--source', E.PROVIDER_COMMIT, '--workers', '2']
            with patch.object(sys, 'path', sys.path.copy()), patch.dict(
                sys.modules, {'trottertracks.pr_rpe_signal_provider': foreign,
                    'numpy': SimpleNamespace(__version__='1.26.4'), 'scipy': SimpleNamespace(__version__='1.14.1')}
            ), patch.object(E.subprocess, 'check_output', return_value=E.PROVIDER_COMMIT + '\n'), patch.object(
                E, 'read_git', side_effect=lambda _commit, path: sources[path].encode()
            ), patch.dict(os.environ, aliases), patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                provider, _, b = E.fixed_runtime(view)
                self.assertEqual(Path(provider.__file__), view / E.PROVIDER_PATH)
                self.assertIs(sys.modules['trottertracks.pr_rpe_signal_provider'], foreign)
                self.assertEqual(list(original.__path__), original_path)
                self.assertEqual(b.release, E.B_RELEASE)
                for name in ('numerical_recipe', 'saved_h4_reference', 'load_h4_context'):
                    setattr(provider, name, getattr(fake, name))
                provider.fixture = fake.evaluate
                leaf_name = provider.__package__ + '.worker_fixture'
                self.assertNotIn(leaf_name, sys.modules)
                self.assertEqual(E.main(), 0)
                result = json.loads((output / 'RESULT.json').read_text())
                self.assertEqual(result['status'], 'REQUESTED_ROUNDS_COMPLETE')
                self.assertEqual(result['new_provider_calls'], 2)
                self.assertTrue(all(p['pid'] != os.getpid() and p['exitcode'] == 0 for p in result['process_exits']))
                raw = json.loads((output / 'raw/m0.json').read_text())
                self.assertEqual(Path(raw['fixture_leaf']), directory / 'worker_fixture.py')
                self.assertNotIn(leaf_name, sys.modules)
                self.assertEqual(E.main(), 0)
                resumed = json.loads((output / 'RESULT.json').read_text())
                self.assertEqual(resumed['reused_terminal_rounds'], 2)
                self.assertEqual(resumed['new_provider_calls'], 0)
                (view / E.PROVIDER_PATH).write_text('# changed despite same HEAD\n')
                with self.assertRaisesRegex(ValueError, 'Fixed signal bytes differ'):
                    E.fixed_runtime(view)

    def test_fake_fork_then_resume_no_duplicate_provider_calls(self):
        cfg,plan,metadata,b=fixture()
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.df-signal-test-')as d:
            out=Path(d)/'output';first=E.execute(cfg,plan,out,FakeProvider(),metadata,b,workers=2)
            old={p.name:p.read_bytes()for p in(out/'raw').glob('*.json')}
            second=E.execute(cfg,plan,out,FakeProvider(),metadata,b,workers=2)
            self.assertEqual(first['new_provider_calls'],2);self.assertEqual(second['new_provider_calls'],0)
            self.assertEqual(second['reused_terminal_rounds'],2);self.assertEqual(second['status'],'REQUESTED_ROUNDS_COMPLETE')
            self.assertEqual(old,{p.name:p.read_bytes()for p in(out/'raw').glob('*.json')})
            self.assertEqual(first['not_requested_rounds'],[2]);self.assertEqual(len(first['process_exits']),2)
            self.assertTrue(all(p['exitcode']==0 for p in first['process_exits']))

    def test_failed_round_and_ambiguous_start_are_not_retried(self):
        cfg,plan,metadata,b=fixture()
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.df-signal-test-')as d:
            out=Path(d)/'output';first=E.execute(cfg,plan,out,FakeProvider(fail=1),metadata,b,workers=2)
            second=E.execute(cfg,plan,out,FakeProvider(),metadata,b,workers=2)
            self.assertEqual(first['status'],'PARTIAL_OR_HELD_NO_RETRY');self.assertEqual(second['new_provider_calls'],0)
            self.assertEqual(second['rows'][1]['status'],'HELD_PRIOR_FAILURE_NO_RETRY')
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='.df-signal-test-')as d:
            out=Path(d)/'output';(out/'started').mkdir(parents=True);(out/'started'/'m0.json').write_text('{}')
            value=E.execute(cfg,plan,out,FakeProvider(),metadata,b,workers=1)
            self.assertEqual(value['new_provider_calls'],1)
            self.assertEqual(value['rows'][0]['status'],'HELD_AMBIGUOUS_PRIOR_START_NO_RETRY')


if __name__=='__main__':unittest.main(verbosity=2)
