"""Fake DF numerical provider only; checks real fork orchestration and resume."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest

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
