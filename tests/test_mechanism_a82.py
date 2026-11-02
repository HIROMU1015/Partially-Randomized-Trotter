"""Invented scalars and mocked execution only; no saved science file access."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import mechanism_a82 as m
import run_mechanism_a82 as cli
from test_mechanism_explanation import invented


def fixture():
    bundle,cores=invented();row=bundle['comparisons'][0]
    row.update(contrast_id=m.TARGET,system='H8',PR_role='OLD_PR',PR_cohort='CONFIRMATION_UNION_POST_SELECTION',
        PR_id=m.CANDIDATE,PF_id='B0I_p11_q1_2nd',binding=dict(system='H8',rank=28,T='0.8',snapshot='INVENTED',compiler='INVENTED'))
    for method in ('PR','PF'):row['sides'][method]['identity']=m.equations.identity(row,method)
    core=cores[0];core.update(contrast_id=m.TARGET,system='H8')
    archive=[];core32=[]
    for i in range(32):
        cid=m.TARGET if i==31 else 'INVENTED_'+str(i)
        c=copy.deepcopy(core);c['contrast_id']=cid;core32.append(c)
        archive.append(dict(contrast_id=cid,system='H8',core_reused_without_recalculation=True,
            explanation=dict(sides={'PR':dict(status='EXPLANATORY_INPUT_MISSING' if cid==m.TARGET else 'EXPLANATION_COMPLETE'),
                'PF':dict(status='EXPLANATION_COMPLETE',token='INVENTED_PF_NOT_EVALUATED')},ratios={})))
    return dict(schema=m.SCHEMA,kind='ARTIFICIAL',comparison=row),core32,archive


class DeltaTests(unittest.TestCase):
    def setUp(self):self.b,self.c,self.e=fixture()
    def run_delta(self):return m.analyze_delta(self.b,self.c,self.e,m.equations.artificial_permit())
    def test_one_PR_call_no_other_science(self):
        before=copy.deepcopy((self.b,self.c,self.e))
        with patch.object(m.equations,'explain_side',wraps=m.equations.explain_side) as side, \
             patch.object(m.equations,'explain',side_effect=AssertionError('pair forbidden')), \
             patch.object(m.equations,'analyze_bundle',side_effect=AssertionError('archive forbidden')), \
             patch.object(m.equations.original,'paired_costs',side_effect=AssertionError('core forbidden')):
            result=self.run_delta();self.assertEqual(side.call_count,1)
            self.assertEqual(side.call_args.args[0],self.b['comparison']['sides']['PR'])
        self.assertEqual(set(result['new_PR_side']['axes']),{'cosine','sine'})
        self.assertEqual(result['inherited_PF_side'],self.e[-1]['explanation']['sides']['PF'])
        self.assertEqual((self.b,self.c,self.e),before)
        self.assertFalse(result['new_comparison_factors_computed'])
    def test_wrong_candidate_rejected_before_science(self):
        self.b['comparison']['PR_id']='OTHER'
        with patch.object(m.equations,'explain_side',side_effect=AssertionError('no scientific call')):
            with self.assertRaisesRegex(ValueError,'A82_FIXED_ENDPOINT'):self.run_delta()
    def test_wrong_PF_rejected(self):
        self.b['comparison']['PF_id']='OTHER'
        with self.assertRaisesRegex(ValueError,'A82_FIXED_ENDPOINT'):self.run_delta()
    def test_wrong_DF_rejected(self):
        self.b['comparison']['binding']['rank']=10
        with self.assertRaisesRegex(ValueError,'A82_DF_T'):self.run_delta()
    def test_cohort_mixing_rejected(self):
        self.b['comparison']['cohorts_pooled']=True
        with self.assertRaisesRegex(ValueError,'COHORT_CLASSIFICATION'):self.run_delta()
    def test_real_kind_without_permit_rejected(self):
        self.b['kind']='SAVED_REAL_VALUES'
        with self.assertRaisesRegex(ValueError,'A82_SCHEMA_KIND'):self.run_delta()
    def test_other_side_status_change_rejected(self):
        self.e[0]['explanation']['sides']['PF']['status']='UNRESOLVED'
        with self.assertRaisesRegex(ValueError,'ARCHIVED_63'):self.run_delta()
    def test_original_A82_completion_cannot_be_assumed(self):
        self.e[-1]['explanation']['sides']['PR']['status']='EXPLANATION_COMPLETE'
        with self.assertRaisesRegex(ValueError,'HISTORICAL_A82'):self.run_delta()
    def test_incomplete_archive_rejected(self):
        self.c.pop(0)
        with self.assertRaisesRegex(ValueError,'FIXED_32_ARCHIVE'):self.run_delta()
    def test_numeric_mismatch_stops(self):
        self.b['comparison']['sides']['PR']['headroom']['cosine']='.1'
        with self.assertRaisesRegex(ValueError,'HEADROOM_MISMATCH'):self.run_delta()
    def test_missing_input_stops(self):
        self.b['comparison']['sides']['PR']['normalization']=None
        with self.assertRaisesRegex(ValueError,'A82_NUMERIC_MISMATCH'):self.run_delta()


class GrantTests(unittest.TestCase):
    def test_invalid_grant_reads_nothing_else(self):
        with patch.object(cli.io,'read',return_value={'approved':False}) as reader:
            with self.assertRaisesRegex(ValueError,'NUMERICAL_AUTHORIZATION'):cli.gate('grant','nonexistent contract')
            reader.assert_called_once_with('grant')
    def test_old_scope_refused(self):
        g=dict(approved=True,approved_by_user=True,runtime_authorization=True,one_shot=True,
            scope='TRACK_A_MECHANISM_EXPLANATION_PRIMARY32',kind='SAVED_REAL_VALUES')
        with patch.object(cli.io,'read',return_value=g):
            with self.assertRaisesRegex(ValueError,'A82_ONLY_SCOPE'):cli.gate('g','c')
    def test_PF_scope_refused(self):
        g=dict(approved=True,approved_by_user=True,runtime_authorization=True,one_shot=True,
            scope=m.SCOPE,kind='SAVED_REAL_VALUES',method_side_ids=[m.TARGET+'.PF'],axes=list(m.AXES))
        with patch.object(cli.io,'read',return_value=g):
            with self.assertRaisesRegex(ValueError,'A82_ONLY_SIDE'):cli.gate('g','c')
    def test_unapproved_CLI_before_contract_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'grant.json').write_text('{"approved":false}')
            r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_mechanism_a82.py'),
                '--grant',str(p/'grant.json'),'--contract',str(p/'DOES_NOT_EXIST')],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('NUMERICAL_AUTHORIZATION_REQUIRED',r.stderr)
            self.assertEqual({f.name for f in p.iterdir()},{'grant.json'})
    def test_existing_and_protected_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaisesRegex(ValueError,'ONE_SHOT_OUTPUT_ALREADY_EXISTS'):cli.io.prepare_output(p,p.parent,[])
            with self.assertRaisesRegex(ValueError,'PROTECTED_PATH'):cli.io.prepare_output(p/'out',p,[p])
    def test_duplicate_JSON_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'
            for text in ['{"a":1,"a":2}','{"a":NaN}']:
                p.write_text(text)
                with self.assertRaises(ValueError):cli.io.read(p)
    def test_own_post_control_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'marker';p.write_text('before');saved={str(p):cli.io.sha(p)};p.write_text('after')
            with patch.object(cli,'gate',return_value=({}, {}, {})):
                with self.assertRaisesRegex(ValueError,'POST_CONTROL_SHA'):cli.post_gate('g','c',saved)


class MockWorkerTests(unittest.TestCase):
    def test_mock_worker_success_inherits_PF_and_archives(self):
        b,c,e=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths={}
            for n,obj in [('bundle',b),('core',c),('explanations',e)]:
                path=root/(n+'.json');path.write_text(json.dumps(obj));paths[n]=str(path)
            binding=dict(payload=paths['bundle'],old_core=paths['core'],old_explanations=paths['explanations'],
                old_core_sha256=cli.io.sha(paths['core']),old_explanations_sha256=cli.io.sha(paths['explanations']))
            grant={};contract=dict(output_root=str(root))
            (root/'ONE_SHOT.json').write_text(json.dumps(dict(parent_pid=123,consumed=True,grant_sha256='gsha',contract_sha256='csha')))
            (root/'ledger.json').write_text('{}')
            real_sha=cli.io.sha
            def hashed(path):return 'gsha' if str(path)=='g' else 'csha' if str(path)=='c' else real_sha(path)
            with patch.object(cli,'gate',return_value=(grant,contract,binding)), \
                 patch.object(cli.os,'getppid',return_value=123),patch.object(cli.io,'sha',side_effect=hashed), \
                 patch.object(cli,'protected_hashes',return_value={}),patch.object(cli,'post_gate'), \
                 patch.object(m.equations,'_authorized_production_permit',return_value=m.equations.artificial_permit()):
                cli.worker('g','c',root)
            result=json.loads((root/'A82_PR_delta.json').read_text())
            self.assertEqual(result['inherited_PF_side'],e[-1]['explanation']['sides']['PF'])
            self.assertEqual(json.loads(Path(paths['core']).read_text()),c)
            self.assertEqual(json.loads(Path(paths['explanations']).read_text()),e)
            cli.accept(root,binding)
            result['inherited_PF_side']['token']='TAMPER';(root/'A82_PR_delta.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'PF_COPY_CHANGED'):cli.accept(root,binding)


if __name__=='__main__':unittest.main()
