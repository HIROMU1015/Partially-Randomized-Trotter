"""Artificial contract/scheduling/binding checks; no molecular calls or compile."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from launch_h46_acquire import validate
from h46_acquire_native import reserve_axis,task_id
from h46_acquire_integration import inspect,compare_after_separate_approval
from h46_higher_pf_adapter import kernel,INPUT_HASH
from run_h46_acquire_comparison import authorize as authorize_integrated
k=kernel()


def fixture():
    groups=[{'id':s+'_'+short+'_q'+str(q),'system':s,'label':label,'q':q,'prefix':3 if s=='H4' else 5}
            for s in ('H4','H6') for short,label,qs in [('S2','2nd',(8,)),('Y4','4th',(4,8)),('N2','4th(new_2)',(2,4,8))] for q in qs]
    plan={'groups':groups,'tasks':[task_id(g,i) for g in groups for i in range(8)],'sample_axes':192,'extra_scientific_compiles':0,'seed_namespace':'ARTIFICIAL'}
    return {'authorized':True,'SOURCE':'fake-SOURCE','plan_sha256':'fake-plan','seed_registry_sha256':'fake-seeds',
            'grant':{'valid':True,'phase':'B_ACQUIRE_MISSING','SOURCE':'fake-SOURCE','plan_sha256':'fake-plan'},
            'plan':plan,'group_binding':{g['id']:{'preparation_hash':'fake-prep'} for g in groups}}


class AcquisitionTests(unittest.TestCase):
    def test_exact_96_tasks_192_axes(self):
        self.assertEqual(len(validate(fixture())),96)

    def test_invalid_grant(self):
        c=fixture();c['grant']['valid']=False
        with self.assertRaisesRegex(ValueError,'APPROVAL'):validate(c)

    def test_scope_q_prefix_extra_compile_rejected(self):
        for field,value in [('q',16),('prefix',6),('label','projected')]:
            c=fixture();c['plan']['groups'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate(c)
        c=fixture();c['plan']['extra_scientific_compiles']=1
        with self.assertRaises(ValueError):validate(c)

    def test_duplicate_task_and_excess_sample_rejected(self):
        c=fixture();c['plan']['tasks'][-1]=c['plan']['tasks'][0]
        with self.assertRaises(ValueError):validate(c)
        with self.assertRaises(ValueError):task_id(fixture()['plan']['groups'][0],8)

    def test_axis_ledger_is_exclusive_no_refund(self):
        with tempfile.TemporaryDirectory() as temp:
            g=fixture()['plan']['groups'][0]
            path=reserve_axis(temp,g,0,'cosine',{'SOURCE':'ARTIFICIAL'})
            self.assertEqual(json.loads(path.read_text())['state'],'ENTERED_NO_REFUND')
            with self.assertRaises(FileExistsError):reserve_axis(temp,g,0,'cosine',{})
            with self.assertRaises(ValueError):reserve_axis(temp,g,0,'third-axis',{})

    def test_seed_uniqueness_and_legacy_collision(self):
        seeds={k.seed_for('ARTIFICIAL','EXPLORATION',g['id'],i,a,b,'fake-input')
               for g in fixture()['plan']['groups'] for i in range(8) for a in range(g['q']) for b in range(3)}
        self.assertEqual(len(seeds),sum(g['q']*3*8 for g in fixture()['plan']['groups']))
        value=k.seed_for('ARTIFICIAL','EXPLORATION','A',0,0,0,'fake-input')
        with self.assertRaises(ValueError):k.seed_for('ARTIFICIAL','EXPLORATION','A',0,0,0,'fake-input',occupied={value})

    def test_integration_requires_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'completion_witness_PRIVATE.json').write_text('{"status":"STOP","all_pool_workers_reaped":true}')
            with self.assertRaisesRegex(ValueError,'NOT_COMPLETE'):inspect(fixture(),p)

    def test_metadata_assembly_artificial_and_pair_mismatch(self):
        c=fixture()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'completion_witness_PRIVATE.json').write_text('{"status":"COMPLETE_ACQUISITION_ONLY","all_pool_workers_reaped":true}')
            for g in c['plan']['groups']:
                for sample in range(8):
                    d=p/g['id']/('sample_%02d'%sample);d.mkdir(parents=True)
                    result={'group':g,'sample':sample,'SOURCE':c['SOURCE'],'plan_sha256':c['plan_sha256'],
                            'seed_registry_sha256':c['seed_registry_sha256'],'status':'ACQUIRED_VALIDATED_NO_ACCOUNTING',
                            'forced_events_enter_cost_sample':False,'input_sha256':INPUT_HASH[g['system']],
                            'preparation_hash':'fake-prep','records':[{'axis':axis,'event_digest':'fake-event','compiler_settings_hash':k.PROFILE}
                                for axis in ('cosine','sine')]}
                    (d/'result.json').write_text(json.dumps(result))
            result=inspect(c,p);self.assertEqual(result['new_cost_axes'],192);self.assertFalse(result['ranking_executed'])
            path=p/c['plan']['groups'][0]['id']/'sample_00/result.json';r=json.loads(path.read_text());r['records'][0]['event_digest']='wrong';path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'UNPAIRED'):inspect(c,p)

    def test_integrated_ranking_cannot_start(self):
        with self.assertRaisesRegex(ValueError,'SEPARATE'):compare_after_separate_approval({},'unused',{'valid':False})
        with self.assertRaisesRegex(ValueError,'SEPARATE'):authorize_integrated({'status':'UNAPPROVED'},{'valid':False},'unused')

    def test_frozen_kernels_and_no_reaccounting(self):
        text=(ROOT/'scripts/h46_acquire_native.py').read_text()
        self.assertNotIn('u_aware_shots(',text);self.assertNotIn('statistics.mean(',text)
        self.assertIn('from h46_new2_native import make_requests, native_evolution',text)
        self.assertIn("if sample == 0:",text)


if __name__=='__main__':unittest.main()
