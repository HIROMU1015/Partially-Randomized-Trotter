"""Published price diagnostics only: synthetic fixtures, no acquisition or Q3."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from publish_h4_search_run import prepare
from trottertracks.h4_search_run_evidence import discover_runs
from trottertracks.h4_search_map import build_index
from trottertracks.h4_search_map_html import render
from trottertracks.saved_rpe_cost_catalog import Reader, PASS


def packet():
    ref = dict(commit='e'*40, path='docs/research/measure/COMPARISON.json')
    point = dict(m=1, q=2, R=3, predicted_RZ={'X': 12, 'Y': 12},
                 observations=[])
    # X/Y share two paths, not four independent samples.
    point['observations'] = [dict(path_id=p, axis=a, compiled_RZ=v, source=ref)
                              for p, v in [('holdout0', 10), ('holdout1', 14)] for a in ('X', 'Y')]
    point.update(observed_mean_RZ={'X': 12, 'Y': 12}, signed_relative_difference={'X': 0, 'Y': 0},
                 independent_paths=2, source=ref)
    return dict(schema='RPE_PRICE_MODEL_VALIDATIONS_V1', models=[dict(model_id='shared-model',
        identity=dict(method='PR_PAULI', L_D=2, frozen_source=ref, actual_step='1/10'),
        status='INDEPENDENT_MEASURED', label='独立実測あり', validation_points=[point],
        calibration_sources=[dict(commit='c'*40, path='docs/research/cal/CELLS.json')])],
        applications=[dict(configuration_id='same-candidate', model_id='shared-model',
            points=[dict(m=0,q=1,R=2),dict(m=1,q=2,R=6)], prices=[1,1,2,2])])


def saved_run():
    doc = dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1', round=dict(round_id='MODEL-DEMO', methods=['PR_PAULI']),
        methods={'PR_PAULI':dict(stop_reason='Q3_BUDGET_EXHAUSTED', evaluated=[dict(
            configuration_id='same-candidate', formula='2nd', L_D=2, requested_delta_pf='1/10',
            N_X=[1,1], N_Y=[1,1], status=PASS, source='SAVED', cost={'numerator':6,'denominator':1}),dict(
            configuration_id='same-candidate', formula='2nd', L_D=2, requested_delta_pf='1/10',
            N_X=[2,1], N_Y=[2,1], status=PASS, source='SAVED', cost={'numerator':8,'denominator':1})])})
    return doc


class PriceValidationMap(unittest.TestCase):
    def test_common_publication_discovery_share_and_source_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);subprocess.run(['git','init','-q',str(root)],check=True)
            run=root/'docs/research/producer/run01';run.mkdir(parents=True)
            (run/'ROUND_RESULT.json').write_text(json.dumps(saved_run()))
            (run/'PRICE_MODEL_VALIDATIONS.json').write_text(json.dumps(packet()))
            pub=root/'docs/research/producer/SEARCH_PUBLICATION.json'
            status,doc=prepare(root,run,pub,run_id='producer',execution_hosts=['host-a','host-b'])
            self.assertEqual(status,'UPDATED')
            self.assertIn('PRICE_MODEL_VALIDATIONS.json',doc['files'])
            subprocess.run(['git','add','.'],cwd=root,check=True)
            subprocess.run(['git','-c','user.name=T','-c','user.email=t@example.test','commit','-qm','fixture'],cwd=root,check=True)
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            reader=Reader(root)
            # A new host/branch is discovered through normal controller publication.
            runs,issues=discover_runs(reader,{'new-environment':head},legacy_heads=set())
            self.assertFalse(issues);self.assertEqual(len(runs),1)
            self.assertEqual(runs[0]['price_validation_source']['commit'],head)
            empty=dict(comparison=dict(accuracy={}),candidates=[])
            index=build_index(empty,runs)
            again=build_index(empty,discover_runs(Reader(root),{'new-environment':head},legacy_heads=set())[0])
            self.assertEqual(index,again)
            self.assertEqual(len(index['price_validation_models']),1)
            self.assertEqual(len(index['points']),2)
            shares=[]
            for point in index['points']:
                view=point['price_validations'][0]
                contribution=view['validation_round_contributions'][0]
                shares.append(Fraction(**contribution['percent_of_model_total']))
                self.assertFalse(contribution['same_application_point'])
                self.assertEqual(contribution['validation_R'],3)
                self.assertEqual(contribution['application_R'],6)
                self.assertIsNone(point['conditional_cost'])  # summary diagnostic never bypasses C price admission
            self.assertEqual(sorted(shares),[Fraction(50),Fraction(200,3)])
            page=render(index)
            self.assertIn('独立実測あり',page)
            self.assertIn('/blob/'+'e'*40+'/docs/research/measure/COMPARISON.json',page)
            self.assertIn('符号付き相対差',page)
            self.assertIn('独立経路数（X/Yは同じ経路を共有）',page)
            self.assertIn('percent_of_model_total',page)
            self.assertEqual(index['new_scientific_calls'],0)
            # Sidecar-only refresh must win even though ROUND_RESULT is unchanged.
            changed=packet();changed['models'][0]['label']='独立実測あり（更新）'
            (run/'PRICE_MODEL_VALIDATIONS.json').write_text(json.dumps(changed))
            prepare(root,run,pub,run_id='producer',execution_hosts=['host-a','host-b'])
            subprocess.run(['git','add','.'],cwd=root,check=True)
            subprocess.run(['git','-c','user.name=T','-c','user.email=t@example.test','commit','-qm','sidecar refresh'],cwd=root,check=True)
            newer=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            refreshed,issues=discover_runs(Reader(root),{'old-host':head,'new-host':newer},legacy_heads=set())
            self.assertFalse(issues);self.assertEqual(len(refreshed),1)
            self.assertEqual(refreshed[0]['price_model_validations']['models'][0]['label'],'独立実測あり（更新）')
            self.assertEqual(refreshed[0]['source']['commit'],head)  # Q3 evidence was preserved
            self.assertEqual(refreshed[0]['price_validation_source']['commit'],newer)
            self.assertEqual(prepare(root,run,pub,run_id='producer',execution_hosts=['host-b','host-a'])[0],'UNCHANGED')
            malformed=deepcopy(changed);malformed['schema']='UNSUPPORTED_DIAGNOSTIC'
            (run/'PRICE_MODEL_VALIDATIONS.json').write_text(json.dumps(malformed))
            prepare(root,run,pub,run_id='producer',execution_hosts=['host-a','host-b'])
            subprocess.run(['git','add','.'],cwd=root,check=True)
            subprocess.run(['git','-c','user.name=T','-c','user.email=t@example.test','commit','-qm','optional malformed diagnostic'],cwd=root,check=True)
            broken=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
            preserved,issues=discover_runs(Reader(root),{'producer':broken},legacy_heads=set())
            self.assertFalse(issues);self.assertEqual(len(preserved),1)
            self.assertEqual(len(preserved[0]['evaluations']),2)
            self.assertTrue(any('PRICE_VALIDATION_METADATA_INCOMPLETE' in reason for reason in preserved[0]['issues']))

            # Check JavaScript syntax if node is available, without a browser or producer.
            import shutil
            if shutil.which('node'):
                script=re.findall(r'<script>(.*?)</script>',page,re.S)[0]
                js=root/'display.js';js.write_text(script)
                subprocess.run(['node','--check',str(js)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    def test_unknown_share_and_no_change_to_saved_qualification(self):
        p=packet();p['applications'][0].pop('prices')
        p['models'][0].update(status='PENDING',label='検証待ち')
        r=dict(id='run',label='run',source={'commit':'x','path':'result'},status='STOP',hosts=[],targets=[],evaluations=[dict(
            method='PR_PAULI',configuration_id='same-candidate',formula='2nd',L_D=2,requested_width='1/10',
            N_X=[1,1],N_Y=[1,1],q3_status=PASS,source={'commit':'x','path':'result'},issues=[])],price_model_validations=p)
        index=build_index(dict(comparison={},candidates=[]),[r])
        point=index['points'][0]
        self.assertEqual(point['q3_status'],PASS)
        self.assertIsNone(point['conditional_cost'])
        self.assertIsNone(point['price_validations'][0]['validation_round_contributions'][0]['percent_of_model_total'])
        self.assertIn('検証待ち',index['views'][0]['cells'][0]['price_validation_labels'])

    def test_model_identity_conflict_is_reported(self):
        one=dict(id='a',label='a',source={},status='STOP',hosts=[],targets=[],evaluations=[],price_model_validations=packet())
        two=deepcopy(one);two.update(id='b',label='b')
        two['price_model_validations']['models'][0]['identity']['L_D']=4
        index=build_index(dict(comparison={},candidates=[]),[one,two])
        self.assertTrue(any('MODEL_IDENTITY_CONFLICT' in i['reason'] for i in index['issues']))
        self.assertEqual(index['price_validation_models']['shared-model']['identity']['L_D'],2)


if __name__=='__main__':unittest.main(verbosity=2)
