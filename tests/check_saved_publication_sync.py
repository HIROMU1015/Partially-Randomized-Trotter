"""Public saved-evidence reconciliation; no producer/scientific APIs invoked."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trottertracks.saved_rpe_cost_catalog import Reader,frac,ref
import sync_saved_rpe_publications as S

ROOT=Path(__file__).resolve().parents[1]
DIR='docs/research/h4_saved_rpe_cost_catalog_20261116'
PF=ref('438d6c9a43351d45c31921a5676359dbc2c80b32','docs/research/h4_pf_optimizer_search_20261116/C_RESULT_JOIN.json')
PAULI=ref('5304fc149ac3e522f806457b36c03d91a170593e','docs/research/h4_optimizer_saved_results_intake_20261116/B2_SAVED_TOTAL_INDEX.json')
PRICE=ref('c7c8c1edd0996ad9ef97629961f67ca508a182f0','docs/research/d_p7_y4_coarse_price_calibration_20261116/PRICE_AND_COST.json')
Q3=ref('8855f420a9fe2577efc77f099c5a39ab0cb9b7e6','docs/research/d_p7_y4_coarse_a_shape_q3_20261116/C_RESULT_JOIN.json')
B2_COMMIT='d7a976a629a6db68197b24f8f86734e109c8bf5d'
B2_BASE='docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round03'
B2=ref(B2_COMMIT,B2_BASE+'/KAPPA_ABS_RESULTS.json')
B2_POINT=ref(B2_COMMIT,B2_BASE+'/best_points/LD5_kappa_x2_edd63e0befd3af591809/REQUEST.json')
B2_K2=ref(B2_COMMIT,B2_BASE+'/best_points/LD5_kappa2_e3cb2cfda719117d6222/REQUEST.json')
B2_ROUND=ref(B2_COMMIT,B2_BASE+'/run01/ROUND_RESULT.json')


class Publications(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=Reader(ROOT)
        cls.seed=cls.reader.read(ref('417a73e7329fd520381dc88cc0de00a11dd0b641',DIR+'/INPUTS.json'))

    def found(self,sources):
        return {str(i):dict(source=s,blob=S.git(ROOT,'rev-parse',s['commit']+':'+s['path']),branches=['synthetic-public-ref'])
                for i,s in enumerate(sources)}

    def test_public_git_discovery_reaches_three_examples_without_private_crawl(self):
        found=S.discover(self.reader,{'B':PF['commit'],'D':PRICE['commit'],'B2':'f01ce68ba918bf372acbefbf5152ccd9be5df390'})
        paths={v['source']['path']for v in found.values()}
        self.assertIn(PF['path'],paths);self.assertIn(PRICE['path'],paths)
        self.assertIn('docs/research/rpe_method_optimizer_h4_20261116/kappa_round01/run01/ROUND_RESULT_INTERRUPTED.json',paths)
        self.assertFalse(any('/jobs/'in p or 'executor_acceptance_'in p for p in paths))

    def test_three_examples_reconcile_and_bad_item_does_not_block(self):
        candidates=self.found([PF,PAULI,PRICE])
        candidates['3']=dict(source=ref(PF['commit'],'docs/research/NOT_PUBLISHED.json'),blob='0'*40,branches=['synthetic'])
        out,records,accepted=S.reconcile(self.reader,self.seed,candidates,{})
        self.assertEqual(frac(out['method_minima']['FULL_PF']['conditional_cost']),480258064)
        self.assertEqual(frac(out['method_minima']['PR_PAULI']['conditional_cost']),frac('2873295767/7'))
        self.assertEqual(frac(out['method_minima']['TRUNCATED_PF']['conditional_cost']),372035410)
        self.assertEqual(records['3']['status'],'PENDING')
        self.assertTrue(all(out[k]==0 for k in ('new_signal','new_compile','new_Q3','new_price_fit')))
        self.assertEqual(len(out['candidates']),51)
        # Imported locators are now normal fixed inputs; repeated discovery must
        # yield identical candidates and minima, not another copy of each point.
        again,_,_=S.reconcile(self.reader,dict(self.seed,sources=self.seed['sources']+accepted),candidates,{})
        self.assertEqual(out,again)

    def test_late_price_and_q3_arrival_both_orders_resolve_one_point(self):
        seed=deepcopy(self.seed);seed['sources']=[s for s in seed['sources']if s['path']not in (PRICE['path'],Q3['path'])]
        results=[]
        for sources in ([Q3,PRICE],[PRICE,Q3]):
            out,_,_=S.reconcile(self.reader,seed,self.found(sources),{})
            points=[r for r in out['candidates']if r['configuration_id']=='dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0'and
                    r['allocation']['allocation_id']=='B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01']
            self.assertEqual(len(points),1);self.assertEqual(frac(points[0]['conditional_cost']),372035410)
            self.assertEqual(points[0]['status'],'FEASIBLE');results.append(points[0])
        self.assertEqual(results[0],results[1])

    def test_cost_only_change_cannot_rewrite_roadmap(self):
        out,records,_=S.reconcile(self.reader,self.seed,self.found([PF,PAULI]),{})
        stage=S.stage_summary(out,list(records.values()));changed=deepcopy(out)
        for row in changed['candidates']:
            if row['conditional_cost']is not None:row['conditional_cost']={'numerator':1,'denominator':1}
        self.assertEqual(stage,S.stage_summary(changed,list(records.values())))
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'ROADMAP.md';p.write_text('POLICY_PRESERVED\n'+S.roadmap_block(stage)+'\nTAIL_PRESERVED\n')
            before=p.read_bytes();self.assertFalse(S.update_roadmap(p,stage));self.assertEqual(before,p.read_bytes())
            new=deepcopy(stage);new['methods']['PR_PAULI']['price_waiting']=True
            self.assertTrue(S.update_roadmap(p,new));self.assertTrue(p.read_text().startswith('POLICY_PRESERVED\n'))

    def test_round03_normal_discovery_exact_join_and_repeated_sync(self):
        found=S.discover(self.reader,{'codex/b2-rpe-split-preparation-20261116':B2_COMMIT})
        selected={k:v for k,v in found.items()if v['source']['path']in (B2['path'],B2_POINT['path'],B2_K2['path'],B2_ROUND['path'])}
        self.assertEqual(len(selected),4)
        seed=dict(self.seed,sources=self.seed['sources']+[PAULI])
        out,records,accepted=S.reconcile(self.reader,seed,selected,{})
        self.assertEqual(frac(out['method_minima']['PR_PAULI']['conditional_cost']),410199237)
        self.assertEqual(frac(out['restricted_minima']['PR_PAULI_KAPPA_ABSOLUTE_2']['conditional_cost']),frac('3508632333/7'))
        points=[r for r in out['candidates']if r.get('search_result_source',{}).get('path')==B2_ROUND['path']]
        self.assertEqual(len(points),2)
        self.assertEqual({r['outcome_sha256']for r in points},{
            '875ccc86920db1ee7051fc83cfd98678b06ec0f246f3e222e5b60962297276d2',
            '9abac97b2cadc5faee22ce420f3c823e5a8dd06822e1643799a30fe84230434f'})
        self.assertTrue(all(len(r['model_coverage_axes'])==26 and r['qualified_for_candidate'] is False and
            r['price_axis_unit_values_persisted'] is False for r in points))
        self.assertTrue(any(r['source']['path']==B2_ROUND['path']and r['status']=='PENDING'and
            'configuration-only match rejected'in r['reason'] for r in records.values()))
        again,_,_=S.reconcile(self.reader,dict(seed,sources=seed['sources']+accepted),selected,{})
        self.assertEqual(out,again)
        self.assertTrue(any(frac(r['conditional_cost'])==frac('2873295767/7')for r in out['candidates']
            if r['method']=='PR_PAULI'and r['conditional_cost']is not None))

    def test_public_selected_point_needs_no_round_specific_index(self):
        # A future round may publish best_points before a final summary.
        out,records,_=S.reconcile(self.reader,self.seed,self.found([B2_POINT,B2_K2]),{})
        self.assertTrue(all(r['status']=='INTEGRATED'for r in records.values()))
        self.assertEqual(frac(out['method_minima']['PR_PAULI']['conditional_cost']),410199237)
        self.assertEqual(frac(out['restricted_minima']['PR_PAULI_KAPPA_ABSOLUTE_2']['conditional_cost']),frac('3508632333/7'))

    def test_bad_outcome_hash_or_price_binding_is_pending_not_false_integration(self):
        original=self.reader.read
        for failure in ('hash','price','allocation'):
            def altered(locator):
                value=deepcopy(original(locator))
                if locator['path']==B2_BASE+'/run01/EVALUATIONS.json' and locator.get('selector') is None and failure=='hash':
                    for row in value['rows']:
                        if row['job']=='edd63e0befd3af591809':row['outcome_sha256']='0'*64
                if locator['path']==B2_BASE+'/run01/NEEDS.json' and failure=='price':
                    for row in value['candidates']:
                        if row['configuration_id']=='cb1aa7487fbfb938c629961014ea5911cc5f3e29b9fc45e16b872ac8fb60c052':
                            row['price_quote']['complete']=False
                if locator==B2_POINT and failure=='allocation':value['N_X'][4]+=1
                return value
            # EVALUATIONS is read by document() to scan a public immutable index.
            document=self.reader.document
            def changed_doc(commit,path):
                value=deepcopy(document(commit,path))
                if path==B2_BASE+'/run01/EVALUATIONS.json'and failure=='hash':
                    for row in value['rows']:
                        if row['job']=='edd63e0befd3af591809':row['outcome_sha256']='0'*64
                return value
            with patch.object(self.reader,'read',side_effect=altered),patch.object(self.reader,'document',side_effect=changed_doc):
                out,records,_=S.reconcile(self.reader,self.seed,self.found([B2_POINT,PF]),{})
            self.assertEqual(records['0']['status'],'PENDING',failure)
            self.assertEqual(frac(out['method_minima']['FULL_PF']['conditional_cost']),480258064)
            self.assertNotEqual(frac(out['method_minima']['PR_PAULI']['conditional_cost']),410199237)


if __name__=='__main__':unittest.main(verbosity=2)
