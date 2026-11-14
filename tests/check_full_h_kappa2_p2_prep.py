"""P-2 metadata/fictional-price tests. No H4 restore, sampling, circuit, vector, compile or fit."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_full_h_kappa2_p2 as P
import revalidate_full_h_kappa2_p1_run02 as R


def fictional_paths(plan, cell):
    return [dict(index=i, cohort=cell['cohort'], row=P.physical_row(cell), seeds=seeds,
                 trial=dict(scalar=.3, prices={a: dict(rz_count=100 + i, cx_count=10, rz_depth=10,
                    cx_depth=10, total_depth=20, circuit_size=200) for a in ('cosine', 'sine')}))
            for i, seeds in enumerate(P.seed_table(plan, cell['cell_id']))]


class Tests(unittest.TestCase):
    def test_child_setup_failure_is_saved_without_scientific_calls(self):
        plan=P.proposal();plan['adopted']=True
        def failed_install():raise ValueError('ARTIFICIAL_IMPORT_STOP')
        with TemporaryDirectory() as temp:
            path=Path(temp)/'job.json';result=Path(temp)/'child.json'
            path.write_text(json.dumps(dict(plan=plan,result=str(result))))
            with patch.dict(sys.modules,{'h46_long_time_signal':SimpleNamespace(install=failed_install)}):
                self.assertEqual(P.child(path),2)
            saved=json.loads(result.read_text())
            self.assertEqual(saved['paths'],[]);self.assertIn('ARTIFICIAL_IMPORT_STOP',saved['error'])

    def test_rank_before_acquisition_and_22_24_26_budgets(self):
        for name, budget in (('D6', 22), ('D7', 24), ('D7_H2', 26)):
            self.assertEqual(P.validate(P.proposal(name))['new_compiles'], budget)
        with patch.object(P.subprocess, 'call') as child:
            with self.assertRaisesRegex(ValueError, 'Unidentified calibration'):
                P.execute(dict(plan=P.proposal('D0')))
            with self.assertRaisesRegex(ValueError, 'not adopted'):
                P.execute(dict(plan=P.proposal('D7')))
            child.assert_not_called()

    def test_q_R_s_time_domain_and_incomplete_plan_rejected(self):
        for field, value in (('q', 2), ('R', 567), ('s', 567), ('T', .2), ('role', 'HOLDOUT'), ('n_paths', 3)):
            p = P.proposal(); p['cells'][0][field] = value
            with self.assertRaises(ValueError): P.validate(p)
        p = P.proposal(); p['cells'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete'): P.validate(p)
        for field in ('law', 'implementation', 'profile'):
            p = P.proposal(); p['cells'][0][field] = dict(p['cells'][0][field], foreign='OTHER')
            with self.assertRaises(ValueError): P.validate(p)

    def test_wiring_and_stage_mix_rejected(self):
        p = P.proposal(); p['cells'][0]['cohort'] = P.P1.COHORT
        with self.assertRaises(ValueError): P.validate(p)
        p['cells'][0]['declared_reuse'] = True
        with self.assertRaises(ValueError): P.validate(p)
        with self.assertRaisesRegex(ValueError, 'stage mismatch'):
            P.selected(P.proposal(), 'CALIBRATION', 'M6_ROUND')

    def test_physical_rows_match_C_rounds_and_seed_pairing(self):
        plan = P.proposal(); prepared = P.P1.plan_row()[0]
        for m in (5, 6):
            c = P.selected(plan, 'HOLDOUT', 'M%d_ROUND' % m)
            self.assertEqual(P.physical_row(c), prepared['plan']['pending']['requests'][m]['physical_candidate'])
        a = P.seed_table(plan, 'G566'); b = P.seed_table(plan, 'P2R566')
        self.assertEqual(b, [[a[0][0], a[1][0]]]); self.assertNotEqual(a[0], a[1])
        self.assertEqual(len(P.seed_table(plan, 'Q8R566')[0]), 8)

    def test_order_and_phase_aware_composition(self):
        parents = [dict(events=[[dict(order=0, phase=1)]], seeds=[11]), dict(events=[[dict(order=2, phase=-1)]], seeds=[12])]
        actual = dict(events=[p['events'][0] for p in parents], seeds=[11, 12])
        self.assertEqual(P.check_composition(actual, parents)['status'], 'PAIRED_EVENTS_IDENTICAL')
        for bad in (dict(actual, seeds=[12, 11]), dict(actual, events=list(reversed(actual['events'])))):
            with self.assertRaises(ValueError): P.check_composition(bad, parents)

    def test_fictional_non_configuration_C_input_is_not_EC(self):
        plan = P.proposal(); cell = plan['cells'][0]
        out = P.records(plan, cell, fictional_paths(plan, cell), source=dict(fixture='ARTIFICIAL'))
        self.assertNotIn('prices', out)
        self.assertEqual({r['axis'] for r in out['C_inputs']}, {'X', 'Y'})
        self.assertTrue(all(r['n_paths'] == 2 and r['finite_sample_mean_is_law_expectation'] is False for r in out['C_inputs']))
        self.assertEqual(out['C_inputs'][0]['rz_count'], dict(numerator=201, denominator=2))
        self.assertFalse(out['model_qualified'])

    def test_C_fixed_guard_round_receipts_and_rejection(self):
        plan = P.proposal(); cell = P.selected(plan, 'HOLDOUT', 'M5_ROUND'); prepared = P.P1.plan_row()[0]
        data = P.records(plan, cell, fictional_paths(plan, cell), source=dict(fixture='ARTIFICIAL'), prepared=prepared)
        guard = R.c_guard()
        with patch.object(guard, '_document', return_value=data['implementation_evidence']):
            out = P.check_round_records(prepared, cell, data['prices'], data['sidecars'], evidence_commit='a' * 40)
        self.assertEqual(len(out['checked']), 2)
        bad = deepcopy(data['sidecars']); bad[0]['implementation']['rte_rotation_lowering'] = 'OLD_RZZ'
        with patch.object(guard, '_document', return_value=data['implementation_evidence']), self.assertRaises(ValueError):
            P.check_round_records(prepared, cell, data['prices'], bad, evidence_commit='a' * 40)

    def test_missing_parent_or_frozen_SOURCE_stops_before_child(self):
        p = P.proposal(); p['adopted'] = True
        with patch.object(P.subprocess, 'call') as child:
            job = dict(plan=p, stage='HOLDOUT', cell_id='M6_ROUND', step_seeds=P.seed_table(p, 'M6_ROUND'), duplicate_preflight_complete=True)
            with self.assertRaisesRegex(ValueError, 'frozen'): P.execute(job)
            job.update(stage='CALIBRATION', cell_id='P2R566', step_seeds=P.seed_table(p, 'P2R566'), output=str(ROOT/'__unused_p2_test'), private=str(ROOT/'__unused_p2_private'))
            with self.assertRaisesRegex(ValueError, 'parent'): P.execute(job)
            child.assert_not_called()

    def test_frozen_model_metadata_without_fit(self):
        plan=P.proposal();D=P.design_api();cal=[c for c in plan['cells'] if c['role']=='CALIBRATION']
        features=list(D.FEATURE_MAPS[plan['feature_map']][0])
        models=[dict(model='PRICE_LINEAR_FEATURE_MODEL_V1',status='CALIBRATED_UNVALIDATED',qualified_for_candidate=False,
            domain=dict(D._domain(cal[0]),axis=a),features=features,coefficients={f:dict(numerator=0,denominator=1) for f in features},
            design_source=plan['design_source'],calibration=[dict(cell_id=c['cell_id'],source=dict(cohort=c['cohort'],step_seeds=P.seed_table(plan,c['cell_id']))) for c in cal]) for a in ('X','Y')]
        ref=dict(commit='a'*40,path='ARTIFICIAL_FROZEN.json')
        original=P.P1.blob
        def reader(value):
            return lambda commit,path: json.dumps(value).encode() if path==ref['path'] else original(commit,path)
        with patch.object(P.P1,'blob',side_effect=reader(models)):
            self.assertEqual(P.frozen_model_gate(plan,ref)['status'],'FROZEN_METADATA_CHECKED_NOT_MODEL_QUALIFICATION')
        for mutation in ('missing_axis','holdout','seed','coefficients'):
            bad=deepcopy(models)
            if mutation=='missing_axis':bad.pop()
            elif mutation=='holdout':bad[0]['calibration'][0]['cell_id']='M6_ROUND'
            elif mutation=='seed':bad[0]['calibration'][0]['source']['step_seeds']=[[0]]
            else:bad[0]['coefficients'].pop(features[0])
            with patch.object(P.P1,'blob',side_effect=reader(bad)),self.assertRaises(ValueError):P.frozen_model_gate(plan,ref)


if __name__ == '__main__': unittest.main(verbosity=2)
