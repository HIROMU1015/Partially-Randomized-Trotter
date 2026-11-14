"""B1 metadata + fictional prices only. No H4 preparation, sampling, build, vector, compile or fit."""
from copy import deepcopy
from pathlib import Path
import json
from unittest.mock import patch
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_pauli_b1_prices as P


def fictional(plan, cell):
    req = P.request_for(plan, cell)[0]
    return [dict(index=i, cohort=cell['cohort'], request=deepcopy(req), result=dict(scalar=.3, seeds=seeds,
                prices={a: dict(rz_count=100+i, cx_count=10, rz_depth=10, cx_depth=10, total_depth=20, circuit_size=200)
                        for a in ('cosine', 'sine')})) for i, seeds in enumerate(P.seed_table(plan, cell['cell_id']))]


class Tests(unittest.TestCase):
    def test_B1_cal_rank_and_24_each_dictionary(self):
        for prefix in (1, 2):
            self.assertEqual(P.validate(P.proposal(prefix))['new_compiles'], 24)
        for prefix, Rs in ((1, [171, 1361, 10885]), (2, [66, 526, 4205])):
            self.assertEqual([c['R'] for c in P.proposal(prefix)['cells'][:3]], Rs)
        wire = P.proposal(2, 'B1-W')
        self.assertEqual(P.validate(wire)['status'], 'WIRING_ONLY_NOT_A_PRICE_MODEL')
        self.assertEqual(P.validate(wire)['identifiability']['rank'], 1)
        with self.assertRaises(ValueError): P.proposal(1, 'B1-W')

    def test_unadopted_and_incomplete_wrong_q_R_law_lowering_rejected_before_child(self):
        with patch.object(P.subprocess, 'call') as child:
            with self.assertRaisesRegex(ValueError, 'not adopted'): P.execute(dict(plan=P.proposal()))
            child.assert_not_called()
        for field, value in (('q', 2), ('R', 172), ('s', 172), ('T', .2), ('role', 'HOLDOUT')):
            plan = P.proposal(); plan['cells'][0][field] = value
            with self.assertRaises(ValueError): P.validate(plan)
        plan = P.proposal(); plan['cells'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete'): P.validate(plan)
        plan = P.proposal(); plan['cells'][0]['implementation']['lowering_id'] = 'CONTROLLED_ZZ_CX_CRZ_CX_V1'
        with self.assertRaises(ValueError): P.validate(plan)

    def test_dictionary_scalar_physical_identity_and_rounds(self):
        reqs = []
        for prefix in (1, 2):
            plan = P.proposal(prefix); req, canonical, packet, dictionary = P.request_for(plan, plan['cells'][0])
            self.assertEqual((req['prefix'], dictionary['prefix']), (prefix, prefix))
            self.assertEqual(req['lambda_P'], dictionary['lambda_P'])
            self.assertEqual(packet['dictionary_identity']['prefix'], prefix)
            reqs.append(req)
            m5 = next(c for c in plan['cells'] if c['cell_id'] == 'M5_ROUND')
            r = P.request_for(plan, m5)[0]
            self.assertEqual(r['candidate_id'], canonical['configuration']['rounds'][5]['candidate']['id'])
            self.assertNotEqual(r['candidate_id'], req['candidate_id'])
            self.assertTrue(req['candidate_id'].startswith('PAULI_PRICE_CELL:'))
        self.assertNotEqual(reqs[0]['candidate_id'], reqs[1]['candidate_id'])

    def test_pair_and_independent_step_seeds(self):
        for prefix in (1, 2):
            plan = P.proposal(prefix); first = plan['cells'][0]['cell_id']; pair = next(c['cell_id'] for c in plan['cells'] if c.get('composed_of'))
            s = P.seed_table(plan, first)
            self.assertEqual(P.seed_table(plan, pair), [[s[0][0], s[1][0]]])
            self.assertNotEqual(s[0], s[1])

    def test_B0_reuse_is_declared_original_path0_not_new_sample_or_holdout(self):
        p = P.proposal(1, 'B1-V', reuse_b0=True); cell = p['cells'][0]
        self.assertEqual(cell['cohort'], P.B0.connection().PILOT_COHORT)
        self.assertEqual(P.seed_table(p, 'M0')[0], P.B0.FIXED_SEEDS)
        data = P.preflight(p)
        self.assertEqual(data['acquisition_budget'], dict(new_compiles=2, reused_compiles=2))
        self.assertEqual(cell['declared_reuse']['source'], P.B0_SOURCE)
        with self.assertRaises(ValueError): P.proposal(1, 'B1-CAL', reuse_b0=True)
        cell['cohort'] = 'RELABELLED'
        with self.assertRaises(ValueError): P.validate(p)

    def test_fictional_C_records_no_promotion_and_path_mix_rejected(self):
        plan = P.proposal(); cell = plan['cells'][0]; paths = fictional(plan, cell)
        out = P.records(plan, cell, paths, source=dict(fixture='ARTIFICIAL'), evidence_path='ARTIFICIAL.json')
        self.assertNotIn('prices', out)
        self.assertTrue(all(r['finite_sample_mean_is_law_expectation'] is False and r['n_paths']==2 for r in out['C_inputs']))
        for mutate in (lambda ps: ps[0].update(cohort='FOREIGN'), lambda ps: ps[0].update(index=1),
                       lambda ps: ps[0]['result'].update(seeds=[0]), lambda ps: ps[0]['request'].update(prefix=2)):
            bad=deepcopy(paths);mutate(bad)
            with self.assertRaises(ValueError): P.records(plan, cell, bad, source=dict(fixture='ARTIFICIAL'), evidence_path='A.json')

    def test_C_fixed_guard_canonical_LD2_wiring_and_wrong_axis_rejection(self):
        p = P.proposal(2, 'B1-W'); c = p['cells'][0]
        out = P.records(p, c, fictional(p,c), source=dict(fixture='ARTIFICIAL',commit='00f74f86d16cc6878e1d4c5977eb9abd9ce9c076'), evidence_path='ARTIFICIAL.json')
        G=P.B0._meta()[2]
        with patch.object(G,'_document',return_value=out['implementation_evidence']):
            result=P.check_round_records(p,c,out['prices'],out['sidecars'],evidence_commit='a'*40)
        self.assertEqual(len(result['checked']),2)
        self.assertEqual(out['C_inputs'], [])  # rank-1 wiring is not a model calibration point
        bad=deepcopy(out['prices']);bad[0]['measurement_axis']='sine'
        with patch.object(G,'_document',return_value=out['implementation_evidence']),self.assertRaises(ValueError):
            P.check_round_records(p,c,bad,out['sidecars'],evidence_commit='a'*40)

    def test_LD2_calibration_requires_published_wiring_metadata_without_replay(self):
        with self.assertRaisesRegex(ValueError, 'LD2 B1-W SOURCE'): P.ld2_wiring_gate(None)
        wire=P.proposal(2,'B1-W'); request=P.request_for(wire,wire['cells'][0])[0]
        checks=['low_control','high_control_leakage','independent_full_space_vector','Hadamard_cosine',
                'Hadamard_sine','compiled_vector_cosine','compiled_vector_sine']
        data=dict(status='FINITE_CELL_COMPLETE',child=dict(paths=[dict(request=request,
            result=dict(status='PASS',checks=[dict(check=c) for c in checks]))]))
        ref=dict(commit='a'*40,path='ARTIFICIAL_WIRING.json'); original=P.B0.blob
        def reader(value):
            return lambda commit,path: json.dumps(value).encode() if path==ref['path'] else original(commit,path)
        with patch.object(P.B0,'blob',side_effect=reader(data)):
            self.assertEqual(P.ld2_wiring_gate(ref)['status'],'LD2_WIRING_METADATA_CHECKED')
        for mutation in ('prefix','compiled_gate','status'):
            bad=deepcopy(data)
            if mutation=='prefix':bad['child']['paths'][0]['request']['prefix']=1
            elif mutation=='compiled_gate':bad['child']['paths'][0]['result']['checks'].pop()
            else:bad['status']='STOP_CELL'
            with patch.object(P.B0,'blob',side_effect=reader(bad)),self.assertRaises(ValueError):P.ld2_wiring_gate(ref)

    def test_frozen_model_metadata_without_fit(self):
        plan=P.proposal();D=P.design_api();cal=[c for c in plan['cells'] if c['role']=='CALIBRATION']
        features=list(D.FEATURE_MAPS[plan['feature_map']][0])
        models=[dict(model='PRICE_LINEAR_FEATURE_MODEL_V1',status='CALIBRATED_UNVALIDATED',qualified_for_candidate=False,
            domain=dict(D._domain(cal[0]),axis=a),features=features,coefficients={f:dict(numerator=0,denominator=1) for f in features},
            design_source=plan['design_source'],calibration=[dict(cell_id=c['cell_id'],source=dict(cohort=c['cohort'],step_seeds=P.seed_table(plan,c['cell_id']))) for c in cal]) for a in ('X','Y')]
        ref=dict(commit='a'*40,path='ARTIFICIAL_FROZEN.json')
        original=P.B0.blob
        def reader(value):
            return lambda commit,path: json.dumps(value).encode() if path==ref['path'] else original(commit,path)
        with patch.object(P.B0,'blob',side_effect=reader(models)):
            self.assertEqual(P.frozen_model_gate(plan,ref)['status'],'FROZEN_METADATA_CHECKED_NOT_MODEL_QUALIFICATION')
        for mutation in ('missing_axis','holdout','seed','coefficients'):
            bad=deepcopy(models)
            if mutation=='missing_axis':bad.pop()
            elif mutation=='holdout':bad[0]['calibration'][0]['cell_id']='M6_ROUND'
            elif mutation=='seed':bad[0]['calibration'][0]['source']['step_seeds']=[[0]]
            else:bad[0]['coefficients'].pop(features[0])
            with patch.object(P.B0,'blob',side_effect=reader(bad)),self.assertRaises(ValueError):P.frozen_model_gate(plan,ref)

if __name__=='__main__':unittest.main(verbosity=2)
