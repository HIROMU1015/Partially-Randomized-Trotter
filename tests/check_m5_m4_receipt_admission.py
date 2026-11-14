"""Artificial receipt values only; pinned real metadata is never numerically fit."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from trottertracks import price_model_validation as V

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('m5_receipt_check', ROOT / 'scripts/check_m5_m4_receipt_admission.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
MOCK = dict(fixture='ARTIFICIAL_RECEIPT_AND_PRICES_ONLY')


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t, cls.f = M.d_callers(ROOT)
        cls.full, cls.trunc = cls.f.plans(ROOT), cls.t.plans(ROOT)

    def receipt(self):
        gates = ('low_control', 'high_control_leakage', 'independent_occupation_vector',
                 'Hadamard_cosine', 'compiled_vector_cosine', 'Hadamard_sine', 'compiled_vector_sine')
        c = deepcopy(self.full[2]['cells'][0]['row'])
        trial = dict(candidate=c, scalar=.125, status='PASS', cohort='EXPLORATION', index=0,
            conditional=False, laws=[], log_B=0, preparation_hash='ARTIFICIAL_PREPARATION',
            prices=dict(cosine=dict(rz_count=111), sine=dict(rz_count=222)),
            checks=[dict(check=k, error=0., tolerance=1e-9, phase_alignment=False) for k in gates])
        carrier = dict(candidate=deepcopy(c), cohort='EXPLORATION', n=1, paired_cosine_sine=True,
            conditional_sampling=False, mean_metrics=deepcopy(trial['prices']))
        catalog = dict(physical_candidate=deepcopy(c), prices=[dict(axes={
            'X': dict(status='EXACT_METADATA_MATCH', rz_count=111),
            'Y': dict(status='EXACT_METADATA_MATCH', rz_count=222)})])
        profile = deepcopy(self.full[2]['plan']['configuration']['spec']['profile'])
        prov = dict(compiler=dict(hash=profile['compiler'], basis=['rz','sx','x','cx'], optimization_level=1, seed_transpiler=17),
            runtime=dict(qiskit='1.3.0'), cost_policy='FULL_COSINE_SINE_HADAMARD_WRAPPER_MEASUREMENT_INCLUDED_P0')
        return [trial, carrier, catalog, c, prov, profile, .125]

    def packet(self):
        trial = self.receipt()[0]; p = self.full[2]
        f = M.d_callers(ROOT)[1]
        f.price_bindings.__globals__['COHORT'] = 'EXPLORATION'
        prices, sides, proofs = f.price_bindings(ROOT, p['plan'], p['cells'][0], trial,
            price_source=MOCK, evidence_commit='a'*40, evidence_path='ARTIFICIAL_PROOFS.json')
        return dict(prices=prices, sides=sides, proofs=proofs, plan=p['plan'])

    def test_artificial_receipt_and_original_cohort(self):
        args = self.receipt(); saved = deepcopy(args)
        self.assertFalse(M.inspect(*args)['recompile']); self.assertEqual(args, saved)
        packet = self.packet()
        self.assertEqual([p['mean']['rz_count'] for p in packet['prices']], [111, 222])
        self.assertTrue(all(p['cohort']=='EXPLORATION' and p['classification']=='DIRECT_COMPILED_DETERMINISTIC'
            and p['expectation']=='DETERMINISTIC' for p in packet['prices']))

    def test_mismatched_receipt_rejected_without_repair(self):
        changes = (lambda a:a[0].update(scalar=.25), lambda a:a[0]['candidate'].update(q=2),
            lambda a:a[1].update(conditional_sampling=True), lambda a:a[0]['prices'].pop('sine'),
            lambda a:a[4]['compiler'].update(seed_transpiler=18),
            lambda a:a[0]['checks'][0].update(error=1.),
            lambda a:a[2]['prices'][0]['axes']['Y'].update(rz_count=111))
        for change in changes:
            args = self.receipt(); change(args)
            with self.subTest(change=change), self.assertRaises(ValueError): M.inspect(*args)

    def test_fixed_metadata_roles_scope_and_no_science_entry(self):
        full = [c for p in self.full for c in p['cells']]
        self.assertEqual(len(full),9); self.assertEqual(sum(c['action']=='NEW_COMPILE' for c in full),8)
        self.assertEqual(sum(len(p['cells']) for p in self.trunc),9)
        for templates in (self.full,self.trunc):
            self.assertEqual([[c['row']['q'] for c in p['cells']] for p in templates], [[7,14,28],[3,6,12],[1,2,4]])
        self.assertEqual([c['role'] for c in self.full[0]['cells']],['CALIBRATION','CALIBRATION','HOLDOUT'])
        self.assertFalse(any(hasattr(self.f,n) or hasattr(self.t,n) for n in ('execute','child','full_trial','checked_trial','cost_trial')))

    def test_existing_guard_and_paper_tier_without_G(self):
        packet = self.packet()
        with patch.object(M.G,'_document',return_value=packet['proofs']), \
             patch.object(M.PN,'paper_cost',side_effect=AssertionError('G forbidden')):
            result = M.probe(ROOT,packet,'a'*40)
        self.assertEqual(len(result['guard']['checked']),2)
        self.assertEqual(result['tiers'],dict(X='DETERMINISTIC_DIRECT',Y='DETERMINISTIC_DIRECT'))
        self.assertFalse(result['comparison_policy_adopted']);self.assertIsNone(result['G'])

    def test_guard_rejects_scalar_axis_source_and_physical_id(self):
        for field in ('scalar','axis','source','physical'):
            packet=self.packet();side=packet['sides'][0]
            if field=='scalar': side['implementation']['resolved_scalar']=.25
            elif field=='axis': side['implementation']['measurement']['measurement_axis']='Y'
            elif field=='source': packet['prices'][0]['source']=dict(fixture='FOREIGN')
            else: side['physical_context']['physical_candidate']['id']='FOREIGN'
            packet['proofs'][side['implementation_evidence']['selector'][0]]={k:deepcopy(side[k]) for k in
                ('price_source','identity','law_binding','axis','cohort','physical_context','profile_extension','implementation')}
            with patch.object(M.G,'_document',return_value=packet['proofs']), self.subTest(field=field), self.assertRaises(ValueError):
                M.probe(ROOT,packet,'a'*40)

    def test_twelve_separate_synthetic_affine_models(self):
        # Values below are invented; no saved RZ value is used in these fits.
        for route,templates in (('FULL_PF',self.full),('TRUNCATED_PF',self.trunc)):
            for p in templates:
                prices=[]
                for c in p['cells']:
                    for axis in ('X','Y'):
                        prices.append(dict(m=c['m'],axis=axis,physical_context=dict(physical_candidate=c['row']),
                            mean=dict(rz_count=10+(2 if axis=='X' else 3)*c['row']['q']),source=MOCK,
                            profile_extension=dict(lowering_id='ARTIFICIAL_NATIVE',source=MOCK),price_profile=MOCK))
                records=self.f.fit_inputs(prices,route=route,formula=p['formula'],width_over_h=p['width_over_h'])
                for axis in ('X','Y'):
                    group=[r for r in records if r['axis']==axis]
                    model=V.fit([r for r in group if r['role']=='CALIBRATION'],features=['one','q'],design_source=MOCK)
                    result=V.validate(model,[r for r in group if r['role']=='HOLDOUT'],
                        tolerance=dict(kind='RELATIVE',value=Fraction(1,50),source=MOCK))
                    self.assertEqual(result['status'],'HOLDOUT_WITHIN_DECLARED_TOLERANCE')
                    self.assertFalse(result['qualified_for_candidate'])

    def test_relative_tolerance_observed_denominator_and_no_refit(self):
        def rec(m,q,rz,role):
            return V.record(cell_id=m,role=role,route='FULL_PF',axis='X',features=dict(one=1,q=q),
                rz_count=rz,kind='DIRECT_COMPILED_DETERMINISTIC',source=MOCK,
                implementation=dict(lowering_id='ART'),profile=MOCK)
        model=V.fit([rec('m4',1,102,'CALIBRATION'),rec('m5',2,102,'CALIBRATION')],features=['one','q'],design_source=MOCK)
        tol=dict(kind='RELATIVE',value=Fraction(1,50),source=MOCK)
        edge=V.validate(model,[rec('m6',4,100,'HOLDOUT')],tolerance=tol)
        self.assertEqual(edge['status'],'HOLDOUT_WITHIN_DECLARED_TOLERANCE')
        failed=V.validate(model,[rec('m6',4,99,'HOLDOUT')],tolerance=tol)
        self.assertEqual(failed['status'],'HOLDOUT_FAILED_NO_REFIT')
        self.assertEqual(failed['coefficients'],model['coefficients'])
        with self.assertRaises(ValueError):V.validate(failed,[rec('m7',8,102,'HOLDOUT')],tolerance=tol)


if __name__=='__main__':unittest.main(verbosity=2)
