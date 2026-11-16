"""Artificial request coordinates only, no provider or numerical science."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
helper=ROOT/'src/trottertracks/pr_candidate_selection.py'
if not helper.exists():helper=Path(__file__).with_name('pr_candidate_selection.py')
s=importlib.util.spec_from_file_location('_draft_selection',helper)
M=importlib.util.module_from_spec(s);s.loader.exec_module(M)


def cfg(cid,kappa):return dict(id=cid,spec=dict(kappa={'value':{'numerator':kappa,'denominator':1}},L_D=4,formula='4th(m5_best)',delta_pf=2,h=1))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.base=cfg('BASE',14);self.context=dict(candidates=[dict(name='BASE',configuration_id='BASE',generated_configuration=self.base)],catalog={'candidates':[]})
        self.selection=dict(kind='KAPPA_PROPOSAL',family=dict(L_D=4,formula='4th(m5_best)',delta_over_h='2'),
            kappa_multiplier='3/2',initial_candidates=[dict(from_configuration='BASE',from_kappa='2')])
        self.defaults=dict(kappa='2',K=2,formula='2nd',delta_over_h='1')
    def normalize(self,s):return M.normalize('PR_PAULI',s,self.context,self.defaults,lambda _:None)
    def test_legacy_coordinate_converted_from_explicit_b2_source(self):
        r=self.normalize(self.selection);self.assertEqual(r['design']['kappa'],'21/2')
        self.assertEqual(r['design']['formula'],'4th(m5_best)');self.assertEqual(r['design']['delta_over_h'],'2')
    def test_missing_base_holds_instead_of_default_kappa(self):
        s=dict(self.selection,initial_candidates=[]);r=self.normalize(s)
        self.assertIn('EXPLICIT_BASE',r['selection_error']);self.assertNotIn('design',r)
    def test_absolute_value_does_not_use_multiplier_coordinate(self):
        s=dict(self.selection,kappa='5/3');r=self.normalize(s);self.assertEqual(r['design']['kappa'],'5/3')
    def test_different_families_and_values_have_separate_request_ids(self):
        ids={M.identity(self.selection),M.identity(dict(self.selection,kappa_multiplier='2')),M.identity(dict(self.selection,family=dict(L_D=6,formula='4th(m5_best)',delta_over_h='2')))}
        self.assertEqual(len(ids),3)
        self.assertEqual(M.identity(self.selection),M.identity(dict(self.selection,priority='CHANGING_COST',family_best_cost=999)))
    def test_generated_b2_configuration_and_its_base_are_reused(self):
        generated=cfg('GENERATED',21);self.context['candidates'].append(dict(configuration_id='GENERATED',generated_from='BASE',generated_configuration=generated))
        s=dict(self.selection,preparation=dict(configuration_id='GENERATED'))
        r=self.normalize(s);self.assertEqual(r['configuration'],generated)
        self.assertEqual(r['kappa_normalization']['base_kappa'],'14');self.assertNotIn('design',r)
    def test_actual_published_preparation_callback_can_supply_canonical(self):
        generated=cfg('GENERATED',21)
        def published(s):
            self.context['candidates'].append(dict(configuration_id='GENERATED',generated_from='BASE',generated_configuration=generated))
            return dict(s,preparation=dict(configuration_id='GENERATED'))
        r=M.normalize('PR_DF',self.selection,self.context,self.defaults,lambda _:None,prepare_kappa=published)
        self.assertEqual(r['configuration'],generated)
    def test_conflicting_source_coordinates_rejected(self):
        self.context['candidates'].append(dict(configuration_id='OTHER',generated_configuration=cfg('OTHER',20)))
        s=deepcopy(self.selection);s['initial_candidates'].append(dict(from_configuration='OTHER',from_kappa='2'))
        self.assertIn('BASES_DISAGREE',self.normalize(s)['selection_error'])


if __name__=='__main__':unittest.main(verbosity=2)
