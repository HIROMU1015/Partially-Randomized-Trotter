"""Saved price replay plus artificial configuration changes; no scientific run."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.pr_price_rules import PriceRules, exact, frac


def change(req, m, *, q=None, r=None, step=None):
    """Artificial metadata only. Never call a mapper or generate a real candidate."""
    row = req['configuration']['rounds'][m]
    q = row['q_PF'] if q is None else q
    r = row['R_outer'] if r is None else r
    step = frac(row['actual_delta_pf']) if step is None else frac(step)
    row.update(q_PF=q, R_outer=r, T=exact(q*step), actual_delta_pf=exact(step))
    row['structure'].update(allocation=[r], event_count=q*r)
    row['candidate'].update(q=q, R_outer=r, T=row['T'], delta=row['actual_delta_pf'], event_count=q*r)
    if row.get('physical_candidate'):
        row['physical_candidate'].update(q=q, R_outer=r, T=row['T'], delta=row['actual_delta_pf'])


class PriceRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = PriceRules.from_repo(ROOT)

    def test_156_saved_model_axes_and_definitions(self):
        count = 0
        for e in self.p.entries:
            if not e['saved']:
                continue
            out = self.p.quote(self.p.template(e['name']))
            self.assertTrue(out['complete'])
            for row in out['rows']:
                self.assertEqual(row['classification'], 'MODEL_REFERENCE')
                self.assertEqual(row['unit_price_RZ'], e['saved'][row['m'], row['axis']]['value'])
                self.assertFalse(row['qualified_E_C'])
                count += 1
        self.assertEqual(count, 156)

    def test_nine_saved_df_totals(self):
        # Read the exact saved costs only; do not parse/re-evaluate giant Q3 bounds.
        text = subprocess.check_output(['git', 'show', 'fd010b111e95411702be51adc73e56b0bba6c093:docs/research/d_pr_df_ld7_price_results_20261116/SUMMARY.json'], cwd=ROOT, text=True)
        saved = json.loads(text, parse_int=str)['menu']
        self.assertEqual(len(saved), 9)
        for row in saved:
            req = self.p.template('DF_LD'+row['L_D'])
            req['shots'] = {a: [int(s['N_'+a]) for s in row['shots']] for a in ('X','Y')}
            self.assertEqual(frac(self.p.quote(req)['total_RZ']), frac(row['cost']))

    def test_three_saved_pauli_totals(self):
        vp = [48,45,42,39,36,33,29,26,23,19,16,13,9]
        for name in ('PAULI_LD2', 'PAULI_LD3', 'PAULI_LD5'):
            e = next(e for e in self.p.entries if e['name'] == name)
            req = self.p.template(name)
            saved = self.p.reader.read(e['price_source'])
            req['shots'] = {a: [next(r['N'] for r in saved['rows'] if r['m']==m and r['axis']==a) for m in range(13)] for a in ('X','Y')}
            out = self.p.quote(req)
            expected = sum(frac(r.get('N_times_C_ref', r.get('N_times_C_ref_exact',0))) for r in saved['rows'])
            self.assertEqual(frac(out['total_RZ']), expected)
            if name == 'PAULI_LD2':
                self.assertEqual(req['shots']['X'], vp)
                self.assertEqual(frac(out['total_RZ']), Fraction(305225682539751998,90343469))
            if name == 'PAULI_LD3':
                self.assertEqual(frac(out['total_RZ']), frac(saved['T_LD3']))

    def test_shots_and_kappa_only_change(self):
        req = self.p.template('DF_LD7')
        req['shots'] = {a: [1]*13 for a in ('X','Y')}
        before = self.p.quote(req)
        req['configuration']['id'] = 'ARTIFICIAL_NEW_ID'
        req['configuration']['spec']['kappa'] = {'value': exact(999)}
        req['configuration']['spec']['delta_pf'] = exact(Fraction(1,20))  # design declaration only
        req['shots']['X'][0] = 2
        after = self.p.quote(req)
        self.assertEqual([r['unit_price_RZ'] for r in before['rows']], [r['unit_price_RZ'] for r in after['rows']])
        self.assertEqual(frac(after['total_RZ'])-frac(before['total_RZ']), frac(before['rows'][0]['unit_price_RZ']))
        req['policy'] = 'DIRECT'
        out = self.p.quote(req)
        self.assertEqual(out['rows'][0]['classification'], 'DIRECT_FINITE_COHORT')

    def test_direct_and_model_are_not_substituted(self):
        req = self.p.template('DF_LD7')
        modeled = self.p.quote(req)['rows'][6]
        req['policy']='AUTO'
        direct = self.p.quote(req)['rows'][6]
        self.assertEqual(direct['classification'], 'DIRECT_FINITE_COHORT')
        self.assertEqual(frac(direct['unit_price_RZ']), 49690)
        self.assertNotEqual(direct['unit_price_RZ'], modeled['unit_price_RZ'])
        self.assertEqual(direct['direct_alternatives'][0]['role'], 'INDEPENDENT_HOLDOUT')

    def test_df9_partial_and_missing_rule(self):
        req = self.p.template('DF_LD9');req['policy']='AUTO'
        out=self.p.quote(req)
        self.assertFalse(out['complete']);self.assertIsNone(out['total_RZ'])
        self.assertEqual(sum(r['classification']=='DIRECT_FINITE_COHORT' for r in out['rows']),2)
        self.assertEqual(frac(out['rows'][12]['unit_price_RZ']),492573)
        self.assertIn('LD9 m6',str(out['rows'][0]['needs']))

    def test_qR_model_extension_and_long_round_flags(self):
        req=self.p.template('DF_LD2');change(req,3,r=6)
        row=self.p.quote(req)['rows'][6]
        self.assertEqual(row['classification'],'MODEL_REFERENCE')
        self.assertEqual(row['model_alternative']['application'],'EXISTING_RULE_APPLICATION')
        self.assertEqual(row['direct_alternatives'],[])
        self.assertIn('q',self.p.quote(req)['rows'][24]['model_alternative']['extrapolation'])
        req=self.p.template('PAULI_LD5');change(req,3,r=5)
        self.assertIn('R',self.p.quote(req)['rows'][6]['model_alternative']['extrapolation'])

    def test_changed_actual_step_needs_calibration(self):
        for name in ('DF_LD1','DF_LD2','DF_LD7','PAULI_LD2','PAULI_LD3','PAULI_LD5'):
            req=self.p.template(name);change(req,3,q=16,step=Fraction(1,20))
            row=self.p.quote(req)['rows'][6]
            self.assertEqual(row['classification'],'NEEDS_PRICE_EVIDENCE')
            self.assertIn('actual PF step',str(row['needs']))

    def test_df7_R_changes_are_underidentified(self):
        req=self.p.template('DF_LD7');change(req,3,r=2)
        row=self.p.quote(req)['rows'][6]
        self.assertEqual(row['classification'],'NEEDS_PRICE_EVIDENCE')
        self.assertEqual(row['needs']['calibration']['coefficients'],['separate head q slope','tail qR slope'])

    def test_structural_input_cannot_be_fabricated(self):
        req=self.p.template('DF_LD1');change(req,3,r=17)
        row=self.p.quote(req)['rows'][6]
        self.assertEqual(row['classification'],'NEEDS_PRICE_EVIDENCE')
        self.assertEqual(row['needs']['calibration']['coefficients'],[])
        self.assertIn('E_I[S]',row['needs']['calibration']['missing_quantity'])

    def test_event_recipe_and_piece_changes_are_not_ignored(self):
        req=self.p.template('DF_LD2')
        req['configuration']['rounds'][3]['law_binding']['definition']['order_probabilities']=[1,0,0]
        row=self.p.quote(req)['rows'][6]
        self.assertEqual(row['classification'],'NEEDS_PRICE_EVIDENCE')
        self.assertIn('event recipe',str(row['needs']))
        req=self.p.template('DF_LD2')
        req['configuration']['rounds'][3]['structure']['signed_weights']=[-1.0]
        self.assertEqual(self.p.quote(req)['rows'][6]['classification'],'NEEDS_PRICE_EVIDENCE')

    def test_nonpositive_model_and_unknown_cohort_do_not_fallback_silently(self):
        p=deepcopy(self.p)
        e=next(e for e in p.entries if e['name']=='DF_LD2')
        e['model']['models']['X']['coefficients']={'one':exact(-1),'q':exact(0),'qR':exact(0)}
        req=p.template('DF_LD2');change(req,3,r=6)
        row=p.quote(req)['rows'][6]
        self.assertEqual(row['classification'],'NEEDS_PRICE_EVIDENCE')
        self.assertIn('nonpositive',str(row['needs']))
        req=self.p.template('DF_LD7');req.update(policy='DIRECT',direct_cohort='UNKNOWN')
        self.assertTrue(all(r['unit_price_RZ']is None for r in self.p.quote(req)['rows']))

    def test_formula_split_dictionary_wrapper_and_sources_do_not_transfer(self):
        mutations=[('formula',lambda r:r['configuration']['spec'].__setitem__('formula','4th(m5_best)')),
                   ('split',lambda r:r['configuration']['spec']['split'].__setitem__('scalar',999)),
                   ('profile',lambda r:r['configuration']['spec']['profile'].__setitem__('seed_transpiler',18)),
                   ('implementation',lambda r:r['implementation'].__setitem__('lowering_id','OTHER')),
                   ('law',lambda r:r['configuration']['spec']['law']['tail_dictionary'].__setitem__('dictionary_sha256','OTHER'))]
        for field,mutate in mutations:
            req=self.p.template('PAULI_LD5');mutate(req)
            if field=='formula':
                for row in req['configuration']['rounds']:row['candidate']['formula']='4th(m5_best)'
            row=self.p.quote(req)['rows'][0]
            self.assertEqual(row['classification'],'NEEDS_PRICE_EVIDENCE')
            self.assertIn('changed: '+field,row['needs']['reasons'])
        req=self.p.template('DF_LD2');req['configuration']['spec']['L_D']=3
        req['configuration']['spec']['split']['L_D']=3
        for row in req['configuration']['rounds']:row['candidate']['prefix']=3
        self.assertTrue(all(r['unit_price_RZ']is None for r in self.p.quote(req)['rows']))

    def test_A_cell_receipts_never_become_round_receipts(self):
        for name,r,expected in [('PAULI_LD3',660,57367),('PAULI_LD5',4,Fraction(68629,2))]:
            req=self.p.template(name);req['policy']='AUTO';change(req,3,r=r)
            out=self.p.quote(req)['rows'][6]
            self.assertEqual(out['classification'],'DIRECT_FINITE_COHORT')
            self.assertEqual(frac(out['unit_price_RZ']),expected)
            self.assertIsNone(out['direct_alternatives'][0]['m'])
            self.assertTrue(out['direct_alternatives'][0]['saved_physical_id'].startswith('PAULI_PRICE_CELL:'))

    def test_paired_cell_dependence_and_wiring_are_preserved(self):
        req=self.p.template('PAULI_LD2');req['policy']='AUTO';change(req,1,r=66)
        cell=self.p.quote(req)['rows'][2]['direct_alternatives'][0]
        self.assertEqual(cell['n_paths'],1);self.assertTrue(cell['paired_dependence'])
        req=self.p.template('PAULI_LD1');req['policy']='AUTO'
        self.assertEqual(self.p.quote(req)['rows'][0]['direct_alternatives'][0]['role'],'B0_WIRING_ONLY_NOT_CALIBRATION')

    def test_malformed_identity_pieces_and_shots_fail_closed(self):
        req=self.p.template('DF_LD7');req['configuration']['rounds'][0]['candidate']['q']=2
        with self.assertRaisesRegex(ValueError,'disagrees'):self.p.quote(req)
        req=self.p.template('DF_LD7');req['configuration']['rounds'][0]['structure']['allocation']=[2]
        with self.assertRaisesRegex(ValueError,'allocation'):self.p.quote(req)
        req=self.p.template('DF_LD7');req['shots']={'X':[1]*13,'Y':[1]*12}
        with self.assertRaisesRegex(ValueError,'shots'):self.p.quote(req)
        req=self.p.template('DF_LD7');req['shots']={'X':[True]*13,'Y':[1]*13}
        with self.assertRaisesRegex(ValueError,'Shot'):self.p.quote(req)

    def test_cli_real_entry_and_all_fixed_locators(self):
        registry=json.loads((ROOT/'configs/pr_price_rules/h4_saved_rules.json').read_text())
        def walk(x):
            if isinstance(x,dict):
                if 'commit'in x and 'path'in x:
                    subprocess.run(['git','cat-file','-e',x['commit']+':'+x['path']],cwd=ROOT,check=True)
                    if x['path'].endswith('.json'):self.p.reader.read(x)
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(registry)
        template=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/quote_pr_price_rules.py'),'--template','DF_LD7']))
        self.assertIn('implementation',template)
        with tempfile.TemporaryDirectory() as tmp:
            inp,out=Path(tmp)/'input.json',Path(tmp)/'output.json'
            inp.write_text(json.dumps(template))
            subprocess.run([sys.executable,str(ROOT/'scripts/quote_pr_price_rules.py'),
                            '--input',str(inp),'--output',str(out)],check=True)
            quoted=json.loads(out.read_text())
            self.assertTrue(quoted['complete']);self.assertEqual(len(quoted['rows']),26)
            template['policy']='INVALID';inp.write_text(json.dumps(template))
            bad=subprocess.run([sys.executable,str(ROOT/'scripts/quote_pr_price_rules.py'),
                                '--input',str(inp)],capture_output=True,text=True)
            self.assertEqual(bad.returncode,2)
            self.assertEqual(json.loads(bad.stderr)['status'],'INVALID_PRICE_INPUT')


if __name__=='__main__':
    unittest.main(verbosity=2)
