"""Artificial numbers only. Never read archived H8 observations in these tests."""
import copy
import csv
from fractions import Fraction as F
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from h8_rare_saved_core import analyze_cell, integrate, near
from h8_rare_saved_verify import validate
from h8_rare_saved_run import exclusive, export_tables
from h8_rare_c05.accounting import Budget, sample_statistics
import h8_rare_saved_run as runner


def fixture():
    candidate='B2_p8_q2_2nd_r4_K4'
    z=[10,12]*24;s=[20+2*(i%3) for i in range(43)]
    probabilities=dict(zip(('S0_ONLY','S2_SINGLE','S2_MULTI','S4_PRESENT'),('0.8','0.15','0.04','0.01')))
    previous=dict(observed_plugin_G='8.8',unknown_mass='0.2',unknown_mixture_mean_threshold_G='456',
                  threshold_over_observed_S0_mean=str(F(456,11)))
    b=dict(condition='H8_0.05',candidate=candidate,system='H8',rank=28,r=4,K=4,q=2,n=48,
           shots={'cosine':1,'sine':1},fixed_PF='ARTIFICIAL',fixed_PF_G_exact='100',
           cohort='ARTIFICIAL',law={'saved_law':{'dummy':True}},conditional_expectation=previous,
           groups=[dict(stratum=k,probability=v,G=dict(mean_exact=sample_statistics(z)['mean'],
                 sample_variance_exact=sample_statistics(z)['variance'],SE=sample_statistics(z)['SE']))
                 for k,v in probabilities.items()],
           tasks=[dict(candidate=candidate,replica=i,stratum='S0_ONLY',order2=0,order4=0,G_exact=str(v),
                       RZ_cosine=v//2,RZ_sine=v//2) for i,v in enumerate(z)])
    rows=[]
    for i,v in enumerate(s):
        rows.append(dict(task_id='ARTIFICIAL_'+str(i),candidate=candidate,stratum='S2_SINGLE',
              scientific_mock=False,normalization_changed=False,event_digest='ARTIFICIAL_'+str(i),
              orders=[0,2,0,0,0,0,0,0],axes={a:dict(event_digest='ARTIFICIAL_'+str(i),metrics={'RZ':v//2})
                                                           for a in ('cosine','sine')}))
    pilot=dict(condition='H8_0.05',candidate=candidate,shots=b['shots'],rows=rows)
    saved=dict(groups={'S2_SINGLE':sample_statistics(s)})
    law=dict(status='PASS',q=2,r=4,K=4,T=.8,law={'dummy':True},
             diagnostic=dict(engineering_gate_passed=True,compiled_cost_bias_bounded=False))
    return b,pilot,saved,law,z,s,probabilities,previous


def csv_text(rows,fields):
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    return buf.getvalue()


class IntegrationTests(unittest.TestCase):
    def calculate(self):
        b,p,s,l,*_=fixture();return analyze_cell(b,p,s,l)

    def test_exact_known_contribution(self):
        r=self.calculate();self.assertTrue(near(r['K_known_G'],F(44,5)+F(3,20)*F(sum(fixture()[5]),43)))

    def test_independent_variance_sum(self):
        r=self.calculate();b,p,s,l,z,x,probs,prev=fixture()
        v0=F(48*sum(i*i for i in z)-sum(z)**2,48*47)
        v1=F(43*sum(i*i for i in x)-sum(x)**2,43*42)
        self.assertTrue(near(r['estimated_variance_K_known_G'],F(4,5)**2*v0/48+F(3,20)**2*v1/43))

    def test_unknowns_retained(self):
        r=self.calculate();self.assertEqual(r['residual_means'],{'S2_MULTI':None,'S4_PRESENT':None})
        self.assertIsNone(r['full_population_variance_G']);self.assertFalse(r['population_ranking_guaranteed'])
        self.assertIsNone(r['statistical_coverage_probability'])

    def test_boundary_conservation(self):
        r=self.calculate()
        for pt in r['two_layer_boundary']['points']:
            self.assertTrue(near(F('0.04')*F(pt['mu_MULTI_G'])+F('0.01')*F(pt['mu_S4_G']),r['remaining_budget_D_G']))

    def test_unknown_mass_reduction_not_variance_claim(self):
        r=self.calculate();self.assertEqual(F(r['residual_mass']),F(1,20))
        self.assertEqual(F(r['stage_B_comparison']['unknown_mass_reduction_factor']),4)
        self.assertIsNone(r['stage_B_comparison']['reduction_in_full_mean_uncertainty'])

    def test_se_shifts_monotone_without_coverage(self):
        rows=self.calculate()['known_error_sensitivity']
        self.assertGreater(F(rows[0]['mixture_threshold_low']),F(rows[3]['mixture_threshold_low']))
        self.assertTrue(all(r['coverage_probability'] is None for r in rows))

    def test_reject_old_value_mismatch(self):
        b,p,s,l,*_=fixture();b['groups'][0]['G']['mean_exact']='12'
        with self.assertRaisesRegex(ValueError,'STATISTICS'):analyze_cell(b,p,s,l)

    def test_reject_duplicate(self):
        b,p,s,l,*_=fixture();p['rows'][1]['task_id']=p['rows'][0]['task_id']
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):analyze_cell(b,p,s,l)

    def test_reject_wrong_stratum(self):
        b,p,s,l,*_=fixture();p['rows'][0]['orders'][3]=4
        with self.assertRaisesRegex(ValueError,'SUPPORT'):analyze_cell(b,p,s,l)

    def test_reject_law_certificate_relabel(self):
        b,p,s,l,*_=fixture();l['diagnostic']['compiled_cost_bias_bounded']=True
        with self.assertRaisesRegex(ValueError,'LAW'):analyze_cell(b,p,s,l)

    def test_reject_shot_change(self):
        b,p,s,l,*_=fixture();p['shots']={'cosine':2,'sine':1}
        with self.assertRaisesRegex(ValueError,'SHOTS'):analyze_cell(b,p,s,l)

    def test_separate_csv_arithmetic(self):
        b,p,s,l,z,x,*_=fixture();out=analyze_cell(b,p,s,l)
        bc=csv_text(b['tasks'],['candidate','stratum','RZ_cosine','RZ_sine','G_exact'])
        cc=csv_text([dict(task=str(i),axis=a,RZ=v//2,paired_G=v) for i,v in enumerate(x) for a in ('cosine','sine')],
                    ['task','axis','RZ','paired_G'])
        self.assertTrue(validate(b,bc,cc,out)['status'].startswith('PASS'))
        out['K_known_G']='1000'
        with self.assertRaises(AssertionError):validate(b,bc,cc,out)

    def test_nonrefundable_budget(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(Path(d)/'budget',{'analysis':1});b.take('analysis',1,'once')
            with self.assertRaises(ValueError):b.take('analysis',1,'retry')
            self.assertEqual(b.used(),{'analysis':1})

    def test_one_shot_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'marker';exclusive(p,{'consumed':True})
            with self.assertRaises(FileExistsError):exclusive(p,{'consumed':False})

    def test_complete_exports_without_overwrite(self):
        oldroot,oldout=runner.ROOT,runner.OUT
        try:
            with tempfile.TemporaryDirectory() as d:
                runner.ROOT=Path(d);runner.OUT='.';rows=[self.calculate()]
                export_tables(rows)
                for n,count in [('known_layers.csv',2),('two_layer_boundary.csv',5),('known_error_sensitivity.csv',4),('condition_summary.csv',1)]:
                    with (Path(d)/n).open() as f:self.assertEqual(len(list(csv.DictReader(f))),count)
                with self.assertRaises(FileExistsError):export_tables(rows)
        finally:runner.ROOT,runner.OUT=oldroot,oldout


if __name__=='__main__':unittest.main()
