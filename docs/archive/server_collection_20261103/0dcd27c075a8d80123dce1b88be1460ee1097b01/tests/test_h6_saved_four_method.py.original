"""Artificial validation/rank fixtures only; no real H6 numeric evaluations."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h6_saved_four_method as c


def fixture():
    methods=[]
    for can,value in zip(c.CANDIDATES,('40','20',30.,10.)):
        methods.append(dict(candidate=can,unit=c.PHYSICAL['unit'],binding=c.BINDING,status='ELIGIBLE',
            cohort='EXPLORATION' if can['method']=='PR' else 'COMMON_PF_DETERMINISTIC',
            compiler=c.h4.COMPILER,native=c.h4.NATIVE,N_cos=3,N_sin=5,C_cos=7,C_sin=11,
            G_saved=value,G_SE_saved=.125 if can['method']=='PR' else None,n=can['n'],paired=True,
            u=1e-9,reference_error_bound=None))
    return dict(schema=c.SERIES+'_INPUTS',physical=c.PHYSICAL,binding=c.BINDING,candidates=c.CANDIDATES,
        cohort_pooling=False,new_accounting=False,limitations=c.LIMITS,methods=methods)


class SavedH6(unittest.TestCase):
    def test_artificial_saved_projection(self):
        d=fixture();self.assertIs(c.validate(d),d)

    def test_H2_H4_DF_or_geometry_mixture(self):
        for key,value in [('system','H4'),('rank',10),('system_qubits',8),('T',1.6),('P',1),('bias','NONCANCELLATION')]:
            with self.subTest(key=key):
                d=deepcopy(fixture());d['physical'][key]=value
                with self.assertRaises(c.Stop):c.validate(d)

    def test_candidate_formula_prefix_q_or_cohort(self):
        for key,value in [('formula','4th(m5_best)'),('prefix',11),('q',2),('method','B0I')]:
            with self.subTest(key=key):
                d=deepcopy(fixture());d['methods'][0]['candidate'][key]=value
                with self.assertRaises(c.Stop):c.validate(d)
        d=deepcopy(fixture());d['methods'][2]['cohort']='CONFIRMATION'
        with self.assertRaises(c.Stop):c.validate(d)

    def test_units_shots_eligibility_missingness(self):
        for key,value in [('unit','T_count'),('N_cos',None),('N_sin',0),('C_cos',float('nan')),
                          ('status','UNDETERMINED'),('u',0.),('reference_error_bound',0.)]:
            with self.subTest(key=key):
                d=deepcopy(fixture());d['methods'][0][key]=value
                with self.assertRaises(c.Stop):c.validate(d)
        for key,value in [('n',32),('paired',False),('G_SE_saved',None)]:
            d=deepcopy(fixture());d['methods'][2][key]=value
            with self.assertRaises(c.Stop):c.validate(d)

    def test_scope_and_legacy_unknown_are_retained(self):
        d=deepcopy(fixture());d['limitations'].remove('H6_original_parent_exit_reap_unknown')
        with self.assertRaises(c.Stop):c.validate(d)
        d=deepcopy(fixture());d['new_accounting']=True
        with self.assertRaises(c.Stop):c.validate(d)

    def test_no_reaccounting_budget(self):
        self.assertEqual(sum(c.counts().values()),3)
        for key in ('allowance','accounting','u_aware_shots','cost_statistics','resource_row','metric_summary',
                    'compile','sampling','simulate','reference_recompute'):
            self.assertEqual(c.counts()[key],0)

    def test_approval_precedes_any_real_input_read(self):
        with patch.object(c,'verify',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(c.Stop,'EXPLICIT_FIXED_H6'):c.execute({}, {}, {})

    def test_old_H4_grant_rejected(self):
        cfg={};approval=dict(issuer='USER_EXPLICIT',execution_authorized=True,
            contract_sha256=c.digest(cfg),accepted_limitations=c.LIMITS)
        with patch.object(c,'verify',side_effect=AssertionError('must not enter')):
            with self.assertRaisesRegex(c.Stop,'UNUSED_H6_RANK'):
                c.authorize(cfg,approval,dict(purpose=c.h4.SERIES))

    def test_artificial_ranking_reuses_code_object(self):
        from h4_archived_exploratory_backend import Backend
        self.assertIs(c.rebound(Backend.rank).__code__,Backend.rank.__code__)
        d=fixture();out=c.rank_rows(d)
        self.assertTrue(out['full_Y4_cheaper']);self.assertTrue(out['PR_Y4_cheaper'])
        self.assertFalse(out['reversal_candidate']);self.assertIsNone(out['causal_attribution'])
        self.assertEqual(out['four_method_point_order'],[c.CANDIDATES[i]['id'] for i in (3,1,2,0)])

    def test_cli_rejects_before_reading_contract(self):
        r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/launch_h6_saved_four_method.py'),
            '--execute','--contract','NO_REAL_CONTRACT'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('EXPLICIT_FIXED_H6',r.stderr)


if __name__=='__main__':unittest.main()
