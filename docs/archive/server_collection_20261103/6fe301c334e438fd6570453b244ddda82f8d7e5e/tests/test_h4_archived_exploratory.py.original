"""Artificial fixtures only. Never read the H4 private manifest or real cost data."""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h4_archived_exploratory as c
from h4_archived_exploratory_backend import Backend, Existing, H2, rebound


def fixture():
    methods={}
    for i,x in enumerate(c.CANDIDATES):
        methods[x.id]=dict(candidate=asdict(x),binding=c.BINDING,unit=c.PHYSICAL['unit'],
            cohort='EXPLORATION' if x.method=='PR' else 'COMMON_PF_DETERMINISTIC',
            compiler=c.COMPILER,native=c.NATIVE,saved_reference=dict(real=.25,imag=.125),
            bias=dict(real=0.,imag=0.),u=1e-9,log_B_margin=1e-12,log_B=0.,
            allowance_recomputed=False,reference_error_bound=None,preparation_hash='a'*64,
            saved_shots=dict(real=None,imag=None),
            costs=[dict(task='SYNTHETIC_'+str(i)+'_'+str(j),cosine=10+i+j,sine=20+i+2*j,
                        receipt_sha256='b'*64) for j in range(x.n)])
    return dict(schema=c.SERIES+'_INPUTS',physical=c.PHYSICAL,binding=c.BINDING,
        candidates=c.candidates(),compiler=c.COMPILER,native=c.NATIVE,
        reference_policy='CONDITIONAL_SAVED_H4_REFERENCE_NOT_ZERO_ERROR_BOUND',
        adopted_commit=c.ADOPTED,saved_commit=c.SAVED,methods=methods)


class Metadata(unittest.TestCase):
    def test_complete_artificial_bundle(self):
        self.assertIs(c.validate_bundle(d:=fixture()),d)

    def test_wrong_scientific_target(self):
        for key,value in [('system','H2'),('rank',12),('rank_policy','screened'),('T',1.6),
                          ('P',1),('epsilon_signal',.005),('control','controlled')]:
            with self.subTest(key=key):
                d=deepcopy(fixture());d['physical'][key]=value
                with self.assertRaises(c.Stop):c.validate_bundle(d)

    def test_candidate_formula_q_prefix_or_law(self):
        for key,value in [('formula','4th(m5_best)'),('prefix',1),('q',4),('K',6),
                          ('r_policy','PER_PIECE'),('r_value',16),('method','FULL')]:
            with self.subTest(key=key):
                d=deepcopy(fixture());d['methods'][c.CANDIDATES[2].id]['candidate'][key]=value
                with self.assertRaises(c.Stop):c.validate_bundle(d)

    def test_binding_reference_allowance_unit_cohort(self):
        for key,value in [('binding',dict(c.BINDING,state='0'*64)),('unit','T_gates'),
                          ('cohort','CONFIRMATION'),('compiler','0'*64),('native','other'),
                          ('saved_reference',dict(real=.5,imag=.125)),('bias',dict(real=0.)),
                          ('u',0.),('u',float('nan')),('log_B_margin',None),('log_B',float('inf')),
                          ('reference_error_bound',0.),('allowance_recomputed',True)]:
            with self.subTest(key=key,value=value):
                d=deepcopy(fixture());d['methods'][c.CANDIDATES[2].id][key]=value
                with self.assertRaises((c.Stop,TypeError)):c.validate_bundle(d)

    def test_pairing_missing_axis_or_missing_cost(self):
        for kind in ('duplicate_task','missing_sine','missing_sample','zero_price','float_price','different_cohort'):
            with self.subTest(kind=kind):
                d=deepcopy(fixture());m=d['methods'][c.CANDIDATES[2].id];pairs=m['costs']
                if kind=='duplicate_task':pairs[1]['task']=pairs[0]['task']
                elif kind=='missing_sine':pairs[0].pop('sine')
                elif kind=='missing_sample':pairs.pop()
                elif kind=='zero_price':pairs[0]['cosine']=0
                elif kind=='float_price':pairs[0]['cosine']=10.
                else:m['cohort']='UNRELATED_OUTER32'
                with self.assertRaises(c.Stop):c.validate_bundle(d)

    def test_authorization_precedes_data_or_backend(self):
        with patch.object(c,'verify',side_effect=AssertionError('must not read real data')):
            with self.assertRaisesRegex(c.Stop,'SEPARATE_H4'):c.execute({}, {}, {})

    def test_H2_grant_cannot_authorize_H4(self):
        cfg=dict(run_root='unused');approval=dict(issuer='USER_EXPLICIT',execution_authorized=True,
            contract_sha256=c.digest(cfg),accepted_limitations=list(c.LIMITS))
        with patch.object(c,'verify',side_effect=AssertionError('must not enter')):
            with self.assertRaisesRegex(c.Stop,'NEW_H4'):c.authorize(cfg,approval,dict(purpose=c.h2.SERIES))

    def test_scientific_functions_are_original_code_objects(self):
        for original in (Existing.account,Existing.rank,H2.statistics):
            self.assertIs(rebound(original).__code__,original.__code__)
        self.assertIs(Backend.statistics.__code__,H2.statistics.__code__)
        self.assertEqual(c.counts()['empirical_allowance'],0)
        self.assertEqual(sum(c.counts().values()),39)

    def test_unauthorized_cli_without_contract_read(self):
        r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/launch_h4_archived_exploratory.py'),
            '--contract','NONEXISTENT_REAL_H4_CONTRACT','--execute'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('SEPARATE_EXPLICIT_H4',r.stderr)


class ArtificialKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.provider=Path(cls.tmp.name)/'provider'
        for path in c.h2.PROVIDER_PATHS:
            p=cls.provider/path;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(subprocess.check_output(['git','show',c.h2.PROVIDER+':'+path],cwd=ROOT))
        sys.path.insert(0,str(cls.provider/'src'))
        cls.shots=importlib.import_module('trottertracks.resource_applicability.ax2b_numerical_accounting')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.provider/'src'));cls.tmp.cleanup()

    def run_fixture(self,ineligible=False):
        data=deepcopy(fixture())
        for i,x in enumerate(c.CANDIDATES):
            d=data['methods'][x.id]
            if ineligible and i==2:d['bias']['real']=.1
            # Artificial input evaluation to establish the synthetic integer expectation only.
            q=self.shots.u_aware_shots({k:abs(v) for k,v in d['bias'].items()},dict(real=d['u'],imag=d['u']),
                epsilon=.0005,log_B_upper=d['log_B']+d['log_B_margin'],evidence_kind='EMPIRICAL',
                evidence_ref='SYNTHETIC_ONLY',alpha_axis=.025)
            d['saved_shots']={a:q['axes'][a]['shots'] for a in ('real','imag')}
        c.validate_bundle(data)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);s=c.Session.__new__(c.Session);s.cfg=dict(budget=c.counts())
            s.root=root;s.used=dict.fromkeys(c.KINDS,0);s.sequence=0;s.active=None
            for name in ('results','ledger','delegated_receipts'):(root/name).mkdir()
            s.take('bootstrap');backend=Backend(s)
            try:
                with patch.object(c,'saved_inputs',return_value=data):result=c.pipeline(s,backend)
                self.assertEqual(s.used,c.counts());self.assertEqual(len(list((root/'ledger').glob('*.json'))),39)
                for d in result['methods']:
                    self.assertEqual(d['series'],c.SERIES)
                    self.assertFalse(d['population_mean_certified'])
                    self.assertFalse(d['unobserved_events_assigned_zero'])
                return result
            finally:backend.close()

    def test_success_exact_budget_pairing_and_scope(self):
        result=self.run_fixture();self.assertEqual(len(result['methods']),4)
        for row,can in zip(result['methods'],c.CANDIDATES):
            self.assertEqual(len(row['saved_cost_sample_tasks']),can.n)
            self.assertIsNone(row['nominal_interval'])
        self.assertIsNone(result['delegated_comparison']['causal_attribution'])
        self.assertEqual(len(result['delegated_comparison']['four_method_point_order']),4)

    def test_ineligible_remains_null_no_ranking(self):
        result=self.run_fixture(True);self.assertFalse(result['methods'][2]['eligible'])
        self.assertIsNone(result['methods'][2]['G_total']);self.assertIsNone(result['methods'][2]['G_axis'])
        self.assertIsNone(result['delegated_comparison']['four_method_point_order'])


if __name__=='__main__':unittest.main()
