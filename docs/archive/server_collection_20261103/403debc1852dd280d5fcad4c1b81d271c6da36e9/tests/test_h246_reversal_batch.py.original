"""Artificial fixtures only; never load molecular numeric input in tests."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib'),str(ROOT/'src')]
import h246_reversal_batch as c
import h246_reversal_batch_backend as b
# The current branch does not vendor this old track kernel. Use its pinned code
# as an artificial fixture, without touching the shared provider worktree.
_provider_fixture=tempfile.TemporaryDirectory()
for _rel in c.PROVIDER_PATHS:
    _path=Path(_provider_fixture.name)/_rel
    _path.parent.mkdir(parents=True,exist_ok=True)
    _path.write_bytes(c.prior.old.recovery.git_bytes(c.PROVIDER,_rel))
sys.path.insert(0,str(Path(_provider_fixture.name)/'src'))
from trottertracks.resource_applicability import ax2b_numerical_accounting as shots


def fixture():
    d=dict(schema=c.SERIES+'_INPUTS',kind='ARTIFICIAL',physical=copy.deepcopy(c.PHYSICAL),
        A_targets=copy.deepcopy(c.A_TARGETS),B_targets=copy.deepcopy(c.B_TARGETS),B_epsilons=c.EPS[:],
        limitations=c.LIMITS[:],bindings={s:dict(system=s,rank=r,artificial=True) for s,r in [('H2',3),('H4',10),('H6',19)]},
        A_rows=[],B_bases=[],quoted_main_comparisons=[],policy_bindings={},account_bindings={},
        cohort_pooling=False,recompute_main_0005=False)
    for scope,targets,name in [('A_SAME_Q',c.A_TARGETS,'A_rows'),('B_H2_PRECISION',c.B_TARGETS,'B_bases')]:
        for t in targets:
            cost=2 if t['formula']=='2nd' else 3
            cohort='COMMON_PF_DETERMINISTIC' if t['method']=='FULL' else 'EXPLORATION'
            a=dict(diagnostic=dict(bias=dict(real=1e-6,imag=2e-6),primary=dict(log_B=0.)),allowance=dict(real=0.,imag=0.,log_B_margin=0.))
            policy=dict(reference='fake_reference_'+t['system'],u=0.,K=4,formula=t['formula'])
            row=dict(candidate=t['id'],system=t['system'],target=copy.deepcopy(t),epsilon=.001 if scope=='A_SAME_Q' else .0005,
                binding=copy.deepcopy(d['bindings'][t['system']]),subscope=scope,policy_binding=policy,
                cohort=cohort,unit=c.prior.UNIT,bias_mode='USUAL_SIGNED',status='ELIGIBLE',
                C=dict(cosine=cost,sine=cost),N=dict(cosine=10,sine=20),G_saved=30*cost,
                pairs=[dict(pair_id='fake_'+str(i),candidate=t['id'],cohort=cohort,axes=dict(cosine=cost,sine=cost)) for i in range(t['n'])],
                uncertainty=dict(G_SE=None if t['method']=='FULL' else 1,saved_axis_statistics='FAKE_LITERAL_SAVED_UNIT_STATS'),source=dict(kind='FAKE'))
            if scope=='B_H2_PRECISION':row['account_inputs']=a;d['account_bindings'][t['id']]=copy.deepcopy(a)
            d[name].append(row);d['policy_bindings'][t['id']]=copy.deepcopy(policy)
    for system in ('H2','H4','H6'):
        for ep in ([.0005] if system=='H2' else c.prior.EPS):
            for method in ('FULL','PR'):
                d['quoted_main_comparisons'].append(dict(system=system,epsilon=ep,method=method,direction='S2_LOWER',
                    S2='fake_S2',Yoshida4='fake_Y4',G_saved_unchanged=['fake_1','fake_2'],R_G='FAKE_LITERAL'))
    return d


class FakeSession:
    def __init__(self,root):
        self.root=Path(root);(self.root/'results').mkdir();self.used=dict.fromkeys(c.KINDS,0);self.used.update(bootstrap=1,bundle_load=1)
    def call(self,kind,fn,*a,**kw):
        c.require(kind in self.used and self.used[kind]<c.counts()[kind],'FAKE_CALL_BUDGET')
        self.used[kind]+=1;return fn(*a,**kw)


class Batch(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.data=fixture()

    def test_exact_scope_and_83_events(self):
        self.assertIs(c.validate(self.data,artificial=True),self.data)
        self.assertEqual(sum(c.counts().values()),83)
        for k in ('sampling','circuit_build','compile','transpile','simulation','signal','reference_recompute','allowance','new_statistics','ranking'):
            self.assertEqual(c.counts()[k],0)

    def test_system_rank_formula_q_prefix_and_subscope_rejected(self):
        for name in ('A_rows','B_bases'):
            for key,value in [('system','OTHER'),('subscope','WRONG'),('candidate','m5_best')]:
                d=copy.deepcopy(self.data);d[name][0][key]=value
                with self.assertRaises(c.Stop):c.validate(d,artificial=True)
            for key,value in [('rank',99),('formula','4th(m5_best)'),('q',16),('prefix',0)]:
                d=copy.deepcopy(self.data);d[name][0]['target'][key]=value
                with self.assertRaises(c.Stop):c.validate(d,artificial=True)

    def test_cohort_reference_allowance_and_epsilon_rejected(self):
        for name in ('A_rows','B_bases'):
            for field,value in [('cohort','CONFIRMATION'),('epsilon',.002),('policy_binding',dict(reference='wrong',u=7)),('bias_mode','ABSOLUTE')]:
                d=copy.deepcopy(self.data);d[name][0][field]=value
                with self.assertRaises(c.Stop):c.validate(d,artificial=True)
        d=copy.deepcopy(self.data);d['B_bases'][0]['account_inputs']['allowance']['real']=1
        with self.assertRaises(c.Stop):c.validate(d,artificial=True)

    def test_pair_breakage_duplicate_and_missing_zero_not_accepted(self):
        for mutation in ('candidate','axis','duplicate','zero_N','missing_G'):
            d=copy.deepcopy(self.data);r=d['A_rows'][4]
            if mutation=='candidate':r['pairs'][0]['candidate']='wrong'
            elif mutation=='axis':r['pairs'][0]['axes']={'cosine':2}
            elif mutation=='duplicate':r['pairs'][1]['pair_id']=r['pairs'][0]['pair_id']
            elif mutation=='zero_N':r['N']['sine']=0
            else:r['G_saved']=None
            with self.assertRaises(c.Stop):c.validate(d,artificial=True)

    def test_only_epsilon_and_output_suffix_adapted(self):
        original=b.Original.account
        for ep in c.EPS:
            adapted=b.adapted_account(ep)
            self.assertEqual(original.__code__.co_code,adapted.__code__.co_code)
            self.assertIs(original.__globals__,adapted.__globals__)
            diff=[(a,z) for a,z in zip(original.__code__.co_consts,adapted.__code__.co_consts) if a!=z]
            self.assertEqual(diff,[(.0005,ep),('_accounting.json','_eps_'+ep.hex()+'_accounting.json')])
        with self.assertRaises(c.Stop):b.adapted_account(.0005)

    def test_full_synthetic_pipeline_actual_kernel_exact_budget(self):
        session=FakeSession(self.tmp.name);old=copy.deepcopy(self.data['quoted_main_comparisons']);headroom=shots.axis_headroom
        result=b.pipeline(session,self.data,artificial=True,shot_module=shots)
        self.assertEqual(session.used,c.counts());self.assertIs(shots.axis_headroom,headroom)
        self.assertEqual(len(result['same_q_comparisons']),8);self.assertEqual(len(result['H2_new_endpoints']),16)
        self.assertEqual(len(result['integration']['main_15_system_precision_cells']),15)
        self.assertEqual(len(result['integration']['same_q_4_system_q_cells']),4)
        self.assertEqual(len(list((session.root/'results').glob('*_accounting.json'))),16)
        self.assertEqual(self.data['quoted_main_comparisons'],old)
        self.assertFalse(result['old_main_comparisons_recomputed'])
        for r in result['H2_new_endpoints']:
            self.assertIsNone(r['uncertainty']['G_SE']);self.assertEqual(r['uncertainty']['saved_axis_statistics'],'FAKE_LITERAL_SAVED_UNIT_STATS')

    def test_ineligible_endpoint_never_zero_cost(self):
        session=FakeSession(self.tmp.name);row=self.data['B_bases'][0];row['account_inputs']['diagnostic']['bias']['real']=1.
        r=b.account(session,row,.001,b.old.Permit(True),shots.u_aware_shots)
        self.assertEqual(r['status'],'INELIGIBLE');self.assertIsNone(r['G_saved']);self.assertIsNone(r['N']['cosine'])
        other=copy.deepcopy(r);other['target']['formula']='4th'
        self.assertIsNone(b.old.decompose(r,other,b.old.Permit(True))['new_ratios'])

    def test_real_values_rejected_by_artificial_permit(self):
        row=copy.deepcopy(self.data['B_bases'][0]);row['binding']['artificial']=False
        with self.assertRaises(c.Stop):b.account(FakeSession(self.tmp.name),row,.001,b.old.Permit(True),shots.u_aware_shots)
        self.data['kind']='SAVED_REAL_VALUES'
        with self.assertRaises(c.Stop):b.pipeline(SimpleNamespace(),self.data,artificial=True)

    def test_no_old_endpoint_reaccount_and_no_forbidden_call(self):
        session=FakeSession(self.tmp.name)
        with self.assertRaises(c.Stop):b.account(session,self.data['B_bases'][0],.0005,b.old.Permit(True),shots.u_aware_shots)
        with self.assertRaises(c.Stop):session.call('compile',lambda:None)
        self.data['recompute_main_0005']=True
        with self.assertRaises(c.Stop):c.validate(self.data,artificial=True)

    def test_approval_rejection_before_real_input_reads(self):
        with patch.object(c,'verify',side_effect=AssertionError('Must not read inputs')):
            with self.assertRaises(c.Stop):c.authorize({},dict(execution_authorized=False),{})
        with self.assertRaises(c.Stop):b.pipeline(SimpleNamespace(),self.data,artificial=False)

    def test_cli_rejects_unauthorized_execute_before_contract_read(self):
        r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/launch_h246_reversal_batch.py'),
            '--execute','--contract','/nonexistent_batch_contract'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('EXPLICIT_FIXED_BATCH_APPROVAL_REQUIRED',r.stderr)

    def test_weak_rank_and_independent_limits_retained(self):
        session=FakeSession(self.tmp.name);result=b.pipeline(session,self.data,artificial=True,shot_module=shots)
        weak=next(x for x in result['integration']['main_15_system_precision_cells'] if x['system']=='H4' and x['epsilon']==.005)
        self.assertTrue(weak['weak_H4_005']);self.assertFalse(weak['independent_standard_Y4_at_this_endpoint'])
        self.assertFalse(result['integration']['cohort_pooling']);self.assertFalse(result['integration']['global_candidate_ranking'])


if __name__=='__main__':unittest.main()
