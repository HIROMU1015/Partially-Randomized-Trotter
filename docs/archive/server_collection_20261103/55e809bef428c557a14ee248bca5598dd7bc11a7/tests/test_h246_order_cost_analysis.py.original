"""Synthetic values only; fixed metadata must never authorize molecular arithmetic."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import h246_order_cost_analysis as c
import h246_order_cost_analysis_backend as b


def fixture(event_path):
    bindings={s:dict(system=s,rank=r,artificial=True) for s,r in [('H2',3),('H4',10),('H6',19)]}
    rows=[];availability=[];structures=[]
    for t in c.TARGETS:
        s=t['system'];cid=t['id'];full=t['method']=='FULL'
        m=dict(target=t,binding=bindings[s],saved_schedule=None,saved_strata=None,saved_cost_statistics=None)
        if not full:m.update(representative_event_file=str(event_path),representative_event_digest='synthetic')
        structures.append(m)
        for ep in c.EPS:
            available=s!='H2' or ep==.0005
            availability.append(dict(candidate=cid,epsilon=ep,available=available))
            if not available:continue
            # An independent synthetic N fixture for the reused continuous-shot identity.
            h=ep/math.sqrt(2);bound=2/h**2*math.log(80);n=math.ceil(bound)
            cost=30 if t['formula']=='2nd' else 20
            cohort='COMMON_PF_DETERMINISTIC' if full else 'EXPLORATION'
            pairs=[dict(pair_id='fake_'+str(i),candidate=cid,cohort=cohort,axes=dict(cosine=cost,sine=cost)) for i in range(t['n'])]
            rows.append(dict(candidate=cid,target=t,system=s,epsilon=ep,binding=bindings[s],unit=c.UNIT,
                cohort=cohort,bias_mode='USUAL_SIGNED',status='ELIGIBLE',N=dict(cosine=n,sine=n),
                C=dict(cosine=cost,sine=cost),G_saved=2*n*cost,pairs=pairs,
                uncertainty=dict(G_SE=None if full else 1,interpretation='SAVED_WITHIN_TRAJECTORY_COST_VARIABILITY'),
                explanation=dict(log_B_base=0.,log_B_margin=0.,log_B_for_shots_saved=None,
                    signed_bias=dict(cosine=0.,sine=0.),allowance=dict(cosine=0.,sine=0.),
                    headroom=dict(cosine=h,sine=h),log_shot_bound={},integer_status={},components={})))
    return dict(schema=c.SERIES+'_INPUTS',kind='ARTIFICIAL',physical=c.PHYSICAL,targets=c.TARGETS,
                epsilons=c.EPS,limitations=c.LIMITS,cohort_pooling=False,bindings=bindings,
                rows=rows,availability=availability,structures=structures)


class Analysis(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.event=Path(self.temp.name)/'events.json'
        self.event.write_text(json.dumps(dict(event_digest='synthetic',conditional=False,cohort='EXPLORATION',
            events=[[dict(taylor_order=0,phase=dict(real=1),application_sequence=[
                dict(is_identity=False,basis_hash='fake'),dict(is_identity=False,basis_hash='fake')])]])))
        self.data=fixture(self.event)

    def test_complete_artificial_metadata(self):
        self.assertIs(c.validate(self.data,artificial=True),self.data)
        self.assertEqual(sum(c.counts().values()),86)

    def test_scope_candidate_formula_rank_q_prefix_and_cohort_rejected(self):
        changes=[('system','H6'),('cohort','CONFIRMATION'),('unit','raw_gate'),('epsilon',.002),
                 ('bias_mode','NONCANCELLATION'),('candidate','4th(m5_best)')]
        for k,v in changes:
            with self.subTest(k=k):
                d=copy.deepcopy(self.data);d['rows'][0][k]=v
                with self.assertRaises(c.Stop):c.validate(d,artificial=True)
        for k,v in [('rank',10),('q',16),('prefix',0),('formula','4th(m5_best)')]:
            with self.subTest(k=k):
                d=copy.deepcopy(self.data);d['rows'][0]['target']=dict(d['rows'][0]['target'],**{k:v})
                with self.assertRaises(c.Stop):c.validate(d,artificial=True)

    def test_epsilon_pairing_missing_price_and_uncertainty_rejected(self):
        for change in ('shots_zero','missing_G','pairing','full_SE','quantum_SE','false_availability'):
            with self.subTest(change=change):
                d=copy.deepcopy(self.data);r=d['rows'][0]
                if change=='shots_zero':r['N']['cosine']=0
                elif change=='missing_G':r['G_saved']=None
                elif change=='pairing':r['pairs'][0]['candidate']='other'
                elif change=='full_SE':r['uncertainty']['G_SE']=3
                elif change=='quantum_SE':r['uncertainty']['interpretation']='SHOT_NOISE'
                else:d['availability'][0]['available']=True
                with self.assertRaises(c.Stop):c.validate(d,artificial=True)

    def test_general_axis_additive_cross_and_multiplicative_closure(self):
        a=copy.deepcopy(self.data['rows'][0]);z=copy.deepcopy(a)
        z['candidate']='synthetic_other';z['target']=dict(a['target'],formula='4th')
        for r,prices,ns,g in [(a,(2,6),(3,4),30),(z,(3,2),(6,3),24)]:
            r['pairs']=[dict(axes=dict(zip(('cosine','sine'),prices)))]
            r['C']=dict(zip(('cosine','sine'),prices));r['N']=dict(zip(('cosine','sine'),ns));r['G_saved']=g
        out=b.decompose(a,z,b.Permit(True))
        self.assertEqual(out['R_G'],dict(numerator=4,denominator=5))
        self.assertEqual(out['additive']['C_at_S2_N'],dict(numerator=-13,denominator=1))
        self.assertEqual(out['additive']['N_at_S2_C'],dict(numerator=0,denominator=1))
        self.assertEqual(out['additive']['cross'],dict(numerator=7,denominator=1))
        self.assertIsNone(out['difference_SE'])

    def test_ineligible_is_missing_not_zero(self):
        a=copy.deepcopy(self.data['rows'][0]);z=copy.deepcopy(a);z['target']=dict(a['target'],formula='4th')
        z['status']='INELIGIBLE';z['N']['cosine']=None;z['G_saved']=None
        self.assertIsNone(b.decompose(a,z,b.Permit(True))['new_ratios'])

    def test_reused_shot_function_and_unresolved_headroom(self):
        row=self.data['rows'][0]
        self.assertEqual(b.explanation(row,b.Permit(True))['relationship']['status'],'EXPLANATION_COMPLETE')
        row['explanation']['headroom']['cosine']=1
        self.assertEqual(b.explanation(row,b.Permit(True))['relationship']['status'],'UNRESOLVED')

    def test_real_input_and_unauthorized_grant_rejected_before_reads(self):
        row=copy.deepcopy(self.data['rows'][0]);row['binding']['artificial']=False
        with self.assertRaises(c.Stop):b.exact_row(row,b.Permit(True))
        with patch.object(c,'verify',side_effect=AssertionError('Must not read real inputs')):
            with self.assertRaises(c.Stop):c.authorize({},dict(execution_authorized=False),{})
        with self.assertRaises(c.Stop):b.Permit(False)

    def test_cli_rejection_before_contract_read(self):
        r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/launch_h246_order_cost_analysis.py'),
                          '--execute','--contract','/nonexistent_h246_contract'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('EXPLICIT_H246_ANALYSIS_APPROVAL_REQUIRED',r.stderr)

    def test_artificial_pipeline_exact_budget_and_representative_scope(self):
        used=dict.fromkeys(c.KINDS,0)
        def call(kind,fn,*args):used[kind]+=1;return fn(*args)
        fake=SimpleNamespace(call=call)
        used.update(bootstrap=1,bundle_load=1)
        schedule=lambda *args:[dict(outer=0,piece=0,signed_time=.125,rte_steps=1)]
        result=b.pipeline(fake,self.data,{},artificial=True,schedule_fn=schedule)
        self.assertEqual(used,c.counts());self.assertEqual(len(result['comparisons']),22)
        self.assertEqual(len(result['shot_relationships']),44)
        pr=next(x for x in result['circuit_structure'] if 'representative_applications' in x)
        self.assertEqual(pr['representative_same_basis_adjacencies'],1)
        self.assertFalse(pr['exhaustive_parameter_stream_identity'])

    def test_no_forbidden_new_numerics_or_old_grant_scope(self):
        for kind in ('sampling','compile','simulation','u_aware_shots','allowance','new_statistics','ranking'):
            self.assertEqual(c.counts()[kind],0)
        with self.assertRaises(c.Stop):b.pipeline(SimpleNamespace(),self.data,{},artificial=False)


if __name__=='__main__':unittest.main()
