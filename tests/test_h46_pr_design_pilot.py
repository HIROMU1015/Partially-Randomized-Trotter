"""Artificial metadata/statistics/process/native tests, never H4/H6 sampling."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_pr_design_pilot as d
import h46_pr_pilot_contract as c
import h46_pr_pilot_native as native
import run_h46_pr_design_pilot as launch

def toy_candidate(formula='2nd',q=1):
    return dict(id='ARTIFICIAL_ONLY',method='PR',formula=formula,q=q,prefix=1,system='H4',
                role='PRIMARY',kind='PR_SIGNAL',K=4,r_policy='ABS_TIME_PER_OUTER',r_value=32)

def toy_task(name='TOY',formula='2nd',q=1):
    return dict(id=name,candidate=toy_candidate(formula,q),sample=0,axes=['cosine','sine'],stage=d.STAGES[0])

def artificial_rows():
    return [dict(trajectory_id=str(i),candidate_id='TOY',campaign_id='ARTIFICIAL',cohort=d.COHORT,
                 conditional=False,paired=True,stratum='S0',cosine_RZ=x,sine_RZ=2*x)
            for i,x in enumerate((10,12,14))]

def fake_worker(task):
    cfg=launch.WORKER_CFG;stop=launch.WORKER_STOP
    if cfg.get('failure')==task['id']:stop.set();raise RuntimeError('SYNTHETIC_STOP')
    if stop.is_set():raise RuntimeError('OWN_STAGE_STOP')
    for kind,n in c.task_limits(task).items():
        for i in range(n):c.reserve(cfg,task,kind,i)
    out=Path(cfg['output'])/task['id'];out.mkdir()
    for i in range(c.task_limits(task)['request_sampling_calls']):c.save(out/f'piece_law_{i}.json',{'synthetic':True})
    c.save(out/'result.json',dict(status='UNIT_COST_PASS_NOT_ACCOUNTING',task=task,SOURCE=cfg['SOURCE'],
      plan_sha256=cfg['plan_sha256'],stage=cfg['stage'],axis_costs=[dict(axis=a,compiler_settings_hash=d.COMPILER_HASH,
      native_policy=d.NATIVE_POLICY) for a in ('cosine','sine')],shot_accounting=False,cost_ranking=False,
      conditional_sampling=False,cohort_pooling=False,seed_cohort=d.COHORT,events=dict(N2=0,N4=0,stratum='S0')))
    return dict(id=task['id'],result_sha256=c.sha(out/'result.json'),peak_RSS_KiB=1)

class DesignTests(unittest.TestCase):
    def test_fixed_domain_and_pending_adoption(self):
        p=d.proposal();self.assertEqual(len(p['groups']),24)
        self.assertEqual([o['n_P_proposed'] for o in p['options']],[9,33])
        self.assertTrue(all(r['n_P_adopted'] is None and r['n_E'] is None for r in p['groups']))
        self.assertFalse(p['scientific_execution_authorized'])

    def test_budgets_derived_not_guessed(self):
        p=d.proposal();b=p['options'][1]['system_call_scope_proposed']['H6']
        self.assertEqual(b['trajectories'],396);self.assertEqual(b['compile_axes'],792)
        self.assertEqual(b['request_sampling_calls'],7425);self.assertEqual(b['rte_events'],47520)
        self.assertEqual(b['shot_calculation'],0);self.assertEqual(b['cost_accounting'],0)

    def test_sample_counts_must_be_adoptable_exact_domain(self):
        rows=[r for r in d.groups() if r['system']=='H4'];counts={r['id']:2 for r in rows}
        self.assertEqual(len(d.tasks_for(d.STAGES[0],counts)),24)
        for value in (None,0,1,True):
            bad=dict(counts);bad[rows[0]['id']]=value
            with self.assertRaises(ValueError):d.tasks_for(d.STAGES[0],bad)
        with self.assertRaises(ValueError):d.tasks_for('H4_PR',counts)

    def test_nominal_precision_formula_and_invalid_targets(self):
        self.assertEqual(d.nominal_variance_sample_size(.25),33)
        for target in (0,1,-.5):
            with self.assertRaises(ValueError):d.nominal_variance_sample_size(target)

    def test_seed_namespaces_and_no_collision_retry(self):
        tasks=[toy_task()]
        regs=[d.registry(tasks,'a'*40,'SYNTHETIC',cohort,[]) for cohort in ('DESIGN_PILOT','EXPLORATION','CONFIRMATION')]
        self.assertEqual(len({json.dumps(r) for r in regs}),3)
        first=next(iter(regs[0]['TOY'].values()))
        with self.assertRaisesRegex(ValueError,'COLLISION'):d.registry(tasks,'a'*40,'SYNTHETIC',d.COHORT,[first])
        with self.assertRaises(ValueError):d.registry(tasks,'a'*40,'SYNTHETIC','OLD_PILOT',[])

    def test_m5_registry_records_all_pieces(self):
        reg=d.registry([toy_task(formula='4th(m5_best)',q=8)],'a'*40,'TOY',d.COHORT,[])
        self.assertEqual(len(reg['TOY']),88)

    def test_strata_priority(self):
        for orders,label in [([0],'S0'),([2],'S2_single'),([2,2],'S2_multi'),([2,4],'S4')]:
            self.assertEqual(d.event_stratum([[{'taylor_order':v} for v in orders]])['stratum'],label)
        with self.assertRaises(ValueError):d.event_stratum([[{'taylor_order':6}]])

class StatisticsTests(unittest.TestCase):
    def describe(self,rows=None):
        return d.describe_pilot(artificial_rows() if rows is None else rows,3,relative_MCSE_targets=[.1],variance_inflations=[1,4])

    def test_axis_covariance_and_paired_SE(self):
        s=self.describe();self.assertEqual(s['covariance'],[[4,8],[8,16]])
        self.assertEqual(s['pair_unit_RZ_variance'],36)
        self.assertAlmostEqual(s['pair_unit_RZ_MCSE'],math.sqrt(12))
        self.assertEqual([p['n_E_nominal_only'] for p in s['n_E_sensitivity']],[3,12])
        self.assertFalse(s['automatic_n_E_adoption'])

    def test_negative_covariance_is_not_dropped(self):
        rows=artificial_rows()
        for r,y in zip(rows,(14,12,10)):r['sine_RZ']=y
        s=self.describe(rows);self.assertEqual(s['covariance'][0][1],-4)
        self.assertEqual(s['pair_unit_RZ_variance'],0)

    def test_unobserved_strata_null_not_zero_cost(self):
        s=self.describe();self.assertEqual(s['strata']['S4']['observed_count'],0)
        self.assertIsNone(s['strata']['S4']['observed_conditional_pair_mean'])
        self.assertIsNone(s['strata']['S4']['population_tail_cost_bound'])
        self.assertTrue(s['rare_tail_unresolved'])

    def test_zero_observed_variance_does_not_set_n_E(self):
        rows=artificial_rows()
        for row in rows:row.update(cosine_RZ=12,sine_RZ=24)
        s=self.describe(rows);self.assertEqual(s['covariance'][0][0],0)
        self.assertTrue(all(p['n_E_nominal_only'] is None for p in s['n_E_sensitivity']))

    def test_incomplete_or_duplicate_pilot_rejected(self):
        with self.assertRaises(ValueError):self.describe(artificial_rows()[:2])
        rows=artificial_rows();rows[1]['trajectory_id']='0'
        with self.assertRaises(ValueError):self.describe(rows)

    def test_exploration_conditioning_or_other_candidate_rejected(self):
        for key,value in [('cohort','EXPLORATION'),('conditional',True),('candidate_id','OTHER'),('paired',False)]:
            rows=artificial_rows();rows[0][key]=value
            with self.assertRaises(ValueError):self.describe(rows)

class ContractTests(unittest.TestCase):
    def test_disabled_config_refused_before_inputs(self):
        with patch.object(c,'verify',side_effect=AssertionError('must not read inputs')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_PILOT'):c.authorize(dict(authorized=False,grant={'valid':False}))

    def test_foreign_stage_or_unadopted_decision_refused(self):
        with self.assertRaisesRegex(ValueError,'PILOT_ONLY'):c.authorize(dict(authorized=True,grant={'valid':True},stage='H4_PR',cohort=d.COHORT))
        with self.assertRaisesRegex(ValueError,'DECISIONS_NOT_ADOPTED'):c.authorize(dict(authorized=True,grant={'valid':True},stage=d.STAGES[0],cohort=d.COHORT,decisions={}))
        dec={k:True for k in ('pilot_design_adopted','nominal_variance_precision_limit_accepted','rare_event_limit_accepted','resources_policy_accepted','unit_cost_only_accepted')}
        dec.update(formal_shot_policy_adopted=False,AD03='DEFERRED',adaptive_top_up=False,pooling=False,
                   n_P_by_candidate={r['id']:2 for r in d.groups() if r['system']=='H4'})
        with self.assertRaisesRegex(ValueError,'FULL_PILOT_SCRATCH'):
            c.authorize(dict(authorized=True,grant={'valid':True},stage=d.STAGES[0],cohort=d.COHORT,decisions=dec,
                             resources={'proxy_policy_adopted':True,'full_output_capacity_adopted':False}))

    def test_two_CLI_paths_no_science_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft.json';c.save(p,dict(authorized=False,grant={'valid':False}))
            for mode in ('--execute','--supervise'):
                r=subprocess.run([sys.executable,'-B',str(c.ROOT/'scripts/run_h46_pr_design_pilot.py'),mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertIn('EXPLICIT_PILOT',r.stdout)
            self.assertEqual([p.name for p in Path(tmp).iterdir()],['draft.json'])

    def test_reserved_calls_are_nonrefundable(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=toy_task();cfg=dict(tasks=[t],SOURCE='a'*40,stage=t['stage'],ledger=tmp,marker=str(Path(tmp)/'CONSUMED.json'),grant={})
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            c.reserve(cfg,t,'compile_axes',0)
            with self.assertRaises(FileExistsError):c.reserve(cfg,t,'compile_axes',0)
            with self.assertRaises(ValueError):c.reserve(cfg,t,'compile_axes',2)

    def test_numeric_boundaries_and_no_phase_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):native.gate(tmp,'at_boundary',1e-7,1e-7)
            self.assertFalse(c.read(Path(tmp)/'at_boundary.json')['phase_alignment'])

    def test_isolated_pool_and_one_shot_marker(self):
        for failure in (None,'A',None):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'run';tasks=[toy_task(n) for n in ('A','B')]
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage=d.STAGES[0],tasks=tasks,run_root=str(root),
                         output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),
                         grant={},runtime={'python':sys.executable},failure=failure)
                with patch.object(c,'authorize'),patch.object(c,'verify'),patch.object(launch,'fresh',return_value={'workers':2}):
                    self.assertEqual(launch.execute(cfg,target=fake_worker),0 if failure is None else 1)
                    report=c.read(root/'completion_witness.json')
                    self.assertTrue(all(w['exit']==0 and w['reaped'] for w in report['workers'].values()))
                    self.assertFalse(report['next_phase_started'])
                    with self.assertRaises(ValueError):launch.execute(cfg,target=fake_worker)

@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'explicit artificial provider required')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))

    def test_three_formulas_preserve_law_and_cohort(self):
        from test_h46_m5_cost_readiness import NativeToyTests
        import h46_m5_native as original
        for formula in ('2nd','4th','4th(m5_best)'):
            with self.subTest(formula=formula),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'output').mkdir();(root/'ledger').mkdir()
                ctx=NativeToyTests.toy(self);task=toy_task(formula=formula);r=task['candidate'];group=dict(r,policy=r['r_policy'])
                sig=original.signal(ctx,group)
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage=d.STAGES[0],tasks=[task],output=str(root/'output'),
                  ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},systems={'H4':{'binding':{'synthetic':True}}},registry_path=str(root/'seeds.json'))
                c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
                c.save(cfg['registry_path'],d.registry([task],cfg['SOURCE'],'ARTIFICIAL',d.COHORT,[]))
                read=c.read
                def read_override(path):
                    if str(path).endswith('/signal_records.json'):return dict(records=[dict(candidate=r,signal=sig)])
                    return read(path)
                with patch.object(c,'authorize'),patch.object(c,'verify'),patch.object(native,'context',return_value=ctx),patch.object(c,'read',side_effect=read_override):
                    native.run_task(cfg,task,threading.Event())
                out=root/'output'/task['id'];report=read(out/'result.json');events=read(out/'events_PRIVATE.json')
                self.assertEqual(report['seed_cohort'],d.COHORT);self.assertEqual(events['cohort'],d.COHORT)
                self.assertEqual(sum(len(piece) for piece in events['events']),32)
                self.assertEqual(len(list(out.glob('piece_law_*.json'))),{'2nd':1,'4th':3,'4th(m5_best)':11}[formula])
                self.assertEqual([r['axis'] for r in report['axis_costs']],['cosine','sine'])
                self.assertFalse(report['shot_accounting'])

if __name__=='__main__':unittest.main()
