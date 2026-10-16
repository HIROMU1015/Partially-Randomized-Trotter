"""Artificial metadata only: guarded reuse, exact selection and scope checks."""
from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import track_a_v1_reaccount as m


class Writer:
    def __init__(self):self.values={}
    def write(self,name,value):
        if name in self.values:raise AssertionError('duplicate output')
        self.values[name]=value


def record(cell,missing_parts=False):
    d=dict(total_signed=dict(real=0.,imag=0.))
    if not missing_parts:d['signed_parts']={'PF':dict(real=0.,imag=0.)}
    return dict(cell=cell,error_decomposition=d,log_B=0.,
        allowance=dict(real=1e-12,imag=1e-12,log_B_margin=0.,kind='EMPIRICAL',certified=False))


def fixture():
    reg=m.c.registrations();old=[record(x,x['method']=='B0I') for x in reg['existing']]
    additional=[record(x) for x in reg['additional']]
    for rec in additional:
        rec['precision_rows']={str(e):{str(f):m.r.accounting.accounting(rec,e,f) for f in m.c.FACTORS} for e in m.c.EPSILONS}
    confirmed_ids={x['id'] for x in reg['existing'] if x['method']=='B2' and x['prefix']==5 and x['r']==4}
    def wrappers(rec,count,phase):
        return [dict(cell_id=rec['cell']['id'],phase=phase,replica=j,seed=j,control='symmetric_directional',
            axis=axis,event_digest='event'+str(j),compiler_hash=m.COMPILER,RZ=10+rec['cell']['prefix']+j)
            for j in range(count) for axis in m.c.AXES]
    exploration=[x for rec in old if rec['cell']['method']!='B0I' for x in wrappers(rec,8 if rec['cell']['method']=='B2' else 1,'exploration')]
    confirmation=[x for rec in old if rec['cell']['id'] in confirmed_ids for x in wrappers(rec,32,'confirmation')]
    old_tables={'H6_EXPLORATION92':[],'H6_CONFIRMATION_UNION8':[]};b0i=[];comparisons=[]
    for e in m.c.EPSILONS:
        for f in m.c.FACTORS:
            det=[];pr=[]
            for rec in old:
                cell=rec['cell'];method=cell['method'];a=m.r.accounting.accounting(rec,e,f)
                value=Fraction(10+cell['prefix'])+(Fraction(7,2) if method=='B2' else 0)
                def saved(v):
                    g=a['N_total']*v
                    return dict(cell_id=cell['id'],method=method,cell=cell,epsilon=e,factor=str(f),accounting=a,
                        metrics={'RZ':dict(G_point=float(g),G_exact=m.r.accounting.rational(g))})
                row=saved(value)
                (b0i if method=='B0I' else old_tables['H6_EXPLORATION92']).append(row)
                if method!='B2':
                    old_tables['H6_CONFIRMATION_UNION8'].append(row);det.append((a['N_total']*value,cell['id']))
                elif cell['id'] in confirmed_ids:
                    row=saved(Fraction(10+cell['prefix'])+Fraction(31,2));old_tables['H6_CONFIRMATION_UNION8'].append(row)
                    pr.append((Fraction(row['metrics']['RZ']['G_point']),cell['id']))
            d=min(det);p=min(pr)
            comparisons.append(dict(epsilon=e,factor=f,metric='RZ',
                best_registered_deterministic=dict(cell_id=d[1],G_point=float(d[0])),
                confirmed_PR_union8=dict(cell_id=p[1],G_point=float(p[0]))))
    axes=m.c.validation_axes()
    rows=[dict(cell_id=a['cell']['id'],axis=a['axis'],status='MEASURED_COMPLETE',
        identity_verified=True,predicted=100,measured=100,model_sha256=m.plan()['frozen_model_sha256']) for a in axes]
    frozen=dict(fitted_after_validation=False,validation_axes=axes,models=[
        dict(stratum=dict(method=method,axis=axis,formula='2nd'),coefficients=['7','3','19'])
        for method in ('B0','B0I') for axis in m.c.AXES])
    b0i_axes=[dict(prefix=str(rec['cell']['prefix']),q=str(rec['cell']['q']),axis=axis,
        RZ=str(10+rec['cell']['prefix']),compiler_hash=m.COMPILER,native_exit_code='0',
        acquisition='REUSE_WITH_DECLARED_LIMITS',origin_SOURCE='artificial')
        for rec in old if rec['cell']['method']=='B0I' for axis in m.c.AXES]
    return dict(frozen=frozen,validation=dict(rows=rows),
        phase_b_terminal=dict(status='COMPLETE_MANDATORY_STOP',exit_code=0),
        phase_b_launcher=dict(status='COMPLETE_MANDATORY_STOP',native_exit_code=0,cleanup_errors=[],all_reaped=True,first_stop=None),phase_a_terminal=dict(status='COMPLETE_MANDATORY_STOP',exit_code=0),
        old_signals=dict(signals=[x for x in old if x['cell']['method']!='B0I']),
        b0i_signals=[x for x in old if x['cell']['method']=='B0I'],additional=dict(status='COMPLETE_MANDATORY_STOP',signals=additional),
        exploration_wrappers=exploration,confirmation_wrappers=confirmation,old_accounting=old_tables,
        b0i_accounting=b0i,b0i_axes=b0i_axes,old_comparison=dict(comparisons=comparisons))


class ReaccountTests(unittest.TestCase):
    def test_scopes_are_nested_and_keep_full_S4(self):
        a,b,d=map(m.scope_ids,m.SCOPES)
        self.assertEqual((len(a),len(b),len(d)),(32,96,136));self.assertTrue(a<b<d)
        self.assertTrue(all('B1_p19_q8_4th' in x for x in (a,b,d)))
        self.assertTrue(all(not x.startswith('B2') for x in d))

    def test_exact_fraction_weighting(self):
        rec=record(m.c.cell('B0',11,1,'4th'));cid=rec['cell']['id']
        a=dict(cost_acquisition_eligible=True,axes={'real':dict(shots=3),'imag':dict(shots=5)},N_total=8)
        axes={(cid,axis):dict(RZ=v,status='VALIDATED_MODEL_ESTIMATE') for axis,v in zip(m.c.AXES,('2/3','7/4'))}
        out=m.value_row(rec,a,axes,epsilon=.01,factor=1.,bias_rule='signed_signal')
        self.assertEqual(Fraction(out['G_exact']),Fraction(43,4));self.assertEqual(out['cost_status'],'VALIDATED_MODEL_ESTIMATE')

    def test_missing_eligible_cost_stops(self):
        rec=record(m.c.cell('B0',11,1,'4th'));a=m.r.accounting.accounting(rec,.05,1.)
        with self.assertRaisesRegex(ValueError,'ELIGIBLE_COST_MISSING'):
            m.value_row(rec,a,{},epsilon=.05,factor=1.,bias_rule='signed_signal')

    def test_unknown_candidate_prohibits_full_domain_winner(self):
        out=m.choose([dict(eligible=None),dict(eligible=True,G_exact='1',cell_id='A')])
        self.assertIsNone(out['best']);self.assertEqual(out['status'],'UNRESOLVED_FULL_DOMAIN')

    def test_exact_ties_and_close_values(self):
        rows=[dict(eligible=True,G_exact=v,G_point=float(Fraction(v)),cell_id=cid,cost_status='VALIDATED_MODEL_ESTIMATE',N_total=2)
              for cid,v in [('A','10000000000000001/10000000000000000'),('B','1'),('C','1')]]
        out=m.choose(rows);self.assertEqual(out['best']['cell_id'],'B');self.assertEqual(out['exact_ties'],['B','C'])

    def test_duplicate_rows_rejected(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE_INPUT_ROW'):m.index_unique([{'id':'A'},{'id':'A'}],lambda x:x['id'])

    def grant(self):
        cpus=sorted(os.sched_getaffinity(0))[:2];p=m.plan()
        config=dict(plan_sha256=m.c.digest(p));grant=dict(schema='track_a_v1_reaccount_grant_v1',approved_by_user=True,
            runtime_authorization=True,scope=p['scope'],one_shot=True,retry=False,resume=False,SOURCE='source',
            config_sha256='config',plan_sha256=m.c.digest(p),resources=p['resources'],call_caps=p['call_caps'],
            frozen_model_sha256=p['frozen_model_sha256'],allowed_cpus=cpus,driver_cpu=cpus[0],monitor_cpu=cpus[1])
        fresh=dict(source='source',config_sha256='config',allowed_cpus=cpus,monotonic_seconds=100.,
            profile_verified=True,resources_idle_verified=True,protected_jobs_unchanged=True,external_RSS_monitor_ready=True)
        return config,grant,fresh

    def test_sealed_grant_and_stale_gate(self):
        config,grant,fresh=self.grant();m.verify_grant(config,grant,fresh,'source','config',now=110)
        with self.assertRaisesRegex(ValueError,'FRESH_GATE'):m.verify_grant(config,grant,fresh,'source','config',now=131)

    def test_mutated_grants_denied(self):
        for key,value in [('approved_by_user',False),('retry',True),('scope','compile'),('SOURCE','other'),('call_caps',{}),('frozen_model_sha256','other')]:
            config,grant,fresh=self.grant();grant[key]=value
            with self.assertRaises(ValueError):m.verify_grant(config,grant,fresh,'source','config',now=110)

    def test_no_scientific_call_budget(self):
        caps=m.plan()['call_caps']
        self.assertTrue(all(caps[x]==0 for x in ('new_signal','new_scalar','new_primitive','new_compile','new_fit','new_sampling','V2')))

    def test_complete_artificial_pass_and_measured_precedence(self):
        data=fixture();writer=Writer();summary=m.evaluate(data,writer)
        self.assertEqual(summary['comparison_rows'],60);self.assertEqual(summary['accounting_rows'],4320)
        self.assertEqual(summary['accounting_evaluations'],4040)
        rows=writer.values['axis_costs.json']['rows'];key=data['validation']['rows'][0]
        measured=next(x for x in rows if (x['cell_id'],x['axis'])==(key['cell_id'],key['axis']))
        self.assertEqual(measured['RZ'],'100');self.assertEqual(measured['source_kind'],'PHASE_B_REGISTERED_VALIDATION')
        comparisons=writer.values['comparisons.json']['rows']
        diag=[x for x in comparisons if x['bias_rule']=='empirical_no_cancellation']
        self.assertTrue(all(x['deterministic']['best'] is None and x['deterministic']['unresolved_candidates']==12 for x in diag))

    def test_partial_phase_b_or_refit_denied(self):
        for key in ('exit','reap','refit'):
            data=fixture()
            if key=='exit':data['phase_b_terminal']['exit_code']=1
            elif key=='reap':data['phase_b_launcher']['all_reaped']=False
            else:data['frozen']['fitted_after_validation']=True
            with self.assertRaises(ValueError):m.evaluate(data,Writer())

    def test_changed_old_baseline_stops(self):
        data=fixture();data['old_comparison']['comparisons'][0]['best_registered_deterministic']['G_point']+=1
        with self.assertRaisesRegex(ValueError,'BASELINE_NOT_REPRODUCED'):m.evaluate(data,Writer())

    def test_changed_phase_a_accounting_stops(self):
        data=fixture();data['additional']['signals'][0]['precision_rows']['0.05']['1.0']['N_total']+=1
        with self.assertRaisesRegex(ValueError,'PHASE_A_ELIGIBILITY_CHANGED'):m.evaluate(data,Writer())

    def test_no_pr_selection_from_unregistered_diagnostic(self):
        data=fixture();data['additional']['signals'][0]['cell']=m.c.cell('B2',3,1,r=4,K=4)
        with self.assertRaisesRegex(ValueError,'REGISTERED_SIGNAL_DOMAIN'):m.evaluate(data,Writer())


if __name__=='__main__':unittest.main()
