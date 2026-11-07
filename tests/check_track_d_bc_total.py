"""Synthetic scalar tests only; no saved molecular totals or circuit calls."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import track_d_bc_total as t


def artificial():
    labels = [(m,a,g) for m in (*t.bc.METHODS,'paired_delta') for a in t.AXES for g in t.bc.METRICS]
    handoff = dict(binding=dict(original_h=t.bc.PACKET,cdf=t.bc.CDF), rows=[
        dict(m=m,q=2**m,N_X=3+m%3,N_Y=3+m%3) for m in range(13)])
    direct = {q:dict(n=16, source=dict(commit='ARTIFICIAL',path='synthetic/q'+str(q)),
        values={(m,a,g):float(q+(m=='reference')) if m!='paired_delta' else 1.
            for m,a,g in labels}) for q in (1,2,4,8,16,32)}
    c74 = {'integrated_statistics.json':dict(quantity_labels=labels,by_q=[dict(q=q,n=16,
        sample_covariance=np.eye(36).tolist()) for q in (2,4,8,16,32)])}
    for name,p in [('affine',2),('sqrt_lane',3)]:
        coefficients = np.zeros((p,36))
        for i,(m,a,g) in enumerate(labels):
            coefficients[0,i] = 1. if m=='paired_delta' else 10.+(m=='reference')
            coefficients[1,i] = 0. if m=='paired_delta' else 2.
            if p==3:coefficients[2,i]=0. if m=='paired_delta' else 1.
        c74['model_'+name+'.json']=dict(parameters=p, design=['1','q']+(['sqrt(q)'] if p==3 else []),
            quantity_labels=labels,coefficients=coefficients.tolist(),coefficient_order='design column, then quantity',
            coefficient_covariance=np.eye(p*36).tolist(),bootstrap_scope='ARTIFICIAL_ONLY')
    flabels=[(m,a,g,param) for m in t.bc.METHODS for a in t.AXES for g in t.bc.METRICS for param in ('slope','intercept')]
    c74['frozen_C6a_future.json']=dict(saved_parameter_covariance=dict(labels=flabels,matrix=np.eye(48).tolist(),formula='ARTIFICIAL_ONLY'))
    return dict(handoff=handoff,direct=direct,c74=c74,premises=t.premises(),
        frozen_coefficients={(m,a,g):(20.+(m=='reference'),3.) for m in t.bc.METHODS for a in t.AXES for g in t.bc.METRICS})


class FixedAccounting(unittest.TestCase):
    def test_three_scenarios_keep_direct_and_q1_event_cut(self):
        x=artificial();rows={s:t.price_records(x,s) for s in t.SCENARIOS}
        for s,rs in rows.items():
            self.assertEqual(len(rs),52)
            self.assertTrue(all(r['cut']==dict(kind='complete_event_within_step',after=16) for r in rs if r['q']==1))
            self.assertTrue(all(r['mean']['cx_count']==1. for r in rs if r['q']==1 and r['method']=='symmetric_directional'))
        a=next(r for r in rows['A'] if r['q']==64 and r['method']=='reference')
        f=next(r for r in rows['F'] if r['q']==64 and r['method']=='reference')
        s=next(r for r in rows['S'] if r['q']==64 and r['method']=='reference')
        self.assertEqual((a['mean']['cx_count'],f['mean']['cx_count'],s['mean']['cx_count']),(139.,213.,147.))

    def test_weighted_totals_pair_and_group_identities(self):
        x=artificial();out=t.evaluate(x)
        for s,r in out['scenarios'].items():
            for g,v in r['totals'].items():
                expected=sum(n['N_X']+n['N_Y'] for n in x['handoff']['rows'])
                self.assertAlmostEqual(v['paired_delta'],expected)
                self.assertAlmostEqual(sum(part[g]['reference'] for part in r['grouped'].values()),v['reference'])
            self.assertEqual(len(r['contributions']),312)
            self.assertEqual(len(r['paired_rounds']),156)

    def test_missing_and_duplicate_rejected(self):
        x=artificial();rows=t.price_records(x,'A')
        for bad in [rows[:-1],rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):t.validate_records(x,bad,'A')

    def test_identity_P0_source_and_scope_rejected(self):
        x=artificial();rows=t.price_records(x,'A')
        for key,val in [('source',dict(commit='other',path='foreign')),('binding',{}),('compiler',{}),('scope','kernel')]:
            bad=copy.deepcopy(rows);bad[0][key]=val
            with self.assertRaises(ValueError):t.validate_records(x,bad,'A')

    def test_nonfinite_negative_or_extra_width_price_rejected(self):
        x=artificial();rows=t.price_records(x,'A')
        for value in [float('nan'),float('inf'),-1.]:
            bad=copy.deepcopy(rows);bad[0]['mean']['cx_count']=value
            with self.assertRaises(ValueError):t.validate_records(x,bad,'A')
        bad=copy.deepcopy(rows);bad[0]['mean']['peak_qubits']=17
        with self.assertRaises(ValueError):t.validate_records(x,bad,'A')

    def test_cut_scenario_and_attenuation_rejected(self):
        x=artificial();rows=t.price_records(x,'A')
        for key,val in [('cut',dict(kind='complete_step',after=0)),('scenario','S'),('attenuation_price_multiplier',True)]:
            bad=copy.deepcopy(rows);bad[0][key]=val
            with self.assertRaises(ValueError):t.validate_records(x,bad,'A')

    def test_signed_shared_covariance_and_cross_resource_retained(self):
        x=np.array([[1.,2.],[1.,2.],[-1.,-2.]])
        r=t.covariance_component([[1.,.5],[.5,4.]],x,'artificial',[])
        c=np.array(r['covariance']);self.assertAlmostEqual(c[0,1],19.)
        self.assertAlmostEqual(c[0,2],-19.)
        v=t.covariance_component([[1.,1.],[1.,1.]],np.array([[1.,-1.]]),'paired',[])
        self.assertEqual(v['nominal_component_SD'],[0.])

    def test_unknown_covariance_not_zero_and_F_overlap_marked(self):
        x=artificial()
        for s in t.SCENARIOS:
            u=t.uncertainty(x,s)
            self.assertIsNone(u['total_covariance']);self.assertIsNone(u['total_SD'])
            self.assertFalse(u['missing_covariance_is_zero'])
            self.assertTrue(u['components_are_not_independent_blocks'])
        f=t.uncertainty(x,'F')['components'][-1]
        self.assertEqual(f['component'],'linked_q2_q4_and_long_q')

    def test_shared_model_rounds_not_independent(self):
        x=artificial();model=x['c74']['model_affine.json'];model['coefficient_covariance']=np.zeros((72,72)).tolist()
        model['coefficient_covariance'][0][0]=4.
        u=t.uncertainty(x,'A')['components'][-1]
        expected=2*sum(r['N_X'] for r in x['handoff']['rows'][6:])
        self.assertAlmostEqual(u['nominal_component_SD'][0],expected)

    def test_invalid_covariance_rejected_and_partial_saved_before_later_failure(self):
        with self.assertRaises(ValueError):
            t.covariance_component([[-1.]],np.ones((1,1)),'invalid',[])
        x=artificial();completed=[]
        def preserve(scenario,data):
            completed.append(scenario)
            if scenario=='F':raise RuntimeError('ARTIFICIAL_IO_FAILURE')
        with self.assertRaises(RuntimeError):t.evaluate(x,on_scenario=preserve)
        self.assertEqual(completed,['A','F'])

    def test_model_schema_rejects_wrong_design_or_missing_axis(self):
        x=artificial();d=x['c74']['model_affine.json'];t.check_model(d,'affine')
        bad=copy.deepcopy(d);bad['design']=['1','sqrt(q)']
        with self.assertRaises(ValueError):t.check_model(bad,'affine')
        bad=copy.deepcopy(d);bad['quantity_labels'][0]=bad['quantity_labels'][1]
        with self.assertRaises(ValueError):t.check_model(bad,'affine')

    def test_actual_and_false_common_N_rejected_by_reused_guard(self):
        x=artificial();rows=t.price_records(x,'A')
        with self.assertRaises(ValueError):t.bc.aggregate_registered(x['handoff'],rows,x['premises'],claim='actual')
        x['premises']['law_is_B4_ideal_product']=False
        with self.assertRaises(ValueError):t.bc.aggregate_registered(x['handoff'],rows,x['premises'])


if __name__=='__main__':unittest.main()
