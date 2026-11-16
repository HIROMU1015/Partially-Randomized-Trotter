"""Metadata and artificial NPZ checks; no real H4 numerical evaluation."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pf_evidence_preparation as E
from trottertracks import pf_candidate_space as P


class Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=P.configure(ROOT,'FULL_PF',10,'2nd','2')
        cls.prepared=E.plan(ROOT,cls.cfg,dict(module='ARTIFICIAL_METADATA_ONLY'))

    def test_selected_plan_calibration_holdout_and_early_width(self):
        plan=E.make_plan(ROOT,self.cfg,dict(module='ARTIFICIAL_METADATA_ONLY'))
        self.assertTrue(self.cfg['id'].startswith('9e96ed75'))
        self.assertEqual(plan['new_signal_means'],8)
        self.assertEqual(len(plan['direct_prices']),8)
        self.assertEqual(plan['calibration_rounds'],[2,3])
        self.assertEqual(plan['holdout_round'],5)
        self.assertEqual(plan['maximum_new_compile'],2)
        self.assertEqual(plan['new_compile_cells'],[dict(m=5,axes=['X','Y'],role='HOLDOUT')])
        self.assertFalse(plan['refit_holdout'])
        self.assertIsNone(plan['tolerance'])
        requests=self.prepared['pending']['requests']
        for m in plan['calibration_rounds']+[plan['holdout_round']]:
            self.assertEqual(E.frac(requests[m]['T'])/requests[m]['q_PF'],E.frac(plan['actual_width']))
        self.assertNotEqual(E.frac(requests[0]['T'])/requests[0]['q_PF'],E.frac(plan['actual_width']))

    def test_full_pilot_scope_normalizes_without_array_read(self):
        original=E.C.blob
        def metadata(root,commit,path):
            if path.endswith('.npz'):raise AssertionError('Real array read')
            return original(root,commit,path)
        with patch.object(E.C,'blob',metadata):
            rows=E.saved_step_descriptors(ROOT,self.prepared)
        self.assertEqual(len(rows),2)
        pilot=next(r for r in rows if r['producer_format']=='FULL_PF_INITIAL_PILOT_SCOPE')
        self.assertEqual(pilot['scope_source']['path'].rsplit('/',1)[-1],'SCOPE.json')
        self.assertEqual(pilot['operator_source']['commit'],'69ee0ff8898960d5f9959af8d34d89f223caec87')
        self.assertEqual(pilot['operator_source']['field'],'PF_step_without_global_scalar')
        self.assertEqual(pilot['prefix'],10)
        self.assertEqual(E.frac(pilot['actual_delta_pf']),P.H)

    def test_prefix_zero_and_eight_saved_step_metadata(self):
        zero=P.configure(ROOT,'B0I',0,'2nd','2')
        eight=P.configure(ROOT,'B0I',8,'2nd','2')
        z=E.plan(ROOT,zero,dict(module='METADATA_ONLY'))
        e=E.plan(ROOT,eight,dict(module='METADATA_ONLY'))
        self.assertEqual(E.saved_step_descriptors(ROOT,z),[])
        descriptors=E.saved_step_descriptors(ROOT,e)
        self.assertEqual(len(descriptors),2)
        self.assertEqual({d['prefix'] for d in descriptors},{8})
        self.assertEqual({E.frac(d['actual_delta_pf'])/P.H for d in descriptors},{1,2})
        saved=E.read(ROOT,E.scalar_source(8))
        self.assertTrue(all(d['tail_scalar']==saved for d in descriptors))

    def test_prefix_eight_t2_means_are_reused_by_physical_schedule(self):
        cfg=P.configure(ROOT,'B0I',8,'2nd','16/7')
        plan=E.make_plan(ROOT,cfg,dict(module='METADATA_ONLY'))
        means={r['m']:r for r in plan['signal_rounds']}
        for m in (4,12):
            self.assertEqual(means[m]['acquisition'],'REUSE_SAVED_MEAN')
            self.assertEqual(means[m]['saved_source']['commit'],P.T2_SOURCE)
            self.assertIn('/cell_p8_2nd_16o7/',means[m]['saved_source']['path'])
            old=E.read(ROOT,means[m]['saved_source'])
            self.assertEqual(old['physical_candidate'],plan['prepared']['pending']['requests'][m]['physical_candidate'])
        self.assertEqual({m for m,r in means.items() if r['acquisition']=='REUSE_SAVED_MEAN'},{0,1,4,12})
        self.assertEqual(plan['new_signal_means'],9)

    def test_binary_piece_times_discriminate_same_q_different_width(self):
        a=dict(prefix=8,formula='4th',T=.2,q=2,delta=.1)
        b=dict(a,T=.4,delta=.2)
        self.assertNotEqual(E.step_key(a,[1.,-0.5,1.]),E.step_key(b,[1.,-0.5,1.]))
        c=dict(a,T=.4,q=4)
        self.assertEqual(E.step_key(a,[1.,-0.5,1.]),E.step_key(c,[1.,-0.5,1.]))

    def test_artificial_npz_field_scalar_prefix_and_cache(self):
        import numpy as np
        scope=dict(first_check=dict(physical_candidate=dict(method='B0I',prefix=8,formula='2nd',T=.2,q=1,delta=.2)),
                   formula='2nd',actual_delta_pf=E.enc(.2),signed_weights=[1.])
        summary=dict(formula='2nd',actual_delta_pf=E.enc(.2),c_tail_ha=.03)
        with patch.object(E,'read',side_effect=[scope,summary]):
            descriptor=E.saved_step_metadata(ROOT,'ARTIFICIAL','toy','PF_step_without_global_scalars')
        buf=io.BytesIO()
        np.savez(buf,PF_step_without_global_scalars=np.eye(36,dtype=complex),scalar_included_unitary=2*np.eye(36))
        ctx=object.__new__(E.SavedOrNativeContext)
        ctx.root=str(ROOT);ctx.prefix=8;ctx.formula='2nd';ctx.constant=2.;ctx.binding={}
        ctx.tail_scalars={8:.03};ctx.saved_steps={};ctx.saved_step_sources=[];ctx.steps={};ctx.new_step_keys=[]
        with patch.object(E.C,'blob',return_value=buf.getvalue()):ctx._load_saved(descriptor)
        c=dict(method='B0I',prefix=8,formula='2nd',T=.8,q=4,delta=.2)
        with patch.object(E.N.Context,'pf_step',side_effect=AssertionError('No new step')):
            np.testing.assert_array_equal(ctx.pf_step(c,[1.]),np.eye(36))
        ctx.tail_scalars[8]=.04
        with patch.object(E.C,'blob',return_value=buf.getvalue()),self.assertRaisesRegex(ValueError,'prefix scalar'):
            ctx._load_saved(descriptor)
        ctx.prefix=0
        with patch.object(E.C,'blob',side_effect=AssertionError('Prefix mismatch must precede array read')),self.assertRaisesRegex(ValueError,'selected prefix'):
            ctx._load_saved(descriptor)

    def test_saved_mean_binding_and_scalar_rejection(self):
        req=self.prepared['pending']['requests'][0]
        origin=E.mean_pool(str(ROOT))[E.physical_key(req['physical_candidate'])][0]['source']
        original_read=E.read;old=original_read(ROOT,origin)
        record=E.saved_record(ROOT,self.prepared,req,origin,dict(module='METADATA_ONLY'))
        mean=E.bind_mean(self.prepared,req,record)
        self.assertEqual(mean['raw'],old['raw'])
        for mutation in ('prefix','width','binding','scalar','B','raw'):
            changed=deepcopy(old)
            if mutation=='prefix':changed['physical_candidate']['prefix']=8
            elif mutation=='width':changed['physical_candidate']['delta']=.2
            elif mutation=='binding':changed['provider_binding']['physical_context']['binding']['state']='OTHER_STATE'
            elif mutation=='scalar':changed['normalization']['numerical']['scalar_once']=False
            elif mutation=='B':changed['normalization']['B']=2.
            else:changed['corrected']['real']+=.01
            def read(root,loc):return deepcopy(changed) if loc==origin else original_read(root,loc)
            with patch.object(E,'read',read),self.assertRaises(ValueError):
                E.saved_record(ROOT,self.prepared,req,origin,dict(module='METADATA_ONLY'))

    def test_saved_price_axis_lowering_and_profile_are_not_interchangeable(self):
        req=E.I.price_requests(ROOT,self.prepared)[0]
        key=(E.physical_key(req['physical_context']['physical_candidate']),req['axis'])
        hit=E.price_pool(str(ROOT))[key]
        for mutation in ('lowering','axis','wrapper','prefix'):
            changed=deepcopy(hit)
            if mutation=='lowering':changed['value']['price_profile']['lowering_id']='OTHER_LOWERING'
            elif mutation=='axis':changed['value']['axis']='Y'
            elif mutation=='wrapper':changed['value']['price_profile']['wrapper']='NO_MEASUREMENT'
            else:changed['value']['physical_context']['physical_candidate']['prefix']=8
            with patch.object(E,'price_pool',return_value={key:changed}),patch.object(E.I,'price_requests',return_value=[req]),self.assertRaises(ValueError):
                E.accepted_direct(ROOT,self.prepared)

    def test_h_direct_receipts_prevent_repeat_compile(self):
        cfg=P.configure(ROOT,'B0I',7,'2nd','1')
        plan=E.make_plan(ROOT,cfg,dict(module='METADATA_ONLY'))
        self.assertTrue(cfg['id'].startswith('e6754564'))
        self.assertEqual(len(plan['direct_prices']),10)
        self.assertEqual({r['m'] for r in plan['direct_prices']},set(range(5)))
        self.assertEqual(plan['calibration_rounds'],[1,2])
        self.assertEqual(plan['holdout_round'],4)
        self.assertEqual(plan['maximum_new_compile'],0)
        self.assertEqual(plan['new_compile_cells'],[])

    def test_later_existing_holdout_avoids_repeat_coarse_compile(self):
        cfg=P.configure(ROOT,'B0I',7,'4th','32/5')
        plan=E.make_plan(ROOT,cfg,dict(module='METADATA_ONLY'))
        self.assertTrue(cfg['id'].startswith('dd8bfc24'))
        self.assertEqual(len(plan['direct_prices']),16)
        self.assertEqual(plan['calibration_rounds'],[5,6])
        self.assertEqual(plan['holdout_round'],9)
        self.assertEqual(plan['maximum_new_compile'],0)
        self.assertEqual(plan['new_compile_cells'],[])

    def test_published_ready_reuses_complete_inputs_without_new_fit(self):
        cfg=P.configure(ROOT,'B0I',7,'4th','32/5')
        ready=E.published_ready(ROOT,cfg)
        self.assertEqual(ready['index'],37)
        self.assertEqual(ready['catalog_source']['selector'],['candidates',37])
        self.assertEqual(len(ready['row']['price_axes']),26)
        self.assertEqual(ready['row']['configuration_id'],cfg['id'])
        self.assertIsNone(E.published_ready(ROOT,P.configure(ROOT,'B0I',7,'2nd','1')))
        original=E.read
        def bad(root,loc):
            value=original(root,loc)
            if loc==E.CATALOG:
                value=deepcopy(value);value['candidates'][37]['price_axes'][0]['physical_id']='OTHER_PHYSICAL_SCHEDULE'
            return value
        with patch.object(E,'read',bad),self.assertRaisesRegex(ValueError,'physical identity'):
            E.published_ready(ROOT,cfg)


if __name__=='__main__':unittest.main(verbosity=2)
