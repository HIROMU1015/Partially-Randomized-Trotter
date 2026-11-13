"""All numerical inputs are fictional; existing B and mean-kernel fixtures only."""
import copy
import json
import unittest
from unittest.mock import patch

import check_rpe_optimization_pipeline as T
from trottertracks import rpe_mean_price_model as P
from trottertracks import mean_compiled_price as K

ROOT=T.ROOT;I=T.I;B=T.B
SOURCE=dict(fixture='ARTIFICIAL_MEAN_PRICE_ONLY')
EXT=dict(lowering_id='ARTIFICIAL_LOWERING',source=SOURCE)
METHODS=('FULL','B0','B0I','PR','FULL_H_RTE')


def model(req,m=0,axis='X'):
    scope=P.context(ROOT,req,m,axis,profile_extension=EXT);terms=[dict(feature='one',coefficient=3,covers=sorted(P.BOUNDARIES))]
    if scope['method'] in ('FULL_PF','TRUNCATED_PF','PR'):
        terms.append(dict(feature='pf_step_count',coefficient=5,covers=['PF_BODY']))
    if scope['method'] in ('PR','FULL_H_RTE'):
        terms.append(dict(feature='event_count',coefficient=2,covers=['RTE_BODY']))
    if scope['method']=='PR':terms.append(dict(feature='q',coefficient=7,covers=['PF_RTE_BOUNDARY']))
    return dict(name='FICTIONAL_ADDITIVE',mode='ADDITIVE',context=scope,source=SOURCE,terms=terms)


def quote(req,spec=None,**kw):
    spec=model(req) if spec is None else spec
    return P.quote_model(ROOT,req,spec['context']['m'],spec['context']['axis'],model=spec,profile_extension=EXT,**kw)


def supplied(scope,quantity,value):return dict(context=copy.deepcopy(scope),quantity=quantity,value=value,source=SOURCE)


class PriceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.requests={method:T.toy(method,M=2)[0] for method in METHODS}

    def test_01_each_method_round_axis_and_q_event_counts(self):
        for method,req in self.requests.items():
            for m in range(3):
                for axis in ('X','Y'):
                    s=model(req,m,axis);r=quote(req,s);scope=s['context'];c=scope['physical_context']['physical_candidate']
                    expected=3+(5*c['q'] if method!='FULL_H_RTE' else 0)
                    if method in ('PR','FULL_H_RTE'):expected+=2*req['configuration']['rounds'][m]['structure']['event_count']
                    if method=='PR':expected+=7*c['q']
                    self.assertEqual(B.a.exact_scalar(r['mean']['rz_count']),expected)
                    self.assertEqual(r['identity'],scope['identity']);self.assertEqual(r['axis'],axis)
                    self.assertFalse(r['model']['qualified_for_candidate']);self.assertEqual(r['price_status'],'UNQUALIFIED_MODEL')
                    self.assertIsNone(r['uncertainty']['total_CI']);json.dumps(r,allow_nan=False)

    def test_02_boundaries_once_and_no_overlap(self):
        req=self.requests['PR'];s=model(req,2);s['terms'][0]['feature']='q'
        with self.assertRaisesRegex(ValueError,'once'):quote(req,s)
        s=model(req);s['terms'].append(copy.deepcopy(s['terms'][1]))
        with self.assertRaisesRegex(ValueError,'double count'):quote(req,s)
        s=model(req);s['terms'].pop(0)
        self.assertIsNone(quote(req,s)['mean']['rz_count'])
        s=model(req);s['terms'][0]['coefficient']=-10**9
        with self.assertRaisesRegex(ValueError,'Negative'):quote(req,s)

    def test_03_unknown_expected_features_do_not_become_zero(self):
        req=self.requests['PR'];s=model(req);s['terms'][2]['feature']='rare_order2_feature'
        self.assertEqual(quote(req,s)['price_status'],'MISSING')
        feature=supplied(s['context'],P.FEATURE,.25)
        feature['uncertainty']=dict(variance=None,law_distance=None)
        result=quote(req,s,features={'rare_order2_feature':feature})
        self.assertEqual(result['uncertainty']['features']['rare_order2_feature'],feature)
        feature['value']=None;self.assertIsNone(quote(req,s,features={'rare_order2_feature':feature})['mean']['rz_count'])
        with self.assertRaisesRegex(ValueError,'replace canonical'):quote(req,s,features={'q':feature})

    def test_04_reuse_mean_kernel_and_single_paired_full_wrapper_residual(self):
        req=self.requests['PR'];s=model(req);s.update(mode='STRUCTURAL_PLUS_RESIDUAL');s.pop('terms')
        scope=s['context'];states=('a',);identity=K.MeanKernel.identity(states)
        piece=K.MeanPiece(1.,3,K.MeanKernel(states,[[1]],[3]),K.MeanKernel(states,[[1]],[2]),
                          K.MeanKernel(states,[[1]],[4]),SOURCE)
        mean=K.full_wrapper_mean(2,[piece],K.MeanKernel(states,[[1]],[5]),identity,[1],[1])
        self.assertEqual(mean['mean'],32);self.assertFalse(mean['compiled_price_qualified'])
        structural=supplied(scope,P.STRUCTURAL,mean['mean'])
        residual=dict(supplied(scope,P.RESIDUAL,-3),mode='C1_AFFINE',coverage='FULL_WRAPPER')
        r=quote(req,s,structural=structural,residuals=[residual])
        self.assertEqual(r['mean']['rz_count'],29);self.assertEqual(r['quantity'],P.QUANTITY)
        self.assertFalse(r['model']['qualified_for_candidate'])
        with self.assertRaisesRegex(ValueError,'one residual'):
            quote(req,s,structural=structural,residuals=[residual,residual])
        residual['coverage']='RTE_INTERNAL_ONLY'
        self.assertIsNone(quote(req,s,structural=structural,residuals=[residual])['mean']['rz_count'])
        structural['quantity']='FULL_WRAPPER_PATH_STRUCTURAL_RZ_COUNT'
        with self.assertRaisesRegex(ValueError,'quantity'):quote(req,s,structural=structural,residuals=[residual])
        with self.assertRaisesRegex(ValueError,'stack'):quote(req,model(req),structural=structural)

    def test_05_wrong_method_law_source_and_profile_rejected(self):
        req=self.requests['PR']
        for change in (lambda s:s['context'].update(method='FULL_H_RTE'),
                       lambda s:s['context']['law_binding'].update(foreign=True),
                       lambda s:s['context']['physical_context']['binding'].update(source={'foreign':True}),
                       lambda s:s['context']['price_profile'].update(lowering_id='FOREIGN'),
                       lambda s:s.update(source=None),lambda s:s.update(qualified_for_candidate=True)):
            s=model(req);change(s)
            with self.assertRaises(ValueError):quote(req,s)
        s=model(req,1)
        with self.assertRaises(ValueError):P.quote_model(ROOT,req,0,'X',model=s,profile_extension=EXT)

    def test_06_large_q_R_path_uses_only_metadata_and_terms(self):
        req=T.toy('PR',M=12)[0];s=model(req,12)
        with (patch.object(T.API.Price,'NativeRuntime',side_effect=AssertionError('no preparation')),
              patch.object(T.API.Price,'Adapter',side_effect=AssertionError('no sampling/compile')),
              patch.object(K,'full_wrapper_mean',side_effect=AssertionError('no new physical E[S]'))):
            r=quote(req,s)
        self.assertEqual(len(r['model']['term_details']),4)
        self.assertEqual(r['model']['term_details'][2]['weight'],req['configuration']['rounds'][12]['structure']['event_count'])
        self.assertNotIn('events',r)

    def test_07_passive_missing_and_model_lookup_no_qualification(self):
        req=self.requests['PR'];records=[quote(req,model(req,m,a)) for m in range(3) for a in ('X','Y')]
        provider=P.PriceProvider(ROOT,req,records,profile_extension=EXT)
        with patch.object(P,'quote_model',side_effect=AssertionError('lookup does not evaluate')):out=provider()
        self.assertEqual(out['prices'],records)
        self.assertTrue(all(p['price_status']=='UNQUALIFIED_MODEL' for p in out['prices']))
        missing=P.PriceProvider(ROOT,req,profile_extension=EXT)()
        self.assertTrue(all(p['mean']['rz_count'] is None and p['price_status']=='MISSING' for p in missing['prices']))

    def test_08_existing_B_accounting_direct_vs_unqualified_models(self):
        for method in ('FULL','B0I','PR','FULL_H_RTE'):
            f=T.fixture(method);req=f[0];direct=copy.deepcopy(f[4])
            for p in direct:
                scope=P.context(ROOT,req,p['identity']['m'],p['axis'],profile_extension=EXT)
                p.update(physical_context=scope['physical_context'],price_profile=scope['price_profile'],normalization_is_price_multiplier=False)
            out=P.PriceProvider(ROOT,req,direct,profile_extension=EXT)()
            compared=I.compare(ROOT,req,*f[1:4],**out,statistical_contract=f[5]['statistical_contract'])
            self.assertEqual([B.a.exact_scalar(a['G']) for a in compared['comparison']['allocations']],[1002,216])
            models=[quote(req,model(req,m,a)) for m in range(2) for a in ('X','Y')]
            out=P.PriceProvider(ROOT,req,models,profile_extension=EXT)()
            compared=I.compare(ROOT,req,*f[1:4],**out,statistical_contract=f[5]['statistical_contract'])
            self.assertIsNone(compared['comparison']['point_minimum']['cost'])

    def test_09_finite_cohort_unknowns_and_failed_values_are_preserved(self):
        f=T.fixture('PR');req=f[0];records=copy.deepcopy(f[4])
        for r in records:
            scope=P.context(ROOT,req,r['identity']['m'],r['axis'],profile_extension=EXT)
            r.update(physical_context=scope['physical_context'],price_profile=scope['price_profile'],
                expectation='FINITE_COHORT_ONLY',n_paths=2,price_status='FAILED',
                normalization_is_price_multiplier=False,
                uncertainty=dict(total_CI=None,joint_covariance=None,law_distance=None))
        before=copy.deepcopy(records);out=P.PriceProvider(ROOT,req,records,profile_extension=EXT)()
        self.assertEqual(out['prices'],before)
        compared=I.compare(ROOT,req,*f[1:4],**out,statistical_contract=f[5]['statistical_contract'])
        self.assertIsNone(compared['comparison']['point_minimum']['cost'])

    def test_10_source_identity_lowering_cohort_and_units_never_relabelled(self):
        req=self.requests['FULL'];record=quote(req)
        for change in (lambda r:r.update(law_binding={}),lambda r:r.update(source=None),
                       lambda r:r['price_profile'].update(lowering_id='FOREIGN'),
                       lambda r:r['physical_context']['physical_candidate'].update(q=999),
                       lambda r:r.update(quantity=P.STRUCTURAL),lambda r:r.update(units={}),
                       lambda r:r['model'].update(qualified_for_candidate=True)):
            r=copy.deepcopy(record);change(r)
            with self.assertRaises(ValueError):P.PriceProvider(ROOT,req,[r],profile_extension=EXT)()
        with self.assertRaisesRegex(ValueError,'Duplicate'):P.PriceProvider(ROOT,req,[record,record],profile_extension=EXT)()
        other=copy.deepcopy(record);other['cohort']='OTHER'
        out=P.PriceProvider(ROOT,req,[record,other],profile_extension=EXT)()
        self.assertEqual(out['prices'][0]['price_status'],'SELECTION_REQUIRED')
        selector={(record['identity']['candidate']['id'],'X'):'OTHER'}
        out=P.PriceProvider(ROOT,req,[record,other],profile_extension=EXT,selections=selector)()
        self.assertEqual(out['prices'][0],other)

    def test_11_independent_R_change_is_not_the_same_price_identity(self):
        req=self.requests['PR'];spec=copy.deepcopy(req['configuration']['spec']);spec['kappa']['value']=10000
        cfg=I.C.configure(ROOT,spec,api=T.API);physical={}
        for r in cfg['rounds']:
            c=copy.deepcopy(req['pr_physical'][r['m']])
            c.update(q=r['q_PF'],R_outer=r['R_outer'],delta=float(B.a.exact_scalar(r['actual_delta_pf'])))
            c['id']=B.p.candidate_id(c);physical[r['m']]=c
        other=I.plan(ROOT,cfg,target=req['target'],signal_source=req['signal_source'],pr_physical=physical)
        s=model(req);s2=model(other)
        self.assertEqual(s['context']['physical_context']['physical_candidate']['q'],
                         s2['context']['physical_context']['physical_candidate']['q'])
        self.assertNotEqual(s['context']['physical_context']['physical_candidate']['R_outer'],
                            s2['context']['physical_context']['physical_candidate']['R_outer'])
        self.assertNotEqual(quote(req,s)['mean'],quote(other,s2)['mean'])
        with self.assertRaises(ValueError):quote(other,s)


if __name__=='__main__':unittest.main(verbosity=2)
