"""Artificial PASS/FAIL receipts -> unchanged generic C PF consumer; no Q3."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import publish_pf_single_q3 as P
from trottertracks import pf_candidate_space as SPACE
from trottertracks.saved_rpe_cost_catalog import Catalog,Reader


class Artificial(unittest.TestCase):
    def fixture(self,status):
        cfg=SPACE.configure(ROOT,'FULL_PF',10,'2nd','2')
        cid=cfg['id'];raw='a'*40;price_commit='b'*40;input_commit='c'*40;published='d'*40
        confsource=P.ref('e'*40,'fixtures/CONFIGURATION.json');signals=P.ref('e'*40,'fixtures/SIGNALS.json')
        prices=P.ref('e'*40,'fixtures/PRICES.json');study='docs/research/ARTIFICIAL_SINGLE_PF';label='AUTO_REQUEST_FROM_NORMAL_OPTIMIZER'
        job=study+'/run01/jobs/FULL_PF/'+label;allocation=dict(allocation_id='RPE_OPT_'+label,shots=[dict(m=m,N_X=1,N_Y=1)for m in range(13)])
        target=dict(kind='SYNTHETIC_METADATA_TARGET',energy_ha=0.)
        case=dict(label=label,configuration_id=cid,configuration=confsource,signal_packet=signals,
            total_view_id='ARTIFICIAL_TOTAL',binding_id='ARTIFICIAL_BINDING',evaluation_target=target,allocation=allocation)
        adopted=dict(estimator='SYNTHETIC_METADATA_DECODER',target='SYNTHETIC',loss='ALL_EVENT_SQUARE',MSE_upper=P.enc('1/10'))
        inputs=dict(cases=[case],adopted_evaluation=adopted,statistical_contract=dict(model='SYNTHETIC_ONLY'),options={})
        bound=P.enc('1/20' if status==P.PASS else '1/5')
        record=dict(configuration_id='ARTIFICIAL_TOTAL',canonical_binding='ARTIFICIAL_BINDING',allocation=allocation,
            evaluation_target=target,diagnostic=dict(deterministic_risk=dict(MSE_lower=bound,MSE_upper=bound)))
        result=dict(status=status,record=record,adopted_q3_assessment=dict(status=status),scientific_evaluation_calls=1)
        outcome=dict(request_id=label,route='pf_b_pf6',status=status,MSE_lower=bound,MSE_upper=bound,scientific_evaluation_calls=1,
            identity=dict(configuration_id=cid,total_configuration_id='ARTIFICIAL_TOTAL',binding_id='ARTIFICIAL_BINDING',
                contract_matches=True,signal_packet=signals,allocation=allocation['shots']))
        request=dict(request_id=label,method='FULL_PF',configuration_id=cid,route=dict(kind='pf_b_pf6'),
            N_X=[1]*13,N_Y=[1]*13,cost=P.enc(260))
        physical={m:'ARTIFICIAL_NATIVE_%d'%m for m in range(13)}
        axes=[dict(m=m,axis=a,physical_id=physical[m],unit_price_RZ=P.enc(10),
            kind='DIRECT_COMPILED_DETERMINISTIC' if m<5 else 'CONDITIONAL_AFFINE_PRICE_REFERENCE',
            price_evidence=dict(source='ARTIFICIAL_ONLY',model_error_bound_RZ=None))for m in range(13)for a in ('X','Y')]
        entry=dict(configuration=confsource,signal_packet=signals,prices=prices,total_configuration_id='ARTIFICIAL_TOTAL',
                   binding_id='ARTIFICIAL_BINDING',price_scope='ARTIFICIAL_CONDITIONAL_MODEL_ONLY')
        docs={(P.D_SOURCE,P.REGISTRY['path']):dict(configurations={cid:entry}),
              (confsource['commit'],confsource['path']):cfg,(prices['commit'],prices['path']):dict(axes=axes)}
        for name,data in [('REQUEST',request),('CASE_INPUTS',inputs),('OUTCOME',outcome),
                          (label+'/RESULT',result),(label+'/API_RETURN',[record])]:docs[raw,job+'/'+name+'.json']=data
        def read(root,source):
            key=source['commit'],source['path']
            if key not in docs:raise FileNotFoundError(str(key))
            value=deepcopy(docs[key])
            for s in source.get('selector',[]):value=value[s]
            return value
        def origin(root,commit,path):
            path=str(path)
            return P.ref(price_commit if path.endswith('/PRICED_AXES.json')else input_commit if path.endswith('/C_INPUTS.json')else raw,path)
        return dict(cfg=cfg,cid=cid,raw=raw,price_commit=price_commit,input_commit=input_commit,published=published,study=study,
            job=job,docs=docs,read=read,origin=origin,physical=physical,adopted=adopted,target=target)

    def run_publish(self,status):
        f=self.fixture(status)
        with TemporaryDirectory(prefix='single_pf_publication_artificial_')as directory:
            root=Path(directory)
            with patch.object(P,'_read',f['read']),patch.object(P,'_origin',f['origin']),patch.object(P,'_files',return_value=[f['job']+'/OUTCOME.json']),patch.object(P.PF,'physical_ids',return_value=f['physical']):
                first=P.publish(root,f['study'],f['raw'])
                self.assertEqual(first['phase'],'PRICE_LEDGER')
                pricepath=f['study']+'/PRICED_AXES.json';price=json.loads((root/pricepath).read_text())
                f['docs'][f['price_commit'],pricepath]=price;f['docs'][f['input_commit'],pricepath]=price
                second=P.publish(root,f['study'],f['price_commit'])
                self.assertEqual(second['phase'],'PRICED_INPUT_VIEW')
                inputpath=f['study']+'/C_INPUTS.json';inputs=json.loads((root/inputpath).read_text())
                f['docs'][f['input_commit'],inputpath]=inputs
                third=P.publish(root,f['study'],f['input_commit'])
                self.assertEqual(third['status'],'PUBLISHED_GENERIC_PF_JOIN')
                for filename in ('SUMMARY.json','C_RESULT_JOIN.json'):
                    f['docs'][f['published'],f['study']+'/'+filename]=json.loads((root/f['study']/filename).read_text())
                self.assertEqual(P.publish(root,f['study'],f['input_commit']),third)
            class ArtificialReader(Reader):
                def commit(self,value):return value
                def read(self,loc):return f['read'](root,loc)
                def document(self,commit,path):return f['read'](root,P.ref(commit,path))
            cfg=f['cfg'];comparison=dict(binding={k:cfg['spec']['binding'][k]for k in ('hamiltonian','state','snapshot')},
                h=cfg['spec']['h'],M=12,compiler_profile=cfg['spec']['profile'],accuracy=f['adopted'],target=f['target'])
            catalog=Catalog(ArtificialReader(root),comparison)
            catalog.add(P.ref(f['published'],f['study']+'/C_RESULT_JOIN.json'))
            output=catalog.result();self.assertEqual(len(output['candidates']),1)
            row=output['candidates'][0]
            self.assertEqual(row['configuration_id'],f['cid']);self.assertEqual(row['N_X'],[1]*13)
            self.assertEqual(len(row['price_axes']),26)
            self.assertEqual(sum(p['price_kind']=='DIRECT_COMPILED_DETERMINISTIC'for p in row['price_axes']),10)
            self.assertEqual(row['conditional_cost'],P.enc(260));self.assertEqual(row['q3_status'],status)
            self.assertEqual(row['MSE_lower'],{k:str(v)for k,v in f['docs'][f['raw'],f['job']+'/OUTCOME.json']['MSE_lower'].items()})
            self.assertTrue(all(x==0 for x in (output['new_signal'],output['new_compile'],output['new_Q3'])))
            return row,output,price

    def test_pass_consumer_is_feasible_with_conditional_quote(self):
        row,out,price=self.run_publish(P.PASS)
        self.assertEqual(row['status'],'FEASIBLE')
        self.assertEqual(out['method_minima']['FULL_PF']['conditional_cost'],P.enc(260))
        self.assertEqual(price['achieved_conditional_cost_RZ'],P.enc(260))
        self.assertIsNone(price['model_error_bound_RZ'])

    def test_fail_consumer_is_infeasible_quote_is_not_achieved(self):
        row,out,price=self.run_publish(P.FAIL)
        self.assertEqual(row['status'],'INFEASIBLE')
        self.assertEqual(row['q3_failure_kind'],'MODEL_THRESHOLD_EXCEEDED_NOT_NUMERICAL_FAILURE')
        self.assertIsNone(out['method_minima']['FULL_PF']['conditional_cost'])
        self.assertIsNone(price['achieved_conditional_cost_RZ'])
        self.assertEqual(price['conditional_model_quote_RZ'],P.enc(260))

    def test_substituted_request_or_api_is_rejected(self):
        for mutation in ('shots','API'):
            f=self.fixture(P.PASS)
            if mutation=='shots':f['docs'][f['raw'],f['job']+'/REQUEST.json']['N_X'][0]=2
            else:f['docs'][f['raw'],f['job']+'/AUTO_REQUEST_FROM_NORMAL_OPTIMIZER/API_RETURN.json'][0]['canonical_binding']='WRONG'
            with patch.object(P,'_read',f['read']),patch.object(P,'_origin',f['origin']),patch.object(P,'_files',return_value=[f['job']+'/OUTCOME.json']),patch.object(P.PF,'physical_ids',return_value=f['physical']),self.assertRaises(ValueError):
                P.publication_views(ROOT,f['study'],f['raw'])


if __name__=='__main__':unittest.main(verbosity=2)
