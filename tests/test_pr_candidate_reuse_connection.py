"""Artificial metadata tests: saved price first and published B2 alias reuse."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_candidate_completion as C,rpe_method_optimizer as O


def config(cid,kappa=2):
    return dict(id=cid,spec=dict(method='PR',L_D=6,formula='4th(m5_best)',finite_K=2,h=.125,
        kappa=dict(value=dict(numerator=kappa,denominator=1)),delta_pf=dict(numerator=1,denominator=4),
        law={'model':'ARTIFICIAL'},split=dict(lambda_R=1,lambda_source={'fixture':'same_tail'})),
        rounds=[dict(m=m,T=m+1,q_PF=m+1,R_outer=1,candidate={'id':'p'+str(m)},
            actual_delta_pf=dict(numerator=1,denominator=4),structure=dict(formula_id='fixed',signed_weights=[1.],allocation=[1],event_count=m+1))for m in range(13)])


def completion(tmp):
    c=C.Completion.__new__(C.Completion);c.directory=tmp/'completion';c.directory.mkdir()
    c.state={'records':{}};c.state_path=c.directory/'STATE.json';c.lock=threading.RLock();c.settings={};c.source='SOURCE'
    c.registry=c.directory/'REGISTRY.json';C.put(c.registry,{'entries':[]});return c


class ReuseTests(unittest.TestCase):
    def test_saved_complete_df_price_skips_unneeded_new_fit_design(self):
        with tempfile.TemporaryDirectory()as d:
            tmp=Path(d);c=completion(tmp);cfg=config('DF');events=[]
            quote=dict(complete=True,rows=[{'unit_price_RZ':dict(numerator=1,denominator=1)}for _ in range(26)])
            def service(mode,request,path,**kwargs):
                events.append(mode)
                if mode=='df-generate':return dict(status='CONFIGURATION_READY',configuration=cfg)
                if mode=='intake':return dict(binding_id='b',total_configuration_id='t')
                raise AssertionError('Unneeded new science/design: '+mode)
            signal=dict(packet_source={'commit':'PACKET','path':'packet.json'})
            with patch.object(C,'ROOT',tmp),patch.object(C,'publish',return_value='COMMIT'),\
                 patch.object(c,'service',side_effect=service),patch.object(c,'initial_quote',return_value=quote)as iq,\
                 patch.object(c,'df_signals',return_value=signal),patch.object(c,'price',return_value=quote)as price:
                result=c.complete(dict(method='PR_DF',configuration=cfg,selection={'kind':'CONFIGURATION_INPUTS_MISSING'}))
            self.assertEqual(result['status'],'READY_FOR_SEARCH')
            self.assertEqual(events,['df-generate','intake']);self.assertEqual(iq.call_count,1)
            self.assertIs(price.call_args.kwargs['initial_quote'],quote)

    def test_original_b2_register_reuses_ready_physical_twin_without_new_route(self):
        with tempfile.TemporaryDirectory()as d:
            tmp=Path(d);c=completion(tmp);original=config('OLD',7);requested=config('NEW',2)
            old=dict(method='PR_DF',family='PR',configuration_id='OLD',name='SAVED_OLD',key=dict(L_D=6),
                caps=[64]*13,ready=True,needs=[],prices=[Fraction(17)]*26,labels=['SAVED_Q3'],saved_points=[{'receipt':'KEPT'}],
                generated_configuration=original)
            placeholder=dict(method='PR_DF',family='PR',configuration_id=None,name='MISSING',key=dict(L_D=6),caps=[64]*13,ready=False)
            context=dict(candidates=[old,placeholder],catalog={'candidates':[]},expected={'OLD':{'binding_id':'ORIGINAL'}})
            record=dict(status='READY_FOR_SEARCH',method='PR_DF',configuration={'commit':'C','path':'new.json'},
                intake={'binding_id':'NEW_BINDING','total_configuration_id':'NEW_TOTAL'},signal={'packet_source':{'commit':'P','path':'packet'}},
                prices=[dict(numerator=999,denominator=1)]*26,quote={'commit':'Q','path':'quote.json'})
            existing_prices=list(old['prices']);saved_points=deepcopy(old['saved_points']);problem={'execution_routes':{'PR_DF':[{'original':'ROUTE'}]}}
            with patch.object(C,'value',return_value=requested),patch.object(C,'git',side_effect=AssertionError('No new route')):
                routes=c.install(context,problem,[record])
            self.assertEqual(record['installed_configuration_id'],'OLD')
            self.assertEqual(record['installation_disposition'],'PHYSICALLY_IDENTICAL_TO:OLD')
            self.assertEqual(context['candidates'],[old]);self.assertEqual(old['prices'],existing_prices)
            self.assertEqual(old['saved_points'],saved_points);self.assertEqual(old['labels'],['SAVED_Q3'])
            self.assertEqual(context['expected'],{'OLD':{'binding_id':'ORIGINAL'}});self.assertEqual(routes,problem['execution_routes'])
            self.assertEqual(old['physical_aliases'][-1]['configuration_id'],'NEW')
            self.assertEqual(old['schedule_keys'],O.schedule_keys(original))


    def test_repeat_install_retains_own_ready_route(self):
        with tempfile.TemporaryDirectory()as d:
            tmp=Path(d);c=completion(tmp);requested=config('NEW',2)
            placeholder=dict(method='PR_DF',family='PR',configuration_id=None,name='MISSING',key=dict(L_D=6),caps=[64]*13,ready=False)
            context=dict(candidates=[placeholder],catalog={'candidates':[]},expected={})
            record=dict(status='READY_FOR_SEARCH',method='PR_DF',configuration={'commit':'C','path':'new.json'},
                intake={'binding_id':'NEW_BINDING','total_configuration_id':'NEW_TOTAL'},signal={'packet_source':{'commit':'P','path':'packet'}},
                prices=[dict(numerator=999,denominator=1)]*26,quote={'commit':'Q','path':'quote.json'})
            problem={'execution_routes':{'PR_DF':[]}}
            def value(locator):return {'fixture':'quote'}if locator.get('commit')=='Q'else requested
            with patch.object(C,'value',side_effect=value),patch.object(C,'git',return_value='SOURCE'):
                first=c.install(context,problem,[record]);second=c.install(context,problem,[record])
            self.assertEqual(first,second);self.assertEqual(len(second['PR_DF']),1)
            self.assertEqual(second['PR_DF'][0]['configuration_ids'],['NEW'])
            self.assertEqual(len(context['candidates']),1);self.assertTrue(context['candidates'][0]['ready'])


if __name__=='__main__':unittest.main(verbosity=2)
