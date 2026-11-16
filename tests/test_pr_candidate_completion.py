"""Common B2 callback tests with artificial acquisition; no scientific providers."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import pr_candidate_completion as C,rpe_method_optimizer as O,rpe_priority_search as P


class CompletionTests(unittest.TestCase):
    def test_saved_ready_resume_does_not_call_acquisition(self):
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.directory=Path(tmp);c.state={'records':{}};c.lock=__import__('threading').RLock();c.state_path=Path(tmp)/'STATE.json'
            r=dict(method='PR_DF',configuration={'fixture':'ARTIFICIAL'})
            c.state['records'][C.key(r)]=dict(status='READY_FOR_SEARCH',configuration_id='ARTIFICIAL')
            with patch.object(c,'pauli_configuration',side_effect=AssertionError('no acquisition')):
                out=c.complete(r)
            self.assertTrue(out['resumed']);self.assertEqual(out['configuration_id'],'ARTIFICIAL')

    def test_published_v2_missing_order_not_reimplemented(self):
        candidates=[dict(name='PR_DF:'+str(n),ready=False,configuration_id=None,key=dict(L_D=n),needs=['SIGNAL'])for n in (9,6,8,5)]
        ready=dict(key=dict(L_D=7));states={'saved':type('State',(),{'c':ready,'best':(Fraction(10),None,None)})()}
        rows=P.new_ld_requests(candidates,states,Fraction(10),{}, {})
        self.assertEqual([r['configuration']['L_D']for r in rows],[6,8,5,9])
        self.assertTrue(all(r['kind']=='NEW_L_D'for r in rows))

    def test_absolute_kappa_is_translated_against_actual_base_without_fixing_width_formula(self):
        request=dict(method='PR_PAULI',design=dict(L_D=4,formula='4th(m5_best)',delta_over_h='2',kappa='2',K=2))
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.settings={}
            base=dict(spec=dict(kappa=dict(value=dict(numerator=7,denominator=1))))
            result=dict(spec=dict(kappa=dict(value=dict(numerator=2,denominator=1))))
            def call(cmd,d,tag,*args,**kwargs):return dict(items=[dict(status='GENERATED',configuration=base if tag=='A_BASE'else result)])
            with patch.object(c,'call',side_effect=call):config,_=c.pauli_configuration(request,Path(tmp))
            sent=json.loads((Path(tmp)/'A_DESIGN.REQUEST.json').read_text())['items'][0]['changes']
            self.assertEqual(sent['kappa_multiplier'],'2/7')
            self.assertEqual(sent['formula'],'4th(m5_best)');self.assertEqual(sent['delta_over_h'],'2')
            self.assertEqual(config,result)

    def test_normal_cli_contains_callback_and_df_route_is_generic(self):
        spec=importlib.util.spec_from_file_location('_completion_cli',ROOT/'scripts/run_rpe_method_optimizer.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        import inspect
        self.assertIn('prepare_for_search',inspect.getsource(cli.round_mode))
        spec=importlib.util.spec_from_file_location('_completion_q3',ROOT/'scripts/run_rpe_optimizer_q3_job.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
        inputs=dict(cases=[dict(evaluation_target={'fixture':1})],statistical_contract={},options={},adopted_evaluation={},acceleration={})
        r=dict(request_id='fixture',configuration_id='cfg',N_X=[1]*13,N_Y=[2]*13,route=dict(kind='prdf_d2_common_entry',contract_template={'fixture':1},configurations={'cfg':dict(configuration={'fixture':1},signal_packet={'fixture':2},binding_id='b',total_configuration_id='t')}))
        with patch.object(q,'git_json',return_value=inputs):case,actual=q.build(ROOT,r)
        self.assertEqual(case['signal_packet_kind'],'A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED')
        self.assertEqual(case['original_configuration_id'],'cfg');self.assertEqual(case['allocation']['shots'][-1]['N_Y'],2)


if __name__=='__main__':unittest.main(verbosity=2)
