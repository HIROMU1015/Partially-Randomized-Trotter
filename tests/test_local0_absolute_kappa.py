import json
from fractions import Fraction as F
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import rpe_schedule_search as W
from trottertracks import rpe_method_optimizer as O
from trottertracks.rpe_local0_search import request_service,resolve,RequestError


class BoundsTests(unittest.TestCase):
    def family(self, c=1):
        return W.Family('f',[dict(m=0,c=F(c),q=1,J=1)])

    def test_closed_boundary_schedules_and_unique_physical_representatives(self):
        f=self.family(F(1,8))
        proposals,_,report=W.plan_wide({'f':f},[],dict(kappa_bounds=['2','64']))
        self.assertTrue(all(F(2)<=p['kappa']<=64 for p in proposals))
        schedules=[f.R(p['kappa']) for p in proposals]
        self.assertIn(f.R(2),schedules)
        self.assertIn(f.R(64),schedules)
        self.assertEqual(len(schedules),len(set(schedules)))
        self.assertEqual(report['families']['f']['requested_kappa_bounds'],[F(2),F(64)])

    def test_single_point_interval_keeps_boundary(self):
        proposals,_,_=W.plan_wide({'f':self.family()},[],dict(kappa_bounds=['4','4']))
        self.assertEqual([p['kappa'] for p in proposals],[F(4)])

    def test_boundary_below_one_and_above_default_upper_are_not_clamped(self):
        for lo,hi in [('1/100','3/100'),('131072','262144')]:
            proposals,_,_=W.plan_wide({'f':self.family()},[],dict(kappa_bounds=[lo,hi]))
            self.assertTrue(proposals)
            self.assertTrue(all(F(lo)<=p['kappa']<=F(hi)for p in proposals))

    def test_saved_same_schedule_does_not_become_another_candidate(self):
        f=self.family(F(1,8))
        points=[dict(family='f',kappa=F(3),status='PENDING',best=None)]
        proposals,_,_=W.plan_wide({'f':f},points,dict(kappa_bounds=['2','64']))
        self.assertNotIn(f.R(3),[f.R(p['kappa'])for p in proposals])

    def test_refinement_and_extensions_stay_in_requested_interval(self):
        f=self.family()
        points=[dict(family='f',kappa=F(k),status='FEASIBLE',best=F(cost))for k,cost in [(2,10),(8,5),(64,12)]]
        for _ in range(5):
            proposals,_,_=W.plan_wide({'f':f},points,dict(kappa_bounds=['2','64']))
            self.assertTrue(all(2<=p['kappa']<=64 for p in proposals))
            points += [dict(p,status='PENDING',best=None)for p in proposals]

    def test_saved_and_newly_installed_candidates_share_the_execution_filter(self):
        problem=dict(methods={'PR_PAULI':dict(L_D_values=[4,5,6,7],absolute_kappa_bounds=['2','64'])})
        catalog=[dict(configuration_id=str(k),kappa={'value':str(k)})for k in [1,2,64,65]]
        context=dict(catalog=dict(candidates=catalog),candidates=[dict(method='PR_PAULI',configuration_id=str(k),key=dict(L_D=5))for k in [1,2,64,65]])
        O.scope_absolute_kappa(context,problem)
        self.assertEqual([c['configuration_id']for c in context['candidates']],['2','64'])
        context['candidates'] += [dict(method='PR_PAULI',configuration_id='installed',key=dict(L_D=5),kappa_value='65'),
                                  dict(method='PR_PAULI',configuration_id='wrongLD',key=dict(L_D=3),kappa_value='4')]
        O.scope_absolute_kappa(context,problem)
        self.assertEqual([c['configuration_id']for c in context['candidates']],['2','64'])

    def test_gateway_selects_schedule_and_preserves_absolute_exact_bounds(self):
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'P.json';r=Path(directory)/'R.json'
            p.write_text(json.dumps(dict(problem=dict(hamiltonian_data=dict(expected=dict(fragment_count=10))),rpe=dict(M=0),
                methods=dict(PR_PAULI=dict(family='PR',L_D_range=dict(min=1,max_offset_from_fragment_count=-1))),search=dict(max_steps_per_method=10))))
            r.write_text(json.dumps(dict(methods=['PR_PAULI'],max_new_q3_per_method=dict(PR_PAULI=0))))
            config=dict(problem=str(p),round=str(r),entry='method',absolute_kappa_route=dict(entry='schedule'))
            request=dict(action='planonly',scope=dict(methods=['PR_PAULI'],per_method=dict(PR_PAULI=dict(L_D=dict(min=4,max=7),kappa=dict(mode='absolute',min='2',max='64'))),shots='auto'))
            selected=request_service(config,request)
            _,spec,canonical=resolve(selected,request)
            self.assertEqual(selected['entry'],'schedule')
            self.assertEqual(spec['method_overrides']['PR_PAULI']['L_D_values'],[4,5,6,7])
            self.assertEqual(spec['method_overrides']['PR_PAULI']['absolute_kappa_bounds'],['2','64'])
            self.assertEqual(spec['search_overrides']['schedule_wide']['kappa_bounds'],['2','64'])
            self.assertEqual(canonical['per_method']['PR_PAULI']['kappa']['mode'],'absolute')
            request['scope']['per_method']['PR_PAULI']['kappa']['mode']='multiplier'
            with self.assertRaises(RequestError):resolve(request_service(config,request),request)


if __name__=='__main__':unittest.main(verbosity=2)
