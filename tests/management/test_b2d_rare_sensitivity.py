"""Toy scalars and mocked IO only; never read the B1 scientific dataset."""
from pathlib import Path
from decimal import Decimal as D,localcontext
from copy import deepcopy
import json,subprocess,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks.b2d_evaluator import rare_sensitivity as r

def toy():
    occurrences=[];stages=[]
    for i,sign in enumerate((1,-2)):
        tau=sign*.1;p,norm=r.finite_law(tau,2)
        spec=dict(duration=str(sign)+'/5',rte_steps=2,cutoff=2,seed=41+i)
        stages.append(dict(payload=spec))
        occurrences.append(dict(occurrence=i,spec=spec,dimensionless_step_time=tau,
            orders=[0,2],order_probabilities=[float(p[k]) for k in (0,2)],exact_finite_distribution=float(norm)))
    row=dict(candidate_id='B3',plan=dict(body=stages),case=dict(n_d=1,macro_steps=2),
        rte_parameters=dict(K=2,r_rte=2),independent_occurrences=True,cost=dict(RTE_short_steps=4))
    return row,occurrences

def mocked_analysis(root,ledger):
    return dict(probabilities={'toy':dict(p_S0='0.9',p_rare='0.1',canonical_p_rare_check='0.1',
        saved_canonical_mass_difference='0',independent_short_steps=1,observed_order0_steps=256)},
        boundaries=[dict(comparison='toy',T='1/5',axis='cosine',metric='rz_count',
            positive_candidate='pos',negative_candidate='neg',sensitivity=[dict(SE_corner=0)])])

class Tests(unittest.TestCase):
    def test_saved_signed_heterogeneous_sequence_product_and_normalization(self):
        with localcontext() as ctx:
            ctx.prec=50;row,occ=toy();p=r.probability(row,occ,D(1))
            expected=r.dec(occ[0]['order_probabilities'][0])**2*r.dec(occ[1]['order_probabilities'][0])**2
            self.assertEqual(p['p_S0'],expected);self.assertGreater(p['p_rare'],0)
            self.assertEqual(p['independent_short_steps'],4)
            self.assertFalse(p['normalization_changed'])

    def test_mismatched_signed_stage_and_law_rejected(self):
        row,occ=toy();bad=deepcopy(occ);bad[1]['spec']['duration']='1/5'
        with self.assertRaises(ValueError):r.probability(row,bad,D(1))
        bad=deepcopy(occ);bad[0]['order_probabilities'][0]=.5
        with self.assertRaises(ValueError):r.probability(row,bad,D(1))

    def test_tiny_positive_probability_preserved_and_K4_supported(self):
        with localcontext() as ctx:
            ctx.prec=50;p,norm=r.finite_law(D('1e-8'),4)
            self.assertGreater(p[4],0);self.assertGreater(1-p[0]**12,0)
            p2,_=r.finite_law(D('-1e-8'),2);self.assertNotIn(4,p2)

    def test_unknown_means_not_zero_filled_and_boundary_direction(self):
        b=r.boundary(D('.1'),D('.2'),dict(mean=D(100),SE=D(2)),dict(mean=D(40),SE=D(1)))
        point=b['sensitivity'][2];self.assertEqual(b['delta_S0'],D(58))
        self.assertEqual(point['u_negative_intercept'],D(290))
        self.assertEqual(point['u_negative_slope'],D('.5'))
        self.assertEqual(b['unknown_means'],dict(positive=None,negative=None))
        for up in (D(0),D(50)):
            un=point['u_negative_intercept']+point['u_negative_slope']*up
            self.assertEqual(b['delta_S0']+D('.1')*up-D('.2')*un,0)
            self.assertLess(b['delta_S0']+D('.1')*up-D('.2')*(un+1),0)
        self.assertFalse(b['nonnegative_cost_alone_fixes_sign'])

    def test_SE_corners_cover_componentwise_boxes_without_independence(self):
        b=r.boundary(D('.1'),D('.2'),dict(mean=D(40),SE=D(2)),dict(mean=D(45),SE=D(3)))
        self.assertEqual(b['delta_S0'],0);self.assertEqual(b['SE_box_width'],D('4.2'))
        self.assertEqual(b['delta_SE2'],[D('-8.4'),D('8.4')])
        self.assertEqual([s['SE_corner'] for s in b['sensitivity']],[-2,-1,0,1,2])
        self.assertIsNone(b['unconditional_ranking']);self.assertIsNone(b['covariance'])

    def test_scope_two_pairs_only_and_unregistered_call_rejected(self):
        s=r.scope();self.assertEqual(s['boundaries'],{'rare.boundary':16,'rare.probability':8})
        with self.assertRaisesRegex(Exception,'UNREGISTERED_FOLLOWUP_CALL'):
            r.FollowupLedger(s).call('rare.probability',lambda:None,identity='F1:9/50')

    def test_mock_success_failure_fresh_output_and_STOP(self):
        with tempfile.TemporaryDirectory() as directory:
            run=dict(root=Path(directory),output=Path(directory)/'fresh',source='MOCK',input_blobs={},scope=r.registered_scope('MOCK',[]))
            status=r.execute(run,mock_only=True,analysis=mocked_analysis)
            self.assertEqual(status['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP')
            self.assertEqual(status['mode'],'MOCK_ONLY')
            self.assertTrue((run['output']/'STOP.json').exists())
            with self.assertRaises(FileExistsError):r.execute(run,mock_only=True,analysis=mocked_analysis)
            run['output']=Path(directory)/'failed'
            def fail(*a):raise RuntimeError('toy interruption')
            status=r.execute(run,mock_only=True,analysis=fail)
            self.assertEqual(status['status'],'FAILED_MANDATORY_STOP')
            self.assertTrue((run['output']/'failure.json').exists())

    def test_import_and_inspect_do_not_load_physics_or_start_analysis(self):
        root=Path(__file__).resolve().parents[2]
        code="import sys;sys.path.insert(0,'src');from trottertracks.b2d_evaluator import rare_sensitivity as r;r.scope();assert not any(m.split('.')[0] in ('numpy','scipy','qiskit','pyscf','mpmath') for m in sys.modules)"
        p=subprocess.run([sys.executable,'-I','-B','-S','-c',code],cwd=root,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run([sys.executable,'-I','-B','-S','scripts/run_b2d_outer_rare_sensitivity.py'],cwd=root,text=True,capture_output=True)
        self.assertNotEqual(p.returncode,0);self.assertIn('required',p.stderr)

if __name__=='__main__':unittest.main()
