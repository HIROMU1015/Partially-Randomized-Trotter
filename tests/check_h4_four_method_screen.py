"""Artificial matrices and passive archived fields only; no real H4 run."""
from copy import deepcopy
import json
import hashlib
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
import numpy as np
from scipy.linalg import expm
from trottertracks import h4_four_method_screen as S
from trottertracks import h4_four_method_screen_mean as N
from trottertracks import pr_rpe_power_mean as old
from trottertracks import h4_pr_rmse_completion as completion
from trottertracks import pr_rpe_rmse as rmse


def artificial(commuting=False):
    z=np.diag([1.,-1.]).astype(complex);x=np.array([[0.,1.],[1.,0.]],complex)
    matrices=[(.001+.0001*i)*(z if commuting or i%2==0 else x) for i in range(11)]
    state=np.array([1.,.2j],complex);state/=np.linalg.norm(state)
    ctx=SimpleNamespace(state=state,matrices=matrices,ham=SimpleNamespace(constant=.07))
    preparation=dict(one_body_policy='RANDOMIZED',scalar=dict(extracted_identity=.013),lambda_R=.9)
    view=N.context(ctx,root=ROOT,preparation=preparation,tail7=dict(scalar=.011))
    return ctx,view,preparation


def reference(ctx,T):
    a=sum(ctx.matrices,np.zeros_like(ctx.matrices[0]))+ctx.ham.constant*np.eye(2)
    z=complex(np.vdot(ctx.state,expm(-1j*T*a)@ctx.state));record=dict(real=z.real,imag=z.imag)
    return dict(reference=record,independent_reference=record,signal_difference=0.)


class ScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=S.plan(ROOT)

    def test_passive_plan_exact20_saved40_and_source_bound_full_h(self):
        with (patch.object(completion,'load_context',side_effect=AssertionError('real arrays')),
              patch.object(S,'evaluate_cell',side_effect=AssertionError('science'))):
            planned=S.plan(ROOT)
        self.assertEqual(len(planned['fresh']),20);self.assertEqual(len(planned['reuse']),40)
        self.assertEqual(planned['maximum_calls']['new_full_H_reference_actions'],0)
        self.assertEqual(planned['full_h_preparation']['one_body_policy'],'RANDOMIZED')
        self.assertEqual(planned['full_h_preparation']['one_body_components'],8)
        self.assertEqual(planned['science_calls'],0)

    def test_wrong_scope_or_saved_duplicate_or_law_is_rejected(self):
        for variant in ('width','prefix','8th','kappa','allocation','law','time','binding','duplicate'):
            p=deepcopy(self.plan)
            if variant in ('width','prefix','8th'):
                row=p['fresh'][0];c=row['candidate']
                if variant=='width':c['q']+=1
                elif variant=='prefix':c['prefix']=9
                else:c['formula']='8th(Yoshida)'
                c['id']=S.p.candidate_id(c)
            elif variant=='kappa':p['fresh'][-1]['kappa']='1'
            elif variant=='allocation':p['fresh'][-1]['structure']['allocation']=[1]
            elif variant=='law':p['fresh'][-1]['structure']['law_model']='DF_B3'
            elif variant=='time':p['fresh'][0]['reference']['T']=.8
            elif variant=='binding':p['fresh'][0]['reference']['binding']['state']='foreign'
            else:p['reuse'].append(dict(physical_candidate=deepcopy(p['fresh'][0]['candidate'])))
            with self.subTest(variant=variant),self.assertRaises(ValueError):S.validate(p)

    def test_final_D_physical_schedule_matches_all18_Full_PF_and_RTE_cells(self):
        api=S.physical_api(ROOT);scope=dict(vars(api))
        scope['SOURCES']=dict(scope['SOURCES'],pf_native=dict(commit='af284d853890155bacd19e61850c196bd1d5c660',path='scripts/h46_long_time_native.py'))
        scope['FORMULAS']={r['candidate']['formula']:dict(formula_id=r['structure']['formula_id']) for r in self.plan['fresh'] if r['comparison_method']=='FULL_PF'}
        scope['B']=SimpleNamespace(_positive=Fraction,_ceil=math.ceil,comparison_method=lambda s:'FULL_PF',
            a=SimpleNamespace(exact_value=lambda x:dict(numerator=x.numerator,denominator=x.denominator)))
        for name in ('_layouts','layout_mapping','physical_schedule'):
            S.M.function(ROOT,'e7dca5f7301c8633b17a59b6a7161331303aef2d',S.D_PATH,name,scope)
        for row in self.plan['fresh']:
            if row['comparison_method']=='TRUNCATED_PF':continue
            c=row['candidate'];full=row['comparison_method']=='FULL_H_RTE'
            spec=dict(method='FULL_H_RTE' if full else 'PF',binding=self.plan['binding'],profile=self.plan['profile'],
                h=.1,M=12,delta_pf=row.get('requested_delta'),formula=c['formula'],L_D=c['prefix'],finite_K=c['K'],kappa=None,
                split=dict(deterministic_scope='FULL_H',lambda_R=self.plan['full_h_preparation']['lambda_R'],preparation=self.plan['full_h_preparation']),
                event_mapping=dict(kind='FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1',source={'fixture':'METADATA_ONLY'}),law={'model':'ideal_finite_RTE_law'})
            kwargs=dict(layout_by_round={row['m']:dict(q=c['q'],R_outer=c['R_outer'])}) if full else {}
            physical=scope['physical_schedule'](spec,**kwargs)['rounds'][row['m']]['physical_candidate']
            self.assertEqual(physical,c)

    def test_pf_and_B0I_power_match_original_signed_vector_propagation(self):
        saved,ctx,prep=artificial()
        for formula in ('2nd','4th','4th(m5_best)'):
            for method,prefix in (('FULL',10),('B0I',7)):
                c=dict(system='H4',method=method,prefix=prefix,formula=formula,T=.8,q=2,delta=.4,R_outer=None,K=None)
                c['id']=S.p.candidate_id(c);st=S.pf_structure(c);row=dict(candidate=c,structure=st,comparison_method='FULL_PF' if method=='FULL' else 'TRUNCATED_PF')
                ref=reference(saved,c['T']);orig=SimpleNamespace(state=saved.state,ham=saved.ham,specs=ctx.specs,
                    tails={7:dict(scalar=.011)},refs={c['T']:complex(ref['reference']['real'],ref['reference']['imag'])})
                prior,_,_=old.kernels(ROOT).evolve(orig,[c])
                result=S.evaluate_cell(ctx,row,ref,root=ROOT,preparation=prep)
                for key in ('raw','corrected'):
                    a=prior[0]['signal'][key];b=result['signal'][key]
                    self.assertLess(abs(complex(a['real'],a['imag'])-complex(b['real'],b['imag'])),1e-12)
                self.assertEqual(result['signal']['log_B'],0.)

    def test_full_h_grouped_power_matches_original_prefix_zero_no_head(self):
        saved,ctx,prep=artificial()
        for K in (2,4):
            row=deepcopy(next(r for r in self.plan['fresh'] if r['comparison_method']=='FULL_H_RTE' and r['m']==6 and r['candidate']['K']==K))
            c=row['candidate'];ref=reference(saved,c['T'])
            orig=SimpleNamespace(state=saved.state,ham=saved.ham,specs=(),
                tails={0:dict(scalar=.013,lam=.9,spec=ctx.full_spec)},refs={c['T']:complex(ref['reference']['real'],ref['reference']['imag'])})
            prior,_,_=old.kernels(ROOT).evolve(orig,[c])
            result=S.evaluate_cell(ctx,row,ref,root=ROOT,preparation=prep)
            self.assertEqual(prior[0]['signal']['log_B'],result['signal']['log_B'])
            for key in ('raw','corrected'):
                a=prior[0]['signal'][key];b=result['signal'][key]
                self.assertLess(abs(complex(a['real'],a['imag'])-complex(b['real'],b['imag'])),1e-11)

    def test_all20_including_long_time_artificial_are_finite_no_N_or_prices(self):
        saved,ctx,prep=artificial(True)
        with patch.object(rmse,'nominal_axis_shots',side_effect=AssertionError('N')):
            records=[S.evaluate_cell(ctx,r,reference(saved,r['candidate']['T']),root=ROOT,preparation=prep) for r in self.plan['fresh']]
        self.assertEqual(len(records),20)
        self.assertEqual(sum(r['finite_normalizer_calls'] for r in records),6)
        for r in records:
            self.assertTrue(r['numerical']['gates_passed']);self.assertIsNone(r['N_X']);self.assertIsNone(r['global_RMSE_bound'])
            self.assertIsNone(r['scientific_PASS_FAIL'])
            if r['candidate'].get('family')=='FULL_H_PURE_RTE':
                self.assertTrue(r['raw_corrected_independent_propagation'])
                self.assertIsNone(r['allowance']['full_h_exact_tail_norm_observation'])
            else:self.assertTrue(r['raw_equals_corrected_at_B1'])

    def test_gate_failure_normalizer_precision_and_B3_are_not_repaired(self):
        saved,ctx,prep=artificial();row=self.plan['fresh'][0];ref=reference(saved,row['candidate']['T'])
        bad=dict(ref,signal_difference=1e-3)
        with self.assertRaises(ValueError):S.evaluate_cell(ctx,row,bad,root=ROOT,preparation=prep)
        bad=deepcopy(ref);bad['reference']['real']=float('nan')
        with self.assertRaises(ValueError):S.evaluate_cell(ctx,row,bad,root=ROOT,preparation=prep)
        row=next(r for r in self.plan['fresh'] if r['comparison_method']=='FULL_H_RTE')
        with patch('trotterlib.rte.finite_rte_distribution',return_value=SimpleNamespace(exact_finite_distribution=1.)):
            with self.assertRaises(ValueError):S.evaluate_cell(ctx,row,reference(saved,row['candidate']['T']),root=ROOT,preparation=prep)
        bad=dict(prep,one_body_policy='DETERMINISTIC')
        with self.assertRaises(ValueError):S.evaluate_cell(ctx,row,reference(saved,row['candidate']['T']),root=ROOT,preparation=bad)

    def test_default_cli_no_execute_and_no_scientific_library_import(self):
        with tempfile.TemporaryDirectory(prefix='four-method-plan-') as tmp:
            path=Path(tmp)/'plan.json'
            command=[sys.executable,'-B','scripts/prepare_h4_four_method_screen.py','--out',str(path)]
            subprocess.run(command,cwd=ROOT,check=True,capture_output=True)
            self.assertEqual(json.loads(path.read_text())['science_calls'],0)
            self.assertNotEqual(subprocess.run(command+['--execute'],cwd=ROOT,capture_output=True).returncode,0)
        subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'src');from trottertracks import h4_four_method_screen;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit'))"],cwd=ROOT,check=True)

    def test_mock_batch_failure_saves_partial_STOP_without_retry_or_real_arrays(self):
        original_read=S.p.SavedArtifacts.read;original_blob=S.M.blob
        with tempfile.TemporaryDirectory(prefix='four-method-mock-STOP-') as tmp:
            context=Path(tmp)/'context.json'
            context.write_text(json.dumps(dict(binding=self.plan['binding'],prefixes={'7':{'scalar':.1}})))
            index=dict(T20_reference_files=dict(H4=dict(path=str(context),sha256=hashlib.sha256(context.read_bytes()).hexdigest())))
            def read(store,ref):return index if ref==completion.INDEX else original_read(store,ref)
            def blob(root,commit,path):
                return (ROOT/path).read_bytes() if commit=='SYNTHETIC_SOURCE' and path in (S.MODULE,S.MEAN) else original_blob(root,commit,path)
            saved=SimpleNamespace(ham=SimpleNamespace(constant=self.plan['full_h_preparation']['scalar']['constant']))
            with (patch.object(S.M,'blob',side_effect=blob),patch.object(S.p.SavedArtifacts,'read',read),
                  patch.object(completion,'load_context',return_value=(saved,{})),patch.object(N,'context',return_value=object()),
                  patch.object(N,'deterministic_step'),patch('os.sched_getaffinity',return_value={0}),
                  patch.object(S,'evaluate_cell',side_effect=ValueError('SYNTHETIC_NUMERICAL_GATE')) as leaf):
                out=S.execute(ROOT,Path(tmp)/'run','SYNTHETIC_SOURCE')
            self.assertEqual(leaf.call_count,1)
            self.assertTrue(out['status'].startswith('STOP'));self.assertFalse(out['automatic_retry'])
            self.assertEqual(out['new_reference_actions'],0);self.assertEqual(out['records'],[])
            self.assertTrue((Path(tmp)/'run/result.json').exists())


if __name__=='__main__':unittest.main()
