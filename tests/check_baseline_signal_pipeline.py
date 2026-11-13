"""Artificial matrices/metadata, fixed SOURCE API; never real H4 inputs."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from trottertracks import baseline_signal_pipeline as P
from trottertracks import baseline_signal_numerics as N

ROOT = Path(__file__).resolve().parents[1]
API = P.published_api(str(ROOT))
SOURCE = dict(fixture='ARTIFICIAL_ONLY')


def fixture(mode='FULL', formula='2nd', prefix=10, M=2, K=2, grouping=(1, 2)):
    binding = dict(system='H4', rank=10, hamiltonian='ARTIFICIAL_H', state='ARTIFICIAL_STATE',
                   snapshot='ARTIFICIAL_SNAPSHOT', decomposition='ARTIFICIAL_DF', source=SOURCE)
    profile = deepcopy(API.B.p.PROFILE)
    policy = dict(family='ARTIFICIAL', id='ARTIFICIAL_POLICY', source=SOURCE,
                  target_rmse_ha=1, error_contract=SOURCE, scientific_status='PROPOSED_NOT_ADOPTED')
    spec = dict(binding=binding, profile=profile, h=dict(numerator=1, denominator=10), M=M,
        method='PF', L_D=prefix, formula=formula, delta_pf=dict(numerator=1, denominator=20),
        finite_K=None, kappa=None, rmse_policy=policy,
        split=dict(L_D=prefix, source=SOURCE, randomized_scope='NONE', deterministic_scope='FULL_H',
            original_PF_method=mode, retained_hamiltonian=dict(construction='FULL_H', source=SOURCE,
            prefix=prefix, target_hamiltonian=binding['hamiltonian'])),
        law=dict(model='deterministic', source=SOURCE), event_mapping=dict(kind='NONE', source=SOURCE))
    layouts = None
    if mode in ('B0', 'B0I'):
        spec['split'].update(deterministic_scope='TRUNCATED_H')
        spec['split']['retained_hamiltonian']['construction'] = mode
    elif mode == 'FULL_H_RTE':
        prep = dict(representation='DF', source=dict(API.D.SOURCES['full_h_prepare'], function='prepare_full_h'),
            one_body_policy='RANDOMIZED', one_body_components=2, deterministic_blocks=0,
            randomized_fragments=list(range(10)), all_fragments=10, lambda_R=.9, one_body_lambda_R=.1,
            lambda_source=SOURCE, normalization_policy='full_h_df_plus_one_body_exact_Z',
            scalar=dict(constant=.07, extracted_identity=.013, total=.083,
                        application='ONCE_AT_TOTAL_T', identity_components_randomized=0),
            preparation_hash='ARTIFICIAL_FULL_H_PREPARATION', binding=binding)
        spec = API.D.full_h_rte_spec(spec, M=M, finite_K=K, preparation=prep)
        layouts = {m:dict(q=grouping[0], R_outer=grouping[1]) for m in range(M+1)}
    config = P.configure(ROOT, spec, layouts=layouts)
    target = dict(quantity='ORIGINAL_DF_H_TARGET_ENERGY', binding=binding, source=SOURCE, energy_shift_ha=0)
    refs = [dict(m=r['m'], T=r['T'], binding=binding, source=dict(SOURCE, reference_round=r['m'])) for r in config['rounds']]
    prepared = P.prepare(ROOT, config, target=target, references=refs)
    x = np.array([[0, 1], [1, 0]], complex); z = np.diag([1., -1.]).astype(complex)
    matrices = [(.001+.0001*i)*(z if i%2 else x) for i in range(11)]
    saved = NS(matrices=matrices, state=np.array([1., .2j])/math.sqrt(1.04), ham=NS(constant=.07))
    ctx = N.Context(ROOT, saved, binding, dict(prefixes={str(p):dict(scalar=.003+p*.001) for p in range(10)}))
    def reference(desc):
        T = float(API.B.a.exact_scalar(desc['T']))
        value = complex(np.vdot(ctx.state, expm(-1j*T*(sum(matrices)+.07*np.eye(2))) @ ctx.state))
        return dict(desc, reference=dict(real=value.real, imag=value.imag),
                    independent_reference=dict(real=value.real, imag=value.imag), signal_difference=0.)
    return prepared, ctx, reference


class Tests(unittest.TestCase):
    def test_metadata_import_has_no_numerical_or_scientific_side_effects(self):
        command = [sys.executable, '-B', '-c',
            "import sys;sys.path.insert(0,'src');from trottertracks import baseline_signal_pipeline as p;"
            "p.published_api('.');assert not any(v in sys.modules for v in ('numpy','scipy','qiskit','pyscf'))"]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)

    def test_formulas_prefixes_all_rounds_and_original_target(self):
        for formula, pieces in [('2nd',1), ('4th',3), ('4th(m5_best)',11)]:
            plan, ctx, ref = fixture(formula=formula, M=12)
            self.assertEqual(len(plan['requests']), 13)
            for req in plan['requests']:
                self.assertEqual(len(req['structure']['signed_weights']), pieces)
                self.assertEqual(req['physical_candidate']['prefix'], 10)
                self.assertEqual(req['q_RPE'], 1 << req['m'])
                self.assertEqual(req['q_PF'], 2*(1 << req['m']))
            if pieces > 1:self.assertTrue(any(w < 0 for w in plan['requests'][0]['structure']['signed_weights']))
            for mode in ('B0', 'B0I'):
                cut, _, _ = fixture(mode, formula, prefix=2)
                self.assertNotEqual(cut['configuration']['id'], plan['configuration']['id'])
                self.assertEqual(cut['target']['binding']['hamiltonian'], 'ARTIFICIAL_H')

    def test_B0I_scalar_for_non7_prefix_and_cache_reuse(self):
        b0, ctx, reader = fixture('B0', prefix=2)
        b0i, _, _ = fixture('B0I', prefix=2)
        for a, b in zip(b0['requests'], b0i['requests']):
            ra = N.evaluate(ctx, a, reader(a['domain']['reference']), source=SOURCE)
            cached = len(ctx.half), len(ctx.steps)
            rb = N.evaluate(ctx, b, reader(b['domain']['reference']), source=SOURCE)
            self.assertEqual(ra['status'], 'NUMERICAL_MEAN_AVAILABLE')
            self.assertEqual(rb['status'], 'NUMERICAL_MEAN_AVAILABLE')
            za = complex(ra['signal']['raw']['real'], ra['signal']['raw']['imag'])
            zb = complex(rb['signal']['raw']['real'], rb['signal']['raw']['imag'])
            self.assertLess(abs(zb - za*np.exp(-1j*a['physical_candidate']['T']*.005)), 1e-13)
            self.assertEqual(cached, (len(ctx.half), len(ctx.steps)))
            self.assertTrue(rb['numerical']['scalar_once'])

    def test_full_H_K_grouping_raw_corrected_and_no_price_equivalence(self):
        for K in (0, 2, 4, 6):
            a, ctx, reader = fixture('FULL_H_RTE', K=K, grouping=(1, 6))
            b, _, _ = fixture('FULL_H_RTE', K=K, grouping=(3, 2))
            ra = N.evaluate(ctx, a['requests'][0], reader(a['requests'][0]['domain']['reference']), source=SOURCE)
            rb = N.evaluate(ctx, b['requests'][0], reader(b['requests'][0]['domain']['reference']), source=SOURCE)
            self.assertEqual(ra['status'], 'NUMERICAL_MEAN_AVAILABLE')
            self.assertEqual(rb['status'], 'NUMERICAL_MEAN_AVAILABLE')
            self.assertNotEqual(a['requests'][0]['physical_candidate']['id'], b['requests'][0]['physical_candidate']['id'])
            self.assertFalse(b['configuration']['rounds'][0]['mapping']['physical_layout']['priced_equivalence_claim'])
            for record in (ra, rb):
                sig = record['signal']; corr = complex(sig['corrected']['real'],sig['corrected']['imag'])
                raw = complex(sig['raw']['real'],sig['raw']['imag'])
                self.assertLess(abs(raw-corr*sig['attenuation']), 1e-12)
                self.assertTrue(record['numerical']['raw_corrected_independent_propagation'])
                self.assertIsNone(record['compiled_C'])

    def test_full_H_matches_original_leaf_on_artificial_input(self):
        from trottertracks.r_price_phase1_science import functions
        plan, ctx, reader = fixture('FULL_H_RTE', grouping=(3, 2))
        req = plan['requests'][1]; ref = reader(req['domain']['reference']); k = N.kernels(str(ROOT))
        scope = dict(np=np, math=math, deepcopy=deepcopy,
            M=NS(V=NS(S0_SCIENCE='ARTIFICIAL_ALLOWANCE_NOT_EVALUATED')),
            finite_rte_distribution=k.finite_rte_distribution,
            kernels=lambda root:NS(evolve=NS(__globals__=dict(polynomial=k.polynomial))),
            allowance_formula=lambda root:lambda *args:dict(u=1e-12, log_B_margin=0))
        functions(ROOT, N.SHARED, 'src/trottertracks/h4_four_method_screen_mean.py', ('evaluate',), scope,
                  strip=(None,'pr_rpe_power_mean','trotterlib.rte'))
        scalar=req['domain']['preparation']['scalar']['extracted_identity']
        oldctx=NS(state=ctx.state,constant=ctx.constant,spectral_residuals=[],
            full_spec=k.spectrum(sum(ctx.matrices)-scalar*np.eye(2)))
        old=scope['evaluate'](oldctx,dict(candidate=req['physical_candidate'],structure=req['structure'],
            comparison_method='FULL_H_RTE'),ref,root=ROOT,preparation=req['domain']['preparation'])
        new=N.evaluate(ctx,req,ref,source=SOURCE)
        self.assertEqual(new['status'],'NUMERICAL_MEAN_AVAILABLE')
        for field in ('raw','corrected','reference','total_signed_bias'):
            self.assertEqual(old['signal'][field],new['signal'][field])
        for field in ('log_B','B','attenuation'):
            self.assertEqual(old['signal'][field],new['signal'][field])

    def test_existing_B_statistics_metadata_accepts_means_and_keeps_missing_N(self):
        from trottertracks.r_price_phase1_science import functions
        plan,ctx,reader=fixture('B0I',prefix=0)
        means=[P.b_mean(req,N.evaluate(ctx,req,reader(req['domain']['reference']),source=SOURCE)) for req in plan['requests']]
        scope=dict(B=API.B,p=API.B.p,deepcopy=deepcopy)
        functions(ROOT,P.B_SOURCE,'src/trottertracks/rpe_decoder_statistics.py',('_integer','prepare'),scope)
        ready=scope['prepare'](plan['configuration'],means,[],target=plan['target'],
            shot_policy=dict(source=SOURCE,meaning='USER_SUPPLIED'))
        self.assertFalse(ready['statistics_executed']);self.assertIsNone(ready['G'])
        self.assertFalse(ready['physical_qualification_complete'])
        self.assertEqual(ready['blockers'],['MISSING_N_X','MISSING_N_Y','PHYSICAL_MEAN_UNQUALIFIED'])

    def test_metadata_CLI_and_fresh_output(self):
        plan,ctx,reader=fixture(M=0)
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            for key,value in [('config',plan['configuration']),('target',plan['target'])]:
                (td/(key+'.json')).write_text(json.dumps(value))
            command=[sys.executable,'-B',str(ROOT/'scripts/prepare_baseline_signals.py'),
                '--config',str(td/'config.json'),'--target',str(td/'target.json'),'--out',str(td/'PLAN.json')]
            subprocess.run(command,check=True,capture_output=True,cwd=ROOT)
            artifact=json.loads((td/'PLAN.json').read_text())
            self.assertEqual(artifact['science_calls'],0)
            self.assertEqual(artifact['requests'][0]['reference_status'],'MISSING_REFERENCE')
            self.assertNotEqual(subprocess.run(command,capture_output=True,cwd=ROOT).returncode,0)
            self.assertNotEqual(subprocess.run(command+['--execute'],capture_output=True,cwd=ROOT).returncode,0)

    def test_one_body_B3_unknown_formula_and_reference_identity_rejected(self):
        plan, _, _ = fixture('FULL_H_RTE')
        spec = deepcopy(plan['configuration']['spec']);spec['split']['preparation']['one_body_policy'] = 'DETERMINISTIC'
        with self.assertRaises(ValueError):P.configure(ROOT, spec, layouts={m:dict(q=1,R_outer=2) for m in range(3)})
        spec = deepcopy(plan['configuration']['spec']);spec['split']['original_PF_method'] = 'B3'
        with self.assertRaises(ValueError):P.configure(ROOT, spec, layouts={m:dict(q=1,R_outer=2) for m in range(3)})
        with self.assertRaises(ValueError):fixture(formula='8th(Yoshida)')
        spec=deepcopy(plan['configuration']['spec']);spec['law']['model']='PCG64_claimed_as_IID'
        with self.assertRaises(ValueError):
            config=P.configure(ROOT,spec,layouts={m:dict(q=1,R_outer=2) for m in range(3)})
            P.prepare(ROOT,config,target=plan['target'])
        refs = [deepcopy(plan['requests'][0]['domain']['reference'])];refs[0]['binding']['state'] = 'FOREIGN'
        with self.assertRaises(ValueError):P.prepare(ROOT, plan['configuration'], target=plan['target'], references=refs)

    def test_batch_and_B_mean_shape_no_N_or_physical_loader(self):
        plan, ctx, reader = fixture(formula='4th(m5_best)', M=12)
        with tempfile.TemporaryDirectory() as td, patch.object(N, 'load_shared_h4', side_effect=AssertionError('No real inputs')):
            result = P.execute(ROOT, plan, Path(td)/'new', source=SOURCE, context_factory=lambda:ctx, reference_reader=reader)
        self.assertEqual(result['status'], 'BASELINE_SIGNALS_COMPLETE_NOT_RMSE_QUALIFIED')
        self.assertEqual(result['new_mean_evaluations'], 13)
        for mean, req in zip(result['means'], plan['requests']):
            self.assertEqual(mean['identity'], req['identity']);self.assertEqual(mean['law_binding'], req['law_binding'])
            self.assertEqual(mean['target'], plan['target']);self.assertEqual(mean['qualification'], 'UNQUALIFIED')
            self.assertIn('log_B',mean['normalization'])
        self.assertIsNone(result['N']);self.assertIsNone(result['G'])

    def test_saved_success_copied_without_call_and_foreign_domain_not_used(self):
        plan, ctx, reader = fixture(M=0)
        req = plan['requests'][0];record = N.evaluate(ctx, req, reader(req['domain']['reference']), source=SOURCE)
        with tempfile.TemporaryDirectory() as td, patch.object(N, 'evaluate', side_effect=AssertionError('No recalc')):
            result = P.execute(ROOT, plan, Path(td)/'reuse', source=SOURCE, saved=[record])
        self.assertEqual(result['saved_copies'], 1);self.assertEqual(result['new_mean_evaluations'], 0)
        for key in ('physical_candidate','law','profile','target','reference'):
            foreign=deepcopy(record);foreign['domain'][key]={'foreign':True}
            with self.assertRaises(ValueError):P.b_mean(req,foreign)
        bad=deepcopy(record);bad['numerical']['gates_passed']=False
        with self.assertRaises(ValueError):P.b_mean(req,bad)

    def test_failure_values_saved_and_remaining_rounds_not_started(self):
        plan, ctx, reader = fixture()
        key = 'power_diagnostics'; original = getattr(N.kernels(str(ROOT)),key)
        def failed(*args,**kwargs):
            powered, metrics = original(*args,**kwargs)
            metrics.update(power_unitarity_error=1e-4,gate_pass=False,unitarity_gate_pass=False)
            return powered,metrics
        with tempfile.TemporaryDirectory() as td, patch.object(N.kernels(str(ROOT)),key,side_effect=failed):
            out=Path(td)/'fail';result=P.execute(ROOT,plan,out,source=SOURCE,context_factory=lambda:ctx,reference_reader=reader)
            saved=json.loads((out/'round_0.json').read_text())
        self.assertEqual(result['status'],'STOP_NUMERICAL_UNRESOLVED');self.assertEqual(result['new_mean_evaluations'],1)
        self.assertEqual(saved['numerical']['power_unitarity_error'],1e-4)
        self.assertIsNotNone(saved['signal']);self.assertIsNone(result['means'][0]['raw'])
        self.assertTrue(all(v['raw'] is None for v in result['means']))
        self.assertIsNone(saved['scientific_PASS_FAIL'])

    def test_missing_reference_is_missing_without_context_or_solver_fallback(self):
        plan,ctx,reader=fixture();plan=P.prepare(ROOT,plan['configuration'],target=plan['target'])
        with tempfile.TemporaryDirectory() as td, patch.object(N,'load_shared_h4',side_effect=AssertionError('No inputs')):
            result=P.execute(ROOT,plan,Path(td)/'missing',source=SOURCE)
        self.assertEqual(result['status'],'INCOMPLETE_REFERENCES');self.assertEqual(result['new_mean_evaluations'],0)
        self.assertEqual(len(result['means']),3)


if __name__ == '__main__':
    unittest.main()
