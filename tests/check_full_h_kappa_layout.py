"""Artificial checks for the Full-H paper-kappa layout helper.

Toy bindings/lambda only, except the last metadata regression, which reproduces
the saved R448 configuration IDs from pinned public JSON. No Hamiltonian/state,
normalizer, signal, sampling, circuit, price, N or G.
"""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import full_h_kappa_layout as K  # noqa: E402

PLAN = ROOT / 'docs/research/d_full_h_kappa_schedule_20261114/BUILD_PLAN.py'
spec_ = importlib.util.spec_from_file_location('_d_full_h_kappa_build_plan', PLAN)
BUILD = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(BUILD)
C = BUILD.pinned_c(); API = C.published_api(str(ROOT)); B, D = API.B, API.D
PYTHON = sys.executable


def toy(lam=3.0, one=1.0, M=6):
    binding = dict(system='H4', hamiltonian='TOY_H', state='TOY_STATE', snapshot='TOY_SNAPSHOT',
                   basis='TOY', geometry='TOY', rank_policy='TOY', series='TOY', rank=10,
                   decomposition=dict(toy=True), source=dict(toy=True))
    base = dict(binding=binding, h=dict(numerator=1, denominator=10),
                rmse_policy=dict(family='PR_PAPER_RMSE', id='TOY', source=dict(toy=True),
                                 target_rmse_ha=dict(numerator=1, denominator=1000),
                                 error_contract=dict(toy=True), scientific_status='SYNTHETIC_ONLY'),
                profile=dict(compiler='TOY', control='symmetric_directional', wrapper='full_Hadamard_with_measurement',
                             scope=B.p.SCOPE, P=0))
    prep = dict(representation='DF', source=dict(D.SOURCES['full_h_prepare'], function='prepare_full_h'),
                one_body_policy='RANDOMIZED', one_body_components=2, deterministic_blocks=0,
                randomized_fragments=list(range(10)), all_fragments=10, lambda_R=lam, one_body_lambda_R=one,
                lambda_source=dict(toy=True), normalization_policy='full_h_df_plus_one_body_exact_Z',
                scalar=dict(constant=.5, extracted_identity=-.25, total=.25, application='ONCE_AT_TOTAL_T',
                            identity_components_randomized=0),
                preparation_hash='TOY_PREP', binding=deepcopy(binding), components=12)
    return base, prep, M


def kspec(value, lam=3.0, M=6):
    base, prep, M = toy(lam, lam / 3, M=M)
    return D.full_h_rte_spec(base, M=M, finite_K=2, preparation=prep,
                             kappa=K.kappa_descriptor(value, proposal_source=dict(toy=True)))


class Tests(unittest.TestCase):
    def test_total_matches_b_and_a_mappers(self):
        from trottertracks import pr_rpe_rmse as A
        for value in (2, Fraction(1, 3), 2.5, 7):
            for lam in (3.0, .7, 13.37):
                s = kspec(value, lam); lay, rows = K.layouts(s, B=B)
                for row in rows:
                    T = Fraction(row['T']['numerator'], row['T']['denominator'])
                    exact = Fraction(value) if not isinstance(value, float) else Fraction.from_float(value)
                    self.assertEqual(row['total_target'], -(-(exact * (Fraction(lam) * T) ** 2).numerator
                                                            // (exact * (Fraction(lam) * T) ** 2).denominator))
                    a = A.paper_event_mapper(kappa_paper=exact, lambda_R=lam, T=T, q_PF=row['q_grouping'], signed_weights=[1.])
                    self.assertEqual((a['r_target'], a['R_outer'], a['event_count'], a['allocation']),
                                     (row['total_target'], row['R_per_group'], row['event_count'], [row['R_per_group']]))

    def test_layout_covers_target_and_envelope(self):
        for value in (Fraction(1, 2), 1, 2, 4):
            lay, rows = K.layouts(kspec(value), B=B)
            for row in rows:
                q, R, s = row['q_grouping'], row['R_per_group'], row['event_count']
                self.assertEqual(q, 1 << row['m']); self.assertEqual(lay[row['m']], dict(q=q, R_outer=R))
                self.assertTrue(row['total_target'] <= s < row['total_target'] + q)
                keff = Fraction(row['kappa_effective']['numerator'], row['kappa_effective']['denominator'])
                env = Fraction(row['paper_logB_envelope']['numerator'], row['paper_logB_envelope']['denominator'])
                self.assertGreaterEqual(keff, Fraction(value)); self.assertLessEqual(env, 1 / Fraction(value))
                self.assertEqual(env * keff, 1)
                self.assertIsNone(row['finite_normalizer']); self.assertIsNone(row['signal'])

    def test_explicit_single_group_and_identity_keeps_q(self):
        s = kspec(2)
        dy, dyrows = K.layouts(s, B=B)
        one, onerows = K.layouts(s, B=B, grouping='EXPLICIT', q_by_round={m: 1 for m in range(7)})
        for a, b in zip(dyrows, onerows):
            self.assertEqual(a['total_target'], b['total_target']); self.assertEqual(b['event_count'], b['total_target'])
        c1 = C.configure(ROOT, s, layouts=dy, api=API); c2 = C.configure(ROOT, s, layouts=one, api=API)
        self.assertNotEqual(c1['id'], c2['id'])
        ids = lambda c: [D.physical_schedule(c['spec'], layout_by_round={r['m']: {k: r['mapping']['physical_layout'][k]
                         for k in ('q', 'R_outer')} for r in c['rounds']})['rounds'][m]['physical_candidate_id'] for m in range(7)]
        self.assertNotEqual(ids(c1)[3], ids(c2)[3])

    def test_rejections(self):
        for bad in (0, -1, True, float('nan'), float('inf'), 'two', None):
            with self.assertRaises(ValueError):
                K.kappa_descriptor(bad, proposal_source=dict(toy=True))
        with self.assertRaises(ValueError):
            K.kappa_descriptor(2, proposal_source=None)
        s = kspec(2)
        for mutate in (lambda v: v.update(kappa=None), lambda v: v['kappa'].update(kind='OTHER'),
                       lambda v: v['kappa'].update(definition='OTHER'), lambda v: v.update(method='PR'),
                       lambda v: v.update(finite_K=3), lambda v: v['split'].update(one_body_policy='DETERMINISTIC'),
                       lambda v: v.update(delta_pf=dict(numerator=1, denominator=10))):
            v = deepcopy(s); mutate(v)
            with self.assertRaises(ValueError):
                K.layouts(v, B=B)
        for kwargs in (dict(grouping='OTHER'), dict(grouping='EXPLICIT'), dict(q_by_round={0: 1}),
                       dict(grouping='EXPLICIT', q_by_round={m: 1 for m in range(6)}),
                       dict(grouping='EXPLICIT', q_by_round={m: 0 for m in range(7)})):
            with self.assertRaises(ValueError):
                K.layouts(s, B=B, **kwargs)

    def test_generated_route_validates_and_echoes_kappa(self):
        s = kspec(2); lay, rows = K.layouts(s, B=B)
        config = C.configure(ROOT, s, layouts=lay, api=API)
        B._validate(config); self.assertTrue(K.check_configuration(config, B=B, rows=rows))
        self.assertEqual(config['spec']['event_mapping']['kind'], D.LAYOUT_KIND)
        for r in config['rounds']:
            self.assertTrue(B.p.same(r['law_binding']['kappa'], config['spec']['kappa']))
            self.assertIsNone(r['q_PF']); self.assertIsNone(r['R_outer'])
        again = C.configure(ROOT, kspec(2), layouts=lay, api=API)
        self.assertEqual(again['id'], config['id'])
        s4 = kspec(4); lay4, _ = K.layouts(s4, B=B)
        self.assertNotEqual(C.configure(ROOT, s4, layouts=lay4, api=API)['id'], config['id'])

    def test_tampering_rejected(self):
        s = kspec(2); lay, rows = K.layouts(s, B=B)
        config = C.configure(ROOT, s, layouts=lay, api=API)
        bad = deepcopy(config); bad['rounds'][3]['mapping']['physical_layout']['R_outer'] += 1
        with self.assertRaises(ValueError):
            B._validate(bad)
        wrong = deepcopy(rows); wrong[4]['R_per_group'] += 1
        with self.assertRaises(ValueError):
            K.check_configuration(config, B=B, rows=wrong)
        base, prep, M = toy()
        free = D.full_h_rte_spec(base, M=M, finite_K=2, preparation=prep)
        with self.assertRaises(ValueError):
            K.check_configuration(C.configure(ROOT, free, layouts=lay, api=API), B=B, rows=rows)

    def test_kappa_free_same_layout_shares_physical_ids_not_configuration(self):
        base, prep, M = toy()
        s = kspec(2); lay, _ = K.layouts(s, B=B)
        free = D.full_h_rte_spec(base, M=M, finite_K=2, preparation=prep)
        ck, cf = C.configure(ROOT, s, layouts=lay, api=API), C.configure(ROOT, free, layouts=lay, api=API)
        self.assertNotEqual(ck['id'], cf['id'])
        pk = D.physical_schedule(s, layout_by_round=lay); pf = D.physical_schedule(free, layout_by_round=lay)
        self.assertEqual([r['physical_candidate_id'] for r in pk['rounds']], [r['physical_candidate_id'] for r in pf['rounds']])
        self.assertEqual({r['parameterization'] for r in pk['rounds']}, {'EXPLICIT_LAYOUT_SUPPLIED_KAPPA'})

    def test_c_price_guard_resolves_kappa_layouts_without_change(self):
        name = 'trottertracks.price_implementation_guard'
        if name not in sys.modules:
            import types
            path = 'src/trottertracks/price_implementation_guard.py'
            mod = types.ModuleType(name); mod.__package__ = 'trottertracks'; sys.modules[name] = mod
            exec(compile(BUILD.blob(BUILD.C_COMMIT, path), BUILD.C_COMMIT + ':' + path, 'exec'), mod.__dict__)
        G = sys.modules[name]
        s = kspec(2); lay, _ = K.layouts(s, B=B); config = C.configure(ROOT, s, layouts=lay, api=API)
        policy = dict(id='TOY_NEW_ZZ_SCENARIO', source=dict(toy=True), profile=deepcopy(config['spec']['profile']),
                      by_method=dict(FULL_H_RTE=dict(native_policy='TOY_NATIVE', rte_rotation_lowering=G.NEW_ZZ,
                                                     scalar_quantity='HAMILTONIAN_CONSTANT_PLUS_EXTRACTED_FULL_H_IDENTITY')))
        out = G.check(ROOT, config, [], cohorts={}, bindings=[], policy=policy)
        self.assertEqual((out['status'], out['checked']), ('IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION', []))
        mixed = deepcopy(policy); mixed['by_method']['PR'] = dict(mixed['by_method']['FULL_H_RTE'], rte_rotation_lowering=G.OLD_ZZ)
        with self.assertRaises(ValueError):
            G.check(ROOT, config, [], cohorts={}, bindings=[], policy=mixed)

    def test_artificial_tau_grid_supports_logB_envelope_gate(self):
        """Pinned K2 normalizer on artificial tau only: b(tau) <= exp(tau^2), log b ~ tau^2."""
        import math, types
        name = '_d_kappa_pinned_rte'
        if name not in sys.modules:
            mod = types.ModuleType(name); sys.modules[name] = mod
            path = 'src/trotterlib/rte.py'; commit = '71ba49bb28b295aa28e51101bc3b1bf07d500767'
            exec(compile(BUILD.blob(commit, path), commit + ':' + path, 'exec'), mod.__dict__)
        rte = sys.modules[name]
        excess = 0
        for e in range(-60, 11):
            tau = 10 ** (e / 10)
            d = rte.finite_rte_distribution(tau, 2); b = d.exact_finite_distribution
            self.assertGreater(b, 1.)
            # binary64 b=1+O(tau^2): exp(tau^2) and b are each rounded, and the relative logB
            # error is ~ulp/tau^2 (1e-4 at tau=1e-6). An execution gate therefore needs the
            # existing one-ULP normalizer sensitivity as slack; exact b<=exp(tau^2) holds.
            self.assertLessEqual(b, math.nextafter(d.paper_upper_bound, math.inf))
            excess += b > d.paper_upper_bound
            ulp = math.log(math.nextafter(b, math.inf)) - math.log(math.nextafter(b, 0.))
            self.assertLessEqual(math.log(b), tau * tau + ulp)
            if tau < 1e-2:
                self.assertAlmostEqual(math.log(b) / (tau * tau), 1., delta=1e-3)
            else:
                self.assertLess(math.log(b), tau * tau)
        print('one-ULP excesses over rounded exp(tau^2):', excess, 'of 71 artificial tau')

    def test_saved_metadata_regression_without_numerical_imports(self):
        code = ('import importlib.util,json,sys;'
                'sp=importlib.util.spec_from_file_location("b",%r);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);'
                'r=m.main();print(json.dumps([r["regression"],[c["configuration_id"] for c in r["candidates"]],'
                '[k for k in ("numpy","scipy","qiskit","pyscf") if k in sys.modules]]))') % str(PLAN)
        out = json.loads(subprocess.check_output([PYTHON, '-B', '-c', code], cwd=ROOT, text=True).strip().splitlines()[-1])
        self.assertEqual(out[0], dict(R448_configuration_id='cfa6aee3260de664de9acd444a628faae388c459a2eaf68a786c707573029d1c',
                                      R448_configuration_reproduced=True, R448_physical_ids_reproduced=13))
        saved = json.loads((PLAN.parent / 'KAPPA_CANDIDATES.json').read_text())
        self.assertEqual(out[1], [c['configuration_id'] for c in saved['candidates']])
        self.assertEqual(out[2], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
