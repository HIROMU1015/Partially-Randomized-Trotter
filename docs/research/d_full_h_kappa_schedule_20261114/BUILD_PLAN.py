"""Metadata-only kappa Full-H plan builder. No input file, normalizer, signal or circuit.

Reuses the pinned C baseline pipeline (3a8b37f1 blob; unchanged since 5cc7c07e),
its published B@437ba4ac / D@e7dca5f7 API and the c546 manifest's base spec,
target and reference descriptors. First reproduces the saved R448 configuration
and 13 physical IDs, then generates proposed (not adopted) kappa layouts.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import full_h_kappa_layout as K  # noqa: E402

C_COMMIT = '3a8b37f1a282c0c06ac1a363b070f4376e9ff41a'
C_NUMERIC = '5cc7c07e112ce97205948465c94441cba0042416'
MANIFEST = dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',
                path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
A_MENU = dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
              path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json')
R448 = dict(commit='367dc98d832ecb4d769f2a3da3d174bc697331a5',
            path='docs/research/truncated_fullh_signal_results_20261114/run01/FULL_H_K2_QDYADIC_R448/PLAN.json')
SCREEN = dict(commit='256caad3414a80de02fb68eca89c13c957ed6354',
              path='docs/research/h4_four_method_signal_screen_preparation_20261113/PLAN.json')
PROPOSAL = dict(path='docs/research/d_full_h_kappa_schedule_20261114/FINITE_SCIENCE_PLAN.md',
                status='PROPOSED_NOT_ADOPTED')
KAPPAS = (2, 1, 4)  # 2 = analytic pure-RTE seed; 1/4 = optional bracket. None adopted.


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def read(d):
    return json.loads(blob(d['commit'], d['path']))


def pinned_c():
    """Execute C's unchanged baseline pipeline blob inside this package."""
    name = 'trottertracks.baseline_signal_pipeline'
    if name not in sys.modules:
        path = 'src/trottertracks/baseline_signal_pipeline.py'
        module = types.ModuleType(name); module.__package__ = 'trottertracks'
        module.__file__ = str(ROOT / path); sys.modules[name] = module
        exec(compile(blob(C_COMMIT, path), C_COMMIT + ':' + path, 'exec'), module.__dict__)
        if blob(C_COMMIT, path) != blob(C_NUMERIC, path):
            raise ValueError('C baseline pipeline differs from the R448 numerical SOURCE')
    return sys.modules[name]


def base_inputs(C, api):
    """Same reads as the c546 BUILD_MANIFEST; nothing is regenerated."""
    manifest = read(MANIFEST); entry = manifest['candidates'][2]
    if entry['label'] != 'FULL_H_K2_QDYADIC_R448':
        raise ValueError('Manifest candidate order changed')
    s = read(A_MENU)['spec']
    prep = api.D.saved_full_h_preparation('H4'); original = deepcopy(prep['binding'])
    if not all(s['binding'][k] == v for k, v in original.items()) or set(s['binding']) - set(original) != {'source', 'decomposition'}:
        raise ValueError('Saved Full-H preparation binding bridge differs')
    prep['binding'] = deepcopy(s['binding'])
    if not C.p.same(prep, manifest['shared']['full_h_preparation']):
        raise ValueError('Full-H preparation descriptor differs from manifest')
    refs = [deepcopy(r['reference_descriptor']) for r in entry['rounds']]
    return manifest, entry, s, prep, refs


def frac(d):
    return Fraction(d['numerator'], d['denominator'])


def existing_schedule_context(api, lam4):
    """kappa_eff = s/(lambda_R T)^2 of saved/planned layouts (metadata arithmetic only)."""
    lam6 = Fraction(api.D.saved_full_h_preparation('H6')['lambda_R'])
    lam4 = Fraction(lam4); rows = []
    def add(label, system, lam, T, q, R, K, status, source):
        x = (lam * Fraction(T)) ** 2; s = q * R
        rows.append(dict(label=label, system=system, T=float(T), q=q, R=R, K=K, s=s,
                         kappa_effective=float(Fraction(s) / x), paper_logB_envelope=float(x / s),
                         status=status, source=source))
    t20 = dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
               path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/REPORT.md')
    light = dict(commit='9b83b96b83f62142c9d87858dc24a94db7ac5d61',
                 path='artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/REPORT.md')
    add('T2_OPT_BEST_EPS_LE_0.005', 'H4', lam4, 2, 8, 448, 2, 'SAVED_T2_GRID_BEST_N9_OLD_LOWERING', t20)
    add('T2_OPT_CONFIRMED_R512', 'H4', lam4, 2, 8, 512, 2, 'SAVED_T2_INDEPENDENT_CONFIRMATION_PAIR', t20)
    add('T2_OPT_BEST_EPS_0.01_K0', 'H4', lam4, 2, 8, 448, 0, 'SAVED_T2_GRID_BEST_N9_OLD_LOWERING', t20)
    add('T2_OPT_BEST_EPS_0.05_K0', 'H4', lam4, 2, 8, 256, 0, 'SAVED_T2_GRID_BEST_N9_OLD_LOWERING', t20)
    add('T2_OPT_BEST_EPS_LE_0.005', 'H6', lam6, 2, 32, 448, 2, 'SAVED_T2_GRID_BEST_N9_OLD_LOWERING', t20)
    add('LIGHT_FIXED', 'H4', lam4, Fraction(4, 5), 16, 16, 4, 'SAVED_LIGHT_N9_OLD_LOWERING', light)
    screen = read(SCREEN)
    for cell in screen['fresh']:
        c = cell['candidate']
        if str(c.get('id', '')).startswith('FULL_H_') and c.get('K') == 2:
            add('R2_ADAPT_SCREEN', 'H4', lam4, Fraction(c['T']), c['q'], c['R_outer'], 2,
                'PLANNED_256CAAD_NEVER_EVALUATED', SCREEN)
    return rows


def wiring_cell(candidate):
    """Proposed P1 cell: kappa2 m3, one fresh normal path, X/Y. Seeds by the ORIGINAL
    Full-H trial_seed (AST-extracted digest/trial_seed only; no setup/context/sample)."""
    import ast, hashlib
    light = '9b83b96b83f62142c9d87858dc24a94db7ac5d61'; path = 'scripts/h46_full_h_pure_rte.py'
    tree = ast.parse(blob(light, path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('digest', 'trial_seed')]
    env = dict(hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), light + ':' + path + ':seed_functions_only', 'exec'), env)
    row = next(r for r in candidate['rounds'] if r['m'] == 3)
    physical = next(r for r in candidate['configuration']['rounds'] if r['m'] == 3)
    cohort = 'D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1'
    seeds = [env['trial_seed'](dict(id=row['physical_candidate_id']), cohort, 0, outer, 0) for outer in range(row['q_grouping'])]
    return dict(cell='P1-KAPPA2-M3', configuration_id=candidate['configuration_id'], m=3,
        canonical_candidate_id=physical['candidate']['id'], physical_candidate_id=row['physical_candidate_id'],
        T=0.8, q_grouping=row['q_grouping'], R_per_group=row['R_per_group'], event_count=row['event_count'], K=2,
        role='ONE_FRESH_NORMAL_PATH_WIRING_AND_NEW_LOWERING_EVIDENCE_NOT_LAW_MEAN_OR_MODEL_POINT',
        axes=['X', 'Y'], normal_paths=1, index=0, cohort=cohort, outer_piece_seeds=seeds,
        seed_source=dict(commit=light, path=path, function='trial_seed', family='DF_FULL_H_LIGHT_20261109'),
        seed_transpiler=17, new_compile_maximum=2,
        lowering=dict(lowering_id='CONTROLLED_ZZ_CX_CRZ_CX_V1',
                      source=dict(commit='01e068af7e06d9f1f5621a5b3800c30eb32ecb06', path='src/trottertracks/efficient_controlled_zz.py')),
        scalar='HAMILTONIAN_CONSTANT_PLUS_EXTRACTED_FULL_H_IDENTITY_ONCE_AT_TOTAL_T',
        execution_caller=None, approved=False,
        proposed_public='docs/research/d_full_h_kappa_execution_20261114/P1-KAPPA2-M3/run01',
        proposed_private='/home/AbeHiromu/projects/d-full-h-kappa-20261114-private/P1-KAPPA2-M3/run01')


def main():
    C = pinned_c(); api = C.published_api(str(ROOT)); B, D = api.B, api.D
    manifest, entry, s, prep, refs = base_inputs(C, api)
    target = deepcopy(manifest['shared']['target'])
    saved = read(R448)

    # 1. Regression: the unchanged route reproduces the saved R448 configuration.
    pure = D.full_h_rte_spec(s, M=12, finite_K=2, preparation=prep)
    old = C.configure(ROOT, pure, layouts={m: dict(q=1 << m, R_outer=448) for m in range(13)}, api=api)
    if old['id'] != saved['configuration']['id'] or not C.p.same(old, saved['configuration']):
        raise ValueError('Saved R448 configuration was not reproduced')
    old_plan = C.prepare(ROOT, old, target=target, references=refs, api=api)
    if [r['physical_candidate']['id'] for r in old_plan['requests']] != [r['physical_candidate']['id'] for r in saved['requests']]:
        raise ValueError('Saved R448 physical IDs were not reproduced')
    lam = Fraction(prep['lambda_R']); h = frac(s['h']) if isinstance(s['h'], dict) else Fraction(s['h'])
    r448 = []
    for r in old_plan['requests']:
        c = r['physical_candidate']; T = h * (1 << r['m']); x = (lam * T) ** 2; n = c['q'] * c['R_outer']
        r448.append(dict(m=r['m'], T=c['T'], q_grouping=c['q'], R_per_group=c['R_outer'], event_count=n,
                         physical_candidate_id=c['id'], kappa_effective=float(Fraction(n) / x),
                         paper_logB_envelope=float(x / n)))

    # 2. Proposed kappa schedules through the same D/B/C route.
    candidates = []
    for value in KAPPAS:
        kappa = K.kappa_descriptor(value, proposal_source=PROPOSAL)
        spec = D.full_h_rte_spec(s, M=12, finite_K=2, preparation=prep, kappa=kappa)
        lay, rows = K.layouts(spec, B=B)
        config = C.configure(ROOT, spec, layouts=lay, api=api)
        B._validate(config); K.check_configuration(config, B=B, rows=rows)
        plan = C.prepare(ROOT, config, target=target, references=refs, api=api)
        phys = D.physical_schedule(config['spec'], layout_by_round=lay)
        for row, req, ph in zip(rows, plan['requests'], phys['rounds'], strict=True):
            c = req['physical_candidate']
            if (c['q'] != row['q_grouping'] or c['R_outer'] != row['R_per_group'] or c['s'] != row['event_count']
                    or c['K'] != 2 or c['family'] != 'FULL_H_PURE_RTE' or req['comparison_method'] != 'FULL_H_RTE'
                    or req['reference_status'] != 'SUPPLIED' or req['q_PF'] is not None):
                raise ValueError('Prepared physical request differs from kappa layout')
            if ph['physical_candidate_id'] != c['id'] or ph['parameterization'] != 'EXPLICIT_LAYOUT_SUPPLIED_KAPPA':
                raise ValueError('D physical schedule differs from prepared request')
            row.update(saved_asset_status=ph['saved_asset_status'], d_parameterization=ph['parameterization'])
            row.update(physical_candidate_id=c['id'], canonical_candidate_id=config['rounds'][row['m']]['candidate']['id'],
                       reference_descriptor=deepcopy(req['domain']['reference']), mean_status=req['mean_status'])
        old_ids = {r['physical_candidate_id'] for r in r448}
        candidates.append(dict(label='FULL_H_K2_QDYADIC_KAPPA' + str(value), kappa=deepcopy(config['spec']['kappa']),
            configuration_id=config['id'], role='PRIMARY_ANALYTIC_SEED' if value == 2 else 'OPTIONAL_BRACKET',
            event_mapping=deepcopy(config['spec']['event_mapping']), rounds=rows,
            shared_physical_ids_with_R448=sorted(old_ids & {r['physical_candidate_id'] for r in rows}),
            configuration=config))
    result = dict(schema='D_FULL_H_KAPPA_SCHEDULE_PROPOSAL_V1', status='METADATA_ONLY_NOT_ADOPTED_NOT_EVALUATED',
        pinned=dict(C_pipeline=C_COMMIT, C_numerical_source=C_NUMERIC, B=C.B_SOURCE, D=C.D_SOURCE,
                    manifest=MANIFEST, A_base_spec=A_MENU, saved_R448_plan=R448),
        lambda_R=dict(value=float(lam), source=deepcopy(prep['lambda_source']), scope='FULL_H_INCLUDING_ONE_BODY',
                      one_body_lambda_R=prep['one_body_lambda_R']),
        regression=dict(R448_configuration_id=old['id'], R448_configuration_reproduced=True,
                        R448_physical_ids_reproduced=13),
        R448_schedule=r448, existing_schedule_context=existing_schedule_context(api, prep['lambda_R']),
        candidates=candidates, proposed_wiring_cell=wiring_cell(candidates[0]), target=target,
        science=dict(input_files_opened=False, finite_normalizer_calls=0, signal_evaluations=0,
                     sampling_build_compile=0, N=None, C=None, G=None, kappa_adopted=False),
        mandatory_stop=True)
    if any(k in sys.modules for k in ('numpy', 'scipy', 'qiskit', 'pyscf')):
        raise ValueError('Metadata builder imported a numerical/quantum library')
    return result


if __name__ == '__main__':
    out = OUT / 'KAPPA_CANDIDATES.json'
    data = main()
    with out.open('w') as f:
        json.dump(data, f, indent=1, ensure_ascii=False, allow_nan=False); f.write('\n')
    print(json.dumps(dict(R448=data['regression'], candidates=[(c['label'], c['configuration_id']) for c in data['candidates']]),
                     ensure_ascii=False))
