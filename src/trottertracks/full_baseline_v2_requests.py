"""Full PF and DF Full-H RTE physical requests for B PR_RPE_RMSE_V2. Metadata only.

The module derives V2 specifications from an existing comparison spec, records the
pinned physical SOURCE, and writes A (N/RMSE) and C (full-wrapper price) requests
that point to the same physical candidate. It never loads a Hamiltonian or
state, evaluates a signal/mean/N/normalization, samples, builds or compiles a
circuit, prices, fits or chooses M, kappa, PF width, formula or RTE layout.
Physical SOURCE is read as pinned Git text only.
"""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess

from . import pr_rpe_rmse_controller as B

p = B.p
ROOT = Path(__file__).resolve().parents[2]
BASE = 'af284d853890155bacd19e61850c196bd1d5c660'          # B V2 + evidence planner
FULL_H = '9b83b96b83f62142c9d87858dc24a94db7ac5d61'        # DF full-H prepare_full_h + light study
FULL_H_T20 = '7a8bc7f132e166390c943e6b32c92cc6abe1baf4'    # later T=2.0 study (s=qR mean sharing)
LIGHT = 'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/'
LAYOUT_KIND = 'D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1'
MODULE = dict(branch='codex/d-full-pf-full-h-rte-v2-paths-20261113', path='src/trottertracks/full_baseline_v2_requests.py',
              module='trottertracks.full_baseline_v2_requests')
SAVED_FULL_H_REFERENCE_T = (.8, 2.)  # light study context and later T=2.0 study


def ref(commit, path, **extra):
    return dict(commit=commit, path=path, **extra)


SOURCES = dict(
    pf_weights=ref(BASE, 'src/trotterlib/h46_m5_plan.py', functions=['weights', 'schedule', 'allocation']),
    pf_signal=ref(BASE, 'scripts/h46_long_time_signal.py', functions=['candidates', 'cid', 'context', 'evolve_batch']),
    pf_native=ref(BASE, 'scripts/h46_long_time_native.py', functions=['compose', 'cost_trial']),
    pf_native_lowering=ref(BASE, 'src/trottertracks/resource_applicability/ax2b_native_df_v5.py',
                           functions=['build_deterministic_native', 'build_partial_native']),
    pf_common_cost=ref(BASE, 'scripts/pf_common_run_native.py', functions=['cost_one']),
    pf_higher_order=ref(BASE, 'src/trotterlib/product_formula.py', functions=['_get_w_list']),
    pf_decomposition=ref(BASE, 'src/trotterlib/pf_decomposition.py', functions=['iter_pf_steps']),
    full_h_prepare=ref(FULL_H, 'scripts/h46_full_h_pure_rte.py', functions=['prepare_full_h', 'context', 'validate']),
    full_h_saved_summary=ref(FULL_H, LIGHT + 'input_and_validation_summary.json'),
    full_h_saved_selection=ref(FULL_H, LIGHT + 'selection.json'),
    full_h_saved_prices=ref(FULL_H, LIGHT + 'price_trials.json'),
    full_h_t20=ref(FULL_H_T20, 'scripts/h46_full_h_t20_opt.py', functions=['evaluate'], priced_equivalence_claim=False),
    rte_law=ref(BASE, 'src/trotterlib/rte.py', functions=['finite_rte_distribution']),
    pauli_hamiltonian=ref(BASE, 'src/trotterlib/pauli_signal_adapter.py', functions=['PauliHamiltonian']),
    pauli_law=ref(BASE, 'src/trotterlib/h4_pauli_pr_search.py', functions=['law_descriptor', 'central_trajectory']),
    pauli_mean=ref(BASE, 'src/trotterlib/h4_pauli_pr_mean.py'),
    pauli_price=ref(BASE, 'scripts/pauli_h2_h6_study.py', functions=['initial', 'price_job']),
)

# Static facts read from the pinned SOURCE; v2_generate mirrors B.generate.
FORMULAS = {
    '2nd': dict(family='S2', coefficient_source='pf_weights', negative_time=False, h46_signal_path=True,
                h46_native_path=True, v2_generate=True),
    '4th': dict(family='STANDARD_YOSHIDA4', coefficient_source='pf_weights', negative_time=True, h46_signal_path=True,
                h46_native_path=True, v2_generate=True, not_alias_of='4th(m5_best)',
                formula_id='PF_STANDARD_YOSHIDA4_BINARY64_V1'),
    '4th(m5_best)': dict(family='M5_BEST_BINARY64', coefficient_source='pf_weights', negative_time=True,
                         h46_signal_path=True, h46_native_path=True, v2_generate=True, formula_id='PF_M5_BEST_20D64C2_BINARY64_V1'),
    '8th(Morales)': dict(family='MORALES8', coefficient_source='pf_higher_order', negative_time=True, h46_signal_path=False,
                         h46_native_path=False, v2_generate=False),
    '8th(Yoshida)': dict(family='YOSHIDA8', coefficient_source='pf_higher_order', negative_time=True, h46_signal_path=False,
                         h46_native_path=False, v2_generate=False),
}
BINDING_KEYS = ('system', 'hamiltonian', 'state', 'snapshot', 'basis', 'geometry', 'rank_policy', 'series')
PREPARATION_KEYS = ('representation', 'source', 'one_body_policy', 'one_body_components', 'deterministic_blocks',
                    'randomized_fragments', 'all_fragments', 'lambda_R', 'one_body_lambda_R', 'lambda_source',
                    'normalization_policy', 'scalar', 'preparation_hash', 'binding')


def _blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT, stderr=subprocess.PIPE)


def _constant(commit, path, name):
    """Literal module constant from pinned SOURCE text; the module is not imported."""
    tree = ast.parse(_blob(commit, path))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == name for t in n.targets))
    return ast.literal_eval(node.value)


def saved_grid():
    path = SOURCES['pf_signal']['path']
    return dict(rank=_constant(BASE, path, 'RANK'), q=_constant(BASE, path, 'Q'), formulas=_constant(BASE, path, 'FORMULAS'))


def catalog():
    """SOURCE catalog with V2 status; weights are the pinned binary64 literals."""
    schedule = p._schedule()
    out = {}
    for label, fact in FORMULAS.items():
        weights = list(schedule.weights(label)) if fact['coefficient_source'] == 'pf_weights' else None
        out[label] = dict(fact, coefficient_source=deepcopy(SOURCES[fact['coefficient_source']]), signed_weights=weights,
            v2_status='SUPPORTED' if fact['v2_generate'] else
                      'B_V2_FORMULA_EXTENSION_REQUIRED' if fact['h46_native_path'] else 'NO_DF_CONTROLLED_FULL_WRAPPER_PATH')
    return dict(formulas=out, sources=deepcopy(SOURCES), grid=saved_grid(), pauli=pauli_reference(),
                science_calls=0, formula_adopted=False)


def pauli_reference():
    """Pauli assets are a different representation; no V2 request is built from them here."""
    return dict(representation='PAULI_JW', status='SOURCE_ONLY_NOT_CONNECTED_BY_D',
        full_h_rule='L_D=0 randomizes every non-identity Pauli term; the identity constant is a global phase',
        law='C_EFFECTIVE_GRID_LAW_EXACT_RETAINED_PRODUCTS_NATIVE_STEP_SCALAR (53-bit uniform grid); not the ideal law',
        sources={k: deepcopy(v) for k, v in SOURCES.items() if k.startswith('pauli_')},
        same_numerical_H_state_as_DF=False)


def check_binding(v2_binding, physical_binding):
    """Saved physical H/state/snapshot must equal the V2 comparison binding."""
    diff = [k for k in BINDING_KEYS if k in physical_binding and not p.same(physical_binding[k], v2_binding.get(k))]
    if diff:
        raise ValueError('Physical H/state/snapshot binding differs from V2 binding: ' + ','.join(diff))
    return True


def check_full_h_preparation(prep, v2_binding):
    """Full-H scope: every non-scalar term randomized, one-body included, scalar once."""
    missing = [k for k in PREPARATION_KEYS if k not in prep]
    if missing:
        raise ValueError('Full-H preparation descriptor incomplete: ' + ','.join(missing))
    if prep['representation'] != 'DF':
        raise ValueError('D adapter builds DF Full-H RTE only; Pauli is a different representation')
    if (prep['source'].get('commit') != FULL_H or prep['source'].get('path') != SOURCES['full_h_prepare']['path']
            or prep['source'].get('function') != 'prepare_full_h'):
        raise ValueError('Full-H preparation must come from prepare_full_h; DF B3 L_D=0 keeps one-body deterministic')
    if (prep['one_body_policy'] != 'RANDOMIZED' or type(prep['one_body_components']) is not int
            or prep['one_body_components'] < 1 or prep['deterministic_blocks'] != 0):
        raise ValueError('One-body components must be randomized with no deterministic block')
    if (list(prep['randomized_fragments']) != list(range(prep['all_fragments']))
            or (v2_binding.get('rank') is not None and prep['all_fragments'] != v2_binding['rank'])):
        raise ValueError('All DF fragments must be randomized in saved order')
    lam, one = prep['lambda_R'], prep['one_body_lambda_R']
    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (lam, one))
            or not 0 < one < lam or not prep['lambda_source']):
        raise ValueError('Full-H lambda must include the one-body l1 and carry its SOURCE')
    if prep.get('tail_only_lambda_R') is not None and prep['tail_only_lambda_R'] >= lam:
        raise ValueError('Tail-only lambda reused as full-H lambda')
    if prep['normalization_policy'] != 'full_h_df_plus_one_body_exact_Z':
        raise ValueError('Full-H normalization policy required')
    scalar = prep['scalar']
    if (scalar.get('application') != 'ONCE_AT_TOTAL_T' or scalar.get('identity_components_randomized') != 0
            or not all(k in scalar for k in ('constant', 'extracted_identity'))):
        raise ValueError('Identity scalar must be extracted and applied once at total T')
    check_binding(v2_binding, prep['binding'])
    return True


def saved_full_h_preparation(system):
    """Transcribe the published light-study preparation facts; nothing is recomputed."""
    summary = json.loads(_blob(FULL_H, SOURCES['full_h_saved_summary']['path']))[system]
    return dict(representation='DF', source=dict(SOURCES['full_h_prepare'], function='prepare_full_h'),
        one_body_policy='RANDOMIZED', one_body_components=summary['one_body_components'],
        deterministic_blocks=summary['deterministic_blocks'], randomized_fragments=list(summary['two_body_fragments']),
        all_fragments=summary['binding']['rank'], lambda_R=summary['lambda_R'],
        one_body_lambda_R=summary['one_body_lambda_R'], lambda_source=dict(SOURCES['full_h_saved_summary'], field=system+'.lambda_R'),
        normalization_policy='full_h_df_plus_one_body_exact_Z',
        scalar=dict(constant=summary['constant'], extracted_identity=summary['extracted_scalar'], total=summary['scalar'],
                    application='ONCE_AT_TOTAL_T', identity_components_randomized=0),
        preparation_hash=summary['preparation_hash'], binding=deepcopy(summary['binding']),
        components=summary['components'], transcribed_not_recomputed=True)


def _common(base):
    return {k: deepcopy(base[k]) for k in ('binding', 'h', 'rmse_policy', 'profile')}


def full_pf_spec(base, *, formula, delta_pf, M, rank):
    """Full-H deterministic PF; caller supplies formula, width and M. No kappa/K."""
    if formula not in FORMULAS:
        raise ValueError('Unknown PF formula label')
    if not FORMULAS[formula]['v2_generate']:
        raise ValueError('Formula ' + formula + ' requires a B V2 formula extension; not aliased')
    if rank != base['binding'].get('rank') or rank != saved_grid()['rank'][base['binding']['system']]:
        raise ValueError('Full PF retains one-body and all saved DF fragments; prefix must equal the full rank')
    source = deepcopy(SOURCES['pf_signal'])
    s = dict(_common(base), method='PF', L_D=rank, delta_pf=delta_pf, formula=formula, M=M, kappa=None, finite_K=None,
        split=dict(L_D=rank, source=source, randomized_scope='NONE', deterministic_scope='FULL_H',
                   original_PF_method='FULL', lambda_R=None, lambda_source=None,
                   retained_hamiltonian=dict(construction='FULL_H', prefix=rank, source=source,
                                             target_hamiltonian=deepcopy(base['binding']['hamiltonian']))),
        law=dict(model='deterministic', source=deepcopy(SOURCES['pf_weights'])),
        event_mapping=dict(kind='NONE', source=deepcopy(SOURCES['pf_weights'])))
    return s


def full_h_rte_spec(base, *, M, finite_K, kappa, preparation):
    """Supplied kappa keeps the legacy mapping; None selects explicit layouts only."""
    check_full_h_preparation(preparation, base['binding'])
    if kappa is not None and (not kappa.get('kind') or not kappa.get('definition_source')):
        raise ValueError('A supplied kappa needs a named source-bound definition')
    split = dict(L_D=0, source=deepcopy(preparation['source']), randomized_scope='FULL_H_INCLUDING_ONE_BODY',
        one_body_policy='RANDOMIZED', deterministic_scope='NONE', original_PF_method='NONE_FULL_H_PREPARATION',
        lambda_R=preparation['lambda_R'], lambda_source=deepcopy(preparation['lambda_source']),
        preparation=deepcopy(preparation))
    return dict(_common(base), method='FULL_H_RTE', L_D=0, delta_pf=None, formula=None, M=M, kappa=deepcopy(kappa),
        finite_K=finite_K, split=split,
        law=dict(model='ideal_finite_RTE_law', sampling='PAIRED_FINITE_TAYLOR_COMPONENT_ORDER_LAW',
                 source=deepcopy(SOURCES['rte_law']), preparation_source=deepcopy(preparation['source'])),
        event_mapping=dict(kind=B.EXPLICIT_LAYOUT if kappa is None else LAYOUT_KIND,
            source=(dict(MODULE, extension_interface=dict(commit='575598983059fdf5238da877f62b40250fd9eae9',
                path='docs/research/abc_d_v2_integration_20261114/INTERFACE.md')) if kappa is None else deepcopy(MODULE))))


def layout_mapper(layout_by_round):
    """Explicit S2 grouping per round: q outer steps x R events. Missing round -> None."""
    layouts = {}
    for m, v in layout_by_round.items():
        if (type(m) is not int or m < 0 or set(v) != {'q', 'R_outer'} or any(type(v[k]) is not int or v[k] < 1 for k in v)):
            raise ValueError('Explicit integer q and R_outer per round required')
        layouts[m] = dict(v)

    def mapper(s, row):
        v = layouts.get(row['m'])
        if v is None:
            return None
        q, R = v['q'], v['R_outer']
        return dict(kind=s['event_mapping']['kind'], source=deepcopy(s['event_mapping']['source']), kappa=deepcopy(s['kappa']),
            R_outer=None, event_count=q*R, physical_layout=dict(formula='2nd', q=q, R_outer=R, pieces_per_outer=1,
                allocation=[R], allocation_policy='ABS_TIME_PER_OUTER', event_positions=q*R,
                event_tau='lambda_R*(T/q)/R_outer', taylor_K=s['finite_K'], signed_times='ALL_POSITIVE_S2',
                scalar='ONCE_AT_TOTAL_T', deterministic_blocks=0, compose_source=deepcopy(SOURCES['pf_native']),
                priced_equivalence_claim=False))
    return mapper


def generate(spec, *, layout_by_round=None):
    """Existing B.generate; Full-H RTE uses the explicit layout mapper only."""
    if spec['method'] == 'FULL_H_RTE':
        if spec['event_mapping']['kind'] not in (LAYOUT_KIND, B.EXPLICIT_LAYOUT) or layout_by_round is None:
            raise ValueError('Full-H RTE needs an explicit physical layout; event_count alone is not a grouping')
        if any(type(m) is not int or m not in range(spec['M'] + 1) for m in layout_by_round):
            raise ValueError('Physical layout round outside the supplied M')
        return B.generate(spec, event_mapper=layout_mapper(layout_by_round))
    if layout_by_round is not None:
        raise ValueError('Full PF has no RTE layout')
    return B.generate(spec)


def _physical(config, row):
    """Physical candidate, saved-asset hint and whether a saved reference exists at T."""
    s = config['spec']; system = s['binding']['system']; T = float(B.a.exact_scalar(row['T']))
    if s['method'] == 'PF':
        c = dict(system=system, method='FULL', formula=s['formula'], prefix=s['L_D'], T=T, q=row['q_PF'],
                 delta=T/row['q_PF'], R_outer=None, K=None)
        c['id'] = p.candidate_id(c)
        grid = saved_grid()
        hint = ('SAVED_LONG_TIME_GRID_POINT_CHECK_CARRIER' if T in grid['q'] and row['q_PF'] in grid['q'][T]
                and s['formula'] in grid['formulas'] else 'NO_SAVED_GRID_POINT')
        return c, hint, T in grid['q']
    layout = (row['mapping'] or {}).get('physical_layout')
    if layout is None:
        return None, 'LAYOUT_NOT_SUPPLIED', T in SAVED_FULL_H_REFERENCE_T
    c = dict(system=system, method='PR', formula='2nd', prefix=0, T=T, q=layout['q'], R_outer=layout['R_outer'],
             K=s['finite_K'], delta=T/layout['q'], s=layout['event_positions'], family='FULL_H_PURE_RTE', L_D=0)
    c['id'] = 'FULL_H_' + p.candidate_id(c)
    saved = {v['id'] for v in json.loads(_blob(FULL_H, SOURCES['full_h_saved_selection']['path']))['candidates']}
    return c, ('SAVED_LIGHT_STUDY_CANDIDATE_FINITE_COHORT' if c['id'] in saved else 'NO_SAVED_EXACT_CANDIDATE'), T in SAVED_FULL_H_REFERENCE_T


def requests(config):
    """A (N/RMSE) and C (price) requests per round, same physical candidate. No values."""
    B._validate(config)
    s = config['spec']
    if s['method'] not in ('PF', 'FULL_H_RTE') or (s['method'] == 'PF' and B.comparison_method(s) != 'FULL_PF'):
        raise ValueError('D requests cover Full PF and Full-H RTE only')
    full_h = s['method'] == 'FULL_H_RTE'
    out = []
    for row in config['rounds']:
        physical, saved, reference = _physical(config, row)
        common = dict(configuration_id=config['id'], m=row['m'], T=deepcopy(row['T']), q_RPE=row['q_RPE'],
            q_PF=row['q_PF'], actual_delta_pf=deepcopy(row['actual_delta_pf']), identity=deepcopy(row['identity']),
            law_binding=deepcopy(row['law_binding']), v2_candidate_id=row['candidate']['id'],
            physical_candidate=deepcopy(physical), physical_candidate_id=None if physical is None else physical['id'],
            physical_schedule_mapping=deepcopy(row['mapping']), saved_asset_status=saved,
            comparison_method=B.comparison_method(s), automatic_execution=False)
        A = dict(common, owner='Codex A', quantity='SUFFICIENT_N_AND_GLOBAL_RMSE_EVIDENCE',
            physics_source=deepcopy(SOURCES['full_h_prepare'] if full_h else SOURCES['pf_signal']),
            signal_functions=(['h46_full_h_pure_rte.context(prepare_full_h)', 'h46_long_time_signal.evolve_batch(prefix=0)']
                              if full_h else ['h46_long_time_signal.context', 'h46_long_time_signal.evolve_batch(method=FULL)']),
            preparation=deepcopy(s['split'].get('preparation')), normalization=('FINITE_K_FULL_H_LAMBDA' if full_h else 'NONE_DETERMINISTIC'),
            saved_reference_time=reference, N_X=None, N_Y=None, rmse_bound_ha=None,
            missing=(['FULL_H_FINITE_MEAN_AND_NORMALIZATION', 'RMSE_SUFFICIENT_N', 'GLOBAL_RMSE_CONTRACT_FOR_FULL_H_RTE',
                      'FINITE_K_AND_SAMPLER_LAW_QUALIFICATION'] if full_h else
                     ['FULL_PF_MEAN_OR_SAVED_SIGNAL_AT_T', 'RMSE_SUFFICIENT_N', 'GLOBAL_RMSE_CONTRACT_FOR_FULL_PF'])
            + ([] if reference else ['NEW_REFERENCE_AT_T']))
        C = []
        for axis in p.AXES:
            C.append(dict(common, owner='Codex C', axis=axis, **p.measurement(row, axis, saved_sampler=full_h), profile=deepcopy(s['profile']),
                quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE', construction=deepcopy(SOURCES['pf_native']),
                construction_mode=('controlled RTE sequences per S2 outer step from prepare_full_h; no deterministic block; scalar phase once'
                                   if full_h else 'deterministic signed S2 pieces of all one-body+DF blocks; scalar phase once'),
                price_law=('SAMPLER_COHORT_DISTANCE_TO_IDEAL_UNQUALIFIED' if full_h else 'DETERMINISTIC'),
                saved_price_source=deepcopy(SOURCES['full_h_saved_prices']) if saved.startswith('SAVED_LIGHT') else None,
                mean=dict(rz_count=None), price_status='NOT_EVALUATED', classification=None, expectation=None,
                model=None, uncertainty=None, event_count_alone_is_price_equivalence=False))
        out.append(dict(m=row['m'], A=A, C=C))
    return dict(configuration_id=config['id'], method=B.comparison_method(s), rounds=out,
        binding=deepcopy(s['binding']), profile=deepcopy(s['profile']), rmse_policy=deepcopy(s['rmse_policy']),
        N_C_values=None, G=None, scientific_conditions_adopted=False, science_calls=0, mandatory_stop=True)


def validate_requests(config, packet):
    """Validate the external physical bridge, not any N/C scientific evidence.

    Full identity includes the whole schedule, H/state/SOURCE and compiler/law;
    a legacy physical ID alone never qualifies a record for this configuration.
    """
    if not p.same(packet, requests(config)):
        raise ValueError('D physical request identity/SOURCE/law/compiler bridge mismatch')
    return True
