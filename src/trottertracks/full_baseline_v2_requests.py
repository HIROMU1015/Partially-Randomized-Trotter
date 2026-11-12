"""Full PF and DF Full-H RTE physical requests for B PR_RPE_RMSE_V2. Metadata only.

A physical schedule is built from the pinned physical SOURCE first. It is bound
to a B V2 configuration only where the published B interface can generate it
(Full PF 2nd / 4th(m5_best)). Standard Yoshida4 Full PF and kappa-free explicit
Full-H RTE layouts are kept as physical requests whose V2 IDs stay null until
B publishes the matching interface; no V2 ID is invented. A (N/RMSE) and C
(full-wrapper price) requests always point to the same physical candidate.

Nothing here loads a Hamiltonian or state, evaluates a signal/mean/N or
normalization, samples, builds or compiles a circuit, prices, fits, or chooses
M, kappa, PF width, formula or RTE layout. Eighth-order PF is out of scope.
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
BASE = 'af284d853890155bacd19e61850c196bd1d5c660'          # B V2 + evidence planner (controller unchanged at 030f3f548)
FULL_H = '9b83b96b83f62142c9d87858dc24a94db7ac5d61'        # DF full-H prepare_full_h + light study
FULL_H_T20 = '7a8bc7f132e166390c943e6b32c92cc6abe1baf4'    # later T=2.0 study (s=qR mean sharing)
LIGHT = 'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/'
LAYOUT_KIND = 'D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1'
MODULE = dict(branch='codex/d-full-pf-full-h-rte-v2-paths-20261113', path='src/trottertracks/full_baseline_v2_requests.py',
              module='trottertracks.full_baseline_v2_requests')
SAVED_FULL_H_REFERENCE_T = (.8, 2.)  # light study context and later T=2.0 study
PENDING_FORMULA = 'PENDING_B_FORMULA_INTERFACE'
PENDING_LAYOUT = 'PENDING_B_KAPPA_FREE_EXPLICIT_LAYOUT_INTERFACE'


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

# Static facts from the pinned SOURCE. b_v2_generate mirrors the published B.generate.
FORMULAS = {
    '2nd': dict(family='S2', formula_id='S2', coefficient_source='pf_weights', negative_time=False,
                h46_signal_path=True, h46_native_path=True, b_v2_generate=True, d_request=True),
    '4th': dict(family='STANDARD_YOSHIDA4', formula_id='YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64',
                coefficient_source='pf_weights', negative_time=True, h46_signal_path=True, h46_native_path=True,
                b_v2_generate=False, d_request=True, not_alias_of='4th(m5_best)'),
    '4th(m5_best)': dict(family='M5_BEST_BINARY64', formula_id='PF_M5_BEST_20D64C2_BINARY64_V1',
                         coefficient_source='pf_weights', negative_time=True, h46_signal_path=True,
                         h46_native_path=True, b_v2_generate=True, d_request=True),
    '8th(Morales)': dict(family='MORALES8', coefficient_source='pf_higher_order', negative_time=True,
                         h46_signal_path=False, h46_native_path=False, b_v2_generate=False, d_request=False),
    '8th(Yoshida)': dict(family='YOSHIDA8', coefficient_source='pf_higher_order', negative_time=True,
                         h46_signal_path=False, h46_native_path=False, b_v2_generate=False, d_request=False),
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


def _status(fact):
    if fact['b_v2_generate']:
        return 'B_V2_SUPPORTED'
    if fact['d_request']:
        return 'D_PHYSICAL_REQUEST_READY_' + PENDING_FORMULA
    return 'OUT_OF_SCOPE_NO_DF_CONTROLLED_FULL_WRAPPER_PATH'


def catalog():
    """SOURCE catalog with V2 status; weights are the pinned binary64 literals."""
    schedule = p._schedule()
    out = {}
    for label, fact in FORMULAS.items():
        weights = list(schedule.weights(label)) if fact['coefficient_source'] == 'pf_weights' else None
        out[label] = dict(fact, coefficient_source=deepcopy(SOURCES[fact['coefficient_source']]),
                          signed_weights=weights, v2_status=_status(fact))
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
    if list(prep['randomized_fragments']) != list(range(prep['all_fragments'])):
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
    if not FORMULAS[formula]['d_request']:
        raise ValueError('Eighth-order PF is out of scope for this adapter')
    if rank != base['binding'].get('rank') or rank != saved_grid()['rank'][base['binding']['system']]:
        raise ValueError('Full PF retains one-body and all saved DF fragments; prefix must equal the full rank')
    source = deepcopy(SOURCES['pf_signal'])
    return dict(_common(base), method='PF', L_D=rank, delta_pf=delta_pf, formula=formula, M=M, kappa=None, finite_K=None,
        split=dict(L_D=rank, source=source, randomized_scope='NONE', deterministic_scope='FULL_H',
                   original_PF_method='FULL', lambda_R=None, lambda_source=None,
                   retained_hamiltonian=dict(construction='FULL_H', prefix=rank, source=source,
                                             target_hamiltonian=deepcopy(base['binding']['hamiltonian']))),
        law=dict(model='deterministic', source=deepcopy(SOURCES['pf_weights'])),
        event_mapping=dict(kind='NONE', source=deepcopy(SOURCES['pf_weights'])))


def full_h_rte_spec(base, *, M, finite_K, preparation, kappa=None):
    """Explicit-layout Full-H pure RTE. No kappa: the layout itself is the input."""
    check_full_h_preparation(preparation, base['binding'])
    if kappa is not None:
        raise ValueError('Explicit-layout Full-H RTE has no kappa; kappa-derived layouts belong to the B generator')
    if type(finite_K) is not int or finite_K < 0 or finite_K % 2:
        raise ValueError('Even nonnegative finite K required (paired finite Taylor)')
    split = dict(L_D=0, source=deepcopy(preparation['source']), randomized_scope='FULL_H_INCLUDING_ONE_BODY',
        one_body_policy='RANDOMIZED', deterministic_scope='NONE', original_PF_method='NONE_FULL_H_PREPARATION',
        lambda_R=preparation['lambda_R'], lambda_source=deepcopy(preparation['lambda_source']),
        preparation=deepcopy(preparation))
    return dict(_common(base), method='FULL_H_RTE', L_D=0, delta_pf=None, formula=None, M=M, kappa=None,
        finite_K=finite_K, split=split,
        law=dict(model='ideal_finite_RTE_law', sampling='PAIRED_FINITE_TAYLOR_COMPONENT_ORDER_LAW',
                 source=deepcopy(SOURCES['rte_law']), preparation_source=deepcopy(preparation['source'])),
        event_mapping=dict(kind=LAYOUT_KIND, source=deepcopy(MODULE), kappa_mode='NOT_DEFINED_EXPLICIT_LAYOUT',
                           paper_kappa_optimality_claim=False))


def _layouts(layout_by_round, M):
    layouts = {}
    for m, v in (layout_by_round or {}).items():
        if (type(m) is not int or m not in range(M + 1) or set(v) != {'q', 'R_outer'}
                or any(type(v[k]) is not int or v[k] < 1 for k in v)):
            raise ValueError('Explicit integer q and R_outer per round required')
        layouts[m] = dict(v)
    return layouts


def layout_mapping(spec, layout):
    """Round mapping in the shape B's event-mapping records use; kappa stays None."""
    q, R = layout['q'], layout['R_outer']
    return dict(kind=spec['event_mapping']['kind'], source=deepcopy(spec['event_mapping']['source']), kappa=None,
        R_outer=None, event_count=q*R, physical_layout=dict(formula='2nd', grouping='S2_OUTER_Q_TIMES_R', q=q,
            R_outer=R, pieces_per_outer=1, allocation=[R], allocation_policy='ABS_TIME_PER_OUTER', event_positions=q*R,
            event_tau='lambda_R*(T/q)/R_outer', taylor_K=spec['finite_K'], signed_times='ALL_POSITIVE_S2',
            scalar='ONCE_AT_TOTAL_T', deterministic_blocks=0, component_law=deepcopy(spec['law']),
            compose_source=deepcopy(SOURCES['pf_native']), priced_equivalence_claim=False,
            paper_kappa_optimality_claim=False))


def physical_schedule(spec, *, layout_by_round=None):
    """Per-round physical recipe and original physical candidate ID. No B call."""
    if spec['method'] not in ('PF', 'FULL_H_RTE') or (spec['method'] == 'PF' and spec['split'].get('deterministic_scope') != 'FULL_H'):
        raise ValueError('D requests cover Full PF and Full-H RTE only')
    full_h = spec['method'] == 'FULL_H_RTE'
    if full_h:
        layouts = _layouts(layout_by_round, spec['M'])
    elif layout_by_round is not None:
        raise ValueError('Full PF has no RTE layout')
    system = spec['binding']['system']; grid = saved_grid(); rounds = []
    saved_light = ({v['id'] for v in json.loads(_blob(FULL_H, SOURCES['full_h_saved_selection']['path']))['candidates']}
                   if full_h else set())
    for m in range(spec['M'] + 1):
        T = B._positive(spec['h']) * (1 << m); Tf = float(T)
        row = dict(m=m, q_RPE=1 << m, T=B.a.exact_value(T), T_float=Tf, scalar='ONCE_AT_TOTAL_T')
        if not full_h:
            q = B._ceil(T / B._positive(spec['delta_pf'])); fact = FORMULAS[spec['formula']]
            weights = list(p._schedule().weights(spec['formula']))
            c = dict(system=system, method='FULL', formula=spec['formula'], prefix=spec['L_D'], T=Tf, q=q,
                     delta=Tf/q, R_outer=None, K=None)
            c['id'] = p.candidate_id(c)
            row.update(q_PF=q, requested_width=deepcopy(spec['delta_pf']), actual_width=B.a.exact_value(T/q),
                formula=spec['formula'], formula_id=fact['formula_id'], signed_weights=weights,
                negative_time=any(w < 0 for w in weights), piece_time='T*w_j/q', outer_steps=q,
                retained='FULL_H_ONE_BODY_PLUS_ALL_FRAGMENTS', prefix=spec['L_D'], fragment_order='SAVED_SNAPSHOT_ORDER',
                law_model='deterministic', normalization='NONE_DETERMINISTIC', control='symmetric_directional',
                physical_candidate=c, physical_candidate_id=c['id'],
                saved_asset_status=('SAVED_LONG_TIME_GRID_POINT_CHECK_CARRIER' if Tf in grid['q'] and q in grid['q'][Tf]
                                    and spec['formula'] in grid['formulas'] else 'NO_SAVED_GRID_POINT'),
                saved_reference_time=Tf in grid['q'])
        else:
            layout = layouts.get(m)
            row.update(q_PF=None, law_model='ideal_finite_RTE_law', normalization='FINITE_K_FULL_H_LAMBDA_NOT_EVALUATED',
                       control='symmetric_directional', saved_reference_time=Tf in SAVED_FULL_H_REFERENCE_T,
                       lambda_R=spec['split']['lambda_R'], K=spec['finite_K'],
                       preparation_hash=spec['split']['preparation']['preparation_hash'])
            if layout is None:
                row.update(mapping=None, physical_candidate=None, physical_candidate_id=None, saved_asset_status='LAYOUT_NOT_SUPPLIED')
            else:
                mapping = layout_mapping(spec, layout)
                c = dict(system=system, method='PR', formula='2nd', prefix=0, T=Tf, q=layout['q'], R_outer=layout['R_outer'],
                         K=spec['finite_K'], delta=Tf/layout['q'], s=mapping['event_count'], family='FULL_H_PURE_RTE', L_D=0)
                c['id'] = 'FULL_H_' + p.candidate_id(c)
                row.update(mapping=mapping, outer_step_width=B.a.exact_value(T/layout['q']), event_count=mapping['event_count'],
                           physical_candidate=c, physical_candidate_id=c['id'],
                           saved_asset_status=('SAVED_LIGHT_STUDY_CANDIDATE_FINITE_COHORT' if c['id'] in saved_light
                                               else 'NO_SAVED_EXACT_CANDIDATE'))
        rounds.append(row)
    return dict(method=B.comparison_method(spec) if spec['method'] == 'PF' else 'FULL_H_RTE', spec=deepcopy(spec),
                binding=deepcopy(spec['binding']), profile=deepcopy(spec['profile']), rounds=rounds)


def _check_against_b(physical, config):
    """A published-B configuration must reproduce the physical recipe exactly."""
    for row, v2 in zip(physical['rounds'], config['rounds']):
        if (v2['m'] != row['m'] or not p.same(v2['T'], row['T']) or v2['q_PF'] != row['q_PF']
                or not p.same(v2['actual_delta_pf'], row['actual_width'])
                or v2['structure']['signed_weights'] != row['signed_weights']
                or v2['structure']['scalar_policy'] != 'scalar_identity_once_at_total_T'):
            raise ValueError('B V2 round differs from the physical recipe at m=' + str(row['m']))


def bind(spec, *, layout_by_round=None):
    """Physical schedule + published-B V2 configuration where the interface exists."""
    physical = physical_schedule(spec, layout_by_round=layout_by_round)
    config = None
    if spec['method'] == 'PF' and FORMULAS[spec['formula']]['b_v2_generate']:
        config = B.generate(spec); _check_against_b(physical, config); status = 'B_V2_GENERATED'
    elif spec['method'] == 'PF':
        status = PENDING_FORMULA
    else:
        status = PENDING_LAYOUT
    rows = [dict(m=r['m'], physical_candidate=deepcopy(r['physical_candidate']), V2_configuration_id=None if config is None else config['id'],
                 V2_candidate_id=None if config is None else config['rounds'][r['m']]['candidate']['id'],
                 source=deepcopy(SOURCES['full_h_prepare'] if spec['method'] == 'FULL_H_RTE' else SOURCES['pf_signal']))
            for r in physical['rounds']]
    return dict(physical=physical, v2_configuration=config, v2_status=status, identity_map=rows,
                b_interface=dict(controller_commit_checked='030f3f548a35d98a88e62e92a35f132e0243a293',
                                 formula_hook=False, kappa_free_explicit_layout=False, physical_candidate_hook=False),
                invented_v2_ids=False)


def generate(spec, *, layout_by_round=None):
    """Published B.generate only (Full PF 2nd / m5_best). Pending cases raise."""
    packet = bind(spec, layout_by_round=layout_by_round)
    if packet['v2_configuration'] is None:
        raise ValueError(packet['v2_status'])
    return packet['v2_configuration']


def requests(packet):
    """A (N/RMSE) and C (price) requests per round, same physical candidate. No values."""
    if packet.get('schema') == B.SCHEMA:   # a published-B configuration
        B._validate(packet)
        packet = bind(packet['spec'])
    physical = packet['physical']; config = packet['v2_configuration']; s = physical['spec']
    full_h = s['method'] == 'FULL_H_RTE'; out = []
    for row in physical['rounds']:
        v2 = None if config is None else config['rounds'][row['m']]
        common = dict(m=row['m'], T=deepcopy(row['T']), q_RPE=row['q_RPE'], q_PF=row['q_PF'],
            v2_status=packet['v2_status'], v2_configuration_id=None if config is None else config['id'],
            v2_candidate_id=None if v2 is None else v2['candidate']['id'],
            identity=None if v2 is None else deepcopy(v2['identity']), law_binding=None if v2 is None else deepcopy(v2['law_binding']),
            binding=deepcopy(s['binding']), physical_round=deepcopy(row),
            physical_candidate=deepcopy(row['physical_candidate']), physical_candidate_id=row['physical_candidate_id'],
            saved_asset_status=row['saved_asset_status'], comparison_method=physical['method'], automatic_execution=False)
        A = dict(common, owner='Codex A', quantity='SUFFICIENT_N_AND_GLOBAL_RMSE_EVIDENCE',
            physics_source=deepcopy(SOURCES['full_h_prepare'] if full_h else SOURCES['pf_signal']),
            signal_functions=(['h46_full_h_pure_rte.context(prepare_full_h)', 'h46_long_time_signal.evolve_batch(prefix=0)']
                              if full_h else ['h46_long_time_signal.context', 'h46_long_time_signal.evolve_batch(method=FULL)']),
            preparation=deepcopy(s['split'].get('preparation')), normalization=row['normalization'],
            rmse_policy=deepcopy(s['rmse_policy']), saved_reference_time=row['saved_reference_time'],
            N_X=None, N_Y=None, rmse_bound_ha=None,
            missing=(['FULL_H_FINITE_MEAN_AND_NORMALIZATION', 'RMSE_SUFFICIENT_N', 'GLOBAL_RMSE_DECODER_CONTRACT_FOR_FULL_H_RTE',
                      'FINITE_K_AND_SAMPLER_LAW_QUALIFICATION'] if full_h else
                     ['FULL_H_ACTION_MEAN_OR_SAVED_SIGNAL_AT_T', 'RMSE_SUFFICIENT_N', 'GLOBAL_RMSE_CONTRACT_FOR_FULL_PF'])
            + ([] if row['saved_reference_time'] else ['NEW_REFERENCE_AT_T'])
            + (['LAYOUT_NOT_SUPPLIED'] if row['physical_candidate_id'] is None else [])
            + ([] if config is not None else [packet['v2_status']]))
        C = []
        for axis in p.AXES:
            C.append(dict(common, owner='Codex C', axis=axis,
                **p.measurement(dict(structure=dict(law_model=row['law_model'])), axis, saved_sampler=full_h),
                profile=deepcopy(s['profile']), quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE',
                construction=deepcopy(SOURCES['pf_native']),
                construction_mode=('controlled RTE sequences per S2 outer step from prepare_full_h; no deterministic block; scalar phase once'
                                   if full_h else 'deterministic signed S2 pieces of all one-body+DF blocks; scalar phase once'),
                price_law=('SAMPLER_COHORT_DISTANCE_TO_IDEAL_UNQUALIFIED' if full_h else 'DETERMINISTIC'),
                expectation_needed=('COHORT_OR_ALL_LAW_MEAN_SEPARATED' if full_h else 'DETERMINISTIC'),
                saved_price_source=deepcopy(SOURCES['full_h_saved_prices']) if row['saved_asset_status'].startswith('SAVED_LIGHT') else None,
                saved_price_match_requires=(['SAME_LAYOUT_q_R_K', 'SAME_T', 'SAME_PREPARATION_HASH', 'SAME_LAW_OR_COHORT_DECLARATION']
                                            if full_h else ['SAME_FORMULA', 'SAME_T', 'SAME_q', 'FULL_PREFIX', 'SAME_PROFILE']),
                mean=dict(rz_count=None), price_status='NOT_EVALUATED', classification=None, expectation=None,
                model=None, uncertainty=None, event_count_alone_is_price_equivalence=False))
        out.append(dict(m=row['m'], A=A, C=C))
    return dict(method=physical['method'], v2_status=packet['v2_status'],
        v2_configuration_id=None if config is None else config['id'], identity_map=deepcopy(packet['identity_map']),
        rounds=out, binding=deepcopy(s['binding']), profile=deepcopy(s['profile']), rmse_policy=deepcopy(s['rmse_policy']),
        N_C_values=None, G=None, scientific_conditions_adopted=False, science_calls=0, mandatory_stop=True)
