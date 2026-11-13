"""Full PF and DF Full-H RTE physical requests for B PR_RPE_RMSE_V2. Metadata only.

A physical schedule is built from the pinned physical SOURCE first and then bound
to the configuration that the published B interface (B_LATEST) generates for the
same spec: Full PF 2nd / standard Yoshida4 / 4th(m5_best), and Full-H RTE with an
explicit q x R layout either without kappa (B.EXPLICIT_LAYOUT) or with a supplied,
source-bound kappa (legacy LAYOUT_KIND). Every round is checked against the
physical recipe; V2 IDs are only those B generated. A (N/RMSE) and C
(full-wrapper price) requests point to the same physical candidate.

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
BASE = 'af284d853890155bacd19e61850c196bd1d5c660'          # pinned physical SOURCE (byte-identical at B_LATEST)
B_LATEST = 'f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8'      # B V2: Yoshida4 PF + kappa-free explicit Full-H layout
B_INTERFACE = dict(commit=B_LATEST, path='docs/research/abc_d_v2_integration_20261114/INTERFACE.md',
                   pre_publication='575598983059fdf5238da877f62b40250fd9eae9')
FULL_H = '9b83b96b83f62142c9d87858dc24a94db7ac5d61'        # DF full-H prepare_full_h + light study
FULL_H_T20 = '7a8bc7f132e166390c943e6b32c92cc6abe1baf4'    # later T=2.0 study (s=qR mean sharing)
LIGHT = 'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/'
LAYOUT_KIND = 'D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1'      # explicit layout WITH a supplied kappa (legacy)
EXPLICIT_LAYOUT = B.EXPLICIT_LAYOUT                        # 'FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1'
MODULE = dict(branch='codex/d-full-pf-full-h-rte-v2-paths-20261113', path='src/trottertracks/full_baseline_v2_requests.py',
              module='trottertracks.full_baseline_v2_requests')
SAVED_FULL_H_REFERENCE_T = (.8, 2.)  # light study context and later T=2.0 study
PENDING_FORMULA = 'PENDING_B_FORMULA_INTERFACE'
PENDING_LAYOUT = 'PENDING_B_KAPPA_FREE_EXPLICIT_LAYOUT_INTERFACE'
# Pending states of D 0536646b, resolved by the B commit above (kept for provenance).
RESOLVED_PENDING = {
    PENDING_FORMULA: dict(resolved_by=B_LATEST, interface=B_INTERFACE, previous_D_commit='0536646b843781a0e0fb6629473526c05f16a30e',
        condition="B pf_pr_rpe_pipeline.structure/generate accept method PF formula '4th' with the pinned "
                  "h46_m5_plan.weights('4th') binary64 tuple and formula_id PF_STANDARD_YOSHIDA4_BINARY64_V1"),
    PENDING_LAYOUT: dict(resolved_by=B_LATEST, interface=B_INTERFACE, previous_D_commit='0536646b843781a0e0fb6629473526c05f16a30e',
        condition='B generate accepts FULL_H_RTE with kappa=None under kind FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1 and '
                  'checks the D q x R layout, K, scalar, preparation and priced_equivalence_claim=false'),
}


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
    b_formula_id=ref(B_LATEST, 'src/trottertracks/pf_pr_rpe_pipeline.py', functions=['structure']),
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

# Static facts from the pinned SOURCE. formula_id is B's canonical ID and is
# checked against B.generate output for every round; physical_formula_label is
# the D 0536646b label of the same h46_m5_plan.weights tuple (provenance only).
FORMULAS = {
    '2nd': dict(family='S2', formula_id='S2', coefficient_source='pf_weights', negative_time=False,
                h46_signal_path=True, h46_native_path=True, b_v2_generate=True, d_request=True),
    '4th': dict(family='STANDARD_YOSHIDA4', formula_id='PF_STANDARD_YOSHIDA4_BINARY64_V1',
                physical_formula_label='YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64', formula_id_source='b_formula_id',
                coefficient_source='pf_weights', negative_time=True, h46_signal_path=True, h46_native_path=True,
                b_v2_generate=True, d_request=True, not_alias_of='4th(m5_best)'),
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
                b_interface=deepcopy(B_INTERFACE), resolved_pending=deepcopy(RESOLVED_PENDING),
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
    """Full-H scope: every non-scalar term randomized, one-body included, scalar once.

    Also used by B's controller (_explicit_preparation) for the kappa-free kind.
    """
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
    """Explicit-layout Full-H pure RTE.

    kappa=None -> B kind FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1 (the layout is the input).
    A supplied, named source-bound kappa keeps the legacy LAYOUT_KIND; D never
    creates or back-solves a kappa.
    """
    check_full_h_preparation(preparation, base['binding'])
    if kappa is not None and (not kappa.get('kind') or not kappa.get('definition_source')):
        raise ValueError('A supplied kappa needs a named source-bound definition')
    if type(finite_K) is not int or finite_K < 0 or finite_K % 2:
        raise ValueError('Even nonnegative finite K required (paired finite Taylor)')
    split = dict(L_D=0, source=deepcopy(preparation['source']), randomized_scope='FULL_H_INCLUDING_ONE_BODY',
        one_body_policy='RANDOMIZED', deterministic_scope='NONE', original_PF_method='NONE_FULL_H_PREPARATION',
        lambda_R=preparation['lambda_R'], lambda_source=deepcopy(preparation['lambda_source']),
        preparation=deepcopy(preparation))
    return dict(_common(base), method='FULL_H_RTE', L_D=0, delta_pf=None, formula=None, M=M, kappa=deepcopy(kappa),
        finite_K=finite_K, split=split,
        law=dict(model='ideal_finite_RTE_law', sampling='PAIRED_FINITE_TAYLOR_COMPONENT_ORDER_LAW',
                 source=deepcopy(SOURCES['rte_law']), preparation_source=deepcopy(preparation['source'])),
        event_mapping=dict(kind=EXPLICIT_LAYOUT if kappa is None else LAYOUT_KIND,
            source=(dict(MODULE, extension_interface=dict(commit=B_INTERFACE['pre_publication'], path=B_INTERFACE['path']))
                    if kappa is None else deepcopy(MODULE))))


def _layouts(layout_by_round, M):
    layouts = {}
    for m, v in (layout_by_round or {}).items():
        if (type(m) is not int or m not in range(M + 1) or set(v) != {'q', 'R_outer'}
                or any(type(v[k]) is not int or v[k] < 1 for k in v)):
            raise ValueError('Explicit integer q and R_outer per round required')
        layouts[m] = dict(v)
    return layouts


def layout_mapping(spec, layout):
    """B round mapping for one explicit S2 grouping (q outer steps x R events); kappa as supplied."""
    q, R = layout['q'], layout['R_outer']
    return dict(kind=spec['event_mapping']['kind'], source=deepcopy(spec['event_mapping']['source']), kappa=deepcopy(spec['kappa']),
        R_outer=None, event_count=q*R, physical_layout=dict(formula='2nd', q=q, R_outer=R, pieces_per_outer=1,
            allocation=[R], allocation_policy='ABS_TIME_PER_OUTER', event_positions=q*R,
            event_tau='lambda_R*(T/q)/R_outer', taylor_K=spec['finite_K'], signed_times='ALL_POSITIVE_S2',
            scalar='ONCE_AT_TOTAL_T', deterministic_blocks=0, compose_source=deepcopy(SOURCES['pf_native']),
            priced_equivalence_claim=False))


def layout_mapper(layout_by_round):
    """B event_mapper for explicit per-round layouts. Missing round -> None (left unresolved)."""
    layouts = {}
    for m, v in layout_by_round.items():
        if (type(m) is not int or m < 0 or set(v) != {'q', 'R_outer'} or any(type(v[k]) is not int or v[k] < 1 for k in v)):
            raise ValueError('Explicit integer q and R_outer per round required')
        layouts[m] = dict(v)

    def mapper(s, row):
        v = layouts.get(row['m'])
        return None if v is None else layout_mapping(s, v)
    return mapper


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
                formula=spec['formula'], formula_id=fact['formula_id'],
                physical_formula_label=fact.get('physical_formula_label', fact['formula_id']), signed_weights=weights,
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
                       preparation_hash=spec['split']['preparation']['preparation_hash'],
                       parameterization='EXPLICIT_LAYOUT_NO_KAPPA' if spec['kappa'] is None else 'EXPLICIT_LAYOUT_SUPPLIED_KAPPA',
                       grouping='S2_OUTER_Q_TIMES_R', component_law=deepcopy(spec['law']), paper_kappa_optimality_claim=False)
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
    """The published-B configuration must reproduce the physical recipe round by round."""
    s = physical['spec']; full_h = s['method'] == 'FULL_H_RTE'
    if len(config['rounds']) != len(physical['rounds']) or B.comparison_method(config['spec']) != physical['method']:
        raise ValueError('B V2 configuration does not cover the physical rounds')
    for row, v2 in zip(physical['rounds'], config['rounds']):
        st, cand = v2['structure'], v2['candidate']
        bad = (v2['m'] != row['m'] or not p.same(v2['T'], row['T']) or v2['q_PF'] != row['q_PF']
               or st['scalar_policy'] != 'scalar_identity_once_at_total_T' or not p.same(v2['identity']['binding'], s['binding']))
        if not full_h:
            bad = bad or (not p.same(v2['actual_delta_pf'], row['actual_width']) or st['signed_weights'] != row['signed_weights']
                          or st['formula_id'] != row['formula_id'] or cand['formula'] != row['formula']
                          or cand['prefix'] != row['prefix'] or cand['K'] is not None or cand['R_outer'] is not None
                          or cand['retained_hamiltonian'].get('construction') != 'FULL_H')
        else:
            # B stores a supplied kappa with an exact value; compare the rest of the mapping as is.
            kappa = config['spec']['kappa']
            same_kappa = (kappa is None) == (s['kappa'] is None) and (kappa is None or all(
                p.same(kappa.get(k), s['kappa'].get(k)) for k in ('kind', 'definition_source')))
            mapping = None if v2['mapping'] is None else dict(v2['mapping'], kappa=None)
            physical = None if row['mapping'] is None else dict(row['mapping'], kappa=None)
            bad = bad or (not same_kappa or not p.same(mapping, physical) or cand['K'] != row['K']
                          or (v2['mapping'] is not None and not p.same(v2['mapping']['kappa'], kappa))
                          or st['event_count'] != row.get('event_count')
                          or v2['resolution'] != ('GENERATED' if row['mapping'] is not None else 'NOT_EVALUATED')
                          or not p.same(v2['law_binding']['kappa'], kappa))
        if bad:
            raise ValueError('B V2 round differs from the physical recipe at m=' + str(row['m']))


def bind(spec, *, layout_by_round=None):
    """Physical schedule + the published-B V2 configuration generated from the same spec."""
    if spec['method'] == 'FULL_H_RTE':
        if spec['event_mapping']['kind'] not in (LAYOUT_KIND, EXPLICIT_LAYOUT) or layout_by_round is None:
            raise ValueError('Full-H RTE needs an explicit physical layout; event_count alone is not a grouping')
    physical = physical_schedule(spec, layout_by_round=layout_by_round)
    if spec['method'] == 'PF':
        config = B.generate(spec)
    else:
        config = B.generate(spec, event_mapper=layout_mapper(layout_by_round))
    _check_against_b(physical, config)
    full_h = spec['method'] == 'FULL_H_RTE'
    rows = []
    for r, v2 in zip(physical['rounds'], config['rounds']):
        rows.append(dict(m=r['m'], physical_candidate=deepcopy(r['physical_candidate']), physical_candidate_id=r['physical_candidate_id'],
            V2_configuration_id=config['id'], V2_candidate_id=v2['candidate']['id'], V2_resolution=v2['resolution'],
            V2_formula_id=v2['structure']['formula_id'], physical_formula_label=r.get('physical_formula_label'),
            source=deepcopy(SOURCES['full_h_prepare'] if full_h else SOURCES['pf_signal']), b_commit=B_LATEST))
    resolved = ([PENDING_LAYOUT] if full_h and spec['kappa'] is None else
                [PENDING_FORMULA] if not full_h and spec['formula'] == '4th' else [])
    return dict(physical=physical, v2_configuration=config, v2_status='B_V2_GENERATED', identity_map=rows,
                b_interface=dict(controller_commit=B_LATEST, interface=deepcopy(B_INTERFACE), formula_hook=True,
                                 kappa_free_explicit_layout=True, physical_candidate_hook=False,
                                 physical_bridge='D_EXTERNAL_REQUEST_TABLE'),
                resolved_pending={k: deepcopy(RESOLVED_PENDING[k]) for k in resolved}, invented_v2_ids=False)


def generate(spec, *, layout_by_round=None):
    """B V2 configuration for a D spec (via bind, so every round is checked)."""
    return bind(spec, layout_by_round=layout_by_round)['v2_configuration']


def _from_config(config):
    """Rebuild the bind() packet of a published-B configuration and require the same ID."""
    B._validate(config)
    s = config['spec']
    if s['method'] not in ('PF', 'FULL_H_RTE') or (s['method'] == 'PF' and B.comparison_method(s) != 'FULL_PF'):
        raise ValueError('D requests cover Full PF and Full-H RTE only')
    layouts = None
    if s['method'] == 'FULL_H_RTE':
        layouts = {r['m']: {k: r['mapping']['physical_layout'][k] for k in ('q', 'R_outer')}
                   for r in config['rounds'] if r['mapping'] is not None}
    packet = bind(s, layout_by_round=layouts)
    if packet['v2_configuration']['id'] != config['id'] or not p.same(packet['v2_configuration'], config):
        raise ValueError('Configuration is not reproduced by the D physical schedule; no relabelling')
    return packet


def requests(packet):
    """A (N/RMSE) and C (price) requests per round, same physical candidate. No values.

    Accepts a bind() packet or a published-B configuration (rebuilt and ID-checked).
    """
    if packet.get('schema') == B.SCHEMA:
        packet = _from_config(packet)
    physical = packet['physical']; config = packet['v2_configuration']; s = physical['spec']
    full_h = s['method'] == 'FULL_H_RTE'; out = []
    for row, v2 in zip(physical['rounds'], config['rounds']):
        common = dict(configuration_id=config['id'], m=row['m'], T=deepcopy(row['T']), q_RPE=row['q_RPE'], q_PF=row['q_PF'],
            actual_delta_pf=deepcopy(v2['actual_delta_pf']), v2_status=packet['v2_status'],
            v2_candidate_id=v2['candidate']['id'], v2_resolution=v2['resolution'],
            identity=deepcopy(v2['identity']), law_binding=deepcopy(v2['law_binding']),
            binding=deepcopy(s['binding']), physical_round=deepcopy(row),
            physical_candidate=deepcopy(row['physical_candidate']), physical_candidate_id=row['physical_candidate_id'],
            physical_schedule_mapping=deepcopy(v2['mapping']), saved_asset_status=row['saved_asset_status'],
            comparison_method=physical['method'], b_commit=B_LATEST, automatic_execution=False)
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
            + (['LAYOUT_NOT_SUPPLIED'] if row['physical_candidate_id'] is None else []))
        C = []
        for axis in p.AXES:
            C.append(dict(common, owner='Codex C', axis=axis, **p.measurement(v2, axis, saved_sampler=full_h),
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
    return dict(configuration_id=config['id'], method=physical['method'], v2_status=packet['v2_status'],
        identity_map=deepcopy(packet['identity_map']), resolved_pending=deepcopy(packet['resolved_pending']),
        rounds=out, binding=deepcopy(s['binding']), profile=deepcopy(s['profile']), rmse_policy=deepcopy(s['rmse_policy']),
        N_C_values=None, G=None, scientific_conditions_adopted=False, science_calls=0, mandatory_stop=True)


def validate_requests(config, packet):
    """Validate the external physical bridge, not any N/C scientific evidence.

    Full identity includes the whole schedule, H/state/SOURCE and compiler/law;
    a legacy physical ID alone never qualifies a record for this configuration.
    """
    if not p.same(packet, requests(config)):
        raise ValueError('D physical request identity/SOURCE/law/compiler bridge mismatch')
    return True
