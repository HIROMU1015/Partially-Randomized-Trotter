"""Pinned A LD1/LD2 menu -> selectable D price requests, metadata only.

No runtime/context/preparation, sampling, circuit, compile, mean or G calls.
Full canonical objects stay at their pinned Git sources; compact publication
indexes are checked by materializing a request only when it is selected.
"""
from copy import deepcopy
from fractions import Fraction

from . import candidate_price_adapter as D

A_SOURCE = 'a9abf8de75bf3432fcd714e868e5c14c8884b8f8'
A_DIR = 'docs/research/pr_rpe_first_signal_menu_20261113/'
D_BASE = '0c6e16342b3c804fa0bd89a4028d6b5bea7b0322'
B_SOURCES = ('437ba4acdb82708b6fc98e3cd119a2a734f7cf64', 'b9b9ffa6233db9dd2b413dc07b27e7039324c666')
LABELS = ('PR-LD1', 'PR-LD2')
P2 = '39ddd59edf54a3a901e5302207b2765ddbd03fbb'
STRUCTURAL = '9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f'


def _ref(path, **selector):
    return dict(commit=A_SOURCE, path=A_DIR+path, **selector)


def _read(root):
    store = D.p.SavedArtifacts(root)
    menu = store.read(_ref('MENU.json'))
    if menu['B_source'] != B_SOURCES[0] or tuple(v['label'] for v in menu['candidates']) != LABELS:
        raise ValueError('Fixed A menu/B source differs')
    configs = {v['label']: store.read(_ref(v['configuration_file'])) for v in menu['candidates']}
    for item in menu['candidates']:
        c = configs[item['label']]; s = c['spec']
        if (c['id'] != item['configuration_id'] or s['L_D'] != item['L_D']
                or s['M'] != 12 or len(c['rounds']) != 13 or s['formula'] != '2nd' or s['finite_K'] != 2
                or s['A_input']['kappa_paper'] != menu['conditions']['kappa_paper_exact']
                or D.p.accounting.exact_scalar(s['split']['lambda_R']) != Fraction.from_float(item['tail_descriptor']['lambda_R'])):
            raise ValueError('Fixed menu configuration/prefix/lambda/kappa differs')
    return menu, configs


def _request(menu, config, label, m, axes, execution_source):
    item = next(v for v in menu['candidates'] if v['label'] == label)
    source = dict(menu=_ref('MENU.json', label=label), tail=deepcopy(item['tail_source']),
        input=deepcopy(config['spec']['binding']['decomposition']['contract']['input_source']))
    return D.from_b(config, m, config['rounds'][m]['physical_candidate'], input_source=source,
        execution_source=execution_source, axes=axes, expected_scalar=item['tail_descriptor']['scalar'])


def request_for(root, label, m, *, axes=('X', 'Y'), execution_source):
    """Materialize only selected metadata. Scientific SOURCE is supplied later.

    The publication uses its own technical metadata SOURCE tag. A later run
    regenerates the request with its actually approved execution SOURCE, without
    editing the historical coverage/index or relabelling previous path prices.
    """
    if label not in LABELS or type(m) is not int or m not in range(13):
        raise ValueError('Select one fixed LD1/LD2 round 0..12')
    menu, configs = _read(root)
    return _request(menu, configs[label], label, m, axes, execution_source)


def prepare(root, *, execution_source):
    """Generate 26 paired requests and index 52 axis cells; execute none."""
    menu, configs = _read(root)
    index = []; cells = []
    for label in LABELS:
        config = configs[label]
        item = next(v for v in menu['candidates'] if v['label'] == label)
        for row in config['rounds']:
            m = row['m']; req = _request(menu, config, label, m, ('X', 'Y'), execution_source)
            pc = req['physical_context']; c = pc['physical_candidate']
            # Use the carrier's literal signed-time expression for one outer only.
            first = [dict(piece=j, signed_time=c['T']*w/c['q'], rte_steps=n)
                for j, (w, n) in enumerate(zip(row['structure']['signed_weights'], row['structure']['allocation']))]
            index.append(dict(label=label, m=m, request_id=req['request_id'], context_key=req['context_key'],
                configuration_id=config['id'], canonical_candidate_id=row['candidate']['id'],
                physical_id=c['id'], axes=req['axes'], configuration=_ref(item['configuration_file']),
                selector=['rounds', m], materialize=dict(function='ld12_price_requests.request_for',
                    label=label, m=m, axes=req['axes'])))
            reused = (dict(status='DEVELOPMENT_PILOT_REFERENCE_ONLY_REQUIRES_NEW_IDENTITY_BRIDGE',
                        source=dict(commit=P2, path='docs/research/d_p2_direct_pilot_20261113/HANDOFF.md'))
                      if label == 'PR-LD1' and m in (0, 6) else
                      dict(status='NO_COMPILED_REFERENCE_IDENTIFIED_IN_FIXED_SOURCES', source=None))
            structural = (dict(status='STRUCTURAL_ONLY_REFERENCE_REQUIRES_DOMAIN_BRIDGE',
                source=dict(commit=STRUCTURAL, path='docs/research/h4_structural_schedule_results_20261113/HANDOFF.md'))
                if label == 'PR-LD1' else dict(status='OLD_LD1_STRUCTURAL_MEAN_NOT_REUSABLE', source=None))
            for axis in ('X', 'Y'):
                measurement = D.p.measurement(row, axis)
                measurement['law_context'].update(price_law=None, ideal_sampler_distance_qualified=None)
                cells.append(dict(label=label, L_D=c['prefix'], m=m, axis=axis,
                    request_id=req['request_id'], context_key=req['context_key'],
                    configuration_id=config['id'], canonical_candidate_id=row['candidate']['id'], physical_id=c['id'],
                    configuration=_ref(item['configuration_file']), identity_selector=['rounds', m, 'identity'],
                    law_selector=['rounds', m, 'law_binding'], law_binding_key=D._key(req['law_binding']),
                    q_RPE=row['q_RPE'], q_PF=c['q'], R_outer=c['R_outer'], K=c['K'], T=c['T'],
                    T_binary64_hex=c['T'].hex(), delta_binary64_hex=c['delta'].hex(),
                    structure=deepcopy(row['structure']),
                    signed_piece_times=[dict(piece=x['piece'], time=x['signed_time'], binary64_hex=x['signed_time'].hex(),
                        events=x['rte_steps']) for x in first],
                    exact_kappa=menu['conditions']['kappa_paper_exact'],
                    preparation=dict(status='NOT_BUILT_METADATA_ONLY', hash=None,
                        deterministic_fragments=list(range(c['prefix'])), randomized_fragments=list(range(c['prefix'], 10)),
                        one_body_policy=pc['one_body_policy'], lambda_R=deepcopy(config['spec']['split']['lambda_R']),
                        scalar_expected=pc['expected_scalar'], source=deepcopy(item['tail_source']),
                        actual_basis_component_tail_hashes=None, runtime_checks='PENDING_SCIENTIFIC_PREPARATION'),
                    price_profile=deepcopy(pc['price_profile']), measurement=measurement,
                    status='REQUEST_READY_SCIENTIFIC_EXECUTION_NOT_AUTHORIZED', unsupported_conditions=[],
                    reuse=dict(compiled=reused, structural=structural, accepted=False,
                        fingerprints_not_constructed=True, cache_hits_not_evaluated=True),
                    acquisition_method='SELECTED_PATH_DIRECT_FULL_WRAPPER_C_AND_S_WITH_EXISTING_ADAPTER',
                    obstacles=['NO_SELECTED_PATH_COUNT_OR_PRECISION', 'TARGET_LAW_RARE_AND_MEAN_CI_UNQUALIFIED',
                        'FULL_NATIVE_WORD_SCALES_WITH_Q_TIMES_R', 'NO_LONG_Q_WINDOW_TAIL_BOUND',
                        'ACTUAL_PREPARATION_INPUT_SCALAR_CHECKS_PENDING'],
                    compile_time=None, peak_RSS=None, N=None, mean_C=None, price_interval=None,
                    selected_for_execution=False))
    return dict(status='FIXED_LD12_PRICE_METADATA_ONLY', A_menu=_ref('MENU.json'), D_adapter_base=D_BASE,
        B_sources=list(B_SOURCES), execution_source=deepcopy(execution_source), request_index=index,
        coverage=cells, paired_round_requests=len(index), axis_cells=len(cells),
        scientific_preparations=0, sampling_build_compile=0, mean_C=None, G=None, mandatory_stop=True)
