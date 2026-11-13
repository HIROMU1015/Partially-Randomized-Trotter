"""Published D physical metadata -> existing B V2 bridge; no scientific leaves."""
from copy import deepcopy
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
import subprocess

from . import full_baseline_v2_requests as D

B, p = D.B, D.p
ROOT = Path(__file__).resolve().parents[2]
PHYSICAL = '0536646b843781a0e0fb6629473526c05f16a30e'
COVERAGE = '92ff17dededea6e1ee273b2eee6a08e292ee9e5b'
PHYSICAL_PATH = 'src/trottertracks/full_baseline_v2_requests.py'
COVERAGE_PATH = 'src/trottertracks/full_pf_saved_inventory.py'
COVERAGE_JSON = 'docs/research/d_full_pf_saved_evidence_coverage_20261113/COVERAGE.json'
EXAMPLE_JSON = 'docs/research/d_full_baseline_interface_update_20261113/REQUEST_EXAMPLE_ARTIFICIAL.json'
RECEIPT_RELEASE = 'e7dca5f7301c8633b17a59b6a7161331303aef2d'
RECEIPT_DIR = 'docs/research/d_full_pf_full_h_v2_final_receipts_20261113/'


@lru_cache(maxsize=2)
def published_module(which):
    """Reuse the two pinned passive D modules; never read a running worktree."""
    commit, path = {'physical': (PHYSICAL, PHYSICAL_PATH), 'coverage': (COVERAGE, COVERAGE_PATH)}[which]
    raw = subprocess.check_output(['git','show',commit+':'+path], cwd=ROOT)
    module = ModuleType('trottertracks._published_D_' + which)
    module.__package__ = 'trottertracks'
    module.__file__ = str(ROOT/path)
    exec(compile(raw, commit+':'+path, 'exec'), module.__dict__)
    return module


def _same(left, right, reason):
    if not p.same(left, right):
        raise ValueError(reason)


def bind_physical(spec, *, layout_by_round=None):
    """Bind a caller-supplied D recipe, never select a real comparison menu.

    The D physical formula ID and layout kind remain in the source view. Only
    the B-generated canonical view uses B's official formula/mapping IDs.
    """
    physical_source = published_module('physical')
    base = {k:deepcopy(spec[k]) for k in ('binding','h','rmse_policy','profile')}
    if spec['method'] == 'PF':
        if spec['formula'] not in ('2nd','4th','4th(m5_best)'):
            raise ValueError('Only the three supplied Full PF formulas are in scope')
        expected = physical_source.full_pf_spec(base, formula=spec['formula'], delta_pf=spec['delta_pf'],
            M=spec['M'], rank=spec['L_D'])
    elif spec['method'] == 'FULL_H_RTE':
        expected = physical_source.full_h_rte_spec(base, M=spec['M'], finite_K=spec['finite_K'],
            preparation=spec['split']['preparation'], kappa=spec['kappa'])
    else:
        raise ValueError('D source view covers Full PF/Full-H RTE only')
    _same(spec, expected, 'D_PHYSICAL_SPEC_SOURCE_LAW_OR_PROFILE_MISMATCH')
    physical = physical_source.physical_schedule(spec, layout_by_round=layout_by_round)
    canonical_spec = deepcopy(spec)
    if spec['method'] == 'PF':
        # Compare the actual pinned coefficient SOURCE and binary64 sequence;
        # agreeing labels alone are insufficient (Yoshida4 is not m5_best).
        ref = physical_source.SOURCES['pf_weights']
        if (ROOT/ref['path']).read_bytes() != subprocess.check_output(['git','show',ref['commit']+':'+ref['path']], cwd=ROOT):
            raise ValueError('PF_COEFFICIENT_SOURCE_DIFFERS')
        config = B.generate(canonical_spec)
    else:
        canonical_spec['event_mapping']['kind'] = B.EXPLICIT_LAYOUT
        canonical_spec['event_mapping']['source']['commit'] = PHYSICAL
        def mapper(s, row):
            old = physical['rounds'][row['m']]['mapping']
            if old is None:
                return None
            out = deepcopy(old)
            out.update(kind=B.EXPLICIT_LAYOUT, source=deepcopy(s['event_mapping']['source']))
            return out
        config = B.generate(canonical_spec, event_mapper=mapper)
    requests = D.requests(config)
    D.validate_requests(config, requests)
    links = []
    for old, row, request in zip(physical['rounds'], config['rounds'], requests['rounds']):
        for key in ('m','q_RPE','q_PF','T'):
            _same(old[key], row[key], 'D_B_SCHEDULE_MISMATCH:' + key)
        _same(old['physical_candidate'], request['A']['physical_candidate'], 'D_B_PHYSICAL_CANDIDATE_MISMATCH')
        if spec['method'] == 'PF':
            _same(old['signed_weights'], row['structure']['signed_weights'], 'D_B_SIGNED_COEFFICIENT_MISMATCH')
            _same(old['actual_width'], row['actual_delta_pf'], 'D_B_ACTUAL_WIDTH_MISMATCH')
        else:
            _same((old['mapping'] or {}).get('physical_layout'),
                (row['mapping'] or {}).get('physical_layout'), 'D_B_PHYSICAL_LAYOUT_MISMATCH')
        links.append(dict(m=row['m'], physical_candidate=deepcopy(old['physical_candidate']),
            V2_configuration_id=config['id'], V2_candidate_id=row['candidate']['id'],
            D_formula_id=old.get('formula_id'), B_formula_id=row['structure']['formula_id'],
            source=dict(commit=PHYSICAL,path=PHYSICAL_PATH)))
    return dict(physical=physical, v2_configuration=config, identity_map=links, requests=requests,
        original_parameterization=deepcopy(spec['event_mapping']), canonical_parameterization=deepcopy(canonical_spec['event_mapping']),
        scientific_conditions_adopted=False, G=None)


def validate_binding(packet):
    """Validate an external identity_map and its A/C requests without relabeling."""
    physical = packet['physical']; spec = physical['spec']
    layouts = None if spec['method'] == 'PF' else {r['m']:dict(q=r['mapping']['physical_layout']['q'],
        R_outer=r['mapping']['physical_layout']['R_outer']) for r in physical['rounds'] if r['mapping'] is not None}
    _same(packet, bind_physical(spec, layout_by_round=layouts), 'D_V2_EXTERNAL_IDENTITY_MAP_MISMATCH')
    return True


def accept_example(entry):
    """Connect D's published artificial request; this is not a real menu."""
    spec = entry['spec']; old_requests = entry['requests']
    layouts = None if spec['method']=='PF' else {r['m']:dict(q=r['A']['physical_round']['mapping']['physical_layout']['q'],
        R_outer=r['A']['physical_round']['mapping']['physical_layout']['R_outer']) for r in old_requests['rounds']
        if r['A']['physical_round']['mapping'] is not None}
    bound = bind_physical(spec,layout_by_round=layouts)
    for old, new in zip(entry['identity_map'], bound['identity_map']):
        _same(old['physical_candidate'],new['physical_candidate'],'D_PUBLISHED_PHYSICAL_IDENTITY_MAP_MISMATCH')
        if old['V2_candidate_id'] is not None:
            for key in ('V2_candidate_id','V2_configuration_id'):
                _same(old[key],new[key],'D_EXISTING_V2_ID_CHANGED')
    for old in old_requests['rounds']:
        row = bound['physical']['rounds'][old['m']]
        _same(old['A']['physical_round'],row,'D_PUBLISHED_PHYSICAL_ROUND_MISMATCH')
        for record in [old['A']] + old['C']:
            _same(record['physical_candidate'],row['physical_candidate'],'D_PUBLISHED_A_C_PHYSICAL_MISMATCH')
            _same(record['binding'],spec['binding'],'D_PUBLISHED_A_C_BINDING_MISMATCH')
            if record['identity'] is not None:
                _same(record['identity'],bound['v2_configuration']['rounds'][row['m']]['identity'],'D_PUBLISHED_A_C_V2_MISMATCH')
        for price in old['C']:
            _same(price['profile'],spec['profile'],'D_PUBLISHED_C_PROFILE_MISMATCH')
            _same(price['mean'],dict(rz_count=None),'D_ARTIFICIAL_EXAMPLE_MUST_NOT_SUPPLY_PRICE')
    return bound


def accept_saved_coverage(cov):
    """Accept D's already classified metadata; do not rerun inventory/receipts."""
    source = published_module('coverage')
    if (cov['status'] != 'FULL_PF_SAVED_EVIDENCE_COVERAGE_METADATA_ONLY' or
            cov['rmse_policy_qualified'] is not False or cov['values_recomputed'] is not False or cov['new_compile'] != 0):
        raise ValueError('D_SAVED_COVERAGE_IS_NOT_CURRENT_RMSE_PRICE_QUALIFICATION')
    ids = [r['physical_candidate_id'] for r in cov['rows']]
    if len(set(ids)) != len(ids) or len(ids) != cov['totals']['candidates']:
        raise ValueError('D_DUPLICATE_OR_INCOMPLETE_COVERAGE')
    ranks = D.saved_grid()['rank']
    facts = published_module('physical').FORMULAS
    for row in cov['rows']:
        c = row['physical_candidate']
        if (c['method'] != 'FULL' or c['formula'] not in source.FORMULAS or c['prefix'] != ranks[c['system']]
                or c.get('K') is not None or c.get('R_outer') is not None or c['id'] != p.candidate_id(c)
                or row['physical_candidate_id'] != c['id'] or row['formula'] != c['formula']
                or row['formula_id'] != facts[c['formula']]['formula_id']
                or row['binding']['status'] != source.EXACT or row['numerical']['rmse_policy_qualified'] is not False):
            raise ValueError('D_FULL_PF_COVERAGE_PHYSICAL_SOURCE_OR_FORMULA_MISMATCH')
        _same(row['binding']['source'], dict(commit=D.BASE,path=source.LONG+'scientific_provenance.json'), 'D_COVERAGE_BINDING_SOURCE_MISMATCH')
        for price in row['prices']:
            if not price['kind'].startswith('DIRECT_COMPILED_DETERMINISTIC'):
                raise ValueError('D_MODEL_OR_STRUCTURAL_VALUE_IS_NOT_SAVED_DIRECT_C')
            if set(price['axes']) != set(p.AXES):
                raise ValueError('D_PRICE_AXES_MUST_REMAIN_SEPARATE')
    for field in ('signal','reference','X','Y'):
        counts = dict(Counter(r[field]['status'] if isinstance(r[field],dict) else r[field] for r in cov['rows']))
        _same(counts,cov['totals'][field],'D_REPORTED_METADATA_COUNTS_DIFFER:' + field)
    return dict(source=dict(commit=COVERAGE,path=COVERAGE_JSON), totals=deepcopy(cov['totals']),
        candidates_adopted_as_menu=False, rmse_policy_qualified=False, qualified_current_mean_C=False,
        sufficient_N=None, global_RMSE_bound=None, G=None,
        rows=[dict(physical_candidate_id=r['physical_candidate_id'], formula=r['formula'], D_formula_id=r['formula_id'],
            signal=r['signal']['status'], reference=r['reference']['status'], X=r['X'], Y=r['Y'],
            B_V2_configuration_id=None, reason='NO_USER_ADOPTED_WHOLE_SCHEDULE',
            price_kinds=[v['kind'] for v in r['prices']]) for r in cov['rows']])


def join_saved_coverage(packet, cov):
    """Join only an explicitly supplied whole schedule, never all 136 as a menu."""
    validate_binding(packet); accept_saved_coverage(cov)
    if packet['physical']['method'] != 'FULL_PF':
        raise ValueError('Full PF saved coverage cannot supply Full-H RTE prices')
    source = published_module('coverage')
    science = p.SavedArtifacts(ROOT).read(dict(commit=D.BASE,path=source.LONG+'scientific_provenance.json'))
    s = packet['physical']['spec']; system = s['binding']['system']
    D.check_binding(s['binding'], science['inputs'][system]['binding'])
    _same(s['profile'], p.PROFILE, 'D_SAVED_COMPILER_PROFILE_MISMATCH')
    byid = {r['physical_candidate_id']:r for r in cov['rows']}
    for row in packet['physical']['rounds']:
        saved = byid.get(row['physical_candidate_id'])
        if saved:
            _same(row['physical_candidate'], saved['physical_candidate'], 'D_SAVED_PHYSICAL_ID_COLLISION')
    return source.for_packet(packet, cov)


def accept_receipt_coverage(previous, current, receipts):
    """Accept D e7dca5f's published 73/0/63; no receipt rerun/RZ recount.

    The old 56/17/63 inventory remains immutable historical evidence. This
    classifies saved physical C, not RMSE adequacy or an adopted V2 menu.
    """
    old=accept_saved_coverage(previous)
    _same(current['previous_coverage_commit'],COVERAGE,'D_PREVIOUS_COVERAGE_SOURCE_CHANGED')
    _same(receipts['previous_coverage_commit'],COVERAGE,'D_RECEIPTS_PREVIOUS_SOURCE_CHANGED')
    _same(current['status'],'FULL_PF_SAVED_EVIDENCE_COVERAGE_RECEIPTS_TRACED_METADATA_ONLY','D_RECEIPT_COVERAGE_KIND')
    for packet in (current,receipts):
        for field,value in dict(values_recomputed=False,new_compile=0,rz_recount=0).items():
            _same(packet[field],value,'D_NEW_NUMERICAL_WORK_NOT_METADATA:' + field)
    _same(current['rmse_policy_qualified'],False,'D_RECEIPTS_ARE_NOT_RMSE_QUALIFICATION')
    _same(current['receipt_source_checks'],receipts['source_checks'],'D_RECEIPT_SOURCE_CHECKS_DIFFER')
    _same(receipts['source_checks']['status'],'EXACT_METADATA_MATCH','D_RECEIPT_SOURCE_UNRESOLVED')
    _same(current['changes'],receipts['changes'],'D_RECEIPT_CHANGE_LIST_DIFFERS')
    for field in ('candidates','groups','signal','reference'):
        _same(current['totals'][field],previous['totals'][field],'D_NONPRICE_TOTALS_CHANGED:' + field)
    byid={r['physical_candidate_id']:r for r in current['rows']}
    if len(byid)!=len(current['rows']) or set(byid)!={r['physical_candidate_id'] for r in previous['rows']}:
        raise ValueError('D_RECEIPT_CANDIDATE_SCOPE_CHANGED')
    changed=[]
    for before in previous['rows']:
        after=byid[before['physical_candidate_id']]
        for field in ('physical_candidate','signal','reference','numerical','binding','physical_reasons','group','missing'):
            _same(after[field],before[field],'D_RECEIPT_NONPRICE_EVIDENCE_CHANGED:' + field)
        expected_formula=p.structure(dict(formula=before['formula'],method='FULL',q=1,R_outer=None))['formula_id']
        _same(after['formula_id'],expected_formula,'D_RECEIPT_CANONICAL_FORMULA_MISMATCH')
        _same(after['physical_formula_label'],before['formula_id'],'D_PHYSICAL_FORMULA_HISTORY_LOST')
        if len(before['prices'])!=len(after['prices']):raise ValueError('D_RECEIPT_PRICE_SCOPE_CHANGED')
        for old_price,new_price in zip(before['prices'],after['prices']):
            for field in ('kind','cohort','n','carrier'):
                _same(new_price[field],old_price[field],'D_SAVED_DIRECT_C_SOURCE_OR_COHORT_CHANGED')
            if not new_price['kind'].startswith('DIRECT_COMPILED_DETERMINISTIC'):
                raise ValueError('D_STRUCTURE_OR_MODEL_IS_NOT_DIRECT_C')
            for axis in p.AXES:
                _same(new_price['axes'][axis]['rz_count'],old_price['axes'][axis]['rz_count'],'D_PRICE_VALUE_CHANGED')
        for axis in p.AXES:
            was=before[axis];now=after[axis]
            _same(now,'EXACT_METADATA_MATCH' if was=='CONDITIONAL_REUSE' else was,'D_RECEIPT_STATUS_UNEXPECTED')
            if was=='CONDITIONAL_REUSE':
                price=next(v for v in after['prices'] if v.get('receipt_trace'))
                if (price['axes'][axis]['status']!='EXACT_METADATA_MATCH'
                        or not any(e['axes'].get(axis,{}).get('status')=='EXACT_METADATA_MATCH' for e in price['receipt_trace']['evidence'])):
                    raise ValueError('D_EXACT_STATUS_REQUIRES_PUBLISHED_AXIS_RECEIPT')
        if before['X']=='CONDITIONAL_REUSE' or before['Y']=='CONDITIONAL_REUSE':changed.append(after['physical_candidate_id'])
    if set(changed)!={v['physical_candidate_id'] for v in current['changes']}:
        raise ValueError('D_RECEIPT_PROMOTION_SCOPE_DIFFERS')
    for axis in p.AXES:
        _same(dict(Counter(r[axis] for r in current['rows'])),current['totals'][axis],'D_RECEIPT_TOTALS_DIFFERS')
    return dict(source=dict(commit=RECEIPT_RELEASE,path=RECEIPT_DIR+'COVERAGE_V2.json'),
        receipt_source=dict(commit=RECEIPT_RELEASE,path=RECEIPT_DIR+'RECEIPTS.json'),
        previous_source=old['source'],previous_totals=old['totals'],totals=deepcopy(current['totals']),
        promoted_physical_ids=changed,receipt_traces_copied_not_reexecuted=True,
        current_RMSE_qualification=False,candidates_adopted_as_menu=False,G=None,
        sufficient_N=None,global_RMSE_bound=None)
