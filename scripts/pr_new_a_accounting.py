"""New A n8 costs plus unchanged saved D comparators. Standard-library only.

Preparation uses audit_structure only. Numerical entry requires a distinct
permit; artificial permits cannot process saved H-chain values. No sampling,
signal, precision optimization, compile, fitting or conditional reweighting.
"""
from copy import deepcopy
from fractions import Fraction as F
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent),
               str(Path(__file__).resolve().parents[1] / 'src/trotterlib')]
import pr_cost_contract as c
import pr_cost_saved as saved

SCHEMA = 'NEW_A_SAVED_TOTAL_COST_V1'
_REAL = object()
OLD_LABELS = dict(H4='SAVED_N32_FIXED_ANCHOR_SELECTION_AND_ESTIMATION',
                  H6='SAVED_INDEPENDENT_CONFIRMATION_N32',
                  H8='SAVED_INDEPENDENT_CONFIRMATION_UNION6_N32')


class Permit:
    def __init__(self, *, synthetic=False, token=None):
        c.require(synthetic is True or token is _REAL, 'NEW_SCOPED_GRANT_REQUIRED')
        self.synthetic = synthetic


def endpoints():
    return [(m, e, f) for m in c.core.MODES for e in c.core.EPS for f in (1., 10.)]


def key(row):
    return row['cell_id'], row['bias_mode'], row['epsilon'], row['factor']


def indexed(rows):
    out = {}
    for row in rows:
        k = key(row)
        c.require(k not in out, 'DUPLICATE_ENDPOINT')
        out[k] = row
    return out


def audit_structure(data):
    """IDs/metadata and saved presence only: no cost/shot arithmetic or ranking."""
    c.require(data['system'] in OLD_LABELS and data['kind'] in
              ('ARTIFICIAL', 'SAVED_REAL_VALUES'), 'SYSTEM_DATA_KIND')
    binding = data['binding']
    c.require(binding['system'] == data['system'] or data['kind'] == 'ARTIFICIAL', 'SYSTEM_BINDING')
    cells = data['cells']
    ids = [x['id'] for x in cells]
    c.require(len(ids) == len(set(ids)) == 16, 'FIXED_SIXTEEN_CANDIDATES')
    for cell in cells:
        c.core.cell_key(cell)
    signal = {x['cell']['id']: x for x in data['signals']}
    c.require(len(signal) == len(data['signals']) == 16 and set(signal) == set(ids), 'SIGNAL_IDS')
    for cell in cells:
        c.require(signal[cell['id']]['cell'] == cell, 'SIGNAL_CELL_PARAMETERS')
        c.require(c.digest(signal[cell['id']]) == data['signal_record_hashes'][cell['id']], 'SIGNAL_RECORD_HASH')
    tasks = data['tasks']
    registry = {(t['cell']['id'], t['replica']): t for t in tasks}
    expected = {(cid, i) for cid in ids for i in range(8)}
    c.require(len(registry) == len(tasks) == 128 and set(registry) == expected, 'EXACT_128_SEED_REGISTRY')
    pairs = {(t['cell_id'], t['replica']): t for t in data['pairs']}
    c.require(len(pairs) == len(data['pairs']) == 128 and set(pairs) == expected, 'EXACT_128_PAIRED_RESULTS')
    c.require(len({t['seed'] for t in tasks}) == 128, 'UNIQUE_REGISTERED_SEEDS')
    for k, pair in pairs.items():
        task = registry[k]
        c.require(task['cell'] == signal[k[0]]['cell'] and pair['seed'] == task['seed'], 'FIXED_SEED_CELL')
        c.core.same_binding(pair['binding'], binding)
        c.require(pair['cohort'] == 'EXPLORATION_N8' and set(pair['axes']) == set(c.core.AXES), 'N8_PAIRED_ONLY')
        for axis in c.core.AXES:
            value = pair['axes'][axis]
            c.require(value['seed'] == pair['seed'] and value['event_digest'] == pair['event_digest'], 'AXIS_PAIRING')
            c.require(value.get('compiler_hash', binding['compiler']) == binding['compiler'], 'PAIR_COMPILER')
    old = data['old_PR_ids']
    c.require(len(set(old)) == len(old) == (6 if data['system'] == 'H8' else 8), 'FIXED_OLD_PR_DOMAIN')
    det = set().union(*map(set, data['deterministic_sets'].values()))
    c.require(len(data['deterministic_sets']) == 2 and not (det & set(old)) and not (set(ids) & (det | set(old))), 'SEPARATE_COMPARISON_DOMAINS')
    old_rows = indexed(data['old_rows'])
    c.require(set(old_rows) == {(cid, *ep) for cid in det | set(old) for ep in endpoints()}, 'COMPLETE_SAVED_D_DOMAIN')
    missing = []
    for record in signal.values():
        for mode, field in [('USUAL_SIGNED', 'precision_rows'), ('EMPIRICAL_NONCANCELLATION',
                'noncancellation_precision_rows' if 'noncancellation_precision_rows' in record
                else 'empirical_noncancellation_precision_rows')]:
            for e in c.core.EPS:
                for f in (1., 10.):
                    if record.get(field, {}).get(str(e), {}).get(str(f)) is None:
                        missing.append([record['cell']['id'], mode, e, f])
    return dict(status='PASS_STRUCTURE_ONLY', candidates=16, pairs=128, measured_axes=256,
                missing_accounting_endpoints=missing, numerical_reaccounting=0, ranking=0)


def old_row(row, binding, old_label, pr_ids):
    out = deepcopy(row)
    eligible = row['eligible']
    c.require(eligible is None or type(eligible) is bool, 'SAVED_ELIGIBILITY_TYPE')
    cost = row.get('G_exact')
    c.require(eligible is True or cost is None, 'NONELIGIBLE_NOT_ZERO_COST')
    if cost is not None:
        c.require(F(cost) > 0, 'POSITIVE_SAVED_COST')
    status = ('NUMERICALLY_UNRESOLVED' if eligible is None else 'INELIGIBLE' if eligible is False
              else 'ELIGIBLE_COST_MISSING' if cost is None else 'SAVED_POINT')
    out.update(binding=deepcopy(binding), status=status, G_RZ_exact=cost,
               G_RZ_SE=row.get('G_SE'), original_cost_status=row.get('cost_status'),
               cohort=old_label if row['cell_id'] in pr_ids else 'SAVED_DETERMINISTIC',
               finite_sample_only=row['cell_id'] in pr_ids)
    return out


def account(data, permit):
    c.require(isinstance(permit, Permit) and permit.synthetic == (data['kind'] == 'ARTIFICIAL'), 'DATA_SCOPE_PERMIT')
    before = c.digest(data)
    audit = audit_structure(data)
    ids = [x['id'] for x in data['cells']]
    pairs = {cid: [p for p in data['pairs'] if p['cell_id'] == cid] for cid in ids}
    projection = dict(kind=data['kind'], binding=data['binding'],
        saved_accounting=saved.signal_accounting(data['signals'], data['binding'], dict.fromkeys(ids, 'EXPLORATION_N8')),
        paired_PR=pairs, cost_evidence=[])
    # Reuses exact saved shots and statistics of each weighted paired trajectory.
    rows = c.core.account_rows(projection, c.core.Permit('D_ACCOUNTING',
        synthetic=permit.synthetic, authorization=None if permit.synthetic else permit))
    old = [old_row(r, data['binding'], OLD_LABELS[data['system']], data['old_PR_ids']) for r in data['old_rows']]
    comparisons = []
    for ep in endpoints():
        new_ep = [r for r in rows if (r['bias_mode'], r['epsilon'], r['factor']) == ep]
        old_ep = [r for r in old if (r['bias_mode'], r['epsilon'], r['factor']) == ep]
        for name, det_ids in data['deterministic_sets'].items():
            deterministic = [r for r in old_ep if r['cell_id'] in det_ids]
            new_cmp = c.core.compare(deterministic + new_ep, det_ids, ids)
            old_pr = [r for r in old_ep if r['cell_id'] in data['old_PR_ids']]
            old_cmp = c.core.compare(deterministic + old_pr, det_ids, data['old_PR_ids'])
            new_best, old_best = new_cmp['best_available_PR'], old_cmp['best_available_PR']
            # The available-set best remains visible; missing endpoints suppress full-set claims.
            complete = new_cmp['comparison_complete'] and old_cmp['comparison_complete']
            versus_old = (str(100 * (1 - F(new_best['G_RZ_exact']) / F(old_best['G_RZ_exact'])))
                          if complete and new_best and old_best else None)
            comparisons.append(dict(bias_mode=ep[0], epsilon=ep[1], factor=ep[2], deterministic_scope=name,
                new_exploration=new_cmp, old_saved=old_cmp,
                new_vs_old_PR_reduction_percent_exact=versus_old,
                selection_basis='exploratory n8 point mean; ID ascending breaks exact ties',
                independent_confirmation_needed_is_not_decided=True))
    c.require(c.digest(data) == before, 'INPUT_MUTATION')
    return dict(status='COMPLETE_MANDATORY_STOP', system=data['system'], new_rows=rows,
        unchanged_old_rows=deepcopy(data['old_rows']), comparisons=comparisons,
        structural_audit=audit, old_PR_cohort_label=OLD_LABELS[data['system']],
        paired_SE_preserves_cosine_sine_covariance=True, cohorts_pooled=False,
        confidence_interval=None, population_mean_certified=False, strict_bias_certificate=None,
        all_candidates_direct_measured=False, conditional_pilots_reweighted=False,
        saved_scalar_passes=1, new_signal=0, compile=0, sampling=0, fit=0,
        shot_recalculation=0, automatic_retry=False, next_stage_authorized=False)
