"""Verify copied fields and immutable inputs only; no numeric recipes/evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import bind_pr_alpha_saved_inputs as b
import pr_alpha_followup_runtime as runtime


def pointer(document, path):
    value = document
    for key in path.split('/')[1:]:
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def same(left, right):
    b.codec.require(b.canonical(left) == b.canonical(right), "COPIED_FIELD_CHANGED")


def check(private, locations):
    plans, payloads, objects, checks = {}, {}, {}, []
    public_catalog = None
    for stage in ('B1', 'B2'):
        plan = b.strict((private/(stage+'_plan_UNAPPROVED.json')).read_bytes())
        b.codec.require(plan['sealed'] and plan['inputs_bound'] and
                        plan['approved_by_user'] is False and plan['execution_authorized'] is False,
                        'INPUT_SEAL_NOT_EXECUTION_APPROVAL')
        runtime.source_gate(plan)  # hashes only, no scalar decoder or scientific call
        raw = Path(plan['input']['path']).read_bytes()
        b.codec.require(len(raw) == plan['input']['bytes'] and b.sha(raw) == plan['input']['sha256'], 'SEALED_INPUT_HASH')
        data = b.strict(raw)
        runtime.scope_gate(plan, data['records'])  # encoded records are not evaluated
        public_catalog = data['original_source_receipts']
        plans[stage], payloads[stage] = plan, data
    for row in public_catalog:
        if row['origin'].startswith('PRIVATE_'):
            raw = Path(locations[row['role']]).read_bytes()
            doc = b.strict(raw)
        else:
            commit, path = row['origin'].split(':', 1)
            raw = b.git(commit, path); doc = b.decode(raw, path)
        b.codec.require(len(raw) == row['bytes'] and b.sha(raw) == row['sha256'], 'ORIGINAL_RECEIPT_CHANGED')
        objects[row['role']] = doc
    for stage, data in payloads.items():
        for r in data['records']:
            source = r['source_fields']; loc = source['signal']
            signal = pointer(objects[loc['receipt']], loc['pointer'])
            b.codec.require(signal['cell']['id'] == r['cell_id'], 'COPIED_SIGNAL_ID')
            loc = source['baseline']; saved = pointer(objects[loc['receipt']], loc['pointer'])
            if r.get('missing_fields'):
                b.codec.require(r['system'] == 'H6' and r['bias_kind'] == 'EMPIRICAL_NONCANCELLATION'
                    and saved is None and not signal['error_decomposition'].get('signed_parts'), 'UNEXPECTED_MISSING_ROW')
                continue
            same(r['baseline_accounting'], saved)
            same(r['corrected'], b.codec.scalar_axes(signal['signals']['corrected']))
            same(r['raw'], b.codec.scalar_axes(signal['signals'].get('raw', signal['signals']['corrected'])))
            same(r['reference'], b.codec.scalar_axes(signal['reference']))
            same(r['allowance'], b.codec.scalar_axes(signal['allowance']))
            same(r['log_B'], signal['log_B']); same(r['log_B_margin'], signal['allowance']['log_B_margin'])
            recipe = r['bias_recipe']
            if recipe['kind'] == 'ABS_SAVED_AXIS_TOTAL':
                same(recipe['values'], b.codec.scalar_axes(signal['error_decomposition']['total_signed']))
            elif recipe['kind'] == 'SUM_ABS_SAVED_PARTS':
                same(recipe['ordered_terms'], [b.codec.scalar_axes(v) for v in signal['error_decomposition']['signed_parts'].values()])
            else:
                b.codec.require(recipe['kind'] == 'COPY_SAVED_MAGNITUDE', 'RECIPE_KIND')
                original = [x for x in objects['H8_saved_bias_magnitudes'] if x['candidate_id'] == r['cell_id']]
                same(recipe['values'], {x['axis']: float(x['empirical_noncancellation_bias']) for x in original})
            cost = r['cost']; loc = source['costs']
            if cost['kind'] == 'PR_PAIRED_MEASURED':
                rows = objects[loc['receipt']]
                b.codec.require(len(cost['pairs']) == 32, 'N32_COPY')
                for pair in cost['pairs']:
                    for axis in ('real', 'imag'):
                        old = rows[pair['CSV_data_rows'][axis]]
                        same(pair[axis], int(old['RZ']))
                        for key in ('seed', 'replica'): same(pair[key], int(old[key]))
                        for key in ('event_digest', 'compiler_hash'): same(pair[key], old[key])
                        b.codec.require(old['cell_id'] == r['cell_id'] and old['phase'] == 'confirmation'
                            and old['control'] == 'symmetric_directional', 'PR_COST_COPY_IDENTITY')
            elif cost['kind'] != 'MISSING':
                if 'pointers' in loc:
                    axes = {axis: pointer(objects[loc['receipt']], p) for axis, p in loc['pointers'].items()}
                else:
                    axes = pointer(objects[loc['receipt']], loc['pointer'])
                same(cost, b.deterministic_cost(axes))
            if stage == 'B1':
                b.codec.require(saved['cost_acquisition_eligible'] is True and saved['alpha_axis'] == .025, 'B1_SAVED_STATUS')
            checks.append(dict(stage=stage, system=r['system'], cell_id=r['cell_id'], epsilon=r['epsilon'],
                               bias_kind=r['bias_kind'], status='EXACT_FIELD_COPY'))
    return dict(status='PASS_METADATA_COPY_ONLY', checked_complete_records=len(checks),
                checked_original_receipts=len(public_catalog), source_commit=plans['B1']['source_commit'],
                B1_rows=len(payloads['B1']['records']), B2_rows=len(payloads['B2']['records']),
                missing_rows=sum(bool(r.get('missing_fields')) for r in payloads['B2']['records']),
                recipe_evaluations=0, real_binomial=0, shot_cost_rank_reanalysis=0,
                execution_authorized=False, independent_review=False), payloads['B1']['records']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-directory', type=Path, required=True)
    parser.add_argument('--local-input-map', type=Path, required=True)
    parser.add_argument('--report-directory', type=Path, required=True)
    args = parser.parse_args()
    result, rows = check(args.private_directory, b.strict(args.local_input_map.read_bytes()))
    b.put(args.report_directory/'copy_verification.json', result)
    path = args.report_directory/'B1_saved_shot_scope.csv'
    with path.open('x', newline='') as stream:
        fields = ('system','cell_id','epsilon','axis','alpha_axis','headroom','allowance','shots','integer_status','evidence_kind','baseline_receipt','baseline_pointer')
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            for axis in ('real','imag'):
                saved = row['baseline_accounting']; a = saved['axes'][axis]; loc = row['source_fields']['baseline']
                writer.writerow(dict(system=row['system'],cell_id=row['cell_id'],epsilon=row['epsilon'],axis=axis,
                    alpha_axis=saved['alpha_axis'],headroom=a['headroom'],allowance=a['allowance'],shots=a['shots'],
                    integer_status=a['integer_status'],evidence_kind=saved['evidence_kind'],
                    baseline_receipt=loc['receipt'],baseline_pointer=loc['pointer']+'/axes/'+axis))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
