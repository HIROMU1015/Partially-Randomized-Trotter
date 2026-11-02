"""Read-only saved-evidence schema/lineage audit; no real bias arithmetic.

Default prints scope. --audit-evidence writes new metadata reports only inside
this isolated snapshot. There is no numerical-analysis or scientific executor.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/trotterlib'))
import h8_bias_evidence as e

ROOT = Path(__file__).resolve().parents[1]
A_SOURCE = 'fc463d9e07825f34e496acac6b40db7aec1aed6d'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
OLD_I_SOURCE = '54ed98c5fa5b7190f77255271f4e3bec8db15630'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'


def strict(data):
    def pairs(items):
        out = {}
        for k, v in items:
            e.require(k not in out, 'DUPLICATE_JSON_KEY'); out[k] = v
        return out
    def number(value):
        out = float(value); e.require(math.isfinite(out), 'NONFINITE_JSON'); return out
    def bad(value):
        raise ValueError('NONFINITE_JSON:' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_float=number, parse_constant=bad)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()


def read_inputs(manifest):
    e.require(manifest['schema'] == 'H8_BIAS_METADATA_AUDIT_INPUTS_V1' and
              manifest['real_numerical_analysis_authorized'] is False, 'METADATA_ONLY_INPUT_CONTRACT')
    receipts = {}; raw = {}
    for row in manifest['files']:
        name = Path(row['snapshot_path'])
        e.require(not name.is_absolute() and '..' not in name.parts and row['role'] not in raw, 'INPUT_PATH_OR_DUPLICATE')
        path = ROOT/name
        e.require(not path.is_symlink() and path.resolve().is_relative_to(ROOT) and
                  path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'FIXED_EVIDENCE_BYTES')
        raw[row['role']] = path.read_bytes()
        receipts[row['role']] = {k: row[k] for k in row if k != 'snapshot_path'}
    return raw, receipts


def audit(manifest):
    raw, receipts = read_inputs(manifest)
    objects = {role: strict(data) for role, data in raw.items()
               if not role.startswith('producer_') and role not in ('A_signals', 'H6_policy', 'H6_result')}
    export = objects['A_export']
    e.require(len(raw['A_signals']) == export['gzip_bytes'] and
              hashlib.sha256(raw['A_signals']).hexdigest() == export['gzip_sha256'], 'A_GZIP_RECEIPT')
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(raw['A_signals'])) as stream:
        decoded = stream.read(export['original_json_bytes']+1)
    e.require(len(decoded) == export['original_json_bytes'] and
              hashlib.sha256(decoded).hexdigest() == export['original_json_sha256'], 'A_EXACT_ORIGINAL_BYTES')
    combined = next(row for row in objects['A_file_receipts']['files']
                    if row['path'] == 'signal_output/H8_signal_complete.json')
    e.require(combined['bytes'] == len(decoded) and combined['sha256'] == export['original_json_sha256'],
              'A_RAW_PUBLIC_EXPORT_BINDING')
    additional = strict(decoded)
    e.require(additional['status'] == objects['A_terminal']['status'] == 'COMPLETE_MANDATORY_STOP' and
              objects['A_terminal']['SOURCE'] == objects['A_binding']['SOURCE'] == A_SOURCE and
              objects['A_binding']['provider_SOURCE'] == PROVIDER and
              objects['A_binding']['profile_digest'] == PROFILE, 'A_COMPLETION_SOURCE_PROFILE')
    reference_receipt = next(row for row in objects['A_binding']['input_receipts']
                             if row['logical_name'] == 'full_H_reference')
    e.require(reference_receipt['sha256'] == receipts['full_reference']['sha256'] ==
              objects['A_reference']['reference_receipt_sha256'] and
              objects['full_reference']['reference_agreement']['certified'] is False and
              objects['full_reference']['numerical_allowance_certified'] is False,
              'SAVED_FULL_H_REFERENCE_EMPIRICAL_LINEAGE')
    workers = objects['A_workers']
    e.require(len(workers['workers']) == 35 and workers['all_reaped'] is True and
              all(w['exit_code'] == 0 and w['reaped'] is True for w in workers['workers']) and
              workers['original_result_hash_missing'] == ['prefix_19'] and
              workers['original_receipts_modified'] is False, 'A_NATIVE_WORKER_PROVENANCE')
    missing_runtime_hash = [w['task_id'] for w in workers['workers'] if not w['original_result_sha256_present']]
    e.require(missing_runtime_hash == ['prefix_19'] and
              all(type(w.get('publication_result_sha256')) is str and len(w['publication_result_sha256']) == 64
                  for w in workers['workers']), 'A_PUBLICATION_HASH_DISTINCT_FROM_RUNTIME')
    binding = objects['old_D0_contract']['binding']
    summary_receipt = next(r for r in objects['old_D0_contract']['input_receipts']
                           if r['logical_name'] == 'signal_summary.json')
    e.require(summary_receipt['bytes'] == receipts['old_signals']['bytes'] and
              summary_receipt['sha256'] == receipts['old_signals']['sha256'] and
              binding['source'] == objects['old_completion']['SOURCE'] == PROVIDER and
              objects['old_completion']['status'] == 'COMPLETE_SAVED_RESULTS_VERIFIED', 'OLD_D0_INPUT_LINEAGE')
    old = objects['old_signals']['signals']
    e.require(len(old) == 92 and len(additional['signals']) == 152, 'INPUT_REGISTRY_COUNTS')
    old_i = objects['old_B0I']
    e.require(len(old_i) == 12 and objects['old_B0I_provenance']['execution_SOURCE'] == OLD_I_SOURCE and
              objects['old_B0I_provenance']['profile_digest'] == PROFILE and
              objects['old_B0I_completion']['SOURCE'] == OLD_I_SOURCE and
              objects['old_B0I_completion']['status'] == 'COMPLETE_MANDATORY_STOP', 'OLD_B0I_PUBLIC_SOURCE')
    origins = {
        'old': dict(kind='OLD_H8_SAVED_SUMMARY_BOUND_BY_D0_INPUT_RECEIPT', SOURCE=PROVIDER,
                    input_sha256=receipts['old_signals']['sha256'], runtime_profile='ORIGINAL_OR_UNKNOWN_NOT_BACKFILLED'),
        'old_i': dict(kind='OLD_B0I_PUBLIC_SCALAR_SIGNAL_RECEIPTS', SOURCE=OLD_I_SOURCE,
                      input_sha256=receipts['old_B0I']['sha256'], runtime_profile=PROFILE,
                      scalar_policy='ROTATE_CORRECTED_AND_TRUNCATED_LEAVE_FULL_H_REFERENCE_UNROTATED'),
        'additional': dict(kind='PHASE_A_EXACT_PUBLIC_GZIP_EXPORT', SOURCE=A_SOURCE, provider_SOURCE=PROVIDER,
                           input_sha256=export['original_json_sha256'], runtime_profile=PROFILE),
    }
    records = [(r, origins['old']) for r in old if r['cell']['method'] in ('B0', 'B1')]
    old_by = {r['cell']['id']: r for r in old if r['cell']['method'] == 'B0'}
    correctness = {(r['prefix'], r['q']): r for r in objects['old_B0I_correctness']}
    e.require(len(correctness) == len(objects['old_B0I_correctness']) == 8 and
              all(r['status'] == 'PASS' and r['certified'] is False for r in correctness.values()),
              'OLD_B0I_SAVED_CORRECTNESS_PROJECTION')
    for wrapper in old_i:
        record = wrapper['signal']; p, q = wrapper['prefix'], wrapper['q']
        e.require(record['cell']['prefix'] == p and record['cell']['q'] == q and
                  wrapper['scalar']['full_Fock_identity'] is True and
                  wrapper['scalar']['new_scientific_certificate'] is False and
                  f'B0_p{p}_q{q}_2nd' in old_by, 'OLD_B0I_PARENT_AND_SCALAR_METADATA')
        origin = dict(origins['old_i'], parent_B0_record_exists=True,
                      saved_scalar_receipt_present=True,
                      native_correctness_projection_present=(p, q) in correctness,
                      no_direct_independent_forward_fields_in_this_overlay_record=True)
        records.append((record, origin))
    records += [(r, origins['additional']) for r in additional['signals']]
    result = e.inventory(records, objects['registry'])
    result['input_receipts'] = list(receipts.values())
    result['lineage_limitations'] = dict(
        A_prefix19_runtime_result_SHA='MISSING_ORIGINAL_FIELD; separately recorded publication hash retained',
        original_runtime_SHA_backfilled=False,
        independent_forward_values='ENDPOINTS_PRESENT_IS_NOT_COMPONENT_AGREEMENT_VERIFIED',
        B0I_overlay='PARENT_FORWARD_ENDPOINTS_SHARE_SCALAR_AND_PHASE_OVERLAY; no independent scalar certificate',
        old_B0I_native_correctness_projection_cases=8, old_B0I_endpoint_cases=12,
        exact_truncated_label='NUMERICALLY_COMPUTED_REFERENCE_NOT_EXACT_ARITHMETIC_CERTIFICATE',
        historical_closure_checks='CITED_PRIOR_RECEIPTS_NOT_RECOMPUTED',
        whole_reference_or_runtime_inventory_rescanned=False)
    selected = objects['old_completion']['selected_ids']
    e.require(len(selected) == len(set(selected)) == 6, 'FIXED_PR_CONFIRMATION_UNION6')
    pr_records = {r['cell']['id']: r for r in old if r['cell']['method'] == 'B2'}
    pr = []
    for cid in selected:
        row = pr_records[cid]
        pr.append(dict(cell_id=cid, axes=[dict(axis=a,
            outer_PF_signed_component_present=e.has_axis(row['error_decomposition'].get('signed_parts', {}).get('outer_PF'), a),
            finite_RTE_signed_component_present=e.has_axis(row['error_decomposition'].get('signed_parts', {}).get('finite_RTE'), a),
            independent_corrected_endpoint_present=e.has_axis(row.get('independent_forward_signals', {}).get('corrected'), a),
            independent_exact_tail_endpoint_present=e.has_axis(row.get('independent_forward_signals', {}).get('exact_tail'), a),
            numerical_reanalysis_executed=False) for a in e.AXES]))
    result['PR_confirmation_context_metadata_only'] = dict(scope='PREVIOUSLY_SELECTED_UNION6_NOT_A_NEW_SELECTION',
        rows=pr, new_PR_noncancellation_shots=0, new_paired_cost_or_mean_computed=0)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-evidence', action='store_true'); p.add_argument('--output', type=Path)
    args = p.parse_args()
    if not args.audit_evidence:
        if args.output: p.error('--audit-evidence required')
        print(json.dumps(dict(scope='SAVED_EVIDENCE_METADATA_ONLY', candidates=184,
              new_bias_arithmetic=0, cost_reaccount=0, runtime_authorization=False))); return
    if args.output is None: p.error('new metadata report path required')
    output = args.output.resolve()
    e.require(output.is_relative_to(ROOT) and not output.exists(), 'NEW_SNAPSHOT_METADATA_OUTPUT')
    result = audit(strict((ROOT/'references/evidence_inputs_v1.json').read_bytes()))
    with output.open('x') as stream: json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(dict(status=result['status'], candidate_counts=result['candidate_counts'],
                          axis_counts=result['axis_counts'], real_numerical_analysis=0)))


if __name__ == '__main__': main()
