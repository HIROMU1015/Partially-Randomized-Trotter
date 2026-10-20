"""Read-only evidence joins and literal recovery. No scientific module imports.

Never evaluate a signal, signed difference, envelope, eligibility, shot count,
cost, minimum, model, statistic or ranking. --check repeats metadata checks only.
Private files are restricted to the exact paths already bound by public receipts.
Their scientific values and grants are not exported by this script.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '85433c48fd4c37d4cb4594391da422558ac3ce7f'
B = 'artifacts/resource_applicability/'
OUT = ROOT / (B + 'track_a_h6_noncancellation_investigation/2026-10-20')
V1 = B + 'track_a_v1_reaccount_result/2026-10-16/'
FIXED = B + 'track_a_h6_fixed_delta_result_v1/2026-10-15/'
PREP = B + 'track_a_pf_selection_comparison_preparation/2026-10-20/'
DONE = B + 'track_a_pf_selection_cost_comparison/2026-10-20/'
PRIVATE = Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261015')
D_CONTRACT = Path('/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/private/sealed03/contracts/H6_D_ACCOUNTING')
PILOT = {(5, 1), (10, 2), (15, 8)}
IDS = {f'B0I_p{p}_q{q}_2nd' for p in (5, 10, 15) for q in (1, 2, 4, 8)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=ROOT)


def text_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def csv_text(rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
                         if isinstance(v, (dict, list, bool)) or v is None else v
                         for k, v in row.items()})
    return out.getvalue()


def collect():
    sources = {}

    def source(path, ref=BASE, decode=True):
        raw = git('show', ref + ':' + path)
        assert (ROOT / path).read_bytes() == git('show', BASE + ':' + path), path
        sources[(ref, path)] = dict(commit=ref, path=path, bytes=len(raw), sha256=sha(raw),
            git_blob_sha=git('rev-parse', ref + ':' + path).decode().strip(),
            url=f'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/{ref}/{path}')
        if not decode:
            return raw
        return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)

    remote = json.loads((OUT / 'github_receipt.json').read_text())
    for record in remote['records']:
        raw = source(record['path'], record['ref'], False)
        s = sources[(record['ref'], record['path'])]
        assert record['remote_ok'] and s['git_blob_sha'] == record['git_blob_sha']
        if 'bytes' in record:
            assert len(raw) == record['bytes']

    binding_path = B + 'track_a_v1_reaccount_readiness/2026-10-16/input_binding.json'
    receipts = source(binding_path)
    expected = {r['logical_name']: r for r in receipts['input_receipts']}
    accounting = source(V1 + 'candidate_accounting.json.gz')['rows']
    excluded = source(DONE + 'excluded_unresolved30.json')['rows']
    assert len(excluded) == 30
    for row in excluded:
        assert row['system'] == 'H6_DF19' and row['bias_mode'] == 'empirical_no_cancellation'
        assert set(row['unresolved_candidate_ids']) == IDS
    unresolved = [r for r in accounting if r['bias_rule'] == 'empirical_no_cancellation'
                  and r['cell_id'] in IDS]
    assert len(unresolved) == 120
    assert all(r['eligible'] is None and r['accounting'] is None and r['G_exact'] is None
               and r['cost_status'] == 'UNRESOLVED_SIGNED_DECOMPOSITION' for r in unresolved)
    assert Counter(r['cell_id'] for r in unresolved) == Counter({cid: 10 for cid in IDS})

    fixed = source(FIXED + 'signal_summary_v1.json')
    fixed_accounting = source(FIXED + 'accounting_v1.json')
    additional = source(B + 'track_a_v1_phase_a_run02/2026-10-16/signal_results_v1.json.gz')['signals']
    assert not IDS.intersection(r['cell']['id'] for r in additional)
    binding = source(PREP + 'input_bindings.json')['H6_DF19']
    prices = source(PREP + 'axis_price_lineage.json')['rows']
    candidates = []
    recovered = []
    private_receipts = []
    for p in (5, 10, 15):
        for q in (1, 2, 4, 8):
            cid = f'B0I_p{p}_q{q}_2nd'
            pilot = (p, q) in PILOT
            relative = ('h6-native-b0i-pilot6-run04-v3/run04' if pilot
                        else 'h6-native-b0i-full-family-run04/run01')
            name = f'case_{p}_{q}_{"scalar_signal" if pilot else "screen"}.json'
            path = PRIVATE / relative / name
            raw = path.read_bytes()
            receipt = expected[f'b0i_saved_signal_p{p}_q{q}']
            assert sha(raw) == receipt['sha256'] and len(raw) == receipt['bytes'], cid
            v = json.loads(raw)
            s = v['signal']
            assert s['cell']['id'] == cid and s['cell']['T'] == 0.8
            assert s['cell']['formula'] == '2nd'
            assert set(s['error_decomposition']) == {'total_signed'}
            assert set(s['signals']) == {'corrected', 'exact_truncated'}
            assert s['allowance']['certified'] is False and s['log_B'] == 0.0
            raw_binding = v['blueprint']['binding']
            for key in ('hamiltonian', 'df', 'state', 'input', 'compiler'):
                assert raw_binding[key] == binding['binding'][key], (cid, key)
            assert raw_binding['source'] == binding['provider_SOURCE']
            matched = [x for x in fixed if x['T'] == 0.8 and x['cell']['id'] == cid]
            if matched:
                assert len(matched) == 1
                old = matched[0]
                assert all(old[k] == s[k] for k in ('signals', 'reference', 'allowance', 'log_B'))
                assert set(old['decomposition']['parts_signed']) == {'discard', 'PF'}
                recovered.append(dict(candidate_id=cid, T=old['T'],
                    source_path=FIXED + 'signal_summary_v1.json', source_index=fixed.index(old),
                    signed_parts_verbatim=old['decomposition']['parts_signed'],
                    literal_identity_fields=['signals', 'reference', 'allowance', 'log_B'],
                    recomputed=False, closure_recomputed=False, eligibility_recovered=False))
            alias = receipt['logical_name']
            private_receipts.append(dict(role=alias, bytes=len(raw), sha256=sha(raw),
                public_receipt_path=binding_path, receipt_match=True, raw_exported=False))
            signed = [r for r in accounting if r['cell_id'] == cid and r['bias_rule'] == 'signed_signal']
            candidates.append(dict(candidate_id=cid, prefix=p, q=q, formula='2nd', T='0.8',
                origin_SOURCE='2e733d8a683d415792ce92a56c7d149b78f42716' if pilot else 'ac6d28e35cf01398f169ce59577182a38da9b959',
                raw_role=alias, raw_sha256=sha(raw), raw_bytes=len(raw), public_receipt_match=True,
                input_H_DF_state_compiler_provider_metadata_match=True,
                compiler_runtime_bit_identity_certified=False,
                old_signal_has_total_signed=True, old_signal_has_signed_parts=False,
                corrected_exact_truncated_reference_allowance_present=True,
                complete_noncancellation_flags_recovered=0,
                saved_price_coverage_reference_only=[dict(axis=u['axis'],
                    baseline_price_present=u['baseline_RZ_literal'] is not None,
                    ALL_SAVED458=u['ALL_SAVED458']) for u in prices
                    if u['system'] == 'H6_DF19' and u['candidate_id'] == cid],
                public_same_signal_decomposition_recovered=bool(matched),
                next_operation='COPY_EXISTING_PARTS_THEN_APPROVED_ENVELOPE_ELIGIBILITY' if matched else
                               'APPROVED_SAVED_SIGNAL_DIFFERENCES_ENVELOPE_ELIGIBILITY',
                ordinary_signed_flags_reference_only=[dict(epsilon=r['epsilon'], factor=r['factor'], eligible=r['eligible']) for r in signed],
                noncancellation_eligible=None, numeric_decision_executed=False))
    assert len(recovered) == 1 and recovered[0]['candidate_id'] == 'B0I_p15_q4_2nd'
    p15account = [r for r in fixed_accounting if r['T'] == 0.8 and r['cell_id'] == 'B0I_p15_q4_2nd']
    assert len(p15account) == 10 and all('bias_mode' not in r for r in p15account)

    # Read the already completed D; do not execute its selection function.
    seal_raw = (D_CONTRACT / 'technical_seal.json').read_bytes()
    seal = json.loads(seal_raw)
    payload_path = Path(seal['payload'])
    payload_raw = payload_path.read_bytes()
    assert sha(payload_raw) == payload_path.name
    result_raw = (D_CONTRACT / 'output/result.json').read_bytes()
    terminal_raw = (D_CONTRACT / 'output/terminal.json').read_bytes()
    terminal = json.loads(terminal_raw)
    assert terminal['result_sha256'] == sha(result_raw)
    assert terminal['SOURCE'] == seal['SOURCE'] == '31ef62a12407010e9c1a07f9b37facedb8dff5fe'
    assert terminal['status'] == 'COMPLETE_MANDATORY_STOP'
    result, payload = json.loads(result_raw), json.loads(payload_raw)
    private_unresolved = [r for r in result['rows'] if r['eligible'] is None]
    assert len(private_unresolved) == 120
    assert all(r['cell_id'] in IDS and r['bias_mode'] == 'EMPIRICAL_NONCANCELLATION'
               and r['shots'] is None and r['G_exact'] is None for r in private_unresolved)
    assert len([r for r in payload['rows'] if r['eligible'] is None]) == 120
    comparisons = [r for r in result['comparisons'] if r['bias_mode'] == 'EMPIRICAL_NONCANCELLATION']
    public = source(B + 'track_a_pr_cost_batch_results/2026-10-18/d_saved_comparison_summary.json')['H6']
    public_rows = [r for r in public['comparisons'] if r['bias_mode'] == 'EMPIRICAL_NONCANCELLATION']
    assert len(comparisons) == len(public_rows) == 20
    assert all(r['unresolved'] == [] for r in comparisons + public_rows)
    science = source('scripts/pr_cost_batch_science.py', seal['SOURCE'], False)
    assert sha(science) == seal['source_hashes']['scripts/pr_cost_batch_science.py']
    assert "and r['eligible'] and r['G_exact'] is None" in science.decode()
    assert "and r['eligible'] and r['G_exact'] is not None" in science.decode()
    d_observation = dict(status='SAVED_NULL_ELIGIBILITY_NOT_RESOLVED_BY_LATER_D', SOURCE=seal['SOURCE'],
        saved_unresolved_rows=120, saved_noncancellation_comparisons=20,
        saved_summary_unresolved_lists_empty=True,
        cause='Truthy eligible filters omit None both from available minimum and unresolved list',
        original_data_or_source_modified=False, no_new_selection_executed=True,
        public_summary_sufficient_to_certify_full_domain=False,
        impact_on_approved70='NONE: approved70 uses V1 signed H6 and H8 six-axis, not this later D',
        private_evidence=[dict(role=role, bytes=len(raw), sha256=sha(raw), raw_exported=False)
            for role, raw in [('D_technical_seal',seal_raw),('D_payload',payload_raw),
                              ('D_saved_result',result_raw),('D_terminal',terminal_raw)]])
    summary70 = source(DONE + 'summary.json')
    checks = dict(status='PASS_METADATA_AND_LITERAL_CHECKS_ONLY',
        github_blob_and_optional_size_checks=len(remote['records']),
        raw12_receipt_checks=12, unchanged_old_unresolved_candidate_endpoints=120,
        excluded_scope_conditions_unchanged=30, public_literal_decomposition_recovered=1,
        saved_noncancellation_eligibility_rows_recovered=0, scope_conditions_resolved=0,
        later_D_null_rows_retained=120, old70_results_recalculated=False,
        scientific_modules_imported=[], scientific_functions_called=[],
        new_differences_envelopes_eligibility_shots_costs_ranking_compile_sampling_fit=0,
        independent_review=False, detailed_scientific_review=False, final_evidence_freeze=False)
    return {
        'source_inventory.json': list(sources.values()),
        'candidate12_evidence.json': dict(rows=candidates),
        'candidate12_evidence.csv': csv_text(candidates),
        'saved_unresolved120.json': dict(status='VERBATIM_OLD_ROWS_NOT_NEW_DECISIONS', rows=unresolved),
        'excluded30_reference.json': dict(status='VERBATIM_EXCLUDED_UNCHANGED', rows=excluded),
        'literal_decomposition_recovery.json': dict(status='LITERAL_COPY_ONLY_NOT_ELIGIBILITY', rows=recovered),
        'private_receipt_observations.json': dict(raw_files=private_receipts,
            private_scientific_arrays_grants_and_raw_not_exported=True),
        'later_D_evidence_gap.json': d_observation,
        'approved70_summary_reference.json': summary70,
        'structural_validation.json': checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = collect()
    for name, value in result.items():
        raw = value if isinstance(value, str) else text_json(value)
        if args.check:
            assert (OUT / name).read_text() == raw, name
        else:
            with (OUT / name).open('x') as f:
                f.write(raw)
    print(text_json(result['structural_validation.json']), end='')


if __name__ == '__main__':
    main()
