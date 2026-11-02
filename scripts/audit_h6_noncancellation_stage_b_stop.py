"""Technical STOP audit only; do not invoke numeric accounting or change a grant."""
import hashlib
import json
from pathlib import Path

from run_h6_noncancellation_stage_b import A, OLD, OUT, PREP, ROOT, read, sha, verify


def main():
    contract = read(OUT / 'sealed_contract.json')
    verify(contract)
    ledger = read(OUT / 'ledger.json')
    native = read(OUT / 'native_exit_evidence.json')
    stop = read(OUT / 'STOP.json')
    consumed = read(OUT / 'grant_consumption_receipt.json')
    assert ledger['status'] == 'STOP_NO_RETRY'
    assert ledger['error_type'] == 'TypeError'
    assert ledger['error'] == "dict() got multiple values for keyword argument 'SOURCE'"
    assert ledger['stages_reserved'] == ledger['stages_completed']
    assert all(v == 0 for v in ledger['prohibited_calls'].values())
    assert stop['status'] == 'STAGE_B_NUMERIC_OR_TECHNICAL_STOP' and stop['retry'] is False
    assert consumed['status'] == 'CONSUMED_STOP' and consumed['further_execution_authorized'] is False
    assert native['native_exit_code'] == 1 and native['same_parent_wait4'] and native['reaped']
    assert native['live_owned_workers'] == 0
    for channel in ('stdout', 'stderr'):
        assert sha(OUT / ('native_' + channel + '.txt')) == native[channel + '_sha256']
    for name in ('launch.marker', 'one_shot.marker'):
        assert (OUT / name).read_text().strip() == contract['SOURCE']

    joined = read(OUT / 'joined_rows.json')['rows']
    original = {r['row_id']: r for r in read(ROOT / PREP / 'saved_condition_rows.json')['rows']}
    stage_a = {(r['candidate_id'], str(r['epsilon']), str(r['allowance_factor'])): r
               for r in read(ROOT / A / 'endpoints.json')['rows']}
    patched = []
    for row in joined:
        old = original[row['row_id']]
        if old['eligible'] is not None:
            assert row == old
        else:
            accepted = stage_a[(row['candidate_id'], row['epsilon'], row['allowance_factor'])]
            assert row['eligible'] is accepted['eligible']
            assert row['saved_shots'] == accepted['shots']
            assert row['saved_bias_and_shot_policy'] == accepted['accounting']
            assert row['saved_total_RZ_literal'] is None
            patched.append(row['row_id'])
    assert len(joined) == 1440 and len(patched) == 120
    assert all(type(r['eligible']) is bool for r in joined)
    partial = read(OUT / 'selection_comparison30.json')['rows']
    combined = read(OUT / 'combined_selection100.json')['rows']
    previous = read(ROOT / OLD / 'selection_comparison.json')['rows']
    assert len(partial) == 30 and len(combined) == 100 and len(previous) == 70
    for a, b in zip(combined[:70], previous):
        assert {k: v for k, v in a.items() if k != 'numeric_origin'} == b
    assert combined[70:] == partial
    assert len({r['condition_id'] for r in combined}) == 100
    assert len({tuple(r[k] for k in ('system', 'bias_mode', 'epsilon', 'allowance_factor'))
                for r in combined}) == 40

    missing = ['summary.json', 'numeric_validation.json', 'numeric_completion.json',
               'completion_audit.json', 'candidate_totals.csv', 'selection_comparison30.csv',
               'candidate_ranks.csv', 'rank_relation_changes.csv', 'combined_selection100.csv',
               'figure_data.json', 'rendering_completion.json']
    assert all(not (OUT / name).exists() for name in missing)
    excluded = {'partial_output_inventory.json', 'stop_audit.json', 'README.md',
                'publication_scope.json', 'preservation_audit.json'}
    inventory = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                 for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in excluded}
    payload = dict(status='PRESERVED_PARTIAL_RUN_FILES_NO_FINAL_NUMERIC_GATE',
                   SOURCE=contract['SOURCE'], files=inventory)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode()
    (OUT / 'partial_output_inventory.json').write_bytes(raw)
    audit = dict(status='PASS_TECHNICAL_STOP_PRESERVATION_ONLY', SOURCE=contract['SOURCE'],
        scientific_task_complete=False, accepted_new_comparison_scopes=0,
        partial_new_scope_rows=30, reference_rows_with_old70=100, unique_reference_endpoints=40,
        original_nonnull_rows_and_PR_verbatim=True, patched_only_unknown120=True,
        stage_A_eligibility_shots_accounting_verbatim=True, previous70_verbatim=True,
        SOURCE_input_runtime_protected_evidence_gate='PASS',
        protected_old_files=len(contract['protected_evidence_hashes']),
        input_files=len(contract['input_hashes']), source_files=len(contract['source_hashes']),
        native_exit_code=1, same_parent_wait4_reaped=True, live_owned_workers=0,
        wall_seconds=native['wall_seconds'], CPU_seconds=native['CPU_seconds'],
        peak_RSS_bytes=native['peak_RSS_bytes'],
        stages_recorded_completed=ledger['stages_completed'],
        numeric_validation_return_recorded=True, numeric_validation_receipt_saved=False,
        final_numeric_gate=False, final_old_evidence_gate_not_reached_in_worker=True,
        post_STOP_evidence_gate='PASS_TECHNICAL_READ_ONLY', missing_completion_files=missing,
        partial_output_inventory_sha256=hashlib.sha256(raw).hexdigest(),
        scientific_functions_invoked_by_this_audit=0, old_marker_STOP_grant_modified=False,
        retry=False, grant_consumed=True, independent_reviewer=False,
        new_publication_authorized=False, detailed_scientific_review=False,
        final_evidence_freeze=False, mandatory_STOP=True)
    (OUT / 'stop_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
