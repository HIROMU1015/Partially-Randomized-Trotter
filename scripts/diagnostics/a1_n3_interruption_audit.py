"""Audit the fixed interrupted journals, never invoke N3/scientific code."""
from pathlib import Path
import hashlib
import json
import tempfile

from trottertracks.algorithm_codesign import a1_n3_interruption_diagnostics as audit

atomic_write_json = audit.atomic_write_json


def main():
    root = Path(__file__).resolve().parents[2]
    bundle = root / 'artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01'
    out = root / 'artifacts/o2_a1_n3_interruption_diagnostic/2026-11-06'
    target = out / 'JOURNAL_DIAGNOSTIC.json'
    if target.exists():
        raise FileExistsError('refusing to replace diagnostic result')
    source = root / 'src/trottertracks/algorithm_codesign/a1_n3_campaign.py'
    lock = json.loads((root / 'docs/research/o2_a1_n3_science_20261106/PROFILE_LOCK.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != lock['source_sha256'][str(source.relative_to(root))]:
        raise ValueError('frozen scientific source mismatch')
    members = audit.verified_members(bundle)
    parsed = {name: audit.parse_records(raw) for name, raw in members.items()}
    result = audit.inspect_records(parsed['JOURNAL.jsonl'][0],
        {name: rows for name, (rows, _) in parsed.items() if name.startswith('worker_')})
    result['incomplete_tails'] = {name: tail for name, (_, tail) in parsed.items() if tail}
    result['original_bundle'] = str(bundle.relative_to(root))
    result['source_reused'] = 'exact frozen reconcile function AST; no engine/module import'
    with tempfile.TemporaryDirectory(prefix='a1-n3-journal-audit-') as directory:
        workspace = Path(directory)
        for name, raw in members.items():
            (workspace / name).write_bytes(raw)
        audit.frozen_reconcile(source, workspace)
        for name in ('ACCOUNTING.json', 'CALL_COUNTS.json'):
            data = json.loads((workspace / name).read_text())
            if name == 'ACCOUNTING.json':
                data['forensic_process_RSS_bytes'] = data.pop('parent_peak_RSS_bytes')
                data['scientific_parent_peak_RSS_bytes'] = 'UNKNOWN'
            original = json.loads((bundle / name).read_text())
            if name == 'CALL_COUNTS.json':
                if any(data[k] != original[k] for k in data):
                    raise ValueError('reconciled counts differ from original')
            elif data['methods'] != original['methods']:
                raise ValueError('reconciled bit/price accounting differs')
            atomic_write_json(out / ('RECONCILED_' + name), data)
    result['reconciliation_against_saved_results'] = 'PASS'
    observed = json.loads((bundle / 'INTERRUPTION_OBSERVATION.json').read_text())
    result['saved_original_file_mtimes'] = observed['files']
    atomic_write_json(target, result)
    print(json.dumps({k: result[k] for k in ('planned', 'bits', 'prices',
        'unfinished_by_cell', 'calls_entered', 'elapsed_engine_seconds_saved_records',
        'reconciliation_against_saved_results', 'new_scientific_calls')}, indent=2))


if __name__ == '__main__':
    main()
