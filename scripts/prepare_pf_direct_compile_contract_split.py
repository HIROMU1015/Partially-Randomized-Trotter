"""Partition the fixed PF proposal into two dormant contracts; metadata only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

import pf_direct_compile_contract as c

ROOT = Path(__file__).resolve().parents[1]
BASE = 'ed3aa50976121bb470b64ebb0e38afeec63ed797'
PLAN = ROOT / 'artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19'
TECH = ROOT / 'artifacts/resource_applicability/track_a_pf_direct_compile_technical_preparation/2026-10-20'
OUT = ROOT / 'artifacts/resource_applicability/track_a_pf_direct_compile_contract_split/2026-10-20'
SCRIPT = 'scripts/prepare_pf_direct_compile_contract_split.py'
REPORT = 'docs/research/track_a_pf_direct_compile_contract_split_20261020.md'
NOTE = 'docs/research/研究ノート/2026-10-20_pf_direct_compile_contract_split.md'
INDEXES = {'PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'docs/README.md',
           'docs/research/README.md', 'docs/research/研究概要・現状.md',
           'scripts/README.md', 'src/trotterlib/README.md', 'artifacts/validation_manifest.json'}
KEY = 'track_a_pf_direct_compile_contract_split_20261020'
COUNTS = {'H2_DF3': 72, 'H4_DF10': 122, 'H6_DF19': 214, 'H8_DF28': 314,
          'H4_LEGACY_DF12': 66}
GROUPS = {'current_4series_722': {'H2_DF3', 'H4_DF10', 'H6_DF19', 'H8_DF28'},
          'legacy_h4_df12_66': {'H4_LEGACY_DF12'}}
USER = '私の推奨は、788軸を計画対象として維持し、現行4系列の722軸とlegacyの66軸を独立した実行契約に分ける'


def blob(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def proposals():
    path = PLAN / 'proposed_compile_axes.csv'
    expected = c.read_json(TECH / 'execution_contract_DRAFT.json')['proposal_sha256']
    rows = c.parse_proposals(path.read_bytes(), expected)
    c.require(len(rows) == 788, 'UNCHANGED_UNIVERSE')
    c.require({key: sum(r['system'] == key for r in rows) for key in COUNTS} == COUNTS,
              'UNCHANGED_SERIES_COUNTS')
    return path, rows


def contract(group, systems, rows, proposal_path):
    selected = c.select_pairs(rows, [r['task_id'] for r in rows if r['system'] in systems])
    result = c.draft(rows, c.sha(proposal_path), c.sha(PLAN / 'system_bindings.json'))
    result.update(schema='PF_DIRECT_SPLIT_CONTRACT_DRAFT_V1',
        technical_base_commit=BASE, contract_id=group, systems=sorted(systems),
        proposal_universe_axes=788, proposed_axes=len(selected),
        proposed_task_ids=[r['task_id'] for r in selected], proposed_rows_digest=c.digest(selected),
        proposed_call_caps=c.call_caps(selected),
        proposed_output_namespace=group,
        scope_status='PROPOSED_PARTITION_ONLY_ACTUAL_ACQUISITION_NOT_APPROVED',
        independence=dict(approval_receipt='separate exact contract digest',
            grant='separate one-shot grant', ledger='separate owned ledger',
            output='separate exclusive output root', marker='separate owned marker',
            STOP='separate contract STOP; no automatic next contract',
            shared_readonly_source_inputs='allowed only after exact byte/profile verification',
            shared_writable_result_or_consumption_state=False),
        production_scope_gate_bound=False,
        next_contract_authorized=False,
        conditional_48_axes_included=False)
    return result, selected


def write():
    OUT.mkdir(parents=True, exist_ok=True)
    path, rows = proposals()
    save('partition_decision.json', dict(user_instruction=USER,
        interpretation='Adopt two independent contract proposals; not acquisition or scientific execution approval.',
        proposed_axes=788, per_series=COUNTS,
        proposed_groups={key: sum(COUNTS[s] for s in systems) for key, systems in GROUPS.items()},
        acquisition_scope_approved=False, execution_authorized=False,
        production_binding_complete=False, conditional_48_axes_included=False))
    for group, systems in GROUPS.items():
        draft, selected = contract(group, systems, rows, path)
        save(group + '_contract_DRAFT.json', draft)
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(selected)
        (OUT / (group + '_proposed_axes.csv')).write_bytes(stream.getvalue().encode())
    preserved = {str(p.relative_to(ROOT)): c.sha(p) for folder in (PLAN, TECH)
                 for p in folder.iterdir() if p.is_file()}
    for item, checksum in preserved.items():
        c.require(hashlib.sha256(blob(item)).hexdigest() == checksum, 'OLD_PACKET_CHANGED:' + item)
    save('preservation.json', dict(base_commit=BASE, files=preserved,
        prior_SOURCE_grants_ledgers_markers_STOP_unchanged=True))
    save('MANDATORY_STOP.json', dict(status='MANDATORY_STOP_CONTRACT_PARTITION_ONLY',
        proposed_axes=788, current_proposed_axes=722, legacy_proposed_axes=66,
        selected_axes=0, execution_authorized=False, grant_issued=False,
        real_input_decodes=0, real_circuit_builds=0, real_control_probes=0, real_compiles=0,
        sampling=0, fit=0, cost_reaccounting=0, next_stage_authorized=False))


def check():
    path, rows = proposals()
    all_ids = set(); sizes = {}
    for group, systems in GROUPS.items():
        actual = c.read_json(OUT / (group + '_contract_DRAFT.json'))
        expected, selected = contract(group, systems, rows, path)
        c.require(actual == expected, 'DRAFT_CHANGED:' + group)
        ids = set(actual['proposed_task_ids'])
        c.require(not (all_ids & ids), 'OVERLAPPING_CONTRACTS')
        all_ids |= ids; sizes[group] = len(ids)
        copied = c.parse_proposals((OUT / (group + '_proposed_axes.csv')).read_bytes(),
                                  c.sha(OUT / (group + '_proposed_axes.csv')))
        c.require(copied == selected, 'COPIED_ROWS_CHANGED')
        c.require(actual['selected_task_ids'] is None
                  and all(value == 0 for value in actual['call_caps'].values()), 'NO_EFFECTIVE_CALLS')
        try:
            c.authorize_metadata(actual, {}, rows)
        except ValueError as error:
            c.require(str(error) == 'SCIENTIFIC_EXECUTION_NOT_AUTHORIZED', 'UNEXPECTED_REJECTION')
        else:
            raise ValueError('DORMANT_CONTRACT_ACCEPTED')
    c.require(all_ids == {r['task_id'] for r in rows}, 'EXACT_788_COVERAGE')
    c.require(sizes == {'current_4series_722': 722, 'legacy_h4_df12_66': 66}, 'PARTITION_COUNTS')
    for item, checksum in c.read_json(OUT / 'preservation.json')['files'].items():
        c.require(c.sha(ROOT / item) == checksum == hashlib.sha256(blob(item)).hexdigest(),
                  'OLD_PACKET_BYTES_CHANGED:' + item)
    # Historical artifact manifests describe their commit snapshots, not current indexes.
    for old in (PLAN, TECH):
        for item, checksum in c.read_json(old / 'artifact_manifest.json')['files'].items():
            snapshot = c.PLAN_COMMIT if old == PLAN else BASE
            raw = subprocess.check_output(['git', 'show', snapshot + ':' + item], cwd=ROOT)
            c.require(hashlib.sha256(raw).hexdigest() == checksum, 'OLD_SNAPSHOT_HASH:' + item)
    allowed = INDEXES | {SCRIPT, REPORT, NOTE} | {str(p.relative_to(ROOT)) for p in OUT.iterdir() if p.is_file()}
    changed = {p for p in subprocess.check_output(['git', 'diff', '--name-only', '-z', BASE], cwd=ROOT).decode().split('\0') if p}
    untracked = {p for p in subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0') if p}
    c.require(changed | untracked <= allowed, 'UNEXPECTED_SOURCE_OR_RESULT_CHANGE')
    for item in INDEXES - {'artifacts/validation_manifest.json'}:
        c.require((ROOT / item).read_bytes().endswith(blob(item)), 'OLD_INDEX_TEXT_CHANGED')
    old_manifest = json.loads(blob('artifacts/validation_manifest.json'))
    new_manifest = c.read_json(ROOT / 'artifacts/validation_manifest.json')
    c.require(set(new_manifest) - set(old_manifest) == {KEY}
              and all(new_manifest[key] == value for key, value in old_manifest.items()), 'OLD_MANIFEST_CHANGED')
    decision = c.read_json(OUT / 'partition_decision.json')
    c.require(decision['user_instruction'] == USER and decision['proposed_groups'] == sizes
              and decision['acquisition_scope_approved'] is False
              and decision['execution_authorized'] is False, 'DECISION_BOUNDARY')
    return dict(status='PASS_METADATA_PARTITION_ONLY', proposed_axes=788,
        current_axes=722, legacy_axes=66, disjoint=True, exact_coverage=True,
        copied_rows_identical=True, cosine_sine_pairs_complete=True,
        both_contracts_reject_execution=True, selected_axes=0, real_compiles=0,
        old_packets_and_snapshot_hashes_preserved=True, production_scope_gate_bound=False)


def publication():
    files = sorted(INDEXES | {SCRIPT, REPORT, NOTE}
                   | {str(p.relative_to(ROOT)) for p in OUT.iterdir() if p.is_file()}
                   | {str((OUT / name).relative_to(ROOT)) for name in ('publication_scope.json', 'artifact_manifest.json')})
    save('publication_scope.json', dict(base_commit=BASE, paths=files,
        new_publication_approved=False, prior_28_file_approval_does_not_cover_this_change=True))
    save('artifact_manifest.json', dict(status='METADATA_PARTITION_NOT_SCIENTIFIC_FREEZE',
        files={item: c.sha(ROOT / item) for item in files if not item.endswith('/artifact_manifest.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write dormant metadata only')
    args = parser.parse_args()
    if args.write:
        write()
    result = check()
    if args.write:
        save('structural_validation.json', result)
        publication()
    else:
        for item, checksum in c.read_json(OUT / 'artifact_manifest.json')['files'].items():
            c.require(c.sha(ROOT / item) == checksum, 'NEW_ARTIFACT_HASH:' + item)
    print(json.dumps(result, indent=2))
