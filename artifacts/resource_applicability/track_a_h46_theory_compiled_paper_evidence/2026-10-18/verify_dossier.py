"""Metadata/document verification only; no scientific imports or arithmetic.

Run with system Python -P -B. --write records technical_checks.json, otherwise
all checks are read-only. Git is used solely to read pinned blobs and file names.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = '6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1'
OLD = '7e454b91185ed7b0f92a3790f800ba5d65d5f156'
CANON = 'artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18'
REL = HERE.relative_to(ROOT).as_posix()
DOC = 'docs/research/track_a_h46_theory_compiled_paper_evidence_20261018.md'
cache = {}
checks = []


def check(name, value, detail=None):
    if not value:
        raise AssertionError((name, detail))
    checks.append({'check': name, 'status': 'PASS', 'detail': detail})


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), '-c',
        'maintenance.auto=false', '-c', 'gc.auto=0', '-c', 'core.quotePath=false', *args])


def pinned(commit, path):
    key = commit, path
    if key not in cache:
        cache[key] = git('show', commit + ':' + path)
    return cache[key]


def read(name):
    return json.loads((HERE / name).read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def resolve(evidence):
    data = json.loads(pinned(evidence['commit'], evidence['path']))
    for part in evidence['json_pointer'].split('/')[1:]:
        key = part.replace('~1', '/').replace('~0', '~')
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def projection(entry):
    original = resolve(entry['source'])
    check('lossless_field_projection:' + entry['row_id'], all(
        key in original and value == original[key]
        for key, value in entry['values'].items()))


def main():
    receipt = HERE / 'technical_checks.json'
    if '--write' in sys.argv and not receipt.exists():
        receipt.write_text('{"status":"PENDING_DOCUMENT_CHECK"}\n')
    tables = {name: json.loads(pinned(BASE, CANON + '/stage3/' + name + '.json'))
        for name in ('expectation_vs_saved', 'same_event_comparisons', 'pf_pr_comparisons')}
    check('canonical_row_counts', [len(v) for v in tables.values()] == [84, 84, 360])
    for name, rows in tables.items():
        path = ROOT / CANON / 'stage3' / (name + '.csv')
        check('canonical_working_json_unchanged:' + name,
            (path.with_suffix('.json')).read_bytes() == pinned(BASE, CANON + '/stage3/' + name + '.json'))
        check('canonical_working_csv_unchanged:' + name,
            path.read_bytes() == pinned(BASE, CANON + '/stage3/' + name + '.csv'))
        with path.open(newline='') as f:
            csv_rows = list(csv.DictReader(f))
        check('csv_json_row_count:' + name, len(csv_rows) == len(rows))
        for i, (csv_row, row) in enumerate(zip(csv_rows, rows)):
            for k, text in csv_row.items():
                expected = row.get(k)
                if isinstance(expected, (dict, list)):
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = ast.literal_eval(text)
                    check('csv_json_nested:' + name + ':' + str(i) + ':' + k,
                        parsed == expected)
                else:
                    check('csv_json_value:' + name + ':' + str(i) + ':' + k,
                        text == ('' if expected is None else str(expected)))
    index = read('row_index.json')['rows']
    check('row_index_count', len(index) == 528)
    check('row_aliases_unique', len({x['row_id'] for x in index}) == 528)
    for entry in index:
        original = resolve(entry['source'])
        data = json.dumps(original, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        check('row_digest:' + entry['row_id'], sha(data) == entry['row_digest_sha256'])
        check('row_source:' + entry['row_id'], entry['SOURCE_accounting'] == original['SOURCE_accounting']
            and entry['SOURCE_counter_evaluation'] == original['SOURCE_counter_evaluation'])
    expected_ids = [('H4', 'B1_p12_q1_2nd'), ('H4', 'B1_p12_q1_4th'),
        ('H4', 'B2_p3_q1_2nd_r4_K2'), ('H4', 'B2_p3_q8_2nd_r2_K4'),
        ('H6', 'B1_p19_q8_4th'), ('H6', 'B2_p10_q8_2nd_r1_K2'),
        ('H6', 'B2_p15_q8_2nd_r2_K4')]
    candidates = read('table_a_conditions.json')['candidates']
    check('fixed_7_candidates', sorted((c['system'], c['candidate_id']) for c in candidates) == sorted(expected_ids))
    for candidate in candidates:
        check('candidate_receipt:' + candidate['candidate_id'], candidate['origin'] == resolve(candidate['source']))
    for name, count, kind in [('table_b_expectation_candidates.json', 28, 'expectation_vs_saved'),
        ('table_b_same_event_candidates.json', 28, 'same_event_comparisons'),
        ('table_c_candidates.json', 30, 'pf_pr_comparisons')]:
        entries = read(name)['rows']
        check('projection_row_count:' + name, len(entries) == count)
        for entry in entries:
            check('estimand_separation:' + entry['row_id'], entry['comparison_kind'] == kind)
            projection(entry)
            if kind == 'pf_pr_comparisons':
                direction = entry['saved_direction_evidence']
                if direction:
                    check('saved_direction:' + entry['row_id'], direction['values'] == resolve(direction['source']))
                for shots in entry['saved_shot_evidence']:
                    check('saved_shots:' + entry['row_id'], shots['values'] == resolve(shots['source']))
    for entry in read('table_d_s4_candidates.json')['rows']:
        check('S4_acceptance_projection:' + entry['row_id'], entry['acceptance'] == resolve(entry['source_acceptance']))
        projection(entry['same_event_row'])
    check('table_D_8_rows', len(read('table_d_s4_candidates.json')['rows']) == 8)
    check('same_event_mixed_saved_count_match', all(r['counter_native_count_match'] is True
        for r in tables['same_event_comparisons'] if r['model'] == 'NATIVE_MIXED'))
    for stage in (1, 2, 3):
        terminal = json.loads(pinned(BASE, CANON + f'/stage{stage}/terminal.json'))
        native = json.loads(pinned(BASE, CANON + f'/stage{stage}/native_exit.json'))
        check('historical_terminal:' + str(stage), terminal['status'] == 'COMPLETE_MANDATORY_STOP'
            and native['native_exit_code'] == 0 and native['wait_reap_completed'] is True)
    claims = read('claim_evidence_matrix.json')['claims']
    aliases = {x['row_id'] for x in index}
    issue_ids = {x['issue_id'] for x in read('limitations.json')['issues']}
    states = {'DIRECTLY_SUPPORTED_WITHIN_FIXED_SCOPE', 'CONDITIONAL_ON_SAVED_MODEL_OR_SAMPLE',
        'TECHNICALLY_VERIFIED_ONLY', 'INSUFFICIENT_EVIDENCE', 'NOT_SUPPORTED', 'REQUIRES_USER_SCIENTIFIC_DECISION'}
    check('claim_12_unique', len(claims) == 12 and len({c['claim_id'] for c in claims}) == 12)
    for c in claims:
        check('claim_schema:' + c['claim_id'], c['status'] in states and not c['adopted_as_paper_claim']
            and c['formal_scientific_adoption_required'] is True and bool(c['evidence'])
            and set(c['row_ids']) <= aliases and set(c['open_issues']) <= issue_ids)
        for e in c['evidence']:
            if e['path'].endswith('.json'):
                resolve(e)
            else:
                pinned(e['commit'], e['path'])
    check('no_new_figures', read('figure_inventory.json')['new_figures_created'] == 0)
    provenance = read('provenance_index.json')
    for e in provenance['artifacts']:
        blob = pinned(e['commit'], e['path'])
        check('file_hash:' + e['commit'] + ':' + e['path'], len(blob) == e['bytes'] and sha(blob) == e['sha256'])
    for e in read('source_closure_checks.json')['checks']:
        check('source_closure:' + e['stage'] + ':' + e['path'],
            sha(pinned(e['SOURCE'], e['path'])) == e['expected_sha256'] == e['actual_sha256'])
    for e in provenance['executed_original_vs_public_mirror']:
        check('original_mirror_tree:' + e['executed_original'],
            git('rev-parse', e['executed_original'] + '^{tree}') == git('rev-parse', e['public_mirror'] + '^{tree}'))
    markdowns = [ROOT / DOC, HERE / 'README.md', HERE / 'table_index.md',
        HERE / 'claim_evidence_matrix.md', HERE / 'limitations.md', HERE / 'figure_inventory.md']
    links = []
    for p in markdowns:
        text = p.read_text()
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', text):
            url = match[1]
            if url.startswith('https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'):
                part = urlsplit(url).path.split('/blob/', 1)[1]
                commit, file = part.split('/', 1)
                pinned(commit, unquote(file))
                links.append(url)
            elif not urlsplit(url).scheme and not url.startswith('#'):
                target = (p.parent / unquote(url.split('#', 1)[0])).resolve()
                check('local_link:' + str(p.relative_to(ROOT)) + ':' + url, target.exists())
        check('document_nonempty:' + str(p.relative_to(ROOT)), bool(text.strip()))
    check('dossier_12_sections', len(re.findall(r'^## \d+\.', (ROOT / DOC).read_text(), re.M)) == 12)
    for name in ['PROJECT_MAP.md', 'docs/research/研究概要・現状.md', 'docs/README.md']:
        original = pinned(BASE, name)
        check('discovery_only_prepend:' + name, (ROOT / name).read_bytes().endswith(original))
    changed = set(git('diff', '--name-only', BASE).decode().splitlines())
    changed.update(git('ls-files', '--others', '--exclude-standard').decode().splitlines())
    allowed = {'PROJECT_MAP.md', 'docs/README.md', 'docs/research/研究概要・現状.md', DOC}
    check('documentation_only_change_scope', all(p in allowed or p.startswith(REL + '/') for p in changed))
    inventory = []
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name != 'technical_checks.json':
            data = p.read_bytes()
            inventory.append({'path': p.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)})
    for name in sorted(allowed):
        data = (ROOT / name).read_bytes()
        inventory.append({'path': name, 'bytes': len(data), 'sha256': sha(data)})
    result = {'status': 'PASS_DOCUMENTATION_METADATA_ONLY', 'base_RESULT': BASE,
        'scope': 'No scientific imports, counter evaluation, sampling, compile, ratio/shot/rank recomputation',
        'canonical_row_counts': [84, 84, 360], 'row_aliases_verified': len(index),
        'pinned_files_verified': len(provenance['artifacts']),
        'source_blob_closure_checks': len(read('source_closure_checks.json')['checks']),
        'claims': len(claims), 'limitations': len(issue_ids), 'checks_passed': len(checks),
        'github_links_resolved_in_local_git': len(links), 'remote_all_files_fetched': False,
        'remote_publication_verification': 'Performed after commit; final SHA and byte verification reported separately',
        'scientific_calls_this_task': {'trajectory': 0, 'compile': 0, 'signal_reference': 0,
            'counter': 0, 'scientific_reanalysis_reaccount': 0, 'grant': 0},
        'independent_reviewer_or_GPT_Claude_scientific_review': False,
        'private_raw_or_credentials_published': False,
        'artifact_hashes_excluding_this_self_referential_receipt': inventory,
        'next_actor': 'ユーザー', 'next_action': '中心claim・論文への位置付け・表図の配置・追加証拠要否を判断',
        'stop': 'MANDATORY_STOP_AFTER_PUBLICATION'}
    if '--write' in sys.argv:
        (HERE / 'technical_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_hashes_excluding_this_self_referential_receipt'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
