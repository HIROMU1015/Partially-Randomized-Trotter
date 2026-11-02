"""Read-only scientific data checks and documentary receipts; no numerical analysis.

--record writes only the new documentary audit/status/SHA manifests.
--verify-committed checks the committed packet and exits without writing anything.
The scientific SOURCE, outputs, consumed grant and STOP are never changed here.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21'
BASE = '42ed0876c0505502428629c1fb4863233247ebaf'
SOURCE = 'e16c3be8a039f6bc468e112f0ed086bde544fef2'
PREEXEC = 'b1dcc900'
INDEXES = ['PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'docs/README.md',
           'docs/research/研究概要・現状.md', 'scripts/README.md',
           'src/trotterlib/README.md', 'docs/research/研究ノート/README.md']
MANIFEST = OUT / 'SHA256_manifest.json'
STRATA = ['S0_ONLY', 'S2_SINGLE', 'S2_MULTI', 'S4_PRESENT']


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def load(path):
    return json.loads(path.read_text())


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def changed_paths():
    tracked = git('diff', '--name-only', '-z', BASE).split(b'\0')
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0')
    return sorted({x.decode() for x in tracked + untracked if x})


def copied(value):
    return '' if value is None else str(value)


def table(name):
    with (OUT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def checks():
    contract = load(OUT / 'contract.json')
    result = load(OUT / 'result.json')
    assert contract['SOURCE'] == result['SOURCE'] == SOURCE
    assert result['contract_sha256'] == sha(OUT / 'contract.json')
    assert result['input_sha256'] == contract['input_sha256']
    for rel, digest in contract['source_hashes'].items():
        assert sha(ROOT / rel) == digest == sha_bytes(git('show', SOURCE + ':' + rel))
    fixed = 'docs/research/pr_rare_rte_stage_b_contract_20261021.md'
    assert (ROOT / fixed).read_bytes() == git('show', SOURCE + ':' + fixed)
    # Pre-execution contract/approval bytes remain as committed before the run.
    for name in ['contract.json', 'execution_approval_receipt.json',
                 'grant_issue_receipt.json', 'preflight_pregrant.json', 'remote_acquisition.json']:
        path = OUT / name
        assert path.read_bytes() == git('show', PREEXEC + ':' + str(path.relative_to(ROOT)))
    approval = load(OUT / 'execution_approval_receipt.json')
    issued = load(OUT / 'grant_issue_receipt.json')
    consumed = load(OUT / 'grant_consumption_receipt.json')
    assert issued['approval_receipt_sha256'] == sha(OUT / 'execution_approval_receipt.json')
    assert issued['grant_sha256'] == consumed['grant_sha256']
    assert consumed['contract_sha256'] == sha(OUT / 'contract.json')
    assert consumed['consumed'] and consumed['nonrefundable']
    assert approval['explicit_stage_B_one_shot_authorized'] and not approval['push_authorized']
    private = ROOT / '.stage_b_private'
    assert sha(private / 'grant_APPROVED_PRIVATE.json') == consumed['grant_sha256']
    assert sha(private / 'inputs.json') == contract['input_sha256']
    assert sha(private / 'original_reads.json') == contract['original_inventory_sha256']
    assert load(private / 'ONE_SHOT.json') == consumed
    assert not git('ls-files', '--', '.stage_b_private').strip()
    assert git('check-ignore', '.stage_b_private/inputs.json').strip()
    assert not (private / 'scientific_driver_stderr.txt').read_bytes()
    calls = load(OUT / 'calls.json')
    assert calls['used'] == calls['budget'] == contract['budget']
    assert load(private / 'analysis_ledger.json') == calls
    native = load(OUT / 'native_exit.json')
    assert native['native_exit_code'] == 0 and native['same_parent_wait4'] and native['reaped']
    assert native['remaining_scientific_workers'] == 0
    pid = native['worker_pid']
    if (Path('/proc') / str(pid) / 'cmdline').exists():
        cmdline = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
        assert b'run_rare_rte_stage_b.py' not in cmdline, 'Original worker still exists'
    assert load(OUT / 'driver_exit_receipt.json')['native_driver_exit_code'] == 0
    completed = load(OUT / 'completion_audit.json')
    assert completed['status'] == 'PASS_STAGE_B_COMPLETE_MANDATORY_STOP'
    assert completed['old_evidence']['files'] == 1895
    assert completed['old_evidence']['status'] == 'PASS_ALL_USED_ORIGINAL_BYTES'
    assert completed['source_input_runtime_final_gate']
    assert completed['scientific_passes'] == 1 and not completed['certificate']
    stop = load(OUT / 'STOP.json')
    assert stop['status'] == 'MANDATORY_STOP_STAGE_B_COMPLETE'
    assert all(v is False for k, v in stop.items() if k != 'status')
    conditions = result['conditions']
    assert len(conditions) == 8
    assert [c['candidate'] for c in conditions] == contract['candidates']
    assert [c['n'] for c in conditions] == [80, 12, 1108, 78, 48, 104, 48, 48]
    assert len({c['condition'] for c in conditions}) == 8
    probability, groups, sensitivity, tasks = [], [], [], []
    for c in conditions:
        assert [g['stratum'] for g in c['groups']] == STRATA
        for g in c['groups']:
            st = g['G']
            probability.append(dict(condition=c['condition'], candidate=c['candidate'],
                                    stratum=g['stratum'], probability=g['probability'],
                                    support_status=g['support_status']))
            groups.append(dict(condition=c['condition'], candidate=c['candidate'],
                               stratum=g['stratum'], n=copied(st['n']),
                               mean_G_exact=copied(st['mean_exact']), SE_G=copied(st['SE']),
                               sample_min_G=copied(st['sample_min']), sample_max_G=copied(st['sample_max']),
                               unknown_population_mean_G=''))
            assert g['conditional_population_mean_G'] is None
            if st['n'] == 0:
                assert all(st[k] is None for k in ['mean_exact', 'SE', 'sample_variance_exact', 'sample_min', 'sample_max'])
            if st['n'] == 1:
                assert st['SE'] is None and st['sample_variance_exact'] is None
            if c['K'] == 2 and g['stratum'] == 'S4_PRESENT':
                assert g['support_status'] == 'OUTSIDE_K2_SUPPORT'
                assert re.fullmatch(r'0(?:\.0*)?(?:[eE][+-]?\d+)?', g['probability'])
            else:
                assert g['support_status'] == 'IN_SUPPORT' and not re.fullmatch(r'0(?:\.0*)?(?:[eE][+-]?\d+)?', g['probability'])
        s = c['conditional_expectation']
        assert not s['zero_filled'] and s['known_means_are_empirical']
        assert all(s[k] is None for k in ['unconditional_population_mean_G',
                                         'mathematically_guaranteed_cost_upper_bound', 'confidence_interval'])
        for u in s['unknown_terms']:
            assert u['mean_G'] is None and u['variance_G'] is None
        sensitivity.append(dict(condition=c['condition'], candidate=c['candidate'],
                                finite_sample_mean_G=c['saved_finite_sample']['mean_exact'],
                                PF_G=c['fixed_PF_G_exact'], observed_plugin_G=s['observed_plugin_G'],
                                unknown_mass=s['unknown_mass'], PF_minus_observed_plugin=s['PF_minus_observed_plugin'],
                                unknown_mixture_mean_threshold_G=s['unknown_mixture_mean_threshold_G'],
                                threshold_over_observed_S0_mean=s['threshold_over_observed_S0_mean'],
                                conditional_direction=s['conditional_direction'], population_guarantee='False'))
        assert len(c['tasks']) == c['n']
        tasks.extend({k: copied(v) for k, v in t.items()} for t in c['tasks'])
    expected = {'probabilities.csv': probability, 'observed_strata.csv': groups,
                'sensitivity.csv': sensitivity, 'trajectory_costs.csv': tasks}
    for name, rows in expected.items():
        assert table(name) == rows, name + ': CSV/JSON transcription differs'
    assert len(tasks) == 1526
    assert len({(t['candidate'], t['replica']) for t in tasks}) == 1526
    independent = load(OUT / 'independent_numeric_audit.json')
    assert len(independent['rows']) == 8 and not independent['independent_AI_reviewer']
    assert [r['condition'] for r in independent['rows']] == [c['condition'] for c in conditions]
    assert all(r['status'] == 'PASS_INDEPENDENT_ARITHMETIC_NOT_POPULATION_CERTIFICATE'
               and r['mean_SE_and_threshold_match'] for r in independent['rows'])
    svg = ElementTree.parse(OUT / 'stratum_probabilities.svg')
    assert svg.getroot().tag == '{http://www.w3.org/2000/svg}svg'
    svg_text = (OUT / 'stratum_probabilities.svg').read_text()
    assert svg_text.count('support-out') == 3
    for c in conditions:
        assert f"{c['system']} eps={c['epsilon']}" in svg_text
    # Old tracked files are immutable except the seven append-only index updates.
    changed_old = git('diff', '--name-only', '--diff-filter=MDRT', '-z', BASE).split(b'\0')
    for rel_bytes in changed_old:
        if not rel_bytes:
            continue
        rel = rel_bytes.decode()
        assert rel in INDEXES, 'Old non-index file modified: ' + rel
        assert (ROOT / rel).read_bytes().endswith(git('show', BASE + ':' + rel))
    candidates = changed_paths()
    assert all(not p.startswith('.stage_b_private/') for p in candidates)
    for rel in candidates:
        if rel in INDEXES:
            continue
        assert (rel.startswith(str(OUT.relative_to(ROOT)) + '/') or
                rel.startswith('docs/research/pr_rare_rte_stage_b_') or
                rel == 'docs/research/研究ノート/2026-10-21_pr_rare_rte_stage_b.md' or
                rel in contract['source_hashes'] or
                rel == 'scripts/audit_rare_rte_stage_b_completion.py'), rel
    # Scan the complete new files and only prepended bytes of inherited indexes.
    prohibited = [re.compile(re.escape('/' + x + '/')) for x in ['home', 'tmp', 'root', 'Users']]
    prohibited += [re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),
                   re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
                   re.compile(r'AKIA[0-9A-Z]{16}'),
                   re.compile('-----BEGIN ' + r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                   re.compile(r'"(?:seed|rng_seed|trajectory_seed|raw_events|private_path)"\s*:')]
    link_count = 0
    for rel in candidates:
        data = (ROOT / rel).read_bytes()
        if rel in INDEXES:
            old = git('show', BASE + ':' + rel)
            data = data[:-len(old)]
        text = data.decode()
        assert not any(pattern.search(text) for pattern in prohibited), 'Private pattern in ' + rel
        if rel.endswith('.json'):
            load(ROOT / rel)
        if rel.endswith('.md'):
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' in target or target.startswith('#'):
                    continue
                path = target.split('#')[0]
                resolved = ((ROOT / rel).parent / path).resolve()
                pending = {OUT / name for name in ['documentation_audit.json', 'status_manifest.json', 'SHA256_manifest.json']}
                assert resolved.exists() or (options.record and resolved in pending), rel + ': missing link ' + target
                link_count += 1
    return dict(status='PASS_DOCUMENTARY_STRUCTURE_PRESERVATION_NOT_SCIENTIFIC_CERTIFICATE',
                SOURCE=SOURCE, predecessor=BASE,
                source_files_unchanged_from_sealed_commit=len(contract['source_hashes']),
                fixed_contract_document_unchanged=True, preexecution_receipts_unchanged=True,
                CSV_rows={name: len(rows) for name, rows in expected.items()},
                CSV_JSON_transcription_exact=True, unknown_values_remain_null=True,
                old_index_previous_bytes_retained_as_suffix=INDEXES,
                old_scientific_files_changed_in_branch=0,
                original_byte_preservation_receipt_sha256=sha(OUT / 'completion_audit.json'),
                original_1895_byte_checks_performed_in_scientific_run=True,
                old_original_files_rehashed_again_by_documentary_checker=False,
                independent_numeric_receipt_sha256=sha(OUT / 'independent_numeric_audit.json'),
                result_sha256=sha(OUT / 'result.json'),
                grant_marker_consumed_once_and_unchanged=True, scientific_retry_or_resume=False,
                native_exit_code=0, worker_reaped=True, original_worker_absent=True,
                driver_tool_exit_code=0, private_files_ignored_and_untracked=True,
                scoped_private_pattern_hits=0, scanner_is_exhaustive_security_guarantee=False,
                local_markdown_links_checked=link_count, SVG_XML_and_condition_coverage=True,
                new_scientific_calls_by_documentary_checker=0,
                documentary_checker_initial_support_label_and_zero_string_expectations_corrected=True,
                scientific_outputs_unchanged_by_checker=True, push=False,
                detailed_scientific_review=False, final_evidence_freeze=False,
                next_stage_authorized=False, mandatory_STOP=True)


def record():
    audit = checks()
    save(OUT / 'documentation_audit.json', audit)
    save(OUT / 'status_manifest.json', dict(
        schema='RARE_RTE_STAGE_B_EVIDENCE_STATUS_V1', date_JST='2026-10-21',
        status='COMPLETE_STAGE_B_CONDITIONAL_NOT_POPULATION_CERTIFICATE', SOURCE=SOURCE,
        result='result.json', report='../../../../docs/research/pr_rare_rte_stage_b_results_20261021.md',
        result_sha256=sha(OUT / 'result.json'), independent_numeric_audit='independent_numeric_audit.json',
        scientific_completion_audit='completion_audit.json', documentary_audit='documentation_audit.json',
        conditions=8, one_shot_scientific_passes=1, unconditionally_certified_population_rankings=0,
        new_mathematical_cost_bound_adopted=False, next_stage_authorized=False,
        existing_validation_manifest_modified=False, new_GitHub_push_authorized=False,
        mandatory_STOP=True))
    paths = [p for p in changed_paths() if ROOT / p != MANIFEST]
    save(MANIFEST, dict(schema='RARE_RTE_STAGE_B_SOURCE_RESULT_SHA256_V1',
                        SOURCE=SOURCE, predecessor=BASE, self_excluded=True,
                        includes='All Stage B added/modified public-candidate files; no private raw, locators or grant body',
                        files=[dict(path=p, bytes=(ROOT / p).stat().st_size,
                                    sha256=sha(ROOT / p)) for p in paths]))
    print('PASS_DOCUMENTARY_AUDIT_NO_NEW_SCIENTIFIC_CALLS')
    print('manifest_files', len(paths))


def verify_committed():
    assert not git('status', '--porcelain').strip(), 'Worktree not clean'
    checks()
    manifest = load(MANIFEST)
    assert manifest['SOURCE'] == SOURCE and manifest['self_excluded']
    expected = set(changed_paths()) - {str(MANIFEST.relative_to(ROOT))}
    assert {r['path'] for r in manifest['files']} == expected
    for row in manifest['files']:
        path = ROOT / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        assert git('show', 'HEAD:' + row['path']) == path.read_bytes()
    print('PASS_COMMITTED_STAGE_B_SOURCE_RESULTS_AND_DOCUMENTATION_SHA')
    print('commit', git('rev-parse', 'HEAD').decode().strip())
    print('MANDATORY_STOP_NO_C_D_RETRY_COMPILE_REVIEW_OR_PUSH')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record', action='store_true')
    mode.add_argument('--verify-committed', action='store_true')
    options = parser.parse_args()
    if options.record:
        record()
    else:
        verify_committed()
