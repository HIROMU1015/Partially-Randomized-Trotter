"""Read-only one-shot evidence audit. Stdlib only; no science imports or calls.

Creates separate audit files; never edits result/source/auth/contract/markers.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
RESULT = ROOT / 'artifacts/track_b_df_correction_comparison_result/2026-10-11/v2'
ART = ROOT / 'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11'
S = '41349b73157c7781e0b534cf0fe61e06bf439954'
A = 'a1f537b68f0f22419dff0202c553b7511653caa3'
BRANCH = 'track-b-df-correction-comparison-one-shot-execution-20261012'
CHECKS = []


def check(label, value):
    CHECKS.append({'check': label, 'passed': bool(value)})
    if not value:
        raise AssertionError(label)


def record(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def load_logical(path):
    index = Path(str(path) + '.review.json')
    if not index.exists():
        return json.loads(path.read_bytes()), record(path)
    d = json.loads(index.read_bytes())
    check('logical identity ' + path.name, d['logical_path'] == path.name)
    check('index size ' + path.name, index.stat().st_size <= 512 * 1024)
    h, size, parts = hashlib.sha256(), 0, []
    permitted = (path.parent / (path.name + '.chunks')).resolve()
    for i, row in enumerate(d['chunks']):
        p = (path.parent / row['path']).resolve()
        check('chunk containment ' + p.name, p.is_relative_to(permitted))
        data = p.read_bytes()
        check('chunk bytes/order ' + str(i), row['order'] == i and
              0 < len(data) == row['bytes'] <= 256 * 1024)
        data.decode('utf-8')
        check('chunk hash ' + str(i), hashlib.sha256(data).hexdigest() == row['sha256'])
        h.update(data); size += len(data); parts.append(data)
    check('logical reconstruction ' + path.name,
          size == d['original_bytes'] and h.hexdigest() == d['original_sha256'])
    return json.loads(b''.join(parts)), {
        'logical_path': str(path.relative_to(ROOT)), 'review_index': str(index.relative_to(ROOT)),
        'bytes': size, 'sha256': h.hexdigest(), 'chunks': len(parts),
        'UTF8_and_ordered_byte_reconstruction_verified': True}


def save(name, data):
    p = AUDIT / name
    with p.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    check('review size ' + name, p.stat().st_size <= 512 * 1024)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    check('audit started at execution authorization HEAD', git('rev-parse', 'HEAD') == A)
    check('execution branch', git('branch', '--show-current') == BRANCH)
    check('direct sole parent S', git('rev-list', '--parents', '-n', '1', A).split() == [A, S])
    changed = git('diff', '--name-only', S, A).splitlines()
    allowed = ['artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/authorization.json',
               'docs/tracks/algorithm_codesign/df_correction_comparison_execution_authorization_v2.md']
    check('authorization-only paths', sorted(changed) == sorted(allowed))
    manifest_path = ROOT / 'artifacts/track_b_df_correction_comparison_integrity_v3/2026-10-11/source_manifest_v3.json'
    manifest = json.loads(manifest_path.read_bytes())
    for row in manifest['files']:
        check('protected source ' + row['path'], record(ROOT / row['path']) == row)
    check('source manifest unchanged', subprocess.check_output(
        ['git', 'show', A + ':' + str(manifest_path.relative_to(ROOT))], cwd=ROOT) == manifest_path.read_bytes())
    contract = json.loads((ART / 'contract_v2.json').read_bytes())
    auth = json.loads((ART / 'authorization.json').read_bytes())
    status = json.loads((RESULT / 'terminal_status.json').read_bytes())
    outer = json.loads((AUDIT / 'outer_process_receipt.json').read_bytes())
    pre = json.loads((AUDIT / 'prelaunch_gate_receipt.json').read_bytes())
    mark = record(RESULT / 'one_shot_marker.json')
    marker = json.loads((RESULT / 'one_shot_marker.json').read_bytes())
    check('marker identity all receipts', mark['sha256'] == status['marker_SHA256'] == outer['one_shot_marker_sha256'])
    check('source/auth execution binding', status['provenance']['source'] == S and status['provenance']['authorization'] == A)
    check('prelaunch identical provenance', pre['provenance'] == status['provenance'])
    check('contract SHA', record(ART / 'contract_v2.json')['sha256'] == auth['contract_sha256'] == status['provenance']['contract_sha256'])
    check('authorization SHA', record(ART / 'authorization.json')['sha256'] == status['provenance']['authorization_sha256'])
    check('auth exact committed bytes', subprocess.check_output(['git', 'show', A + ':' + allowed[0]], cwd=ROOT) == (ART / 'authorization.json').read_bytes())
    receipt = (ROOT / auth['review_original_container']).read_bytes()
    original = receipt[auth['review_original_byte_offset']:auth['review_original_byte_offset'] + auth['review_original_bytes']]
    check('embedded GPT review original', len(original) == auth['review_original_bytes'] and hashlib.sha256(original).hexdigest() == auth['review_original_sha256'])
    check('once/retry0/STOP', marker['runs'] == auth['runs'] == 1 and
          marker['retries'] == status['retries'] == outer['retries'] == 0 and
          marker['mandatory_STOP'] and status['mandatory_STOP'] and outer['mandatory_STOP'])
    check('outer failure consistent', outer['exit_status'] == 2 and status['classification'] == 'TECHNICAL_INCONCLUSIVE')
    stdout = (AUDIT / 'runner.stdout.log').read_bytes()
    check('outer stdout identity', len(stdout) == outer['stdout_bytes'] and hashlib.sha256(stdout).hexdigest() == outer['stdout_sha256'])
    printed = json.loads(stdout)
    check('stdout/status agreement', printed['classification'] == status['classification'] and printed['reason'] == status['technical_reason'] and printed['mandatory_STOP'])
    identities, logical_data = [], {}
    for p in sorted(RESULT.glob('*.review.json')):
        logical = Path(str(p)[:-len('.review.json')])
        data, identity = load_logical(logical)
        logical_data[logical.name] = data; identities.append(identity)
    cd = logical_data['CD.json']
    check('CD completed six registered rows', len(cd['rows']) == 6 and
          {(r['arm'], r['delta']) for r in cd['rows']} ==
          {(a, d) for a in ('A', 'B') for d in ('1/8', '1/16', '1/32')})
    summary_rows = []
    for i, row in enumerate(cd['rows']):
        check('row measured total unavailable ' + str(i), row['PR_measured_total_cost'] is None)
        for k in ('RTE_executed_order_law',):
            check('saved law sum ' + str(i) + k, sum(F(p) for p in row[k].values()) == 1)
        check('saved event law sum ' + str(i), sum(F(e['p']) for e in row['executed_slot_events']) == 1)
        moment = row['exact_moment_factorization']
        exact = F(moment['Gamma_exec'])**2 * F(moment['slot_second_moment'])**moment['slot_count']
        iv = row['executed_detailed_second_moment_interval']
        check('saved moment factorization inside interval ' + str(i), F(iv['lo']) <= exact <= F(iv['hi']))
        rng = row['exact_range_factorization']
        exact_range = F(rng['Gamma_exec']) * F(rng['slot_range'])**rng['slot_count']
        iv = row['executed_detailed_range_interval']
        check('saved range factorization inside interval ' + str(i), F(iv['lo']) <= exact_range <= F(iv['hi']))
        check('saved positive shot width ' + str(i), F(row['statistical_halfwidth']) > 0 and row['shots_per_axis'] > 0)
        summary_rows.append({'original_pointer': '/rows/' + str(i),
            **{k: v for k, v in row.items() if k not in ('executed_slot_events', 'pauli_coefficients')},
            'event_count': len(row['executed_slot_events']),
            'event_pointer': '/rows/' + str(i) + '/executed_slot_events',
            'Pauli_table_pointer': '/rows/' + str(i) + '/pauli_coefficients'})
    check('CD no whole-PR winner', cd['whole_PR_resource_winner'] is None)
    e_prefix = json.loads((RESULT / 'E-r3.json.prefix.json').read_bytes())
    p = e_prefix['prefix']
    check('E-r3 only partial', e_prefix['partial_not_scientific_outcome'] and
          p['active_modes'] == 3 and p['compile_calls'] == 1 and p['native'] == [] and p['all_B_event_costs'] == [])
    elog = (RESULT / 'E-r3.log').read_text()
    check('underlying fingerprint guard recorded', 'ArithmeticError: Frozen E whole fingerprint mismatch; no synthesis' in elog)
    for start in sorted(RESULT.glob('*.started.json')):
        d = json.loads(start.read_bytes()); job = RESULT / (d['label'] + '.job.json')
        canonical = hashlib.sha256(json.dumps(json.loads(job.read_bytes()), sort_keys=True).encode()).hexdigest()
        check('started canonical parsed-job hash ' + d['label'], canonical == d['job_sha256'] and d['retries'] == 0)
        check('job marker binding ' + d['label'], json.loads(job.read_bytes())['marker_sha256'] == mark['sha256'])
    children = status['resource_receipt']['child_receipts']
    check('child exits and stop', {r['label']: r['exit_status'] for r in children} ==
          {'CD': 0, 'E-inputs': 0, 'E-r3': 1, 'E-r6': -9})
    check('E-r6 guard termination', next(r for r in children if r['label'] == 'E-r6')['guard_terminated'])
    check('no E-r6 retained prefix/result', not (RESULT / 'E-r6.json').exists() and
          not (RESULT / 'E-r6.json.review.json').exists() and not (RESULT / 'E-r6.json.prefix.json').exists())
    check('no T calls/inventory/accounting', not list(RESULT.glob('T-*')) and
          not list(RESULT.glob('*inventory*')) and not (RESULT / 'T_accounting.json').exists() and
          status['synthesis_inventory_binding'] is None)
    check('no forbidden stages recorded', status['G10_RA_D0_molecule_GPU_LP_trajectory_calls'] == 0)
    # Read-only process metadata, no GPU query or calculation.
    running = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if b'trottertracks.algorithm_codesign.df_correction_comparison.worker' in cmd and str(ROOT).encode() in cmd:
            running.append(int(proc.name))
    check('no execution coordinator remains', not Path('/proc/' + str(outer['pid'])).exists())
    check('no matching execution worker remains', running == [])
    result_files = [record(p) for p in sorted(RESULT.rglob('*')) if p.is_file()]
    check('all result files review-sized', all(r['bytes'] <= 512 * 1024 for r in result_files))
    check('result below output cap', sum(r['bytes'] for r in result_files) < contract['caps']['output_bytes'])
    save('CD_saved_summary_v1.json', {
        'kind': 'READ_ONLY_SAVED_CD_FIELD_EXTRACTION_NOT_FINAL_SCIENTIFIC_COMPARISON',
        'final_classification': status['classification'],
        'scientific_winner': None, 'interpretation': 'INCOMPLETE_RUN_DIAGNOSTIC_ONLY',
        'source': S, 'authorization': A, 'CD_identity': next(r for r in identities if r['logical_path'].endswith('/CD.json')),
        'domain': cd['domain'], 'rows': summary_rows, 'process_receipt': cd['_process_receipt'],
        'native_cost_unavailable': True, 'E_cost_not_transplanted': True,
        'rational_strings_preserved': True, 'additional_science_calls': 0})
    save('saved_output_audit_v1.json', {
        'kind': 'SAVED_OUTPUT_AND_PROVENANCE_AUDIT', 'source': S, 'authorization': A,
        'classification_unchanged': status['classification'],
        'protected_source_files_unchanged': len(manifest['files']),
        'logical_artifact_identities': identities, 'result_file_inventory': result_files,
        'result_bytes': sum(r['bytes'] for r in result_files),
        'outer_process_receipt': outer, 'frozen_terminal_receipt': status,
        'native_compile_calls': {'E_r3_saved_attempts': 1, 'E_r6_attempts': None,
            'total_attempts': None, 'reason': 'parallel E-r6 killed without prefix; exact count not recorded'},
        'T_synthesis_calls': 0, 'audit_science_matrix_compile_synthesis_solver_calls': 0,
        'retries': 0, 'mandatory_STOP': True, 'active_execution_workers': running,
        'checks': CHECKS.copy(), 'all_checks_passed': True,
        'limitations': ['No failed circuit fingerprint/IR saved; mismatch cause cannot be localized from this record.',
            'No E-r6 retained prefix; in-flight compile count is unknown.',
            'RSS is a sampled aggregate, not a zero-overshoot ceiling.',
            'CD completed rows are an incomplete-run diagnostic, not final A/B evidence.']})
    # Ensure results were not touched by this audit.
    check('result bytes unchanged through audit', result_files ==
          [record(p) for p in sorted(RESULT.rglob('*')) if p.is_file()])
    print(json.dumps({'all_checks_passed': True, 'check_count': len(CHECKS),
        'protected_files': len(manifest['files']), 'result_files': len(result_files),
        'result_bytes': sum(r['bytes'] for r in result_files), 'mandatory_STOP': True}))


if __name__ == '__main__':
    main()
