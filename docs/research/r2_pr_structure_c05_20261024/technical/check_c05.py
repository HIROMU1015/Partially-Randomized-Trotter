#!/usr/bin/env python3
"""C0.5 documentary checks and reused abstract fixtures; no physical kernels."""
import ast
import contextlib
import csv
from fractions import Fraction as F
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parents[2]
BASE = '8cb41c40aec9c31c69a5a50e8e5fe93aa6d2eb88'
PREFIX = ROOT.relative_to(WORKTREE).as_posix() + '/'
checks = []

# Same stdlib helpers used in C0; extract only the functions, never its main.
old = subprocess.check_output(['git', '-C', str(WORKTREE), 'show',
    '501b4232b3e9ef9778e917f0a86f1d6c1d05f189:docs/research/prior_art_novelty_20261021/audit/check_documents.py'])
helpers = ast.Module(body=[n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)
                         and n.name in ('git', 'digest', 'check')], type_ignores=[])
exec(compile(helpers, '<reused-documentary-helpers>', 'exec'), globals())


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    catalog = load('source_catalog.json')
    def raw(key):
        r = catalog[key]
        return git('show', r['commit'] + ':' + r['path'])

    paths = [p for p in git('diff', BASE, '--name-only', '-z').decode().split('\0') if p]
    paths += [p for p in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0') if p]
    old_paths = set(git('ls-tree', '-r', '--name-only', BASE).decode().splitlines())
    check('new_namespace_only', bool(paths) and all(p.startswith(PREFIX) for p in paths),
          'No first-study, C0, Track B, original science/STOP or other worktree edits')
    check('no_existing_file_rewritten', not set(paths) & old_paths, sorted(set(paths) & old_paths))

    bad = []
    for key, ref in catalog.items():
        b = raw(key)
        if digest(b) != ref['sha256']:
            bad.append(key)
        if ref.get('symbols'):
            nodes = {n.name: n for n in ast.walk(ast.parse(b)) if isinstance(n, ast.FunctionDef)}
            if any(n not in nodes or nodes[n].lineno != v['line'] for n, v in ref['symbols'].items()):
                bad.append(key + ':symbol')
    check('fixed_references_and_existing_symbols', not bad, {'source_count': len(catalog), 'errors': bad})
    with (ROOT / 'algorithmic_evidence_crosswalk.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    by_id = {r['series_id']: r for r in rows}
    check('crosswalk_22_distinct_series_complete', len(rows) == len(by_id) == 22 and all(all(r.values()) for r in rows),
          'Question/structure/math/cost/info/result/status/delta/reuse/scope/source retained')
    check('crosswalk_fixed_refs_resolve', all(set(r['evidence_refs'].split(';')) <= set(catalog)
          and set(r['source_refs'].split(';')) <= set(catalog) for r in rows), 'No unsupported path/name guesses')

    state = load('scope_and_state.json'); stop = load('STOP.json')
    check('five_branch_observations', len(state['remote_heads_observed']) == 5
          and state['observation_only_other_branch_changes_do_not_block'] is True,
          'Observations, not a freeze of unrelated worktree HEADs/status')
    check('P1_held_not_adopted', state['P1_status'] == 'HOLD_BY_CURRENT_USER_INSTRUCTION'
          and stop['P1'] == 'HOLD_NOT_ADOPTED_NOT_EXECUTED', 'Current C0.5 instruction supersedes suggestion only')
    p1 = json.loads(raw('C0_P1'))
    local_p1 = WORKTREE / catalog['C0_P1']['path']
    check('C0_draft_and_STOP_preserved', local_p1.read_bytes() == raw('C0_P1')
          and not p1['execution_authorized'] and p1['grant'] is None
          and (WORKTREE / catalog['C0_STOP']['path']).read_bytes() == raw('C0_STOP'), 'Old draft/STOP unchanged')
    check('proposal_B_not_formal_decision_or_science', state['proposed_option'] == stop['proposed_option'] == 'B'
          and state['formal_research_decision'] is None and stop['formal_decision'] is None
          and state['numerical_validation_contract'] is None and not stop['next_scientific_execution_authorized'],
          'No speculative contract or automatic P1/C1 restart')

    marker = json.loads(raw('M2_MARKER'))
    check('M2_completed_not_old_implementation_pending', marker['status'] == 'TRANSFER_SUPPORTED'
          and marker['mandatory_stop_reached'] and not marker['next_stage_authorized'], 'Old pre-execution labels remain history')
    check('C1_latest_pause_separate_from_G6', 'C1_PAUSED_AFTER_EXPLORATORY_RESOURCE_ASSESSMENT' in raw('C1_PAUSE').decode()
          and by_id['C1']['current_state'] == 'EXPLORATORY_COMPLETED_PAUSED', 'G6 retained; no H4 full-native claim')
    check('parent_S4_closure_overrides_v3_pending', 'D＋E' in raw('PA_CLOSE').decode()
          and by_id['PARENT_S4']['current_state'] == 'LIMITED_POSITIVE_CURRENT_SEARCH_CLOSED', 'Corrected S4 vs old raw-C proxy retained')
    check('BF_negative_does_not_rewrite_original_failure', 'BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY' in raw('BF_CLOSE').decode()
          and by_id['BF']['current_state'] == 'NEGATIVE_CLOSED_WITH_ORIGINAL_TECH_STOP', 'Result and policy layers distinguished')
    check('Gaussian_unknown_not_performance_negative', by_id['GAUSSIAN']['current_state'] == 'METHOD_DELTA_ABSENT_PERFORMANCE_UNKNOWN',
          'No gain/loss was measured')
    check('counterexample_and_quarantine_retained', 'DISPROVED' in raw('PU').decode()
          and by_id['G6_PU']['current_state'] == 'UNIVERSAL_CLAIM_DISPROVED'
          and by_id['OLD_SCREENING']['current_state'] == 'QUARANTINED_NOT_REUSABLE', 'No rescue or reused training labels')
    check('PR2_primary_retention_and_frontier_loss_distinct', 'primary RZ最小を落とした' not in by_id['PR2_M1_SELECTOR']['verified_result']
          and '1/2' in by_id['PR2_M1_SELECTOR']['verified_result']
          and 'primary RZ点最小を保持' in by_id['PR2_M1_SELECTOR']['verified_result'], 'Quote original PM0, no new regret computed')

    # Re-run the already existing abstract BM fixture exactly as written. No
    # molecular matrices, chemistry input, native compilation, or result writing.
    bm = raw('BM_FIXTURE')
    tree = ast.parse(bm)
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    check('reused_BM_fixture_stdlib_only', imports <= {'fractions', 'hashlib', 'json', 'math'}, sorted(imports))
    ns = {'__name__': 'r2_c05_abstract_fixture'}
    exec(compile(bm, '<fixed-BM-abstract-word-fixture>', 'exec'), ns)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        ns['main']()
    bm_result = json.loads(captured.getvalue())
    check('reused_nine_abstract_word_fixtures', bm_result['fixture_count'] == 9
          and all(r['all_exact_residuals_zero'] for r in bm_result['fixtures']),
          'Literal/compact/BM degree3 equality, not a new theorem or physical validation')
    check('reused_six_wrong_factor_mutations_detected', bm_result['factor_mutation_controls_passed'] == 6,
          'Existing 1/(4m²) versus 1/m² sensitivity check')

    # Reuse only frozen pure allocation, with synthetic coefficients unrelated
    # to any molecular record or standard Yoshida formula.
    allocation_source = ast.parse(raw('ALLOCATION'))
    fn = next(n for n in allocation_source.body if isinstance(n, ast.FunctionDef) and n.name == 'allocation')
    alloc_ns = {'math': math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<fixed-pure-allocation>', 'exec'), alloc_ns)
    allocate = alloc_ns['allocation']
    t = F(7, 11); lam = F(5, 13); R = 12
    microsteps = [lam * t / (q * (R // q)) for q in (1, 2, 4)]
    check('synthetic_S2_same_R_law_key', len(set(microsteps)) == 1,
          'Exact scalar key only; no claim of equal signal or compiled price')
    coefficients = (2, -3, 2)
    a = allocate(coefficients, 12); b = allocate(coefficients, 6)
    check('synthetic_integer_allocation_breaks_proportionality', a != tuple(2*x for x in b)
          and sum(a) == 12 and sum(b) == 6, {'synthetic_coefficients': coefficients, 'q1': a, 'q2': b})
    check('synthetic_cutoff_and_sign_keys_stay_distinct', (microsteps[0], 2) != (microsteps[0], 4)
          and microsteps[0] != -microsteps[0], 'Law/phase inputs cannot be discarded by an R-only cache key')
    try:
        allocate(coefficients, 2)
    except ValueError as exc:
        check('synthetic_minimum_piece_allocation_guard', str(exc) == 'INSUFFICIENT_OCCURRENCE_STEPS', str(exc))
    else:
        check('synthetic_minimum_piece_allocation_guard', False, 'Missing rejection')

    check('no_real_numerical_or_scientific_execution', all(state[k] == 0 for k in (
          'science_module_imports', 'new_scientific_calls', 'new_saved_value_numerical_analysis', 'new_scientific_review', 'new_grants')),
          'Only the explicitly synthetic abstract fixtures and metadata were evaluated')
    check('policy_version_gap_is_recorded', not state['formal_policy']['expected_original_obtained']
          and state['formal_policy']['available_sha256'] != state['formal_policy']['expected_sha256'],
          'Exact formal original not claimed as read')

    errors = []
    for p in ROOT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        s = p.read_text()
        if re.search(r'/h[o]me/|/Us[e]rs/|/t[m]p/|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY', s):
            errors.append(p.relative_to(ROOT).as_posix())
    check('new_public_files_no_private_locator_or_credential_patterns', not errors, errors)
    broken = []
    for p in ROOT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if target.startswith(('http:', 'https:')) or target == 'technical_checks.json':
                continue
            if not (p.parent / target.split('#')[0]).exists():
                broken.append(target)
    check('local_links_resolve', not broken, broken)
    result = dict(schema='R2_CS_C05_TECHNICAL_CHECKS_V1',status='PASS' if all(r['pass'] for r in checks) else 'FAIL',
        check_count=len(checks),checks=checks,source_count=len(catalog),crosswalk_series=len(rows),
        reused_abstract_BM_fixtures=9,reused_factor_mutations=6,new_scientific_calls=0,new_real_numerical_analysis=0,
        new_scientific_reviews=0,author_check_not_independent_review=True)
    (ROOT / 'technical_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('status','check_count','source_count','crosswalk_series','new_scientific_calls','new_real_numerical_analysis')}))
    for r in checks:
        if not r['pass']:
            print(json.dumps(r, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
