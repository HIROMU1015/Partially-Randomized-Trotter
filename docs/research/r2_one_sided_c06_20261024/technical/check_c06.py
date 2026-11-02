#!/usr/bin/env python3
"""Documentary checks plus scalar synthetic controls; no selection/science runner."""
import ast
import csv
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parents[2]
BASE = '25cbd26d27057bd3d5af9c18d1c4b7be13d9cedf'
PREFIX = ROOT.relative_to(WORKTREE).as_posix() + '/'
checks = []


def load(name):
    return json.loads((ROOT / name).read_text())


CATALOG = load('source_catalog.json')
helper = CATALOG['DOC_HELPERS']
old = subprocess.check_output(['git', '-C', str(WORKTREE), 'show',
                               helper['commit'] + ':' + helper['path']])
# Reuse the same stdlib helpers as C0/C0.5, without running the old main.
nodes = [n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)
         and n.name in ('git', 'digest', 'check')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<fixed-document-helpers>', 'exec'), globals())


def raw(key):
    r = CATALOG[key]
    return git('show', r['commit'] + ':' + r['path'])


def main():
    check('dedicated_branch', git('branch', '--show-current').decode().strip()
          == 'codex/r2-one-sided-c06-20261024', 'Only the dedicated C0.6 worktree')
    check('fixed_base_ancestor', subprocess.run(['git', '-C', str(WORKTREE),
          'merge-base', '--is-ancestor', BASE, 'HEAD'], check=False).returncode == 0, BASE)
    paths = [p for p in git('diff', BASE, '--name-only', '-z').decode().split('\0') if p]
    paths += [p for p in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0') if p]
    old_paths = set(git('ls-tree', '-r', '--name-only', BASE).decode().splitlines())
    check('only_new_C06_namespace', bool(paths) and all(p.startswith(PREFIX) for p in paths),
          'No first-study/index, old SOURCE/results/grant/ledger/marker/STOP changes')
    check('no_existing_tracked_file_rewritten', not old_paths.intersection(paths),
          sorted(old_paths.intersection(paths)))
    bad = []
    for key, r in CATALOG.items():
        b = raw(key)
        if digest(b) != r['sha256']:
            bad.append(key)
        if r.get('symbols'):
            functions = {n.name: n for n in ast.walk(ast.parse(b))
                         if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            if any(n not in functions or functions[n].lineno != v['line']
                   for n, v in r['symbols'].items()):
                bad.append(key + ':symbol')
    check('fixed_source_SHA_and_AST_symbols', not bad, {'sources': len(CATALOG), 'errors': bad})
    state, stop = load('scope_and_state.json'), load('STOP.json')
    check('three_candidates_no_new_method_claim', state['candidate_count'] == 3
          and {p['id'] for p in state['propositions']} == {'P-A', 'P-B', 'P-C'}
          and all(not p['independent_method_delta'] for p in state['propositions']),
          'Conditional existing results, not newly adopted theorems')
    check('C_is_recommendation_not_formal_closure', state['proposed_option'] == stop['proposed_option'] == 'C'
          and state['formal_research_decision'] is None and stop['formal_decision'] is None
          and stop['R2_CS_closure'] == 'RECOMMENDATION_ONLY_NOT_FORMALLY_ADOPTED',
          'GPT/user retains the scientific decision')
    check('no_repeat_pilot_survey_or_plan', not state['repeat_same_survey_or_pilot_recommended']
          and state['scientific_validation_plan'] is None and state['numerical_validation_contract'] is None
          and state['new_scientific_source'] is None and not stop['next_execution_authorized'],
          'No new science implementation, contract or automatic next task')
    preserved = ['C0_STOP', 'C0_P1', 'C05_STOP', 'R3']
    check('old_STOP_P1_R3_bytes_preserved', all((WORKTREE / CATALOG[k]['path']).read_bytes() == raw(k)
          for k in preserved) and stop['P1'] == 'HOLD_NOT_ADOPTED_NOT_EXECUTED', preserved)
    check('C1_pause_and_G6_remain_separate', 'C1_PAUSED_AFTER_EXPLORATORY_RESOURCE_ASSESSMENT'
          in raw('C1_PAUSE').decode() and state['G6'] == 'RETAINED_SEPARATE_RESEARCH'
          and not stop['other_research_reopened'], 'No closure of other research or C1 restart')
    check('PU_and_sufficient_policy_scope_retained', 'DISPROVED' in raw('PU').decode()
          and '必要最小trial数' in raw('W2B_BUDGET').decode()
          and 'U.loをB_Fの下界としない' in raw('W2B_BUDGET').decode(),
          'Existing PU counterexample; budget upper/lower meanings unchanged')
    with (ROOT / 'information_access_matrix.csv').open() as f:
        rows = list(csv.DictReader(f))
    by_id = {r['information_id']: r for r in rows}
    check('access_matrix_18_complete_distinct_entries', len(rows) == len(by_id) == 18
          and all(all(r.values()) for r in rows), 'Access categories are not an automatic cost order')
    check('oracle_not_cheap_and_all_acquisition_costs_present', by_id['O_ORACLE']['available_before_signal'] == 'NO'
          and by_id['S_SIGNAL']['available_before_signal'] == 'NO_AS_CHEAP_FEATURE'
          and all(r['computational_cost'] for r in rows), 'Includes DF, local/full support, native and setup')

    # Scalar fixtures only: controls for invalid certificate inferences. These
    # are not a new certificate engine and never consume molecular records.
    f_actual, f_upper, c_actual, c_upper = F(4), F(5), F(1), F(6)
    check('upper_vs_upper_can_reverse_actual_order', f_upper < c_upper
          and f_actual > c_actual and f_actual <= f_upper and c_actual <= c_upper,
          'An upper-bound ordering alone cannot exclude a candidate')
    check('strict_separation_and_overlap_distinct', F(5) > F(4)
          and not F(4) > F(4) and max(F(2), F(3)) <= min(F(5), F(6)),
          'Strict exclusion requires a strict gap; overlap is not tie')
    epsilon, bias_upper, possible_bias = F(1, 2), F(3, 4), F(0)
    check('failed_bias_upper_is_not_infeasibility_proof', possible_bias <= bias_upper
          and bias_upper >= epsilon and possible_bias < epsilon,
          'Certification failure retains UNDETERMINED')
    laws = [((F(1, 2), F(-1)), (F(1, 2), F(1))), ((F(1), F(1)),)]
    means = [sum(p*x for p, x in law) for law in laws]
    seconds = [sum(p*x*x for p, x in law) for law in laws]
    check('range_and_moment_do_not_determine_mean', seconds == [1, 1]
          and means == [0, 1] and all(abs(x) <= 1 for law in laws for _, x in law),
          'Bounded scalar variables, not a molecular or quantum simulation')

    # Reuse ONLY exact scalar interval/dyadic code from the frozen G6 SOURCE.
    allowed = {'Interval', 'root_interval', 'dyadic_distribution'}
    scalar_nodes = [n for n in ast.parse(raw('G6_CORE')).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in allowed]
    ns = {'__name__': __name__, 'F': F, 'dataclass': dataclass, 'isqrt': math.isqrt}
    exec(compile(ast.Module(body=scalar_nodes, type_ignores=[]), '<fixed-G6-scalar-fixture>', 'exec'), ns)
    check('only_scalar_G6_functions_reused', {n.name for n in scalar_nodes} == allowed,
          'No ReturnKernel/P5/event generator/budget/provider import or execution')
    Interval = ns['Interval']
    enclosure = ns['root_interval'](F(2), 8)
    check('reused_outward_root_on_synthetic_scalar', enclosure.lo**2 <= 2 <= enclosure.hi**2,
          'Square root of abstract scalar 2, no physical input')
    target = (F(1, 3), F(2, 3))
    law = ns['dyadic_distribution'](tuple(Interval(v, v) for v in target), 4, F(1, 4))
    q = tuple(v/2 for v in law)
    weights = tuple(a/p for a, p in zip(target, q))
    total_mean = sum(p*w for p, w in zip(q, weights))
    check('actual_dyadic_proposal_restores_synthetic_target', law != target
          and total_mean == sum(target) == 1 and all(p*w == a for p, w, a in zip(q, weights, target)),
          'Rational correction uses actual proposal, no random draws')
    check('accepted_only_mean_changes_denominator', sum(q) == F(1, 2)
          and total_mean/sum(q) == 2 and total_mean != total_mean/sum(q),
          'Zero trial remains in the full-trial denominator')
    try:
        ns['dyadic_distribution']((Interval(F(0), F(0)), Interval(F(1), F(1))), 4, F(1, 4))
    except ValueError as exc:
        rejected = str(exc) == 'positive support required'
    else:
        rejected = False
    check('synthetic_bad_support_rejected', rejected, 'No silent loss of positive-support assumptions')

    alloc_node = next(n for n in ast.parse(raw('ALLOCATION')).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'allocation')
    alloc_scope = {'math': math}
    exec(compile(ast.Module(body=[alloc_node], type_ignores=[]), '<fixed-pure-allocation>', 'exec'), alloc_scope)
    allocate = alloc_scope['allocation']
    a, b = allocate((2, -3, 2), 12), allocate((2, -3, 2), 6)
    check('reused_C05_artificial_integer_allocation_control', a == (4, 5, 3) and b == (2, 2, 2)
          and a != tuple(2*x for x in b), 'Same prior synthetic fixture, no real PF coefficients')
    microsteps = {F(5, 13)*F(7, 11)/(q*(12//q)) for q in (1, 2, 4)}
    check('synthetic_same_R_key_does_not_drop_sign_or_cutoff', len(microsteps) == 1
          and next(iter(microsteps)) != -next(iter(microsteps))
          and (next(iter(microsteps)), 2) != (next(iter(microsteps)), 4),
          'Scalar law reuse only, no signal or compiled-price equality')
    try:
        allocate((2, -3, 2), 2)
    except ValueError as exc:
        rejected = str(exc) == 'INSUFFICIENT_OCCURRENCE_STEPS'
    else:
        rejected = False
    check('minimum_occurrence_guard_preserved', rejected, 'Original pure allocation rejects too few steps')

    zero_keys = ['new_scientific_calls', 'new_real_numerical_reanalysis', 'new_sampling',
                 'new_compile', 'new_simulation', 'new_scientific_module_imports',
                 'new_grants', 'new_detailed_scientific_reviews']
    check('all_unauthorized_scientific_work_zero', all(state[k] == 0 for k in zero_keys), zero_keys)
    check('policy_original_unobtained_recorded', not state['formal_policy']['expected_original_obtained']
          and state['formal_policy']['available_sha256'] != state['formal_policy']['expected_sha256'],
          'No claim to have obtained the exact formal original')
    private, broken = [], []
    for p in ROOT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        s = p.read_text()
        if re.search(r'/h[o]me/|/Us[e]rs/|/t[m]p/|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY', s):
            private.append(p.relative_to(ROOT).as_posix())
        if p.suffix == '.md':
            for link in re.findall(r'\]\(([^)]+)\)', s):
                if link.startswith(('https:', 'http:')) or link == 'technical_checks.json':
                    continue
                if not (p.parent/link.split('#')[0]).exists():
                    broken.append(link)
    check('no_private_locator_or_credential_patterns', not private, private)
    check('local_document_links_resolve', not broken, broken)
    result = dict(schema='R2_CS_C06_TECHNICAL_CHECKS_V1', status='PASS' if all(x['pass'] for x in checks) else 'FAIL',
                  check_count=len(checks), checks=checks, source_count=len(CATALOG), information_rows=len(rows),
                  candidate_count=3, new_scientific_calls=0, new_real_numerical_reanalysis=0,
                  author_check_not_independent_review=True, mathematical_proof_certification=False)
    (ROOT/'technical_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'check_count', 'source_count', 'information_rows',
                                          'candidate_count', 'new_scientific_calls', 'new_real_numerical_reanalysis')}))
    for item in checks:
        if not item['pass']:
            print(json.dumps(item, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
