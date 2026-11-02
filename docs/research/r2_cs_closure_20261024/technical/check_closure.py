#!/usr/bin/env python3
"""Verify closure metadata only; never import or run scientific/test pipelines."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parents[2]
BASE = 'dee25ba13ee8ebcb23c7d294c659ea678d818277'
BRANCH = 'codex/r2-cs-closure-20261024'
PREFIX = ROOT.relative_to(WORKTREE).as_posix() + '/'
STATUS = 'R2_CS_INDEPENDENT_SELECTION_RESEARCH_CLOSED'
checks = []


def load(name):
    return json.loads((ROOT/name).read_text())


record = load('decision_record.json')
helper = record['fixed_references']['DOC_HELPERS']
body = subprocess.check_output(['git', '-C', str(WORKTREE), 'show',
                                helper['commit'] + ':' + helper['path']])
# Existing stdlib document helpers used by C0/C0.5/C0.6. No old main is run.
nodes = [n for n in ast.parse(body).body if isinstance(n, ast.FunctionDef)
         and n.name in ('git', 'digest', 'check')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<fixed-document-helpers>', 'exec'), globals())


def main():
    refs = record['fixed_references']
    stop = load('STOP.json')
    check('dedicated_branch_and_base', git('branch', '--show-current').decode().strip() == BRANCH
          and subprocess.run(['git', '-C', str(WORKTREE), 'merge-base', '--is-ancestor', BASE, 'HEAD'],
                             check=False).returncode == 0, BRANCH)
    paths = [p for p in git('diff', BASE, '--name-only', '-z').decode().split('\0') if p]
    paths += [p for p in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0') if p]
    old = set(git('ls-tree', '-r', '--name-only', BASE).decode().splitlines())
    check('only_new_closure_directory', bool(paths) and all(p.startswith(PREFIX) for p in paths)
          and not old.intersection(paths), 'No prior scientific SOURCE/result/grant/ledger/marker/STOP or first-study edit')
    check('explicit_user_adoption_not_recommendation_only', record['user_adoption']['adopted'] is True
          and record['user_adoption']['authority'] == 'EXPLICIT_USER_DECISION'
          and record['user_adoption']['adopted_C06_option'] == 'C'
          and bool(re.fullmatch('[0-9a-f]{64}', record['user_adoption']['instruction_sha256'])),
          'Current user instruction adopts the fixed C0.6 recommendation')
    check('closure_status_is_R2_only', record['status'] == stop['status'] == STATUS
          and record['research_id'] == stop['research_id'] == 'R2-CS-20261024'
          and record['closure_scope']['id'] == 'R2_CS_INDEPENDENT_CONFIGURATION_SELECTION_ALGORITHM'
          and not record['closure_scope']['propagates_to_other_research']
          and not stop['closure_propagates_to_other_research'], 'Dedicated closure schema; no propagation')
    expected = {'C0': '8cb41c40aec9c31c69a5a50e8e5fe93aa6d2eb88',
                'C0.5': '25cbd26d27057bd3d5af9c18d1c4b7be13d9cedf', 'C0.6': BASE}
    check('fixed_stages_and_direct_C06_basis', record['fixed_evidence_commits'] == expected
          and record['direct_scientific_basis'] == 'C0.6'
          and set(record['closure_scope']['included_stages']) == set(expected), 'No scientific reassessment')
    bad = []
    for key, r in refs.items():
        b = git('show', r['commit'] + ':' + r['path'])
        if digest(b) != r['sha256']:
            bad.append(key)
    check('fixed_reference_SHA_matches', not bad, {'reference_count': len(refs), 'errors': bad})
    check('eight_asset_groups_resolve', len(record['preserved_assets']) == 8
          and len({a['id'] for a in record['preserved_assets']}) == 8
          and all(a['references'] and set(a['references']) <= set(refs) for a in record['preserved_assets']),
          'Existing SOURCE/math/results referenced, not duplicated or executed')
    historical = ['C0_pilot_contract_DRAFT_json', 'C0_STOP_json', 'C05_STOP_json', 'C06_STOP_json', 'R3']
    check('old_P1_HOLD_STOP_R3_preserved', all((WORKTREE/refs[k]['path']).read_bytes()
          == git('show', refs[k]['commit'] + ':' + refs[k]['path']) for k in historical), historical)
    p1 = record['P1']
    draft = json.loads(git('show', refs['C0_pilot_contract_DRAFT_json']['commit'] + ':'
                          + refs['C0_pilot_contract_DRAFT_json']['path']))
    check('P1_removed_unexecuted_and_ungranted', p1['id'] == draft['pilot'] == 'P1_FULL_SAVED_INFORMATION_VALUE'
          and p1['status'] == stop['P1_status'] == 'REMOVED_FROM_R2_CS_EXECUTION_PLAN_UNEXECUTED'
          and all(p1[k] is False for k in ('research_continuation_candidate', 'executed', 'scientific_source_fixed',
                                          'grant_issued', 'new_numerical_reanalysis', 'automatic_execution_allowed'))
          and draft['runner_SOURCE'] is None and draft['grant'] is None
          and draft['execution_authorized'] is False and draft['enabled'] is False,
          'New final status; old draft/HOLD records retain their original bytes')
    states = {s['id']: s['state'] for s in record['preserved_research_scopes']}
    check('other_research_scopes_unchanged', states == {'TRACK_A': 'UNCHANGED', 'PR2': 'UNCHANGED',
          'G6_P5': 'UNCHANGED', 'TRACK_B': 'UNCHANGED', 'R3': 'STOP_R3_NO_METHOD_DELTA_UNCHANGED',
          'OTHER_PR_PF': 'UNCHANGED'}, 'No other research closure, reopening or reclassification')
    check('no_general_impossibility_claim_or_new_plan', record['general_selection_impossibility_claimed'] is False
          and record['future_conditions_are_new_plan_or_authorization'] is False
          and p1['future_similar_validation_requires'] == 'SEPARATE_RESEARCH_CONTRACT_AND_EXPLICIT_USER_APPROVAL',
          'Future reconsideration conditions confer no execution authority')
    check('no_new_science_reanalysis_review_or_state_change', all(v == 0 for v in record['actual_activity'].values()),
          record['actual_activity'])
    check('mandatory_STOP_no_next_work', stop['mandatory_STOP'] is True
          and record['next_stage_authorized'] is stop['next_stage_authorized'] is False
          and record['next_action'] == stop['next_action'] == 'NONE'
          and stop['next_work'] == '次の作業なし', 'No automatic pilot, review, science or new research')
    policy = record['formal_policy']
    check('formal_policy_version_gap_retained', not policy['expected_original_obtained']
          and policy['expected_sha256'] != policy['available_sha256']
          and policy['explicit_current_user_instruction_applied'], 'Exact formal original not claimed as read')
    private, broken = [], []
    for p in ROOT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        s = p.read_text()
        if re.search(r'/h[o]me/|/Us[e]rs/|/t[m]p/|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY', s):
            private.append(p.relative_to(ROOT).as_posix())
        if p.suffix == '.md':
            for link in re.findall(r'\]\(([^)]+)\)', s):
                if link.startswith(('https:', 'http:')) or link == 'verification.json':
                    continue
                if not (p.parent/link.split('#')[0]).exists():
                    broken.append(link)
    check('no_private_locator_or_credential_patterns', not private, private)
    check('local_links_resolve', not broken, broken)
    result = dict(schema='R2_CS_CLOSURE_METADATA_VERIFICATION_V1',
                  status='PASS' if all(c['pass'] for c in checks) else 'FAIL', check_count=len(checks),
                  checks=checks, fixed_reference_count=len(refs), preserved_asset_groups=8,
                  new_scientific_calls=0, new_saved_value_numerical_analysis=0,
                  prior_scientific_or_synthetic_pipelines_executed=False,
                  author_metadata_check_not_independent_scientific_review=True)
    (ROOT/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'check_count', 'fixed_reference_count',
                                          'preserved_asset_groups', 'new_scientific_calls')}))
    for c in checks:
        if not c['pass']:
            print(json.dumps(c, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
