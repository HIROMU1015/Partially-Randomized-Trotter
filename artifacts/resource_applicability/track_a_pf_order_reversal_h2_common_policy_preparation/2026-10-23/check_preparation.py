"""Check preparation structure/provenance, without numerical science execution."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = '366b2fd24f7e86aa82b5b518b6e3774520f65596'
AUDIT = ROOT / 'artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    # These two documented outputs are finalized after the checks below.
    # A failed check leaves an explicit pending record rather than a stale PASS.
    save(HERE/'technical_verification.json', dict(status='CHECKS_PENDING'))
    save(HERE/'file_inventory.json', dict(status='INVENTORY_PENDING'))
    checks = []
    def check(name, condition, detail=None):
        checks.append(dict(check=name, PASS=bool(condition), detail=detail))
        if not condition:
            raise AssertionError(name)

    check('exact_base_HEAD', git('rev-parse','HEAD').decode().strip() == BASE)
    check('detached_worktree', subprocess.run(['git','symbolic-ref','-q','HEAD'],cwd=ROOT,capture_output=True).returncode == 1)
    check('all_preexisting_tracked_bytes_unchanged', git('diff','--name-only',BASE) == b'')
    check('index_unchanged_no_staging', git('diff','--cached','--name-only') == b'')
    own = str(HERE.relative_to(ROOT)) + '/'
    new_paths = git('ls-files','--others','--exclude-standard').decode().splitlines()
    check('new_files_confined_to_preparation_directory', all(p.startswith(own) for p in new_paths))
    for r in read(HERE/'inherited_audit_files.json')['files']:
        raw = (ROOT/r['path']).read_bytes()
        check('inherited_audit_unchanged:'+Path(r['path']).name,
              sha(raw) == r['sha256'] and raw == git('show',BASE+':'+r['path']))

    sources = read(HERE/'source_manifest.json')
    for r in sources:
        raw = (ROOT/r['local_path']).read_bytes()
        check('source_hash:'+r['commit'][:8]+':'+r['path'], len(raw) == r['bytes'] and sha(raw) == r['sha256'])
        check('fixed_git_blob:'+r['commit'][:8]+':'+r['path'],
              raw == git('show',r['commit']+':'+r['path']))
        if r['path'].endswith('.py'):
            ast.parse(raw, filename=r['path'])

    plan = read(HERE/'execution_plan_DRAFT.json')
    records = list(csv.DictReader((HERE/'candidate_records_4.csv').open()))
    original = list(csv.DictReader((AUDIT/'evidence_rows_75.csv').open()))
    check('four_explicit_candidates_no_reselection', [r['candidate_id'] for r in records] == plan['candidate_ids'])
    check('four_rows_copied_as_strings_without_numeric_reanalysis', all(r in original for r in records) and len(records)==4)
    check('single_requested_precision_only', {r['epsilon'] for r in records} == {'0.0005'})

    private = read(AUDIT/'private_metadata_audit.json')
    provider = read(HERE/'provider_git_comparison.json')['files']
    check('only_previously_identified_provider_paths', [r['path'] for r in provider] == [r['path'] for r in private['provider_file_comparisons']])
    check('git_byte_comparison_not_runtime_attestation', all(r['actual_new_loaded_provider_verified'] is False for r in provider))
    check('four_library_Git_matches_eight_external_provider_paths',
          len([r for r in provider if r['git_bytes_match_old_seal'] is True])==4 and
          len([r for r in provider if r['new_git_sha256'] is None])==8)
    loc = read(HERE/'exact_locator_checks.json')
    check('private_gap_and_user_deferral_explicit', loc['new_private_root'] is None and
          loc['remote_access_attempted'] is False and loc['user_reply']=='一旦なしで進めて後で渡す')
    check('three_known_locators_only', len(loc['files']) == 3 and
          {r['path'] for r in loc['files']} == set(read(HERE/'missing_evidence.json')['files'][5]['exact_known_paths']))

    def get_source(commit, path):
        r = next(r for r in sources if (r['commit'],r['path']) == (commit,path))
        return ast.parse((ROOT/r['local_path']).read_bytes())
    a = get_source('217c971dd537b1cc7fbc13742bc0f6c263ff7f45', 'src/trottertracks/resource_applicability/h6_matched_accounting_v1.py')
    n = next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='empirical_allowance')
    check('old_generator_component_signature',
          [v.arg for v in n.args.kwonlyargs] == ['reference_error','path_errors','normalization_error','raw_closure','maximum_norm','work','settings'])
    b = get_source('4ec26921c6872ed6e8667f167074e1e405cf1a3f', 'scripts/h46_m5_signal_contract.py')
    n = next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='eligibility')
    # Structural dataflow observation: the generator takes signal record/group;
    # the H2 independent forward diagnostics are not arguments to this call.
    names = {x.id for x in ast.walk(n) if isinstance(x,ast.Name)}
    check('new_generator_not_mistaken_for_constant_assignment', 'roundoff' in names and 'closure' in names and 'max' in names)
    check('new_generator_does_not_consume_forward_or_reference_error_fields',
          'independent_forward_abs_errors' not in ast.dump(n) and 'reference_error' not in ast.dump(n))
    runtime = get_source('4ec26921c6872ed6e8667f167074e1e405cf1a3f','scripts/h2_outer32_runtime.py')
    signal = next(n for n in runtime.body if isinstance(n,ast.FunctionDef) and n.name=='signal_stage')
    check('H2_signal_calls_same_new_eligibility_generator',
          any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='eligibility' for x in ast.walk(signal)))

    for name in ('prepare_static_evidence.py','check_preparation.py'):
        tree = ast.parse((HERE/name).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.extend(x.name.split('.')[0] for x in node.names)
            if isinstance(node,ast.ImportFrom): imports.append(node.module.split('.')[0])
        check('own_checker_stdlib_imports_only:'+name,
              set(imports) <= {'ast','csv','hashlib','json','pathlib','subprocess','re'})

    for name in ('README.md','REPORT.md','ALLOWANCE_OPTIONS.md','EXECUTION_PLAN_DRAFT.md'):
        body = (HERE/name).read_text()
        links = re.findall(r'\]\(([^)]+)\)',body)
        check('local_document_links_exist:'+name,
              all((HERE/x.split('#')[0]).exists() for x in links if not x.startswith(('https:','http:','#'))))
    check('three_policy_options_recommendation_unadopted', len(plan['policy_options'])==3 and
          plan['recommended_policy_option']==plan['policy_options'][0] and plan['policy_adopted'] is False)
    check('execution_disabled_no_SOURCE_no_grant_no_reservations',
          plan['authorized'] is False and plan['science_GO'] is False and
          plan['SOURCE_for_future_execution'] is None and plan['future_grant'] is None and
          plan['valid_grants_created']==plan['scientific_calls_reserved']==plan['scientific_calls_consumed']==0)
    check('uncertain_extra_call_counts_not_invented', plan['conditional_missing_diagnostic_calls']['currently_required_call_count'] is None)
    status = read(HERE/'completion_status.json')
    zeros = ('new_scientific_calls','saved_value_scientific_reanalysis','shot_recalculations',
             'cost_reaccounting','rankings_computed','new_sampling','new_signals','new_simulation',
             'new_build_compile','new_valid_grants','scientific_calls_reserved','commits_created','pushes')
    check('zero_current_execution_counters', all(status[k]==0 for k in zeros))
    stop = read(HERE/'STOP.json')
    check('mandatory_STOP_all_boundaries', stop['status']=='MANDATORY_STOP' and
          all(stop[k] is False for k in ('science_authorized','policy_adopted','valid_grant','science_GO','commit_push_authorized','old_STOP_modified','next_stage_authorized')))
    check('completion_is_preparation_not_scientific_validation', status['mandatory_STOP'] and
          not status['science_GO'] and not status['detailed_science_review_started'] and
          not status['external_reproduction_or_CI_claimed'])

    save(HERE/'technical_verification.json', dict(status='STATIC_PREPARATION_CHECKS_PASS',
        checks=checks, passed=len(checks), failed=0,
        method='primary-author static byte/structure checks; no separate independent scientific reviewer',
        historical_source_imports=0, scientific_tests_executed=0,
        scientific_numerical_assertions_executed=0, scientific_execution_authorized=False,
        previous_75_row_audit_reexecuted=False, commit_created=False, push=False))
    files = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.name != 'file_inventory.json':
            raw = path.read_bytes()
            files.append(dict(path=str(path.relative_to(HERE)),bytes=len(raw),sha256=sha(raw)))
    save(HERE/'file_inventory.json',dict(status='UNCOMMITTED_LOCAL_PREPARATION_FILES',
         inventory_self_excluded=True, files=files))
    print(json.dumps(dict(status='STATIC_PREPARATION_CHECKS_PASS',passed=len(checks),failed=0,
                         scientific_tests_executed=0,files=len(files),mandatory_STOP=True)))


if __name__=='__main__':
    main()
