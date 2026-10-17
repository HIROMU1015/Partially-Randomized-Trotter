"""Contract-only metadata checks. No runner, input decoder, certificate or auth.

The existing preflight is NOT called. Contract-only statements are selected from
its frozen AST; statements involving auth/pins are excluded and listed. Execution
guard checks are static. This is neither a numerical test nor execution approval.
"""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT/'docs/research/qc_a2_pilot_phase_a_20261018'
CANONICAL = Path('/tmp/prt-qc-a2-pilot-phase-a-20261018')
BASE = 'a0629ca65142644a38cef545a9ad1244e6387eec'
SOURCE = '8d483f9bfa5983bb52fef55887e2f9f73a790511'
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'src'))
from trottercert.a2_pilot import contract as cm
from trottercert.a2_spec01_03.common import canonical_bytes, content_id

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, env=ENV)

def digest(b):
    return hashlib.sha256(b).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    checks = []
    def check(name, condition):
        checks.append({'check': name, 'PASS': bool(condition)})
        if not condition:
            raise AssertionError(name)

    raw = (OUT/'execution_contract_C.json').read_bytes()
    c = cm.loads_strict(raw.decode())
    draft = cm.loads_strict((OLD/'execution_contract_draft.json').read_text())
    prop = json.loads((OLD/'candidate_conditions_proposal.json').read_text())
    check('adopted_C_not_execution_permission', c['adopted'] is True and c['contract_grants_execution'] is False)
    check('no_UNSET', not cm._unset(c))
    check('same_exact_top_level_schema_as_fixed_draft', set(c) == set(draft))
    check('frozen_SOURCE_object_unchanged', c['source'] == draft['source'] and c['source']['commit'] == SOURCE)
    check('frozen_input_except_adopted_mapping', {k:v for k,v in c['input'].items() if k!='mapping'} ==
          {k:v for k,v in draft['input'].items() if k!='mapping'})
    check('mapping_as_adopted_proposal', c['input']['mapping'] == prop['mapping_proposal'])
    check('trial_as_adopted', c['task']['occupied_modes'] == ['0','1','2','3'])
    check('full_PF_order_as_adopted', c['task']['fragment_order'] == ['one_body']+['df:block'+str(i) for i in range(12)])
    check('sqrt_resolution_as_adopted', c['task']['sqrt_bits'] == '96')
    check('exact_Rat_policy_as_adopted', c['task']['comparison_policy'] == {
        'd': {'num':'1','den':'1000'}, 'alpha_spec': {'num':'1','den':'10'},
        'root_width': {'num':'1','den':'1099511627776'}, 'max_bisections':'256'})
    check('cap_guarantees_information_policy_unchanged', all(c['task'][k] == draft['task'][k]
          for k in ('cap_policy','guarantees','information_policy')))
    check('record_and_log_limits_exact', c['limits']['output_bytes'] == '1099511627776' and c['limits']['log_bytes'] == '1073741824')
    check('other_limits_unchanged', all(c['limits'][k] == draft['limits'][k] for k in draft['limits'] if k not in ('output_bytes','log_bytes')))
    check('one_shot_failure_state_exclusions_store_unchanged', all(c[k] == draft[k] for k in ('failure_policy','state','excluded','execution_store','scope','schema')))

    # Run ONLY actual contract metadata statements, never fabricated auth/pins.
    path = ROOT/'src/trottercert/a2_pilot/contract.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    fun = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
    forbidden = {'auth', 'auth_bytes', 'pins', 'expected_auth', 'artificial_test'}
    allowed_calls = {'keys_exact','require','_unset','root_precision','integer','trial','policy_validate',
                     'content_id','digest','type','len','set','all'}
    selected, skipped = [], []
    for statement in fun.body:
        names = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)}
        if isinstance(statement, ast.Return) or names & forbidden:
            skipped.append({'line': statement.lineno, 'references': sorted(names & forbidden),
                            'reason': 'AUTH_PINS_OR_RETURN_NOT_EXECUTED'})
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                allowed = (isinstance(node.func, ast.Name) and node.func.id in allowed_calls) or (
                    isinstance(node.func, ast.Attribute) and node.func.attr in ('startswith','split')) or (
                    isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 're' and node.func.attr == 'fullmatch')
                assert allowed, ast.unparse(node)
        selected.append(statement)
    namespace = dict(vars(cm), contract=c)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 '<frozen-contract-metadata-statements-only>', 'exec'), namespace)
    metadata_require_count = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'require'
                                 for statement in selected for n in ast.walk(statement))
    check('all_selected_frozen_contract_metadata_checks_pass', True)
    check('SOURCE_declared_commit_shape_and_identity', len(SOURCE) == 40 and c['source']['identity'] == cm.source_identity())
    cm.verify_git_sources(CANONICAL, c)
    check('SOURCE_Git_blobs_at_S_match', True)
    source_paths = c['source']['identity']['modules']
    for name, expected in source_paths.items():
        check('SOURCE_SHA256_and_AST:'+name, digest((ROOT/name).read_bytes()) == expected
              and digest((CANONICAL/name).read_bytes()) == expected)
        ast.parse((ROOT/name).read_text(), filename=name)
    check('SOURCE_module_inventory', len(source_paths) == 31)
    for ancestor, descendant in [(c['input']['commit'], SOURCE), (SOURCE, BASE)]:
        check('ancestry:'+ancestor+'<='+descendant,
              subprocess.run(['git','merge-base','--is-ancestor',ancestor,descendant],cwd=ROOT,env=ENV).returncode==0)
    check('canonical_checkout_HEAD_at_Phase_A', git('rev-parse','HEAD',cwd=CANONICAL).decode().strip()==BASE)
    check('canonical_checkout_clean', not git('status','--porcelain','--untracked-files=all',cwd=CANONICAL).strip())
    check('canonical_origin_correct', git('remote','get-url','origin',cwd=CANONICAL).decode().strip()==
          'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git')
    input_path = c['input']['repository_path']
    blob = git('show',c['input']['commit']+':'+input_path)
    check('input_container_bytes_only_no_decode', (CANONICAL/input_path).read_bytes()==blob and (ROOT/input_path).read_bytes()==blob
          and digest(blob)==c['input']['manifest']['container']['sha256']
          and str(len(blob))==c['input']['manifest']['container']['bytes'])

    runner_path = ROOT/'src/trottercert/a2_pilot/runner.py'
    runner = runner_path.read_text()
    ast.parse(runner)
    check('guard_order_preflight_source_claim_input_adapter', runner.index('contract, auth = preflight(') <
          runner.index('source_check(contract)') < runner.index('store.claim(claim)') <
          runner.index('blob = load_input(contract)') < runner.index('raw = invoke("adapter"'))
    check('production_default_execute_closed', 'def run_files(repository, contract_path, authorization_path, pins, *, execute=False):' in runner
          and 'if execute is not True:' in runner)
    check('production_no_artificial_auth_override', 'return _run(contract_bytes, auth_bytes, pins, load_input=load, source_check=check_source, store_factory=store_factory)' in runner)
    check('canonical_path_and_ledger_guard_present', 'Different checkout cannot bypass the fixed one-shot ledger' in runner
          and 'artifacts/qc_a2_pilot_execution' in runner and 'Canonical owned ledger only' in runner)
    storage = (ROOT/'src/trottercert/a2_pilot/storage.py').read_text()
    check('claim_and_output_exclusive_no_release', 'parents=False, exist_ok=False' in storage
          and 'os.O_WRONLY | os.O_CREAT | os.O_EXCL' in storage
          and not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('unlink','rmdir','remove') for n in ast.walk(ast.parse(storage))))
    check('storage_per_record_or_each_log_cap', 'name in ("stdout.txt", "stderr.txt")' in storage
          and 'len(data) <= limit' in storage)
    check('positive_S2_order_scalar_semantics_in_checker', all(v in (ROOT/'src/trottercert/a2_pilot/core01_checker.py').read_text()
          for v in ('POSITIVE_PALINDROMIC_S2_V1','DISPLAY_LEFT_TO_RIGHT_TIME_RIGHT_TO_LEFT','EXP_MINUS_I_H_ENERGY','SCALAR_IDENTITY')))
    auth_segment = ast.get_source_segment(path.read_text(), fun)
    check('separate_auth_and_external_pin_binding_static', all(v in auth_segment for v in (
        'auth["schema"] == expected_auth','auth["execution_authorized"] is True',
        'auth["contract_sha256"] == pins["contract_sha256"]','auth["input_manifest_id"]',
        'sha256(auth_bytes).hexdigest() == pins["authorization_sha256"]')))
    ledger = CANONICAL/c['execution_store']['ledger_relative']
    claim_path = ledger/digest(raw)
    check('no_ledger_or_claim_exists_or_created', not ledger.exists() and not ledger.is_symlink() and not claim_path.exists())
    check('no_ledger_in_management_checkout', not (ROOT/c['execution_store']['ledger_relative']).exists())
    check('canonical_artifacts_not_symlink', not (CANONICAL/'artifacts').is_symlink())
    gitdir = Path(git('rev-parse','--absolute-git-dir',cwd=CANONICAL).decode().strip())
    locks = [str(p.relative_to(gitdir)) for p in gitdir.rglob('*.lock')]
    check('no_canonical_Git_locks_observed', not locks)
    before = json.loads(Path('/tmp/qc-a2-contract-adoption-before-20261018.json').read_text())
    check('latest_common_instruction_bytes_match_archived_reference',
          digest((Path(before['parent']['root'])/'research_project_instructions_unified_2026-10-17.md').read_bytes()) ==
          digest((ROOT/'docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md').read_bytes()))
    manifest_id = content_id('a2_input_manifest_v1', c['input']['manifest'])
    identity = {'schema':'a2_phase_b_contract_identity_v1', 'contract_path':str((OUT/'execution_contract_C.json').relative_to(ROOT)),
        'contract_raw_bytes':len(raw), 'contract_raw_sha256_for_future_authorization':digest(raw),
        'contract_canonical_json_sha256_not_raw_pin':digest(canonical_bytes(c)), 'phase_A_commit':BASE,
        'source_commit':SOURCE, 'source_identity':c['source']['identity'],
        'input_commit':c['input']['commit'], 'input_repository_path':input_path,
        'input_container_bytes':len(blob), 'input_container_sha256':digest(blob), 'input_manifest_id':manifest_id,
        'mapping_id_for_raw_provenance':content_id('a2_input_mapping_v1', c['input']['mapping']),
        'mapping_metadata_id':content_id('a2_adopted_mapping_metadata_v1', c['input']['mapping']),
        'PF_order_metadata_id':content_id('a2_adopted_PF_order_metadata_v1', {'formula':'POSITIVE_PALINDROMIC_S2_V1','fragment_order':c['task']['fragment_order']}),
        'trial_metadata_id':content_id('a2_adopted_trial_metadata_v1', {'occupied_modes':c['task']['occupied_modes'],'sector_N':'4','mode0':'LSB'}),
        'task_metadata_id':content_id('a2_adopted_task_metadata_v1',c['task']),
        'expected_model_id_policy':c['input']['expected_model_id'],
        'actual_model_sector_partition_PF_certificate_ids':'NOT_GENERATED_NO_COEFFICIENT_DECODE',
        'canonical_execution_repository':str(CANONICAL), 'ledger_path':str(ledger), 'future_claim_path':str(claim_path),
        'contract_grants_execution':False,
        'note':'Metadata IDs bind choices only; they are not numerical certificates or a consumed claim.'}
    dump(OUT/'identity_manifest.json',identity)
    memory = {k:v.strip() for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemTotal','MemAvailable')}
    usage = shutil.disk_usage(CANONICAL)
    observation = {'schema':'a2_execution_store_read_only_observation_v1',
        'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'canonical_repository':str(CANONICAL),'canonical_HEAD':BASE,'ledger_exists':False,'claim_exists':False,
        'ledger_path':str(ledger),'future_claim_path':str(claim_path),'Git_locks':locks,
        'logical_CPUs_observed':os.cpu_count(),'affinity_CPUs_observed':len(os.sched_getaffinity(0)),
        'memory_observation':memory,'disk_bytes':{'total':usage.total,'used':usage.used,'free':usage.free},
        'resource_reservation':False,'capacity_equal_to_1TiB_required':False,'resource_or_shared_settings_changed':False,
        'limitation':'Point-in-time host observations, not a reservation, quota attestation, runtime guarantee or process-use clearance. Recheck actual availability/noninterference before separate authorized execution.'}
    dump(OUT/'execution_store_observation.json', observation)
    report = {'schema':'a2_phase_b_contract_static_metadata_inspection_v1','status':'PASS',
        'passed_named_checks':len(checks),'checks':checks,
        'preflight_metadata_AST_selection':{'frozen_source_sha256':digest(path.read_bytes()),
            'selected_statement_lines':[s.lineno for s in selected], 'direct_require_calls_executed':metadata_require_count,
            'skipped_auth_pins_return_statements':skipped,
            'full_preflight_called':False,'authorization_created_or_supplied':False},
        'source_AST_parsed_and_sha_verified_files':len(source_paths),
        'input_check_scope':'Raw container bytes/SHA256 and existing manifest only, no unzip/header/value decode.',
        'guard_assessment':'STATIC_ONLY_NOT_A_RUNTIME_EXECUTION_TEST',
        'Phase_A_artificial_tests':'INHERITED_EVIDENCE_NOT_RERUN_THIS_TASK',
        'scientific_source_modified':False,'scientific_evaluations':0,'certificate_generation_calls':0,
        'adapter_calls':0,'runner_calls':0,'authorization_grant_marker_issued':0,'claims_consumed':0,
        'contract_grants_execution':False,'mandatory_STOP':True}
    dump(OUT/'static_inspection.json', report)
    print(json.dumps({'status':'PASS','named_checks':len(checks),'selected_metadata_require_calls':metadata_require_count,
                      'SOURCE_files':len(source_paths),'C_sha256':digest(raw),'input_manifest_id':manifest_id,
                      'Phase_B_executed':False,'ledger_exists':False}))

if __name__ == '__main__':
    main()
