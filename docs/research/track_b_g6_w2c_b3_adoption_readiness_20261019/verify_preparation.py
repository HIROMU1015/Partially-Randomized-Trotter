"""Static metadata + artificial tests only. No scientific source execution."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE.relative_to(ROOT).as_posix()+'/'
NEW_SOURCE = 'src/trottertracks/algorithm_codesign/g6_w2c_b3_readiness.py'
NEW_TEST = 'tests/tracks/algorithm_codesign/test_g6_w2c_b3_readiness.py'
sys.path.insert(0, str(ROOT/'src'))
M = importlib.import_module('trottertracks.algorithm_codesign.g6_w2c_b3_readiness')
BASE_PATH = ROOT/'docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.json'
base = json.loads(BASE_PATH.read_text())
c = json.loads((HERE/'execution_contract_candidate_refined.json').read_text())
inv = json.loads((HERE/'source_inventory.json').read_text())
checks = []
def check(ok, name):
    checks.append({'name': name, 'passed': bool(ok)})
    if not ok:raise AssertionError(name)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name, value):(HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

result = M.validate_refined(c, base)
check(result['technical_candidate_consistent'] and result['science_calls'] == 0, 'refined preparation candidate consistent')
check(M.D.identity(base) == M.BASE_CONTRACT_ID, 'immutable base contract exact identity')
check(c['fixed_science']['call_budget']['total_sampling_calls'] == 31710848, 'budget unchanged, includes correctness256')
for p, h in c['fixed_science']['source_pins'].items():check(digest(ROOT/p) == h, 'old source bytes unchanged '+p)
for p, h in c['auxiliary_source_pins'].items():check(digest(ROOT/p) == h, 'new helper/test full SHA '+p)
for row in inv['frozen_public_evidence']:
    blob = subprocess.check_output(['git','-C',str(ROOT),'show',row['commit']+':'+row['path']])
    check(hashlib.sha256(blob).hexdigest() == row['sha256'], 'fixed Git blob '+row['path'])
    check(row['public_bytes_equal_git_blob'] is True, 'public retrieval '+row['path'])
    if row['base_worktree_equal']:
        check((ROOT/row['path']).read_bytes() == blob, 'fixed original unchanged '+row['path'])
for p, ceiling in zip(M.ANGLE_PROFILES, (20, 37, 53)):
    row = M.angle_bound(p)
    check(row['sufficient'] and row['radius_upper'] < M.F(1, 1 << ceiling) < row['tau'],
          'parameter-only sufficient inequality '+p['id'])
check(not any('trottertracks.algorithm_codesign.'+n in sys.modules for n in
              ('g7_generator','g6_f3_generator','g6_f3_precision','return_aggregation')), 'no real science modules imported')
try:M.reject_science(c)
except PermissionError:check(True, 'preparation science entry rejects')
else:check(False, 'preparation science entry rejects')
preflight = {'status':'CONSISTENT_DRAFT_ONLY', **result,
             'base_driver_blockers_unchanged':M.D.candidate_errors(base),
             'execution_blockers':['USER_PACKAGE_ADOPTION_PENDING','PRODUCTION_PORT_NOT_IMPLEMENTED_OR_ENABLED',
                                   'FORMAL_PRODUCTION_SOURCE_CONTRACT_NOT_FROZEN','SEPARATE_EXPLICIT_EXECUTION_APPROVAL_PENDING'],
             'new_real_sampling':0,'new_real_G7_correctness':0,'real_CPU_profiling':0,'real_authority_or_claims':0,
             'mandatory_STOP':True}
write('preflight.json',preflight)
for p in HERE.rglob('*.json'):
    json.loads(p.read_text());check(True, 'JSON parses '+p.name)
for p in (ROOT/NEW_SOURCE, ROOT/NEW_TEST, Path(__file__)):
    ast.parse(p.read_text());check(True, 'AST parses '+p.name)
for p in HERE.glob('*.md'):
    for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
        if '://' not in link and not link.startswith('#'):
            check((p.parent/link.split('#')[0]).exists(), 'document link '+p.name+':'+link)
changed = subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',M.BASE_COMMIT]).decode().splitlines()
extra = subprocess.check_output(['git','-C',str(ROOT),'ls-files','--others','--exclude-standard']).decode().splitlines()
for p in set(changed+extra):check(p.startswith(PREFIX) or p in (NEW_SOURCE, NEW_TEST), 'dedicated new-file scope '+p)

paths = (NEW_TEST, 'tests/tracks/algorithm_codesign/test_g6_w2c_b3_driver.py',
         'tests/tracks/algorithm_codesign/test_g6_w2c_cost_preparation.py')
def test_run(path):
    command = [sys.executable, path, '-v']
    run = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                         capture_output=True, text=True)
    text = run.stdout+run.stderr
    found = re.search(r'Ran (\d+) tests', text)
    count = int(found.group(1)) if found else 0
    log = Path(path).stem+'.log';(HERE/log).write_text(text)
    return {'path':path,'command':command,'exit_code':run.returncode,'tests':count,
            'group':'new_readiness' if path == NEW_TEST else 'existing_artificial',
            'log':log,'log_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'scope':'private temp/mock endpoints/fake clocks; no G7/F3 numeric execution or real bitstream'}
if '--static-only' in sys.argv:
    runs = json.loads((HERE/'artificial_test_results.json').read_text())['runs']
    for row in runs:check(digest(HERE/row['log']) == row['log_sha256'], 'saved local test log '+row['log'])
else:
    with ThreadPoolExecutor() as pool:runs = list(pool.map(test_run, paths))
check(all(row['exit_code'] == 0 for row in runs), 'all artificial test suites PASS')
check([row['tests'] for row in runs] == [30,36,48], 'new30 and existing84 distinct')
write('artificial_test_results.json',{'status':'PASS','new_readiness_tests':30,'existing_artificial_tests':84,
    'total_tests':114,'runs':runs,'scope':'new30 + existing36/48, local preparation checks only',
    'mock_thread_race_attempts':32,'mock_thread_race_winners':1,
    'mock_owned_process_race_attempts':32,'mock_process_race_winners':1,'production_admission_enabled':False,
    'real_G7_sampling_correctness_CPU_profiling':0,'real_authorization_claim_issuance_consumption':0,
    'scientific_SOURCE_executed':False,'local_observation_not_CI_or_external_reproduction':True})
old = json.loads((HERE/'verification.json').read_text())
report = {'schema':'b3_adoption_readiness_verification_v1','passed':True,'checks':checks,'check_count':len(checks),
    'base_commit':M.BASE_COMMIT,'old_driver_science_sources_contracts_modified':False,
    'new_real_science_calls':0,'saved_scientific_numerical_reanalysis':0,'production_authority_enabled_or_issued':False,
    'other_worktree_interiors_index_lock_process_output_accessed':False,'mandatory_STOP':True}
if 'preservation' in old:report['preservation'] = old['preservation']
write('verification.json',report)
print(json.dumps({'passed':True,'management_checks':len(checks),'new_tests':30,'existing_tests':84,'total':114,'science_calls':0}))
