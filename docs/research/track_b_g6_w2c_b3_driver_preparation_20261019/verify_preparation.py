"""Management/mock verification only. No real scientific module is imported."""
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
BASE = 'c8cbeb18f76aef2cb1329d8c810036372fa07cc0'
DRIVER = 'src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py'
TEST = 'tests/tracks/algorithm_codesign/test_g6_w2c_b3_driver.py'
PREFIX = HERE.relative_to(ROOT).as_posix()+'/'
sys.path.insert(0, str(ROOT/'src'))
M = importlib.import_module('trottertracks.algorithm_codesign.g6_w2c_b3_driver')
checks = []
def check(ok, name):
    checks.append({'name': name, 'passed': bool(ok)})
    if not ok:raise AssertionError(name)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name, value):(HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

c = json.loads((HERE/'execution_contract_candidate.json').read_text())
inv = json.loads((HERE/'source_inventory.json').read_text())
check(c['adopted'] is False and c['contract_grants_execution'] is False, 'candidate is not an authority')
check(c['cells'] == list(M.schedule()), '16 cells preserve arm/layer/axis/saved N')
check(c['call_budget']['total_sampling_calls'] == 256+8*sum(M.SAVED_N.values()), 'call budget bookkeeping')
check(c['call_budget']['total_sampling_calls'] == 31710848, 'exact scheduled sample budget')
check(c['call_budget']['budget_recomputation_calls'] == 0, 'no budget reexecution')
check(len(M.candidate_errors(c)) == 6, 'preflight rejects adoption and all five unfilled decisions')
for p, expected in c['source_pins'].items():check(digest(ROOT/p) == expected, 'source pin '+p)
for r in inv['new_source_and_tests']:check(digest(ROOT/r['path']) == r['sha256'], 'new source/test SHA '+r['path'])
for r in inv['fixed_read_only_public_evidence']:
    data = subprocess.check_output(['git','-C',str(ROOT),'show',r['commit']+':'+r['path']])
    check(hashlib.sha256(data).hexdigest() == r['sha256'], 'fixed evidence '+r['path'])
    check(r['public_bytes_match_git_blob'] is True, 'retrieved public fixed bytes '+r['path'])
    if r['base_worktree_bytes_match']:
        check((ROOT/r['path']).read_bytes() == data, 'fixed original unchanged '+r['path'])
raw = c['saved_evidence']
rows = M.load_saved((ROOT/raw['path']).read_bytes(), raw['sha256'])
check([rows[a]['budget']['N_per_axis'] for a in M.ARMS] == list(M.SAVED_N.values()), 'saved N quoted verbatim only')
for p in HERE.rglob('*.json'):
    json.loads(p.read_text())
    check(True, 'JSON parses '+p.name)
for p in (ROOT/DRIVER, ROOT/TEST, Path(__file__)):
    ast.parse(p.read_text());check(True, 'AST parses '+p.name)
for path in HERE.glob('*.md'):
    for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if '://' not in link and not link.startswith('#'):
            check((path.parent/link.split('#')[0]).exists(), 'document link '+path.name+':'+link)
changed = subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',BASE]).decode().splitlines()
extra = subprocess.check_output(['git','-C',str(ROOT),'ls-files','--others','--exclude-standard']).decode().splitlines()
for p in set(changed+extra):check(p.startswith(PREFIX) or p in (DRIVER, TEST), 'dedicated change scope '+p)
check(not any('trottertracks.algorithm_codesign.'+s in sys.modules for s in
              ('g7_generator','g6_f3_generator','g6_f3_precision','return_aggregation')), 'no real science modules imported')

def test_run(path):
    command = [sys.executable, path, '-v']
    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True)
    log = result.stdout+result.stderr
    match = re.search(r'Ran (\d+) tests', log)
    tests = int(match.group(1)) if match else 0
    name = Path(path).stem+'.log'
    (HERE/name).write_text(log)
    return {'path': path, 'command': command, 'exit_code': result.returncode, 'tests': tests,
            'log': name, 'log_sha256': hashlib.sha256(log.encode()).hexdigest(),
            'scope':'artificial fixtures/mock modules/fake clocks/private temp only; not real G7 correctness or CPU evidence'}
if '--static-only' in sys.argv:
    runs = json.loads((HERE/'artificial_test_results.json').read_text())['runs']
    for row in runs:check(digest(HERE/row['log']) == row['log_sha256'], 'retained artificial log '+row['log'])
else:
    with ThreadPoolExecutor() as pool:
        runs = list(pool.map(test_run, (TEST, 'tests/tracks/algorithm_codesign/test_g6_w2c_cost_preparation.py')))
check(all(r['exit_code'] == 0 for r in runs), 'new and inherited artificial tests PASS')
check(sum(r['tests'] for r in runs) == 84, '36 new + 48 inherited artificial tests')
write('artificial_test_results.json', {'status':'PASS', 'total_tests':84, 'new_tests':36,'inherited_artificial_tests':48,
    'runs':runs,'local_observation_not_CI':True,'real_G7_scientific_execution':False,
    'mock_full_schedule_trials':800,'mock_admission_only':True,'real_authority_issued_or_consumed':False})
old = json.loads((HERE/'verification.json').read_text()) if (HERE/'verification.json').exists() else {}
result = {'schema':'g6_w2c_b3_driver_preparation_verification_v1','passed':True,'checks':checks,
          'check_count':len(checks),'base_commit':BASE,'new_scientific_calculations':0,'saved_scientific_reanalysis':0,
          'new_real_correctness_sampling_profiling':0,'circuit_compile_synthesis':0,
          'authority_grant_marker_claim_issued_or_consumed':0,'parallel_worktree_interiors_accessed':False,
          'local_tests_not_CI_or_external_reproduction':True,'mandatory_STOP':True}
if 'preservation' in old:result['preservation'] = old['preservation']
write('verification.json',result)
print(json.dumps({'passed':True,'checks':len(checks),'artificial_tests':84,'science_calls':0}))
