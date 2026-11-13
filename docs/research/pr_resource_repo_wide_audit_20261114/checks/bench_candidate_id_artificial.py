"""Audit microbenchmark (artificial candidate): cost of pf_pr_rpe_pipeline.candidate_id per call,
and of the same function with the parsed carrier `cid` cached (identical output required)."""
import sys, time, ast, json, hashlib
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root/'src'))
import importlib.util
spec = importlib.util.spec_from_file_location('pipeline_under_test', root/'src/trottertracks/pf_pr_rpe_pipeline.py',
                                              submodule_search_locations=None)
# import as part of the package so relative imports resolve
from trottertracks import pf_pr_rpe_pipeline as common
c = dict(system='H4', method='PR', formula='2nd', prefix=1, T=6.4, q=64, R_outer=128, K=2, delta=.1)
n = 300; t = time.perf_counter(); ids = [common.candidate_id(c) for _ in range(n)]; per = (time.perf_counter() - t) / n
path = root/'scripts/h46_long_time_signal.py'
node = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.FunctionDef) and x.name == 'cid')
scope = {'json': json, 'hashlib': hashlib}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
m = 100000; t = time.perf_counter(); fast = [scope['cid'](c) for _ in range(m)]; per_fast = (time.perf_counter() - t) / m
assert set(ids) == set(fast) and len(set(ids)) == 1
print('candidate_id: %.2f ms/call; cached parse: %.4f ms/call; ratio %.0fx; same id %s' % (per * 1e3, per_fast * 1e3, per / per_fast, ids[0]))
