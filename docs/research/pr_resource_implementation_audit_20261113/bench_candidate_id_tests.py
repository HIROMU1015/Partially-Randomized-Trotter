"""Run from the root of a B 3216d27a-based tree: python -B <this file> <test module> original|cached.
Run an existing unittest module with pf_pr_rpe_pipeline.candidate_id memoised (in-process patch only)."""
import ast, hashlib, json, sys, time, unittest
from functools import lru_cache
from pathlib import Path
sys.path[:0] = ['src', 'tests']
from trottertracks import pf_pr_rpe_pipeline as p
if sys.argv[2] == 'cached':
    @lru_cache(maxsize=1)
    def _cid():
        path = Path(p.__file__).resolve().parents[2] / 'scripts/h46_long_time_signal.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'cid')
        scope = {'json': json, 'hashlib': hashlib}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        return scope['cid']
    p.candidate_id = lambda candidate: _cid()(candidate)
module = __import__(sys.argv[1])
t = time.perf_counter()
r = unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromModule(module))
print(sys.argv[1], sys.argv[2], 'tests', r.testsRun, 'ok', r.wasSuccessful(), 'seconds', round(time.perf_counter() - t, 1))
