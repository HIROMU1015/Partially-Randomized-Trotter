"""Run the unchanged artificial native test suite against the pinned counter.

Only the source-file lookup is redirected; tests/counter bytes are preserved.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'artifacts/resource_applicability/track_a_current_s2_theory_compiled/2026-10-22/run01/fixed_source'
text=(FROZEN/'test_native_df_gate_model.py').read_text()
old='ROOT / "src/trotterlib/native_df_gate_model.py"'
assert text.count(old)==1
text=text.replace(old,repr(str(FROZEN/'native_df_gate_model.py')))
module=types.ModuleType('_pinned_native_artificial_tests');module.__file__=str(FROZEN/'test_native_df_gate_model.py')
sys.modules[module.__name__]=module
exec(compile(text,module.__file__,'exec'),module.__dict__)
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
print('ARTIFICIAL_PINNED_COUNTER_TESTS='+str(result.testsRun))
raise SystemExit(not result.wasSuccessful())
