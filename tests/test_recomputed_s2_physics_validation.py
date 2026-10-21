"""Artificial native-to-independent-physics check, no molecule or run unit."""
import importlib.util
from pathlib import Path
import runpy
import sys


def test_gaussian_onebody_and_squared_primitive():
    root=Path(__file__).resolve().parents[1]
    fixture=runpy.run_path(str(root/'tests/test_recomputed_s2_comparison_v2.py'))
    ctx=fixture['artificial']()
    p=root/'src/trotterlib/recomputed_s2_physics_validation.py'
    spec=importlib.util.spec_from_file_location('artificial_native_physics',p)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.CTX=ctx;module.FULL=ctx['preps'][2];module.TIMES=[.2,.4]
    rows=[module.check_block(i) for i in range(3)]
    assert all(r['PASS'] for r in rows)
    assert max(c['error'] for r in rows for c in r['checks'])<1e-12
