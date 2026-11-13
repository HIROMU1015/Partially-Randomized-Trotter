"""Pinned B functions for artificial integration tests; no physical data reads."""
import ast
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks import pr_rpe_rmse as A

ROOT = Path(__file__).resolve().parents[1]
B_SOURCE = '170047ef6e7a72aef5f4c6ff9c73f942097a694e'
B_PATH = 'src/trottertracks/rpe_decoder_statistics.py'
PATCH = ROOT/'docs/research/rpe_decoder_r3_20261113/B_S3_PATCH.diff'


def b_statistics(patched=False):
    source = subprocess.check_output(['git', 'show', B_SOURCE+':'+B_PATH], cwd=ROOT, text=True)
    if patched:
        with tempfile.TemporaryDirectory(prefix='r3-B-hook-') as tmp:
            path = Path(tmp)/B_PATH
            path.parent.mkdir(parents=True)
            path.write_text(source)
            subprocess.run(['git', 'apply', '--unidiff-zero', str(PATCH)], cwd=tmp, check=True, capture_output=True)
            source = path.read_text()
    tree = ast.parse(source)
    # B controller is not present on A's branch. Supply only its unchanged
    # exact-scalar conversion and existing pipeline, never a replacement solver.
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or n.level == 0]
    code = subprocess.check_output(['git', 'show', B_SOURCE+':src/trottertracks/h4_df10_rpe_connection.py'], cwd=ROOT)
    functions = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in ('exact_scalar', 'exact_value')]
    exact = dict(Fraction=Fraction, math=__import__('math'))
    exec(compile(ast.Module(body=functions, type_ignores=[]), B_SOURCE+':exact', 'exec'), exact)
    scope = dict(A=A, B=SimpleNamespace(p=p, a=SimpleNamespace(**{k:exact[k] for k in ('exact_scalar','exact_value')})))
    exec(compile(tree, B_SOURCE+':'+B_PATH+(':hook' if patched else ''), 'exec'), scope)
    return SimpleNamespace(**{k:v for k,v in scope.items() if not k.startswith('__')})


def strip_acceleration(result):
    result = deepcopy(result)
    for row in result['trials']:
        row.pop('decoder_acceleration_source', None)
    return result
