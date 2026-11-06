"""Reuse verbatim arithmetic ASTs, without importing science dependencies/data."""
import ast
from fractions import Fraction as F
import hashlib
from pathlib import Path
from types import SimpleNamespace

FIXED_COMMIT = "560c6092bfb5e71af8eedc4a4be5416922764f98"
SOURCE_PATH = "src/trottertracks/algorithm_codesign/a1_engine_connection.py"
SOURCE_SHA256 = "3fb23b8f6e838441645708bece6b1bc1d5de4baf394190e7853f66090d8a7bd9"
NAMES = ("multiply", "add", "norm_squared", "apply_exact",
         "require_probability", "born_probability", "threshold_count")


def load_oracle():
    root = Path(__file__).resolve().parents[3]
    path = root / SOURCE_PATH
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("fixed dense oracle SOURCE mismatch")
    tree = ast.parse(source, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    zero = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ZERO" for t in n.targets)]
    if len(nodes) != len(NAMES) or len(zero) != 1:
        raise ValueError("fixed arithmetic definitions missing")
    scope = {"F": F}
    # Original AST bodies, arguments, checks, loop order and ZERO are unchanged.
    exec(compile(ast.Module(body=zero + nodes, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**{n: scope[n] for n in ("ZERO",) + NAMES})


old = load_oracle()
