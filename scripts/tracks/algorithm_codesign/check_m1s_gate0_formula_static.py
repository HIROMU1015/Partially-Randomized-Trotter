#!/usr/bin/env python3
"""Static/JSON/authority checks only. NEVER evaluates a numerical formula."""
from pathlib import Path
import ast
import copy
import importlib.abc
import json
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


class BlockNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {"numpy", "scipy", "openfermion", "qiskit"} or fullname.endswith('.science'):
            raise RuntimeError("Forbidden numerical import in static checker: " + fullname)


def main():
    sys.meta_path.insert(0, BlockNumerics())
    from trottertracks.algorithm_codesign.m1s_gate0_v3 import control as c
    from trottertracks.algorithm_codesign.m1s_gate0_v3.budget import FORMULA_IDS, formula_policy
    checks = []
    def check(name, ok):
        checks.append({"name": name, "PASS": bool(ok)})
        if not ok:
            raise AssertionError(name)
    trees = {}
    paths = [*c.SOURCE_PATHS, "scripts/tracks/algorithm_codesign/check_m1s_gate0_formula_static.py"]
    for path in paths:
        source = (ROOT / path).read_text()
        tree = ast.parse(source, filename=path)
        compile(tree, path, "exec")  # compilation only; no module execution
        trees[path] = tree
        check("syntax: " + path, True)
    bp = "src/trottertracks/algorithm_codesign/m1s_gate0_v3/budget.py"
    sp = "src/trottertracks/algorithm_codesign/m1s_gate0_v3/science.py"
    cp = "src/trottertracks/algorithm_codesign/m1s_gate0_v3/control.py"
    bt, st, ct = trees[bp], trees[sp], trees[cp]
    check("budget module has no top-level calculation", all(
        not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)) or
        (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)) or
        (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict))
        for node in bt.body))
    functions = {n.name: n for n in ast.walk(bt) if isinstance(n, ast.FunctionDef)}
    constructor = functions['__init__']
    check("budget constructor requires permit first", ast.unparse(constructor.body[0]) == "require_permit(permit)")
    check("formula inputs exclude observations", ast.unparse(functions['law_limits'].args) == 'self, coefficients, lam, precision')
    check("global inputs are raw scales", ast.unparse(constructor.args) == 'self, permit, constant, one, lambdas, factors')
    check("budget arithmetic uses exact rational type", 'Fraction' in (ROOT / bp).read_text())
    check("budget conversion rounds outward", 'math.nextafter' in ast.unparse(functions['up']))
    check("no dynamic expression evaluation", not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in {'eval', 'exec', '__import__'} for tree in (bt, st, ct) for n in ast.walk(tree)))
    check("9 allowlisted formula IDs", len(FORMULA_IDS) == 9)
    check("17 original precision items retained", len(c.EPS_KEYS) == 17)
    check("new schema", c.candidate()['schema'] == 'm1s_gate0_execution_contract_v3')
    check("new source binding includes budget", bp in c.SOURCE_PATHS)
    check("template cannot execute", not c.validate_config(c.candidate())['execution_ready'])
    check("template no structural errors", c.validate_config(c.candidate())['valid_static_shape'])
    for key, fid in FORMULA_IDS.items():
        config = c.candidate()
        config['precision'][key] = {'formula_id': fid}
        check("selector recognized: " + key, c.validate_config(config)['valid_static_shape'])
        config['precision'][key] = {'formula_id': fid, 'expression': 'result-dependent'}
        check("extra selector field rejected: " + key, not c.validate_config(config)['valid_static_shape'])
    for key in set(c.EPS_KEYS) - set(FORMULA_IDS):
        config = c.candidate()
        config['precision'][key] = {'formula_id': 'unsupported'}
        check("unsupported formula rejected: " + key, not c.validate_config(config)['valid_static_shape'])
    config = c.candidate()
    for key, fid in FORMULA_IDS.items():
        config['precision'][key] = {'formula_id': fid}
    result = c.validate_config(config)
    check("oracle quality remains pending", 'oracle_relative_residual_quality_scientific_decision' in result['pending'])
    check("coverage adoption remains pending", 'conditional_roundoff_model_adoption' in result['pending'])
    config['formula_policy']['assumptions'].append('POSTHOC_FIT')
    check("model conditions cannot be edited", not c.validate_config(config)['valid_static_shape'])
    config = c.candidate()
    config['precision']['W_error_budget'] = 2
    config['precision']['eps_W'] = 1
    check("fixed scalar coverage cannot exceed ceiling", not c.validate_config(config)['valid_static_shape'])
    config = c.candidate()
    config['limits']['wall_time_seconds'] = 1
    check("no scientific wall cap", not c.validate_config(config)['valid_static_shape'])
    config = c.candidate()
    config['decisions']['automatic_GO_STOP'] = True
    check("automatic scientific decision rejected", not c.validate_config(config)['valid_static_shape'])
    for tree, method in ((st, '__init__'), (st, 'run_authorized')):
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method]
        node = nodes[0]
        body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        check(method + " requires permit first", ast.unparse(body[0]) == 'require_permit(permit)')
    split = next(n for n in ast.walk(st) if isinstance(n, ast.FunctionDef) and n.name == 'split')
    text = ast.unparse(split)
    check("law budget before hpsi and W", text.index('law_limits(') < text.index('hpsi =') < text.index('W ='))
    check("resolution failure before metrics", text.index('PRECISION_UNRESOLVED_BEFORE_METRICS') < text.index('hpsi ='))
    check("budgets resolved per split without updating global precision", 'self.eps.update(resolved)' not in text)
    check("no target widening in resolver", 'eps_W' not in ast.unparse(functions['law_limits']) and 'eps_v' not in ast.unparse(functions['law_limits']))
    check("rho interval rounded outward", 'down(0.5' in text and 'up(0.5' in text)
    prepare = next(n for n in ast.walk(st) if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    check("formula scales resolved before H reconstruction", ast.unparse(prepare).index('global_limits()') < ast.unparse(prepare).index('self.bilinears ='))
    auth = next(n for n in ast.walk(ct) if isinstance(n, ast.FunctionDef) and n.name == 'authorize')
    a = ast.unparse(auth)
    check("separate v3 execution schema required", 'm1s_gate0_separate_execution_authorization_v3' in a)
    check("scope adoption cannot authorize", 'SCOPE_ADOPTION_NOT_EXECUTION_AUTHORIZATION' in a)
    check("source prior to adopted contract prior to authority", 'sc in (head, cc)' in a and 'cc == head' in a)
    check("raw NPZ identity before import retained", 'Frozen input byte identity mismatch before numerical import' in a)
    try:
        c.require_permit(None)
    except ValueError:
        check("absent authority denied", True)
    else:
        check("absent authority denied", False)
    check("numeric libraries never imported", not any(n in sys.modules for n in ('numpy','scipy','openfermion','qiskit')))
    check("scientific module never imported", not any(n.endswith('.m1s_gate0_v3.science') for n in sys.modules))
    return {"schema": "m1s_gate0_formula_static_checks_v1", "PASS": len(checks), "checks": checks,
            "numerical_formula_evaluations": 0, "numerical_correctness_tests": 0,
            "new_scientific_calculations": 0, "NPZ_arrays_decoded": 0,
            "scientific_execution_authorized": False, "mandatory_STOP": True}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
