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
    from trottertracks.algorithm_codesign.m1s_gate0_v4 import control as c
    from trottertracks.algorithm_codesign.m1s_gate0_v4.budget import FORMULA_IDS, formula_policy
    def deny_budget_call(frame, event, arg):
        if (event == 'call' and frame.f_code.co_filename.endswith('/m1s_gate0_v4/budget.py')
                and frame.f_code.co_name in {'__init__', 'q', 'l1', 'gamma', 'up', 'global_limits', 'law_limits'}):
            raise RuntimeError('Numerical formula evaluation forbidden in static checker')
    sys.setprofile(deny_budget_call)
    checks = []
    def check(name, ok):
        checks.append({"name": name, "PASS": bool(ok)})
        if not ok:
            raise AssertionError(name)
    trees = {}
    paths = [*c.SOURCE_PATHS, "scripts/tracks/algorithm_codesign/check_m1s_gate0_v4_static.py"]
    for path in paths:
        source = (ROOT / path).read_text()
        tree = ast.parse(source, filename=path)
        trees[path] = tree
        check("syntax: " + path, True)
    bp = "src/trottertracks/algorithm_codesign/m1s_gate0_v4/budget.py"
    sp = "src/trottertracks/algorithm_codesign/m1s_gate0_v4/science.py"
    cp = "src/trottertracks/algorithm_codesign/m1s_gate0_v4/control.py"
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
    check("new schema", c.candidate()['schema'] == 'm1s_gate0_execution_contract_v4')
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
    check("oracle quality no longer has an unset parameter", 'oracle_relative_residual_quality_scientific_decision' not in result['pending'])
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
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_Engine') if method == '__init__' else tree
        nodes = [n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == method]
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
    check("separate v4 execution schema required", 'm1s_gate0_separate_execution_authorization_v4' in a)
    check("scope adoption cannot authorize", 'SCOPE_ADOPTION_NOT_EXECUTION_AUTHORIZATION' in a)
    check("source prior to adopted contract prior to authority", 'sc in (head, cc)' in a and 'cc == head' in a)
    check("raw NPZ identity before import retained", 'Frozen input byte identity mismatch before numerical import' in a)
    try:
        c.require_permit(None)
    except ValueError:
        check("absent authority denied", True)
    else:
        check("absent authority denied", False)
    check("absolute quality has frozen provenance value", formula_policy()["oracle_absolute_residual_quality_hartree"] == 1e-8)
    check("quality policy uses absolute original solver", formula_policy()["oracle_quality_policy"] == "absolute_original_solver_acceptance_v1")
    check("evidence label is explicit", c.EVIDENCE_CLASSIFICATION == "CONDITIONAL_ENGINEERING_NOT_CERTIFIED")
    check("relative quality field removed", "oracle_relative_residual_quality" not in (ROOT / bp).read_text())
    for bad in (None, 1e-7, 1e-9, True, float('nan')):
        config = c.candidate()
        config['formula_policy']['oracle_absolute_residual_quality_hartree'] = bad
        check("frozen absolute quality modification rejected: " + repr(bad), not c.validate_config(config)['valid_static_shape'])
    config = c.candidate()
    config['precision']['eps_state_residual'] = 1e-8
    check("bare scalar cannot replace residual formula", not c.validate_config(config)['valid_static_shape'])
    config = c.candidate()
    config['precision']['eps_state_residual'] = {'formula_id': 'oracle_quality_plus_roundoff_v1'}
    check("v3 residual rule cannot be reused", not c.validate_config(config)['valid_static_shape'])
    config = c.candidate()
    config['evidence_classification'] = 'CERTIFIED'
    check("certificate evidence upgrade rejected", not c.validate_config(config)['valid_static_shape'])
    config = c.candidate()
    config['formula_policy']['coverage_model_adopted'] = True
    check("model acknowledgement alone does not authorize", not c.validate_config(config)['execution_ready'])
    global_text = ast.unparse(functions['global_limits'])
    check("absolute quality added to separate arithmetic budget", 'residual = self.q(quality) + self.residual_roundoff' in global_text)
    check("residual ledger retains both components", 'oracle_quality_component_hartree' in global_text and 'residual_arithmetic_error_upper' in global_text)
    check("old-new arithmetic difference unclaimed", 'DIFFERENT_PATHS_NOT_DIFFERENCE_CERTIFIED' in global_text)
    check("budget evidence stays conditional", 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED' in global_text)
    prep_text = ast.unparse(prepare)
    check("residual receipt precedes failing gate", prep_text.index('self.state_receipt =') < prep_text.index("self.gate('full_H_state_residual_2'"))
    run = next(n for n in ast.walk(st) if isinstance(n, ast.FunctionDef) and n.name == 'run_authorized')
    run_text = ast.unparse(run)
    check("failed state receipt preserved", "hasattr(engine, 'state_receipt')" in run_text)
    check("failed nonfinite diagnostics retain strict JSON", 'json.dumps(json_safe(report)' in run_text and 'allow_nan=False' in run_text)
    check("precision unresolved is technical stop", "r.get('technical_stop', False)" in run_text and 'TECHNICAL_STOP' in run_text)
    check("all three split rows retained", "for s in ('G3', 'G6', 'G9')" in run_text)
    check("oracle failure categorized", 'ORACLE_QUALITY_UNRESOLVED' in ast.unparse(st))
    check("input model failure categorized", 'INPUT_MODEL_NOT_APPLICABLE_TECHNICAL_STOP' in ast.unparse(st))
    check("reconstruction failure categorized", 'RECONSTRUCTION_UNRESOLVED' in ast.unparse(st))
    old_budget = ast.parse((ROOT / 'src/trottertracks/algorithm_codesign/m1s_gate0_v3/budget.py').read_text())
    old_funcs = {n.name: n for n in ast.walk(old_budget) if isinstance(n, ast.FunctionDef)}
    for name in ('__init__', 'q', 'l1', 'gamma', 'up'):
        check('unchanged arithmetic AST: ' + name, ast.dump(old_funcs[name]) == ast.dump(functions[name]))
    old_law = ast.unparse(old_funcs['law_limits']).replace('CONDITIONAL_MODEL_NOT_PLATFORM_CERTIFIED','CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
    check("W/v/floor law arithmetic preserved", old_law == ast.unparse(functions['law_limits']))
    check("numeric libraries never imported", not any(n in sys.modules for n in ('numpy','scipy','openfermion','qiskit')))
    check("scientific module never imported", not any(n.endswith('.m1s_gate0_v4.science') for n in sys.modules))
    sys.setprofile(None)
    return {"schema": "m1s_gate0_v4_static_checks_v1", "PASS": len(checks), "checks": checks,
            "evidence_classification": c.EVIDENCE_CLASSIFICATION,
            "runtime_or_numerical_correctness_verified": False,
            "numerical_formula_evaluations": 0, "numerical_correctness_tests": 0,
            "new_scientific_calculations": 0, "NPZ_arrays_decoded": 0,
            "scientific_execution_authorized": False, "mandatory_STOP": True}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
