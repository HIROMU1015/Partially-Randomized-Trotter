#!/usr/bin/env python3
"""Non-scientific checks only. Never call the numerical engine or load an NPZ.

This is deliberately separate from pytest and all historical science runners.
The import blocker and audit hook reject numerical modules and NPZ access.
"""
import ast
import contextlib
import importlib.abc
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
BLOCKED = {"numpy", "scipy", "openfermion", "openfermionpyscf", "pyscf", "qiskit", "numba"}


class BlockScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED or fullname.endswith("m1s_gate0.science"):
            raise AssertionError(f"Non-scientific check attempted scientific import: {fullname}")


def audit(event, args):
    if event == "open" and str(args[0]).endswith(".npz"):
        raise AssertionError("Non-scientific check attempted NPZ access")


def main():
    sys.meta_path.insert(0, BlockScientificImports())
    sys.addaudithook(audit)
    from trottertracks.algorithm_codesign.m1s_gate0 import control
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append({"check": name, "status": "PASS"})

    for path in (*control.SOURCE_PATHS, str(Path(__file__).relative_to(ROOT))):
        source = (ROOT / path).read_text()
        tree = ast.parse(source, filename=path)
        compile(tree, path, "exec")  # Syntax only; never execute the code object.
        check("syntax:" + path, True)
        if path.endswith(("control.py", "science.py", "__init__.py")):
            check("no_top_level_science:" + path,
                  all(not isinstance(node, ast.Import) or all(x.name.split(".")[0] not in BLOCKED for x in node.names)
                      for node in tree.body))
    science = ast.parse((ROOT / control.SOURCE_PATHS[2]).read_text())
    entry = next(n for n in science.body if isinstance(n, ast.FunctionDef) and n.name == "run_authorized")
    check("numerical_entry_first_action_requires_permit", isinstance(entry.body[1], ast.Expr)
          and isinstance(entry.body[1].value, ast.Call) and entry.body[1].value.func.id == "require_permit")
    engine = next(n for n in science.body if isinstance(n, ast.ClassDef) and n.name == "_Engine")
    constructor = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    check("engine_constructor_first_action_requires_permit", constructor.body[0].value.func.id == "require_permit")

    def cli(argv):
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = control.main(argv)
        except SystemExit as exc:
            code = exc.code
        return code, out.getvalue(), err.getvalue()
    for argv in ([], ["status"], ["--help"], ["execute", "--help"], ["schema"]):
        code, out, err = cli(argv)
        check("CLI:" + repr(argv), code == 0 and bool(out) and not err)
    check("input_alone_is_not_execution", cli([control.INPUT_PATH])[0] == 2)
    check("execute_missing_authority_is_refused", cli(["execute"])[0] == 2)
    template = control.candidate()
    result = control.validate_config(template)
    check("candidate_valid_shape_not_executable", result["valid_static_shape"] and not result["execution_ready"])
    malformed = dict(template, split_ids=["G6"])
    check("split_change_rejected", not control.validate_config(malformed)["valid_static_shape"])
    malformed = dict(template, adopted=True)
    check("adoption_status_inconsistency_rejected", not control.validate_config(malformed)["valid_static_shape"])
    try:
        control.read_json_bytes(b'{"x":1,"x":2}')
    except ValueError:
        check("duplicate_JSON_rejected", True)
    else:
        check("duplicate_JSON_rejected", False)
    try:
        control.read_json_bytes(b'{"x":NaN}')
    except ValueError:
        check("nonfinite_JSON_rejected", True)
    else:
        check("nonfinite_JSON_rejected", False)
    for invalid in (None, object()):
        try:
            control.require_permit(invalid)
        except ValueError:
            check("absent_permit_rejected", True)
        else:
            check("absent_permit_rejected", False)
    # Deliberately fail before any scientific module can be imported.
    try:
        control.authorize(ROOT, "missing_contract.json", "0" * 40, "missing_auth.json")
    except ValueError:
        check("wrong_authority_HEAD_rejected", True)
    else:
        check("wrong_authority_HEAD_rejected", False)
    check("no_numerical_modules_imported", not any(x.split(".")[0] in BLOCKED or x.endswith("m1s_gate0.science") for x in sys.modules))
    report = {"schema": "m1s_gate0_non_scientific_static_checks_v1", "checks": checks,
              "passed": len(checks), "scientific_numerical_tests_run": 0,
              "numerical_correctness": "UNTESTED_BY_EXPLICIT_SCOPE",
              "NPZ_access_in_this_checker": 0, "scientific_execution_authorized": False,
              "status": control.STATUS, "mandatory_STOP": True}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
