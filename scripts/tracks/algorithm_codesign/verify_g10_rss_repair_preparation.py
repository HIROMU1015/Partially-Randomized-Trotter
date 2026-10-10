"""Static/saved-only G10 v2 preparation audit; never invokes either runner."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from trottertracks.algorithm_codesign.g10_launch import verify_launch

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT/'artifacts/track_b_g10_rss_repair_preparation/2026-10-10/v2'
BOOKKEEPING_KEYS = {'schema', 'branch', 'base_commit', 'result_directory', 'authorization_path',
                    'optional_receipt_path', 'source_manifest', 'protected_ledger', 'io_revision'}


def identity(path, limit=None):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        raise PermissionError('NPZ rejected before any access')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while limit is None or size < limit:
            chunk = stream.read(32768 if limit is None else min(32768, limit-size))
            if not chunk: break
            size += len(chunk); digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


class StripTechnical(ast.NodeTransformer):
    """Explicit allowlist: only del, snapshots, pending.clear may disappear."""
    def visit_Delete(self, node): return None
    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and (node.value.func.value.id, node.value.func.attr) in
                    {('io', 'snapshot'), ('pending', 'clear')}):
            return None
        return self.generic_visit(node)


def scientific_ast_audit():
    old_path = ROOT/'scripts/tracks/algorithm_codesign/g10_degree_matched_native.py'
    new_path = ROOT/'scripts/tracks/algorithm_codesign/g10_degree_matched_native_v2.py'
    old = ast.parse(old_path.read_text()); new = ast.parse(new_path.read_text())
    execute = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
    old_with = next(n for n in ast.walk(execute) if isinstance(n, ast.With)
                    and ast.unparse(n.items[0].context_expr) == 'guard')
    old_body = old_with.body
    stop = next(i for i, n in enumerate(old_body) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "result['resource']")
    # The trailing guard.check is preserved in v2 execute rather than collect.
    old_body = old_body[:stop-1]
    collect = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    start = next(i for i, n in enumerate(collect.body) if isinstance(n, ast.Expr)
                 and ast.unparse(n.value) == "configure(c['interval_dps'])")
    normalized = StripTechnical().visit(ast.Module(body=copy.deepcopy(collect.body[start:]), type_ignores=[]))
    before = ast.dump(ast.Module(body=old_body, type_ignores=[]), include_attributes=False)
    after = ast.dump(normalized, include_attributes=False)
    if before != after:
        raise AssertionError('scientific statements changed beyond explicit lifetime/telemetry allowlist')
    old_clocks = [n for n in execute.body if isinstance(n, ast.FunctionDef)]
    new_clocks = [n for n in collect.body if isinstance(n, ast.FunctionDef)]
    for left, right in zip(old_clocks, new_clocks):
        if ast.dump(left) != ast.dump(StripTechnical().visit(right)):
            raise AssertionError('classical stage clock expression changed')
    # Critical ordering checks make the otherwise permissive deletion allowlist
    # concrete: no required row/event/cache field is deleted or emptied.
    source = new_path.read_text()
    expected_deletes = {'old_raw', 'old', 'gs', 'bs', 'ce', 'cert', 'g', 'b', 'es',
                        'bits', 'event', 'keys', 'pending', 'm', 'arm', 'events',
                        'saved', 'angle', 'needed', 'r'}
    deleted = {target.id for node in ast.walk(collect) if isinstance(node, ast.Delete)
               for target in node.targets if isinstance(target, ast.Name)}
    if deleted != expected_deletes:
        raise AssertionError('unexpected lifetime deletion set')
    if any(not isinstance(t, ast.Name) for n in ast.walk(collect) if isinstance(n, ast.Delete) for t in n.targets):
        raise AssertionError('required field deletion forbidden')
    if not (source.index('old = json.loads(old_raw)') < source.index('del old_raw') < source.index('pending = []')
            and source.index("[rebudget_anchor(r, c)") < source.index('del old\n') < source.index('for m, arm, events, b in pending:')
            and source.index("result['rows'].append(row(") < source.index('pending.clear()') < source.index('expected =')):
        raise AssertionError('lifetime cleanup ordering changed')
    if 'serial(result)' in source or 'json.dumps(result)' in source:
        raise AssertionError('whole-result fallback restored')
    return {'normalized_scientific_body_AST_equal': True,
            'classical_clock_expressions_equal': True,
            'normalization_allowlist': ['local del', 'io.snapshot', 'pending.clear'],
            'scientific_AST_sha256': hashlib.sha256(before.encode()).hexdigest(),
            'old_runner': identity(old_path), 'new_runner': identity(new_path),
            'm5_deepcopy_implementation_unchanged': True,
            'required_payload_fields_not_removed': True}


def audit():
    old_contract = json.loads((ROOT/'artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json').read_text())
    c = json.loads((PREP/'contract_v2.json').read_text())
    changed = {k for k in old_contract.keys() | c.keys() if old_contract.get(k) != c.get(k)}
    if not changed <= BOOKKEEPING_KEYS or old_contract['caps'] != c['caps']:
        raise AssertionError('science/caps contract changed')
    old_manifest = json.loads((ROOT/'artifacts/track_b_g10_degree_preparation/2026-10-10/source_manifest_v1.json').read_text())
    # Use the explicit ledger rather than guess the prefix receipt schema.
    ledger = json.loads((ROOT/c['protected_ledger']).read_text())
    violations = []
    for path, record in ledger.items():
        got = identity(ROOT/path, record['bytes'] if path in c['append_only_paths'] else None)
        if got != {'bytes': record['bytes'], 'sha256': record['sha256']}:
            violations.append(path)
    if violations: raise AssertionError('protected identity changed: '+repr(violations))
    for path, digest in old_manifest['sha256'].items():
        if path not in c['append_only_paths'] and identity(ROOT/path)['sha256'] != digest:
            raise AssertionError('old critical source changed: '+path)
    auth = json.loads((ROOT/c['authorization_path']).read_text())
    if auth.get('science_execution_authorized') is not False or auth.get('source_commit') is not None:
        raise AssertionError('preparation must remain unauthorized')
    try:
        verify_launch(ROOT, PREP/'contract_v2.json', '0'*40)
    except PermissionError as exc:
        refusal = str(exc)
    else:
        raise AssertionError('pending launch accepted')
    if (ROOT/c['result_directory']).exists():
        raise AssertionError('fresh production output/marker may not be created in preparation')
    source_manifest = ROOT/c['source_manifest']
    if source_manifest.exists():
        manifest = json.loads(source_manifest.read_text())
        if manifest.get('focused_tests_passed') is not True:
            raise AssertionError('focused source verification incomplete')
        for path, digest in manifest['sha256'].items():
            if identity(ROOT/path)['sha256'] != digest:
                raise AssertionError('new critical identity changed: '+path)
    forbidden_imports = [name for name in sys.modules if name.startswith(('numpy', 'mpmath', 'pygridsynth'))]
    if forbidden_imports: raise AssertionError('science dependency unexpectedly imported')
    return {'kind': 'G10_V2_STATIC_SAVED_ONLY_PREPARATION', 'passed': True,
            'science_contract_and_caps_identical': True, 'bookkeeping_changed_keys': sorted(changed),
            'scientific_AST': scientific_ast_audit(), 'protected_paths': len(ledger),
            'protected_violations': [], 'old_critical_source_paths': len(old_manifest['sha256']),
            'pending_launch_refusal': refusal, 'fresh_marker_result_absent': True,
            'science_runner_synthesis_matrix_sampling_LP_calls': 0,
            'production_512_MiB_completion_proven': False,
            'science_execution_authorized': False, 'mandatory_STOP': True}


if __name__ == '__main__': print(json.dumps(audit(), indent=2, ensure_ascii=False))
