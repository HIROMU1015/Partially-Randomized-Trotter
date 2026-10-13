#!/usr/bin/env python3
"""Read-only D3 source/contract/identity verifier, never evaluates saved rows."""
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_robustness import launch
from trottertracks.algorithm_codesign.rte_development.evidence import digest


def verify(root):
    root=Path(root);c=json.loads((root/launch.CONTRACT).read_text())
    launch.validate_contract(root,c)
    a=json.loads((root/c['execution']['authorization_path']).read_text())
    if (a['science_execution_authorized'] is not False or a['runs']!=0
        or a['source_commit'] is not None or a['retries']!=0):
        raise PermissionError('source preparation requires pending authority')
    if a['contract_sha256']!=digest(root/launch.CONTRACT):
        raise PermissionError('pending contract binding mismatch')
    manifest=launch.check_identities(root,c)
    if (root/c['execution']['result_directory']).exists():
        raise FileExistsError('production output namespace must remain absent')
    forbidden={'aggregate_pairs','transpile','native_ir','collect_pending','reference_events',
               'finite_target','exact_target','sample','matrix_power','solve'}
    sources=list((root/'src/trottertracks/algorithm_codesign/rte_robustness').glob('*.py'))
    sources+=list((root/'scripts/tracks/algorithm_codesign').glob('run_d3_*.py'))
    calls=0
    for p in sources:
        tree=ast.parse(p.read_text())
        compile(tree,str(p),'exec')  # Parse/compile only, never executes a source.
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else None
                if name in forbidden:raise PermissionError('forbidden scientific call: '+name)
                calls+=1
    try:launch.verify_launch(root,'1'*40)
    except PermissionError as exc:
        if 'source preparation only' not in str(exc):raise
    else:raise AssertionError('pending runner launch accepted')
    return dict(status='D3_SOURCE_PREPARATION_STATIC_VERIFICATION_PASS',
                sources_compiled_without_execution=len(sources),AST_calls_inspected=calls,
                critical_files=len(manifest['critical_sha256']),pending_authority_rejects=True,
                scientific_design_projection_unchanged=True,real_backend_calls=0,
                registered_budget_or_price_evaluations=0,registered_pair_evaluations=0,
                production_result_namespace_exists=False,mandatory_STOP=True)


if __name__=='__main__':
    print(json.dumps(verify(ROOT),indent=2))
