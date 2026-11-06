"""Fix H6 code/config/input identities; no numerical library or input decode."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE='68634ef468eb6c912de7bf569484fc454fb32548'
CONTRACT='docs/research/research_c_h6_contract_unapproved_20261106.json'
SOURCE='artifacts/research_c_h6_runner/2026-11-06/execution_preflight/SOURCE.json'


def closure(root,roots):
    pending=list(roots);seen=set()
    def module(name):
        if not name.startswith(('trottertracks.','trotterlib.')):return
        path=Path('src')/Path(*name.split('.'))
        for candidate in (path.with_suffix('.py'),path/'__init__.py'):
            if (root/candidate).is_file():pending.append(str(candidate));break
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=root/name
        if not path.is_file():raise ValueError('MISSING_SOURCE:'+name)
        seen.add(name);text=path.read_text();tree=ast.parse(text)
        parts=Path(name).with_suffix('').parts
        package=list(parts[1:-1]) if parts[0]=='src' else []
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names:module(alias.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=(package[:len(package)-node.level+1]+(node.module.split('.') if node.module else [])) if node.level else (node.module or '').split('.')
                module('.'.join(prefix))
                for alias in node.names:module('.'.join(prefix+[alias.name]))
        for legacy in re.findall(r'_legacy\([\'"]([^\'"]+)[\'"]\)',text):module('trotterlib.'+legacy)
    return sorted(seen)


def main():
    c=json.loads(subprocess.check_output(['git','show',BASE+':'+CONTRACT],cwd=ROOT,text=True))
    c.update(status='UNAPPROVED_SCIENCE_EXECUTION',execution_ready=True,
        scientific_execution_authorized=False,execution_entry='scripts/run_research_c_h6.py',
        execution_SOURCE=SOURCE,execution_SOURCE_sha256=None,
        fresh_output='artifacts/research_c_h6_independent_validation/2026-11-06-runner/run01',
        technical_blockers=[],decisions_before_execution=[
            'Separate explicit user permission for this fixed H6 scientific SOURCE/input/C0/scope; technical readiness is not permission.'])
    old=json.loads((ROOT/'artifacts/research_c_h6_validation_preparation/2026-11-06/technical/environment.json').read_text())
    c['environment']['package_pins']={
        role:{k:v for k,v in old[role].items() if v is not None and k!='pytest'}
        for role in ('fit','evaluation')}
    c['environment']['python_version']='3.12.3'
    c['native']['correctness']['coverage_choice']='ADOPTED_D_PLUS_C'
    c['native']['correctness'].pop('alternative_every_kernel')
    c['native']['correctness'].pop('requires_explicit_acceptance_of_weaker_coverage_than_H4')
    c['order'][0]='Fixed adopted H6 conditions; completed synthetic E2E and passive checks; wait for separate scientific execution permission'
    c['order'][1]='Verify connected executor SOURCE and byte-fixed H6 input/C0, exact environments, fresh output and current resources'
    c['runtime_batch_columns']=16
    c['storage_plan']=dict(planning_envelope_GiB=60,resource_cap=False,
        maximum_native_QPY=2242,checked_reference_matrices=26,
        maximum_physical_assessments=391,physical_export_bytes_model_per_assessment=40968064,
        notes='raw/compiled each1092 + inverse2 each26 + shell6; physical J/U/G/H exports and optimizer raw are additional; first registered batch revises estimate, no raw deletion')
    sys.path.insert(0,str(ROOT/'src'))
    from trottertracks.c_h6_contract import configuration
    roots=['scripts/run_research_c_h6.py','scripts/seal_research_c_h6.py',
        'src/trotterlib/df_hamiltonian.py','src/trotterlib/rte.py',
        'src/trottertracks/c_h6_runtime.py','src/trottertracks/c_h6_downstream.py',
        'src/trottertracks/c_h6_parallel.py','src/trottertracks/c_h6_contract.py',
        'tests/management/test_research_c_h6_runner.py']
    files=closure(ROOT,roots)
    manifest=dict(schema='H6_execution_SOURCE_v1',base_commit=BASE,
        files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        contract=configuration(c),dynamic_modules=['trotterlib.df_hamiltonian','trotterlib.rte','literal _legacy targets recursively included'],
        scope='connected scientific execution code/config; permission is separate',
        old_technical_SOURCE='artifacts/research_c_h6_validation_preparation/2026-11-06/technical/SOURCE.json')
    path=ROOT/SOURCE;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    c['execution_SOURCE_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/CONTRACT).write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),SOURCE_sha256=c['execution_SOURCE_sha256'],scientific_calls=0)))


if __name__=='__main__':main()
