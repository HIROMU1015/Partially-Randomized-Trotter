#!/usr/bin/env python3
"""Explicit one-shot authority for the fixed saved-data C/N/G package only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# Private namespace avoids trotterlib.__init__ and its chemistry/compiler imports.
pkg=types.ModuleType('h4_cng_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['h4_cng_private']=pkg
from h4_cng_private import h4_certificate_control as C

PLAN = ROOT/'artifacts/non_df_h4_conditional_cng_plan/2026-10-24'
BASE = ROOT/'artifacts/non_df_h4_conditional_cng_results/2026-10-24'
os.environ['MPLCONFIGDIR']=str(BASE/'mplconfig')
CODE = ['scripts/run_h4_conditional_cng.py','src/trotterlib/h4_conditional_cng.py',
        'src/trotterlib/h4_certificate_control.py','src/trotterlib/fixed_signal_accounting.py',
        'src/trotterlib/fixed_signal_evaluation.py','src/trotterlib/rte_event_audit.py',
        'tests/test_h4_conditional_cng.py']
LIMITS = dict(saved_input_binding_batches=1,base_error_and_scale_batches=1,
              cost_summary_batches=1,conditional_package_batches=1,comparison_region_batches=1,
              PR_cost_uncertainty_batches=2,PR_rare_stratum_mass_and_envelope_batches=2,
              scalar_rounding_verification_batches=1)


def runtime():
    # Existing project environment is read only. Bind actual selected packages,
    # standard library and linked native objects; no package installation.
    paths = {Path(sys.executable).resolve()}
    stdlib = Path(sysconfig.get_path('stdlib'))
    for p in stdlib.rglob('*'):
        if 'site-packages' in p.parts or '__pycache__' in p.parts:
            continue
        if p.is_file() and p.suffix in ('.py','.so'):
            paths.add(p.resolve())
    for name in ('numpy','scipy','matplotlib','contourpy','cycler','dateutil','PIL',
                 'packaging','pyparsing','kiwisolver','six'):
        spec = importlib.util.find_spec(name)
        C.require(spec is not None,'RUNTIME_MISSING:'+name)
        if spec.submodule_search_locations:
            for d in spec.submodule_search_locations:
                for p in Path(d).rglob('*'):
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                        paths.add(p.resolve())
            # numpy/scipy wheels may keep BLAS in sibling .libs directories.
            for d in Path(spec.origin).parent.parent.glob(name+'.libs'):
                paths.update(p.resolve() for p in d.rglob('*') if p.is_file())
        else:
            paths.add(Path(spec.origin).resolve())
    native = [p for p in paths if p.suffix == '.so' or '.so.' in p.name or p == Path(sys.executable).resolve()]
    for p in native:
        r = subprocess.run(['ldd',str(p)],capture_output=True,text=True)
        for line in r.stdout.splitlines():
            for word in line.split():
                if word.startswith('/') and Path(word).is_file():
                    paths.add(Path(word).resolve())
    return dict(python=sys.executable,version=sys.version,
                files={str(p):C.sha(p) for p in sorted(paths)},
                package_installations=0)


def verify(seal):
    C.require(seal['kind']=='H4_CONDITIONAL_CNG_SEAL_V1','WRONG_SEAL')
    C.require(C.sha(sys.executable)==seal['python_sha256'],'EXECUTABLE')
    C.hashes({str(ROOT/p):h for p,h in seal['code_files'].items()})
    for p,h in seal['code_files'].items():
        data = subprocess.check_output(['git','show',seal['SOURCE']+':'+p],cwd=ROOT)
        import hashlib
        C.require(hashlib.sha256(data).hexdigest()==h,'GIT_SOURCE_BINDING:'+p)
    C.require(C.sha(BASE/'runtime.json')==seal['runtime_sha256'],'RUNTIME_MANIFEST')
    C.hashes(C.read(BASE/'runtime.json')['files'])
    C.require(C.sha(PLAN/'inputs.json')==seal['input_manifest_sha256'],'INPUT_MANIFEST')
    C.require(C.sha(PLAN/'axis_conditions.json')==seal['axis_catalog_sha256'],'AXIS_CATALOG')
    C.hashes({str(ROOT/p):h for p,h in C.read(PLAN/'inputs.json')['files'].items()})
    C.require(C.sha(BASE/'approval.txt')==seal['approval_sha256'],'APPROVAL')


def prepare():
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(),'ALREADY_PREPARED')
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code = {p:C.sha(ROOT/p) for p in CODE}
    for p in CODE:
        C.require(subprocess.check_output(['git','show',source+':'+p],cwd=ROOT)==(ROOT/p).read_bytes(),'UNCOMMITTED_CODE')
    rt = runtime(); C.save(BASE/'runtime.json',rt)
    seal = dict(kind='H4_CONDITIONAL_CNG_SEAL_V1',SOURCE=source,code_files=code,
                python_sha256=C.sha(sys.executable),runtime_sha256=C.sha(BASE/'runtime.json'),
                input_manifest_sha256=C.sha(PLAN/'inputs.json'),axis_catalog_sha256=C.sha(PLAN/'axis_conditions.json'),
                approval_sha256=C.sha(BASE/'approval.txt'),UTC=C.utc())
    C.save(BASE/'seal.json',seal)
    draft = C.read(PLAN/'execution_contract_DRAFT.json')
    contract = dict(draft)
    contract.update(kind='H4_CONDITIONAL_CNG_SAVED_REANALYSIS_ACTIVE_V1',version='CONDITIONAL_CNG_V2',
                    execution_authorized=True,grant_valid=True,dispatch_enabled=True,N_G_authorized=True,
                    numerical_runner_SOURCE=source,runtime_manifest='runtime.json',
                    approval_required='Explicit current user approval in approval.txt received',
                    current_stage='LIMITED_SAVED_DATA_REANALYSIS',approval_sha256=seal['approval_sha256'],
                    seal_sha256=C.sha(BASE/'seal.json'),namespace=str(BASE/'run01'),
                    authority_batch_limits=LIMITS,
                    authority_batch_mapping='logical batches reserve the unchanged finite planned_reanalysis_budget; output cardinalities checked before completion',
                    grant_id=None)
    C.save(BASE/'active_contract.json',contract)
    verify(seal)
    ns = BASE/'run01';ns.mkdir()
    grant = dict(kind='CNG_SAVED_DATA_ONE_SHOT',namespace=str(ns),contract_sha256=C.sha(BASE/'active_contract.json'),
                 seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=seal['approval_sha256'],one_shot=True)
    grant['grant_id'] = C.digest(grant)
    C.save(ns/'grant.json',grant)
    print(json.dumps(dict(SOURCE=source,grant_id=grant['grant_id'],status='SEALED_NOT_EXECUTED')))


def check(ns, consume=False):
    C.require(ns.resolve()==BASE/'run01','NAMESPACE')
    c=C.read(BASE/'active_contract.json');seal=C.read(BASE/'seal.json');g=C.read(ns/'grant.json')
    C.require(c['kind']=='H4_CONDITIONAL_CNG_SAVED_REANALYSIS_ACTIVE_V1','CONTRACT_KIND')
    d=C.read(PLAN/'execution_contract_DRAFT.json')
    allowed={'kind','version','execution_authorized','grant_valid','dispatch_enabled','N_G_authorized',
             'numerical_runner_SOURCE','runtime_manifest','approval_required','current_stage','grant_id'}
    C.require(all(c[k]==v for k,v in d.items() if k not in allowed),'FIXED_DRAFT_CONDITION_CHANGED')
    C.require(all(c[k] is True for k in ('execution_authorized','grant_valid','dispatch_enabled','N_G_authorized')),'NOT_AUTHORIZED')
    C.require(c['mandatory_STOP'] is True and not c['automatic_retry'] and not c['formal_certificate_accepted'],'STOP_CERTIFICATE')
    C.require(c['authority_batch_limits']==LIMITS and all(v==0 for v in c['forbidden_calls'].values()),'BUDGET')
    C.require(g['kind']=='CNG_SAVED_DATA_ONE_SHOT' and g['namespace']==str(ns),'GRANT_NAMESPACE')
    C.require(g['grant_id']==C.digest({k:v for k,v in g.items() if k!='grant_id'}),'GRANT_DIGEST')
    C.require(g['contract_sha256']==C.sha(BASE/'active_contract.json') and g['seal_sha256']==C.sha(BASE/'seal.json')==c['seal_sha256'],'GRANT_BINDING')
    C.require(c['approval_sha256']==g['approval_sha256']==seal['approval_sha256'],'APPROVAL_BINDING')
    verify(seal)
    C.require(not (ns/'STOP.json').exists(),'ALREADY_STOPPED')
    if consume:
        C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id'],UTC=C.utc(),nonrefundable=True))
    else:
        C.require(C.read(ns/'CONSUMED.json')['grant_id']==g['grant_id'],'NOT_CONSUMED')
    return c,seal


def worker():
    ns=BASE/'run01'; c,seal=check(ns)
    C.require((ns/'LAUNCH_CLAIM.json').is_file(),'NOT_LAUNCHED')
    C.save(ns/'WORKER_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    from h4_cng_private.h4_conditional_cng import run
    run(ns,c,C.CallBudget(ns,LIMITS))
    verify(seal)


def launch():
    ns=BASE/'run01';c,seal=check(ns,consume=True)
    C.save(ns/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker']
    # Reuse native subreaper/wait unchanged except the scope of *new* receipts.
    tree=__import__('ast').parse((ROOT/'src/trotterlib/h4_certificate_control.py').read_text())
    node=next(n for n in tree.body if isinstance(n,__import__('ast').FunctionDef) and n.name=='witness')
    for n in __import__('ast').walk(node):
        if isinstance(n,__import__('ast').Constant) and n.value=='NEW_C_WORKER_ONLY':
            n.value='NEW_CNG_SAVED_REANALYSIS_ONLY'
    env=dict(vars(C));exec(compile(__import__('ast').Module(body=[node],type_ignores=[]),'<reused scoped witness>','exec'),env)
    status=env['witness'](ns/'native',command)
    if status==0:
        verify(seal)
        ledger=C.read(ns/'call_ledger.json')
        C.require(ledger['used']==LIMITS,'INCOMPLETE_BATCHES')
    C.save(ns/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP' if status==0 else 'FAILED_NO_RETRY',
        native_exit=C.read(ns/'native/worker_native_exit.json'),all_owned_reaped=C.read(ns/'native/STOP.json')['all_owned_reaped'],
        original_H4_native_verified=False,automatic_retry=False,UTC=C.utc()))
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','worker'):g.add_argument('--'+mode,action='store_true')
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.launch:sys.exit(launch())
    else:worker()
