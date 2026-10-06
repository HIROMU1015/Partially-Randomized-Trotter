#!/usr/bin/env python3
"""Guarded preparation/tests only. Production runners and workers are forbidden."""
import argparse
import builtins
from datetime import datetime, timezone
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).absolute().parents[2]
BUNDLE = 'artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06'
OUTPUT = '/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run01'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare','tests'), required=True)
    parser.add_argument('--explicit-cpu-list', default='')
    parser.add_argument('--cpu-evidence', default='')
    args = parser.parse_args(argv)
    expected_env = {k:'1' for k in ('PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS',
        'OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS')}
    expected_env['QISKIT_PARALLEL'] = 'false'
    if not all(os.environ.get(k) == v for k,v in expected_env.items()):
        raise RuntimeError('absolute interpreter with process-only single-thread environment required')
    counts = dict(molecular_access=0, molecular_import=0, scientific_processing=0, trajectory_seed=0,
        signal_sampling_build_compile=0, transpile=0, GPU=0, production_runner_launch=0,
        production_worker_launch=0, shared_environment_changes=0, other_job_changes=0)
    attempted = []
    def refuse(scope, value):
        counts[scope] += 1; attempted.append({'scope':scope,'attempt':str(value)})
        raise RuntimeError('zero-science boundary: '+str(value))
    original_import = builtins.__import__
    def guarded_import(name, *a, **kw):
        if name.split('.')[0] in ('pyscf','openfermion','openfermionpyscf','trotterlib','cupy','torch','pynvml'):
            refuse('molecular_import' if name.split('.')[0] not in ('cupy','torch','pynvml') else 'GPU', name)
        return original_import(name,*a,**kw)
    builtins.__import__ = guarded_import
    def protected_path(path):
        if not isinstance(path,(str,bytes,os.PathLike)):
            return False
        text = os.fsdecode(path)
        if text == OUTPUT or text.startswith(OUTPUT+'/'):
            return True
        return text.endswith(('.npz','.npy','.pkl','.pickle','.db','.sqlite','.sqlite3')) or any(
            part in text.split('/') for part in ('.runtime','runtime','checkpoint','checkpoints','cache','caches','registry','registries'))
    def check_path(path):
        if protected_path(path):
            refuse('molecular_access',os.fsdecode(path))
    original_stat, original_lstat, original_resolve = os.stat, os.lstat, Path.resolve
    def stat(path,*a,**kw):
        check_path(path);return original_stat(path,*a,**kw)
    def lstat(path,*a,**kw):
        check_path(path);return original_lstat(path,*a,**kw)
    def resolve(self,*a,**kw):
        check_path(self);return original_resolve(self,*a,**kw)
    os.stat,os.lstat,Path.resolve = stat,lstat,resolve
    def audit_hook(event, arguments):
        if event == 'open':
            check_path(arguments[0])
        if event == 'subprocess.Popen':
            command = arguments[1]
            # Metadata checkout_gate uses Git subprocesses; no other child allowed.
            valid = isinstance(command,(list,tuple)) and len(command)>3 and command[0]=='git' and command[1]=='-C'
            action = command[3] if valid else None
            valid = valid and action in ('show','rev-parse','remote','diff','ls-tree','merge-base')
            if action=='remote':valid=valid and list(command[4:])==['get-url','origin']
            if action=='merge-base':valid=valid and command[4]=='--is-ancestor'
            if not valid:refuse('production_worker_launch',command)
        if event in ('os.system','os.posix_spawn','os.fork','os.forkpty','os.exec'):
            refuse('production_worker_launch',event)
        if event in ('os.kill','os.killpg','os.setpriority','os.sched_setaffinity'):
            refuse('other_job_changes',event)
    sys.addaudithook(audit_hook)
    # Only Qiskit metadata is allowed. Preserve its callable signature for gates.
    import qiskit
    @functools.wraps(qiskit.transpile)
    def no_transpile(*a,**kw):
        refuse('transpile','qiskit.transpile')
    qiskit.transpile = no_transpile
    prefix = 'trottertracks.resource_applicability.h4_geometry.'
    mocked = {}
    for module_name, symbols in {
        'execution':('launch','generation_stage','signal_stage','OwnedRun','_generate_worker','_compile_worker','load_new_input','input_boundary'),
        'workers':('OwnedPool','owned_worker_main','private_dispatch'),
        'inputs':('generate_input','freeze_input','validate_frozen_state'),
        'signal':('prepare','corrected_signal','trajectory_seeds','sample_events','candidate_identity'),
        'circuits':('build_evolution','wrapper','numerical_fingerprint'),
        'parallel':('compile_wrappers',)}.items():
        module = types.ModuleType(prefix+module_name)
        for name in symbols:
            setattr(module,name,Mock(side_effect=lambda *a,_name=module_name+'.'+name,**kw:refuse('scientific_processing',_name)))
        mocked[prefix+module_name] = module
    helper_file = ROOT/'scripts/resource_applicability/prepare_h4_input_generation_authorization.py'
    spec = importlib.util.spec_from_file_location('h4_input_authorization_preparation',helper_file)
    helper = importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    started = time.monotonic()
    with patch.dict(sys.modules,mocked):
        gates,identity,_resources = helper.production_modules()
        with patch.object(identity,'trajectory_seed',side_effect=lambda *a,**kw:refuse('trajectory_seed','trajectory_seed')),\
             patch.object(identity,'step_seed',side_effect=lambda *a,**kw:refuse('trajectory_seed','step_seed')):
            if args.mode=='prepare':
                helper.main(['--explicit-cpu-list',args.explicit_cpu_list,'--cpu-evidence',args.cpu_evidence])
                result_data = {'tests':0,'failures':0,'errors':0,'skipped':0}
            else:
                file=ROOT/'tests/tracks/resource_applicability/test_h4_input_generation_authorization.py'
                test_spec=importlib.util.spec_from_file_location('h4_input_authorization_zero_science_tests',file)
                module=importlib.util.module_from_spec(test_spec);sys.modules[test_spec.name]=module;test_spec.loader.exec_module(module)
                module.PREPARATION=helper
                result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
                result_data={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}
    passed = not attempted and not any(result_data[k] for k in ('failures','errors','skipped'))
    payload={'schema_version':'h4-input-authorization-zero-science-audit-v1','status':'PASS' if passed else 'FAIL',
        'mode':args.mode,'observed_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,
        'python_executable':sys.executable,'python_version':sys.version,'thread_environment':expected_env,
        'command_argv':sys.argv,'counts':counts,'protected_attempts':attempted,**result_data,
        'private_science_and_output_boundaries_mocked':True,'production_source_modified':False,
        'simulated_approval_in_memory_only':args.mode=='tests','saved_review_approved':False,
        'simulated_cpu_list_used_in_memory_only':args.mode=='tests',
        'additional_transpile':0,'preserved_source_series_transpile_cumulative':28,
        'final_review_performed':False,'user_explicit_launch_performed':False,'mandatory_stop':True}
    if args.mode=='prepare':name='preparation_guard_audit_v1.json'
    else:
        attempt=1+len(list((ROOT/BUNDLE).glob('gate-tests-attempt-*.json')))
        name='gate-tests-attempt-%02d.json'%attempt
    helper.write_new(BUNDLE+'/'+name,payload)
    print('H4_INPUT_AUTHORIZATION_ZERO_SCIENCE_AUDIT '+json.dumps(payload,sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
