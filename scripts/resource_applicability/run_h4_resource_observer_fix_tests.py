#!/usr/bin/env python3
"""Guarded zero-science observer/gate tests and read-only preparation actions."""
import argparse
import builtins
from datetime import datetime,timezone
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).absolute().parents[2]
SOURCE_BUNDLE='artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06'
AUTH_BUNDLE='artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2'
OUTPUT='/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run01'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('observer-tests','binding-tests','source-audit','observe','draft'),required=True)
    parser.add_argument('--source-commit');parser.add_argument('--output')
    args=parser.parse_args(argv)
    expected={k:'1' for k in ('PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS',
        'OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS')}
    expected['QISKIT_PARALLEL']='false'
    if not all(os.environ.get(k)==v for k,v in expected.items()):raise RuntimeError('thread/interpreter environment')
    counts=dict(molecular_access=0,molecular_import=0,science_processing=0,seed_generation=0,transpile=0,
        GPU=0,production_runner_worker_launch=0,shared_environment_changes=0,other_job_changes=0)
    attempted=[]
    def refuse(scope,value):
        counts[scope]+=1;attempted.append({'scope':scope,'attempt':str(value)})
        raise RuntimeError('zero-science boundary '+str(value))
    original_import=builtins.__import__
    def guarded_import(name,*a,**kw):
        top=name.split('.')[0]
        if top in ('pyscf','openfermion','openfermionpyscf','trotterlib'):refuse('molecular_import',name)
        if top in ('cupy','pynvml','torch'):refuse('GPU',name)
        return original_import(name,*a,**kw)
    builtins.__import__=guarded_import
    def check_path(path):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        text=os.fsdecode(path)
        if text==OUTPUT or text.startswith(OUTPUT+'/') or text.endswith(('.npz','.npy','.pkl','.pickle','.db','.sqlite','.sqlite3')) or any(
                part in text.split('/') for part in ('.runtime','runtime','checkpoint','checkpoints','cache','caches','registry','registries')):
            refuse('molecular_access',text)
    stat,lstat,resolve=os.stat,os.lstat,Path.resolve
    def guarded_stat(path,*a,**kw):check_path(path);return stat(path,*a,**kw)
    def guarded_lstat(path,*a,**kw):check_path(path);return lstat(path,*a,**kw)
    def guarded_resolve(self,*a,**kw):check_path(self);return resolve(self,*a,**kw)
    os.stat,os.lstat,Path.resolve=guarded_stat,guarded_lstat,guarded_resolve
    def hook(event,arguments):
        if event=='open':check_path(arguments[0])
        if event=='subprocess.Popen':
            command=arguments[1]
            good=isinstance(command,(list,tuple)) and len(command)>3 and command[0]=='git' and command[1]=='-C'
            action=command[3] if good else None
            good=good and action in ('show','rev-parse','diff','ls-tree','merge-base')
            if action=='merge-base':good=good and command[4]=='--is-ancestor'
            if not good:refuse('production_runner_worker_launch',command)
        if event in ('os.system','os.posix_spawn','os.fork','os.forkpty','os.exec'):
            refuse('production_runner_worker_launch',event)
        if event in ('os.kill','os.killpg','os.setpriority','os.sched_setaffinity'):
            refuse('other_job_changes',event)
    sys.addaudithook(hook)
    import qiskit
    @functools.wraps(qiskit.transpile)
    def no_transpile(*a,**kw):refuse('transpile','qiskit.transpile')
    qiskit.transpile=no_transpile
    prefix='trottertracks.resource_applicability.h4_geometry.'
    mocked={}
    for name,symbols in {
        'execution':('launch','generation_stage','signal_stage','OwnedRun','_generate_worker','_compile_worker','input_boundary','load_new_input'),
        'workers':('OwnedPool','owned_worker_main','private_dispatch'),
        'inputs':('generate_input','freeze_input'),
        'signal':('prepare','corrected_signal','trajectory_seeds','sample_events'),
        'circuits':('build_evolution','wrapper','numerical_fingerprint'),
        'parallel':('compile_wrappers',)}.items():
        module=types.ModuleType(prefix+name)
        for symbol in symbols:setattr(module,symbol,Mock(side_effect=lambda *a,_name=name+'.'+symbol,**kw:refuse('science_processing',_name)))
        mocked[prefix+name]=module
    file=ROOT/'scripts/resource_applicability/prepare_h4_resource_observer_fix.py'
    spec=importlib.util.spec_from_file_location('h4_resource_observer_preparation',file)
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper
    started=time.monotonic();exit_code=0
    with patch.dict(sys.modules,mocked):
        spec.loader.exec_module(helper)
        with patch.object(helper.identity,'trajectory_seed',side_effect=lambda *a,**kw:refuse('seed_generation','trajectory_seed')),\
             patch.object(helper.identity,'step_seed',side_effect=lambda *a,**kw:refuse('seed_generation','step_seed')):
            if args.mode.endswith('-tests'):
                file=ROOT/'tests/tracks/resource_applicability/test_h4_resource_observer_fix.py'
                test_spec=importlib.util.spec_from_file_location('h4_resource_observer_zero_science_tests',file)
                module=importlib.util.module_from_spec(test_spec);sys.modules[test_spec.name]=module;test_spec.loader.exec_module(module)
                module.HELPER=helper
                cls=module.ObserverTests if args.mode=='observer-tests' else module.BindingTests
                result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
                result_data={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}
            else:
                command=['--mode',args.mode]
                if args.source_commit:command+=['--source-commit',args.source_commit]
                if args.output:command+=['--output',args.output]
                exit_code=helper.main(command)
                result_data={'tests':0,'failures':0,'errors':0,'skipped':0}
    success=not attempted and not any(result_data[k] for k in ('failures','errors','skipped')) and exit_code==0
    payload={'schema_version':'h4-resource-fix-zero-science-audit-v1','status':'PASS' if success else 'FAIL_OR_BLOCKED',
        'mode':args.mode,'observed_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,
        'python_executable':sys.executable,'python_version':sys.version,'thread_environment':expected,
        'command_argv':sys.argv,'counts':counts,'protected_attempts':attempted,**result_data,
        'metadata_only_positive_approval':args.mode=='binding-tests','simulated_approved_review_saved':False,
        'simulated_cpu_permission_saved':False,'private_science_output_worker_boundary_mocked':True,
        'production_runner_worker_called':False,'additional_transpile':0,'source_series_transpile_cumulative':28,
        'saved_allowed_cpus':[],'saved_review_approved':False,'execution_ready':False,'mandatory_stop':True}
    bundle=AUTH_BUNDLE if args.mode in ('binding-tests','draft') else SOURCE_BUNDLE
    attempt=1+len(list((ROOT/bundle).glob('guard-'+args.mode+'-attempt-*.json')))
    helper.write_new(bundle+'/guard-'+args.mode+'-attempt-%02d.json'%attempt,payload)
    print('H4_RESOURCE_FIX_ZERO_SCIENCE_AUDIT '+json.dumps(payload,sort_keys=True))
    return 0 if success else 1


if __name__=='__main__':sys.exit(main())
