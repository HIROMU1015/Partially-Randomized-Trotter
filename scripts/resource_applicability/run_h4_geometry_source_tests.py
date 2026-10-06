#!/usr/bin/env python3
"""Dedicated synthetic suite; never invokes either production runner."""
import argparse
import builtins
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest
import functools
import hashlib

ROOT=Path(__file__).absolute().parents[2]
BUNDLE='artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06'
PRIOR_BUNDLE='artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06'
sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',required=True)
    args=parser.parse_args(argv)
    audit=Path(args.audit_dir).absolute()
    from trottertracks.resource_applicability.h4_geometry import gates
    from trottertracks.resource_applicability.h4_geometry.identity import require
    require(audit==ROOT/BUNDLE,'new parallel-source audit scope only')
    prior_raw=(ROOT/PRIOR_BUNDLE/'synthetic_transpile_reservations.jsonl').read_bytes()
    require(hashlib.sha256(prior_raw).hexdigest()=='1d16b9f10840316eefbb64270ef8d5f3f054d98f186db716002269264cb9d9f6','old reservation ledger identity')
    prior_rows=[json.loads(line) for line in prior_raw.splitlines()]
    require(len(prior_rows)==25 and [r['invocation'] for r in prior_rows]==list(range(1,26)),'old25 invocations')
    prior_audit=(ROOT/PRIOR_BUNDLE/'source_freeze_v1.json').read_bytes()
    require(hashlib.sha256(prior_audit).hexdigest()=='8c7d68c2f50753e22152079206e6d9a8e6e174a87f4b940b4cfed218c92e5eef','old source audit identity')
    require(all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()),'single-process test environment')
    audit.mkdir(parents=True,exist_ok=True)
    counts={'molecular_access':0,'molecular_import':0,'actual_science_transpile':0,
            'synthetic_transpile_this_attempt':0,'synthetic_operator_checks':0}
    protected=('pyscf','openfermion','openfermionpyscf','trotterlib')
    attempts=[]
    original_import=builtins.__import__
    def guarded_import(name,*a,**kw):
        if name.split('.')[0] in protected:
            counts['molecular_import']+=1;attempts.append(name)
            raise RuntimeError('forbidden molecular/legacy import')
        return original_import(name,*a,**kw)
    builtins.__import__=guarded_import
    original_stat,original_lstat,original_resolve=os.stat,os.lstat,Path.resolve
    def protected_path(path):
        if not isinstance(path,(str,bytes,os.PathLike)):
            return False
        p=os.fsdecode(path)
        if p.startswith('/tmp/'):
            return False
        return p.endswith(('.npz','.npy','.sqlite','.sqlite3','.db','.pkl','.pickle')) or any(x in p.split('/') for x in ('.runtime','runtime','checkpoint','checkpoints','cache','caches'))
    def check_path(path):
        if protected_path(path):
            counts['molecular_access']+=1;attempts.append(os.fsdecode(path));raise RuntimeError('forbidden scientific path')
    def audit_hook(event,args):
        if event=='open':
            check_path(args[0])
        if event in ('subprocess.Popen','os.system'):
            raise RuntimeError('no subprocess or production launch in synthetic suite')
    sys.addaudithook(audit_hook)
    def stat(path,*a,**kw):
        check_path(path);return original_stat(path,*a,**kw)
    def lstat(path,*a,**kw):
        check_path(path);return original_lstat(path,*a,**kw)
    def resolve(self,*a,**kw):
        check_path(self);return original_resolve(self,*a,**kw)
    os.stat,os.lstat,Path.resolve=stat,lstat,resolve
    import qiskit
    original_transpile=qiskit.transpile
    invocation_file=audit/'synthetic_transpile_reservations.jsonl'
    @functools.wraps(original_transpile)
    def counted(circuit,*a,**kw):
        require(not a and circuit.num_qubits<=4 and len(circuit.data)<=500,'small synthetic circuit only')
        require(kw.get('num_processes')==1,'synthetic compiler one process')
        prior=invocation_file.read_text().splitlines() if invocation_file.exists() else []
        require(25+len(prior)<64,'old25 plus new cumulative synthetic transpile cap64 before invocation')
        record={'invocation':len(prior)+1,'source_series_cumulative':25+len(prior)+1,
                'scope':'SYNTHETIC_ONLY','qubits':circuit.num_qubits,'pid':os.getpid()}
        with invocation_file.open('a') as f:
            f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        counts['synthetic_transpile_this_attempt']+=1
        return original_transpile(circuit,*a,**kw)
    qiskit.transpile=counted
    file=ROOT/'tests/tracks/resource_applicability/test_h4_geometry_source.py'
    spec=importlib.util.spec_from_file_location('h4_geometry_source_synthetic_tests',file)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.COUNTS=counts
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    total=len(invocation_file.read_text().splitlines()) if invocation_file.exists() else 0
    payload={'status':'PASS' if result.wasSuccessful() and not attempts else 'FAIL','tests':result.testsRun,
             'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'synthetic_transpile_prior':25,'synthetic_transpile_new_total':total,
             'prior_reservation_ledger_sha256':hashlib.sha256(prior_raw).hexdigest(),
             'synthetic_transpile_cumulative':25+total,'wall_seconds':time.monotonic()-start,
             'python_executable':sys.executable,'thread_environment':gates.THREAD_ENV,'protected_attempts':attempts,**counts}
    print('H4_SOURCE_SYNTHETIC_AUDIT '+json.dumps(payload,sort_keys=True))
    # One new immutable attempt result per run; prior failures are retained.
    attempt=len(list(audit.glob('test-attempt-*.json')))+1
    with (audit/('test-attempt-%02d.json'%attempt)).open('x') as f:
        json.dump(payload,f,indent=2,sort_keys=True);f.write('\n')
    return 0 if payload['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(main())
