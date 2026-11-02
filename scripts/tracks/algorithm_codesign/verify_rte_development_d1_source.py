#!/usr/bin/env python3
"""Off-domain tests and runtime lock only. No registered acquisition."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))


class Recorder(unittest.TextTestResult):
    def __init__(self,*a,**k):super().__init__(*a,**k);self.rows=[]
    def addSuccess(self,t):super().addSuccess(t);self.rows.append({'test':t.id(),'status':'PASS'})
    def addFailure(self,t,e):super().addFailure(t,e);self.rows.append({'test':t.id(),'status':'FAIL'})
    def addError(self,t,e):super().addError(t,e);self.rows.append({'test':t.id(),'status':'ERROR'})


def main(output):
    from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime
    from trottertracks.algorithm_codesign.rte_development.evidence import digest
    c=json.loads((ROOT/'artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json').read_text())
    runtime=verify_runtime(ROOT,c)
    if digest(Path(sys.executable).resolve())!=c['runtime_executable_sha256']:raise PermissionError('runtime mismatch')
    path=ROOT/'tests/tracks/algorithm_codesign/test_rte_development_d1.py'
    spec=importlib.util.spec_from_file_location('focused_rte_development',path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from trottertracks.algorithm_codesign import g9_matrix,g9_native
    from trottertracks.algorithm_codesign.rte_reallocation import numeric
    denied=lambda *a,**k:(_ for _ in ()).throw(AssertionError('registered/science acquisition forbidden in preparation'))
    start=time.monotonic();cpu=time.process_time()
    with patch.object(g9_matrix,'Q_matrices',denied),patch.object(g9_native,'cts_events',denied),patch.object(numeric,'synthesize',denied):
        result=unittest.TextTestRunner(verbosity=1,resultclass=Recorder).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    report={'kind':'D1_OFF_DOMAIN_FOCUSED_VALIDATION','fixture':'2-label p=(1/3,2/3), x=1/7, generic X/Z 2x2; r=3 product check; invented bookkeeping/mock inputs',
        'tests':result.rows,'executed':result.testsRun,'PASS':sum(r['status']=='PASS' for r in result.rows),
        'failures':len(result.failures),'errors':len(result.errors),'runtime':runtime,
        'test_source_sha256':digest(path),'python':sys.version,'numeric_threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,
        'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'registered_provider_generated':0,'registered_Q_T_U':0,'registered_events_IR':0,
        'native_compile':0,'synthesis':0,'sampling':0,'trajectory':0,'LP':0,'molecule_NPZ_GPU':0,
        'runtime_identity_match':True,'mandatory_STOP':True,'science_execution_authorized':False}
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();sys.exit(main(a.output))
