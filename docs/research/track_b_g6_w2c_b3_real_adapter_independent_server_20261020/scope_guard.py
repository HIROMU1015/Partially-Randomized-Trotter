"""Preload guard for the six modules still prohibited by the user's amendment."""
import atexit
import importlib.abc
import json
import os
from pathlib import Path
import sys

CORE={'g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision','g6_f3_generator'}
ATTEMPTS=[]
def loaded():return sorted(n for n in sys.modules if n.rsplit('.',1)[-1] in CORE)

def snapshot(stage):
    directory=os.environ.get('B3_INDEPENDENT_IMPORT_LOGS')
    if directory:
        p=Path(directory);p.mkdir(parents=True,exist_ok=True)
        (p/(str(os.getpid())+'.json')).write_text(json.dumps(dict(pid=os.getpid(),stage=stage,
            guarded=sorted(CORE),blocked_attempts=ATTEMPTS,core_scientific_modules_loaded=loaded(),
            allowed_driver_helper_loaded=[n for n in sys.modules if n.rsplit('.',1)[-1] in ('g6_w2c_b3_driver','g6_w2c_cost_preparation')]),indent=2)+'\n')

class CoreGuard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.rsplit('.',1)[-1] in CORE:
            ATTEMPTS.append(fullname);snapshot('DENIED_BEFORE_LOAD')
            raise RuntimeError('CORE_SCIENCE_DEPENDENCY_STOP:'+fullname)

assert loaded()==[]
sys.meta_path.insert(0,CoreGuard())
snapshot('INSTALLED_BEFORE_SUBJECT_IMPORTS')
atexit.register(lambda:snapshot('PROCESS_EXIT'))
