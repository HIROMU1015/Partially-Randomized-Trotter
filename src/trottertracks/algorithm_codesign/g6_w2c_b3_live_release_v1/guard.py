"""Parent/spawn preload denial and process-local import observation."""
import importlib.abc
import os
import sys
from ..g6_w2c_b3_live_preflight_v1.guard import CORE, Closed, require

ATTEMPTS=[]


class CoreGuard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.rsplit('.',1)[-1] in CORE:
            ATTEMPTS.append(fullname)
            raise Closed('RELEASE_CORE_LOAD_DENIED:'+fullname)
        return None


def install():
    require(not [n for n in sys.modules if n.rsplit('.',1)[-1] in CORE],'CORE_ALREADY_LOADED')
    # Move this observer ahead of historical guards, without removing them.
    existing=next((g for g in sys.meta_path if type(g) is CoreGuard),None)
    if existing is not None:sys.meta_path.remove(existing)
    sys.meta_path.insert(0,existing if existing is not None else CoreGuard())


def observation():
    install()
    return dict(pid=os.getpid(),preload_guard=True,core_loaded=[n for n in sys.modules
        if n.rsplit('.',1)[-1] in CORE],core_import_attempts=list(ATTEMPTS),artificial_only=True)


def science_dispatch(*args,**kwargs):
    raise Closed('LIVE_RELEASE_CANDIDATE_UNAUTHORIZED_NO_ACTIVATION_PATH')
