"""Install before importing the artificial driver dependency closure."""
import importlib.abc
import os
import sys

CORE = frozenset(('g7_generator', 'g7_provider', 'return_aggregation',
                  'g6_f3_contract', 'g6_f3_precision', 'g6_f3_generator'))
ATTEMPTS = []


class Closed(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise Closed(reason)


class BlockCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.', 1)[-1] in CORE:
            ATTEMPTS.append(fullname)
            raise Closed('CORE_LOAD_DENIED:' + fullname)
        return None


def loaded():
    return sorted(n for n in sys.modules if n.rsplit('.', 1)[-1] in CORE)


def install():
    require(not loaded(), 'CORE_ALREADY_LOADED')
    if not any(type(g) is BlockCore for g in sys.meta_path):
        sys.meta_path.insert(0, BlockCore())


def observation():
    install()
    return dict(pid=os.getpid(), core_loaded=loaded(), blocked_attempts=list(ATTEMPTS),
                preload_guard=True, artificial_only=True)
