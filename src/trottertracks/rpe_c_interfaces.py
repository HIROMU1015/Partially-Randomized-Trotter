"""SOURCE-pinned C interfaces bound to B's live modules, without C providers.

C's planner, physical-ID bridge, input checker and paper accounting are loaded
unchanged. Only dependency wiring is substituted: the numerical/provider loader
is never needed to call these metadata and supplied-value interfaces.
"""
import ast
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import sys
import types

from . import rpe_price_implementation_guard as R
from . import rpe_shot_allocation as W
from . import pr_rpe_rmse as A

SOURCE = R.GUARD['commit']


@lru_cache(maxsize=None)
def interfaces(root):
    root = str(root); guard = R.guard(root); name = guard.__package__
    package = sys.modules[name]; shim = package.baseline_signal_pipeline
    api = shim.published_api(root)
    api.W = W; api.C = shim
    # C.paper_nominal_cost obtains the existing nominal function from B's
    # package rather than implementing the formula again.
    setattr(sys.modules[R.B.__package__], 'pr_rpe_rmse', A)

    def register(short):
        module = types.ModuleType(name + '.' + short); module.__package__ = name
        module.__file__ = str(Path(root)/'src/trottertracks'/(short+'.py'))
        sys.modules[module.__name__] = module; setattr(package, short, module)
        return module

    def selected(commit, path, names, scope):
        tree = ast.parse(R.blob(root, commit, path))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise ValueError('C pinned interface unavailable')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     commit+':'+path, 'exec'), scope)

    loader = register('r_price_phase1_science')
    loader.__dict__.update(ast=ast, m=types.SimpleNamespace(blob=R.blob))
    selected(SOURCE, 'src/trottertracks/r_price_phase1_science.py', ('functions',), loader.__dict__)
    for short in ('rpe_optimization_pipeline', 'paper_nominal_cost'):
        module = register(short); path = 'src/trottertracks/'+short+'.py'
        exec(compile(R.blob(root, SOURCE, path), SOURCE+':'+path, 'exec'), module.__dict__)
        if short == 'rpe_optimization_pipeline':
            pipeline = module
            # Explicit live-B dependency adapter, not a second C planner.
            # Fixed providers() is retained for provenance but is not invoked.
            pipeline.pinned_providers = pipeline.providers
            def live_api(value):
                if str(value) != root:
                    raise ValueError('C interfaces bound to another root')
                return api
            pipeline.providers = live_api
    return types.SimpleNamespace(pipeline=pipeline, paper=package.paper_nominal_cost,
                                 guard=guard, source=SOURCE)
