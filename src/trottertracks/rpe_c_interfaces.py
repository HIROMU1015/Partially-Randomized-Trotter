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
PR_DF_METADATA_SOURCE = 'fefd548ecd56f60d72031b3d385f6f5e6ee95d06'


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

    # Baseline planning needs only C's existing passive prepare(), not its
    # numerical producers or private environment. Keep its implementation exact.
    baseline_path = 'src/trottertracks/baseline_signal_pipeline.py'
    constants = {'B_SOURCE', 'D_SOURCE', 'FORMULAS', 'MODULE', 'NUMERIC_POLICY'}
    tree = ast.parse(R.blob(root, SOURCE, baseline_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    shim.__dict__[target.id] = ast.literal_eval(node.value)
    shim.deepcopy = deepcopy
    selected(SOURCE, baseline_path, ('prepare',), shim.__dict__)

    # C's PR route needs A's metadata planner/recipe only. Exclude every signal
    # producer, matrix builder and numerical provider from this private shim.
    plan_scope = dict(deepcopy=deepcopy, common=R.B.p)
    selected(PR_DF_METADATA_SOURCE, 'src/trottertracks/pr_rpe_signal_plan.py', ('plan',), plan_scope)
    recipe_path = 'src/trottertracks/pr_rpe_signal_provider.py'
    recipe_tree = ast.parse(R.blob(root, PR_DF_METADATA_SOURCE, recipe_path))
    recipe = next(ast.literal_eval(n.value) for n in recipe_tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'RECIPE' for t in n.targets))
    recipe_scope = dict(deepcopy=deepcopy, RECIPE=recipe)
    selected(PR_DF_METADATA_SOURCE, recipe_path, ('numerical_recipe',), recipe_scope)
    api.A = types.SimpleNamespace(metadata=types.SimpleNamespace(plan=plan_scope['plan']),
                                  numerical_recipe=recipe_scope['numerical_recipe'])

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
