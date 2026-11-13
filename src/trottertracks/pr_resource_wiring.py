"""Definition loading and metadata binding only, before scientific entry points.

Run in a fresh process for each source tree. Existing fixed loaders and
validators do the work; no physical context, kernel, sampler or compiler runs.
This technical check neither consumes a grant nor reopens a historical STOP.
"""
import ast
import builtins
import dis
import importlib
import importlib.util
import inspect
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from types import CodeType, SimpleNamespace


def check_globals(function):
    """Check LOAD_GLOBAL in actual extracted definitions, including nested code."""
    scope = function.__globals__
    def loaded(code):
        names = {i.argval for i in dis.get_instructions(code) if i.opname == 'LOAD_GLOBAL'}
        for value in code.co_consts:
            if isinstance(value, CodeType): names.update(loaded(value))
        return names
    missing = loaded(function.__code__) - set(scope) - set(vars(builtins))
    if missing:
        raise ValueError(function.__qualname__ + ': unsupplied globals: ' + ','.join(sorted(missing)))


def bind_calls(caller, attribute, api):
    """Bind actual call-site argument names; never invoke the numerical API."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(caller)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == attribute]
    if not calls: raise ValueError('Expected API call missing: ' + attribute)
    for call in calls:
        if any(k.arg is None for k in call.keywords) or any(isinstance(a, ast.Starred) for a in call.args):
            raise ValueError('Explicit API arguments required: ' + attribute)
        inspect.signature(api).bind(*[object() for _ in call.args],
                                    **{k.arg: object() for k in call.keywords})


def context_definition(loader, root, plan):
    """Run the real S-only loader through exec, stop before calling context().

    Only its final call is replaced by returning that very callable. This
    retains the real scope construction (the former missing-ast failure).
    A changed loader boundary fails closed instead of executing a context.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(loader)))
    function = tree.body[0]; last = function.body[-1]
    if not (isinstance(last, ast.Return) and isinstance(last.value, ast.Call)
            and ast.unparse(last.value.func) == "scope['context']"):
        raise ValueError('S-only loader boundary changed')
    function.body[-1] = ast.Return(value=last.value.func)
    scope = dict(loader.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(loader), 'exec'), scope)
    definition = scope[loader.__name__](root, plan, 'H4')
    check_globals(definition)
    inspect.signature(definition).bind(SimpleNamespace(root=Path(root), plan=plan, contexts={}), 'H4')
    return definition


def signal_input_metadata(loader, root, config):
    """Actual H4 loader's JSON/hash/binding prefix; stop before matrix restore."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(loader)))
    function = tree.body[0]
    stops = [i for i,n in enumerate(function.body) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'load_context']
    if len(stops) != 1: raise ValueError('Signal input loader boundary changed')
    function.body = function.body[:stops[0]] + ast.parse('return tails').body
    scope = dict(loader.__globals__)
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(loader), 'exec'), scope)
    return scope[loader.__name__](root, config)


def check_price_request(root, request):
    """Validate scalar meaning as well as hashes, using D's existing reader."""
    from . import ld12_price_requests as L
    L.D.check_request(request)
    prefix = request['physical_context']['physical_candidate']['prefix']
    expected = L.request_for(root, 'PR-LD'+str(prefix), request['m'], axes=request['axes'],
                            execution_source=request['physical_context']['execution_source'])
    if not L.D.p.same(request, expected):
        raise ValueError('Price request/configuration/SOURCE/total scalar differs from fixed inputs')


def check_backend_plan(root, plan):
    from . import p1_efficient_zz as P
    missing = P.backend_plan_keys(root) - set(plan)
    if missing: raise ValueError('Fixed Backend plan lacks keys: '+','.join(sorted(missing)))


def preflight(root, series, source, *, request=None, plan=None):
    root = Path(root).resolve()
    if len(source) != 40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('Full fixed SOURCE commit required')
    if (request is not None or plan is not None) and series != 'D':
        raise ValueError('Price request/plan requires series D')
    checked = []
    def module(name):
        value = importlib.import_module('trottertracks.'+name)
        path = Path(value.__file__).resolve()
        relative = str(path.relative_to(root))
        expected = subprocess.check_output(['git','show',source+':'+relative], cwd=root)
        if path.read_bytes() != expected: raise ValueError('SOURCE bytes differ: '+relative)
        checked.append(relative)
        return value

    # These live shared modules are consumed by every profile. Historical
    # provenance is checked separately; today's imports must match SOURCE too.
    module('pf_pr_rpe_pipeline')
    module('h4_df10_rpe_connection')
    relative = 'scripts/h46_long_time_signal.py'
    if (root/relative).read_bytes() != subprocess.check_output(['git','show',source+':'+relative],cwd=root):
        raise ValueError('SOURCE candidate ID rule differs')

    if series == 'A':
        V = module('pr_rpe_rmse_s0_v2'); provider = module('pr_rpe_signal_provider')
        power = module('pr_rpe_power_mean')
        # The actual fixed-blob definition loader; no spectrum/evolve call.
        kernels = power.kernels(root)
        for function in (kernels.spectrum, kernels.evolve): check_globals(function)
        spec = importlib.util.find_spec('h46_new2_core')
        if spec is None or Path(spec.origin).resolve() != root/'scripts/h46_new2_core.py':
            raise ValueError('Actual loader needs this SOURCE scripts/ import path')
        script = importlib.import_module('h46_new2_core')
        if Path(script.__file__).read_bytes() != subprocess.check_output(
                ['git','show',source+':scripts/h46_new2_core.py'], cwd=root):
            raise ValueError('SOURCE bytes differ: scripts/h46_new2_core.py')
        inspect.signature(script.occupation_one_body).bind(object(), object())
        menu_source = 'a9abf8de75bf3432fcd714e868e5c14c8884b8f8'
        directory = 'docs/research/pr_rpe_first_signal_menu_20261113/'
        store = V.p.SavedArtifacts(root); menu = store.read(dict(commit=menu_source,path=directory+'MENU.json'))
        b = V.b_metadata(root)
        for item in menu['candidates']:
            config = store.read(dict(commit=menu_source,path=directory+item['configuration_file']))
            b.validate(config)
            batch = provider.SignalBatch(config, {r['m']:r['physical_candidate'] for r in config['rounds']},
                b_validate=b.validate, b_exact_scalar=provider._scalar, source=menu['signal_source'], root=root,
                context_factory=lambda c: provider.load_h4_context(root, c),
                reference_provider=lambda j: provider.saved_h4_reference(root, j))
            inspect.signature(batch.factory).bind(config)
            inspect.signature(provider.load_h4_context).bind(root, config)
            inspect.signature(provider.evaluate).bind(object(), batch.pending['jobs'][0], object(), root=root)
            signal_input_metadata(provider.load_h4_context, root, config)
        boundary = 'SignalBatch.run / saved H4 restore / spectrum / evolve: NOT CALLED'
    elif series == 'B':
        H = module('h4_s0_v2_intake')
        result = H.prepare(root)  # existing passive SOURCE/identity/lookup planner
        if result['science_calls'] != 0 or not result['mandatory_stop']:
            raise ValueError('Passive intake boundary changed')
        boundary = 'N/C/G providers: NOT CALLED'
    elif series == 'C':
        C = module('h4_structural_kernel'); loader = module('s0a_structure_context')
        C.check_inputs(root)  # actual fixed-commit configuration checks
        # Reuse the same fixed C1 plan used by the existing S-only loader.
        from . import s0a_saved_structure as saved
        fixed = saved.saved(root, 'PLAN.json')
        context_definition(loader.load_context, root, fixed)
        from . import h4_pr_physical_closure as closure
        import trotterlib.rte as rte
        if Path(rte.__file__).read_bytes() != loader.m.blob(root, loader.m.PROVIDER, 'src/trotterlib/rte.py'):
            raise ValueError('Pinned descriptor API SOURCE differs')
        bind_calls(closure.make_law, 'make_rte_config', rte.make_rte_config)
        boundary = 'S-only context / preparation / law descriptor / mean S: NOT CALLED'
    elif series == 'D':
        P = module('p1_efficient_zz'); D = module('candidate_price_adapter')
        module('ld12_price_requests')
        if plan is None:
            plan = P.execution_plan(P.make_plan(root, 'WIRING_PREFLIGHT_ONLY'), dict(commit=source), root)
        check_backend_plan(root, plan)
        if plan['source'] not in (source, dict(commit=source)):
            raise ValueError('Backend plan SOURCE differs')
        science = P.load_c1_backend(root)['r_price_phase1_science']
        backend = science.Backend(root, plan)  # copies metadata only
        for name in P.USED_BACKEND_METHODS: check_globals(getattr(science.Backend, name))
        inspect.signature(backend.context).bind('H4')
        inspect.signature(backend.price_requests).bind(object(), 0, Path('.'), (), (), (), conditional=False)
        cfg = science.read(plan['runtime']['input_config'])['systems']['H4']
        science.m._binding_matches(cfg['binding'], plan['menus']['H4']['binding'])
        old = json.loads(P._blob(root, science.m.H4V1,
            'docs/research/h4_df10_price_v1_20261110/DEVELOPMENT.json'))
        for name, digest in old['input_sha256'].items():
            item = cfg['inputs'][name]; data = Path(item['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest or item['sha256'] != digest or len(data) != item['bytes']:
                raise ValueError('Fixed Backend input bytes differ: '+name)
        if request is not None: check_price_request(root, request)
        else:
            from . import ld12_price_requests as L
            for label in L.LABELS:
                for m in (0, 12):
                    req = L.request_for(root, label, m, execution_source=dict(commit=source))
                    check_price_request(root, req)
        boundary = 'NativeRuntime.context / restore / sampling / build / compile: NOT CALLED'
    else:
        raise ValueError('Choose A, B, C or D')
    return dict(status='TECHNICAL_WIRING_PASS_NOT_SCIENTIFIC_QUALIFICATION', series=series,
                source=source, checked_modules=checked, boundary=boundary,
                science_calls=0, mandatory_stop=True)
