"""Concrete scientific resolver. Only static_binding is used in preparation.

resolve is implemented but never called by artificial tests. Every import is
after the closed activation gate. Scientific files are read as text/AST only.
"""
import ast
import importlib
import inspect
from pathlib import Path
from hashlib import sha256
from dataclasses import dataclass
from .guard import CORE,require
from .gate import require_live
from ..g6_w2c_b3_live_preflight_v1 import spec as S

MODULES=tuple(sorted(CORE))+('g6_w2c_cost_preparation',)
EXTRA={'g7_generator':['DeterministicBits.__init__','_law'],
       'g6_f3_generator':['_state','GeneratorSession.status'],
       'return_aggregation':['dyadic_index'],
       'g6_w2c_cost_preparation':['canonical_state']}


def shape(node):
    a=node.args
    positional=a.posonlyargs+a.args; required=len(positional)-len(a.defaults)
    rows=[(p.arg,'POSITIONAL_ONLY' if i<len(a.posonlyargs) else 'POSITIONAL_OR_KEYWORD',i>=required)
          for i,p in enumerate(positional)]
    if a.vararg:rows.append((a.vararg.arg,'VAR_POSITIONAL',False))
    rows.extend((p.arg,'KEYWORD_ONLY',d is not None) for p,d in zip(a.kwonlyargs,a.kw_defaults))
    if a.kwarg:rows.append((a.kwarg.arg,'VAR_KEYWORD',False))
    return [list(x) for x in rows]


def static_binding(root):
    root=Path(root).resolve();symbols={};files={}
    for module in MODULES:
        path=S.SOURCE+module+'.py';raw=(root/path).read_bytes()
        files[module]=dict(path=path,sha256=sha256(raw).hexdigest())
        wanted=set(S.API.get(module,()))|set(EXTRA.get(module,()))
        def visit(body,prefix=''):
            for node in body:
                if isinstance(node,ast.ClassDef):visit(node.body,prefix+node.name+'.')
                elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and prefix+node.name in wanted:
                    symbols[module+'.'+prefix+node.name]=dict(path=path,qualname=prefix+node.name,
                        arguments=ast.unparse(node.args),shape=shape(node),line=node.lineno)
        visit(ast.parse(raw,filename=path).body)
        require(all(module+'.'+n in symbols for n in wanted),'MISSING_RESOLVER_SYMBOL:'+module)
    tree=ast.parse((root/S.SOURCE/'g6_f3_contract.py').read_bytes())
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('SPEC_COMMIT','PROOF_COMMIT','G7_COMMIT','POLICY','OLD_PINS','F3_FILES')}
    return dict(files=files,symbols=symbols,f3_constants=constants,science_reference=S.SCIENCE,
        compatibility='NOT_ESTABLISHED_NO_LIVE_IMPORT',loader_called=False)


def bind_loaded(modules,binding,root):
    """Shared location/signature verifier; fixture tests use non-science modules."""
    root=Path(root).resolve()
    for name,pin in binding['files'].items():
        module=modules[name];expected=root/pin['path']
        require(Path(module.__file__).resolve()==expected,'RESOLVER_LOCATION:'+name)
        require(sha256(expected.read_bytes()).hexdigest()==pin['sha256'],'RESOLVER_HASH:'+name)
    for qualified,spec in binding['symbols'].items():
        module,_,symbol=qualified.partition('.');obj=modules[module]
        for part in symbol.split('.'):
            obj=inspect.getattr_static(obj,part)
            if isinstance(obj,(classmethod,staticmethod)):obj=obj.__func__
        require(callable(obj) and Path(obj.__code__.co_filename).resolve()==root/spec['path']
                and obj.__qualname__==spec['qualname'],'RESOLVER_CALLABLE_SOURCE:'+qualified)
        actual=[[p.name,p.kind.name,p.default is not inspect.Parameter.empty]
                for p in inspect.signature(obj,follow_wrapped=False).parameters.values()]
        require(actual==spec['shape'],'RESOLVER_SIGNATURE:'+qualified)
    return True


@dataclass(frozen=True)
class ScientificSymbols:
    g7: object
    f3: object
    precision: object
    audit: object
    dyadic_index: object
    permit: object


_RESOLVED={}


def resolve(c,m,permit):
    require_live(permit,c,m)
    key=S.digest([c,m])
    if key in _RESOLVED:
        symbols,modules=_RESOLVED[key]
        bind_loaded(modules,c['final_runtime_binding'],c['runtime_root'])
        from .wrapper import source_checkpoint
        source_checkpoint(c)
        return symbols
    from .gate import assert_spent
    assert_spent(permit,c,m)
    from .contract import preflight_runtime
    preflight_runtime(c,m,permit)
    # Only these project-owned preparation guards are removed, in this admitted
    # process, immediately before imports. This function is not called today.
    import sys
    guard_modules={
        'trottertracks.algorithm_codesign.'+n for n in (
        'g6_w2c_b3_final_live_v1.guard','g6_w2c_b3_live_release_v1.guard',
        'g6_w2c_b3_live_preflight_v1.guard','g6_w2c_b3_real_adapter_v1.guard',
        'g6_w2c_b3_api_binding_v1.ports','g6_w2c_b3_parallel_scheduler')}
    sys.meta_path[:]=[g for g in sys.meta_path if not (
        type(g).__module__ in guard_modules and type(g).__name__ in ('CoreGuard','BlockScience','BlockCore'))]
    modules={name:importlib.import_module('trottertracks.algorithm_codesign.'+name) for name in MODULES}
    bind_loaded(modules,c['final_runtime_binding'],c['runtime_root'])
    symbols=ScientificSymbols(modules['g7_generator'],modules['g6_f3_generator'],
        modules['g6_f3_precision'],modules['g6_w2c_cost_preparation'],
        modules['return_aggregation'].dyadic_index,permit)
    _RESOLVED[key]=(symbols,modules)
    return symbols
