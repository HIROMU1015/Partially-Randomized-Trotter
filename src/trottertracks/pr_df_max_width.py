"""Explicit PR-DF maximum-width adapter; old fixed-grid contracts stay unchanged.

The original A law, B identity assembly, and finite-RTE numerical kernel are
reused. The requested width and exact T/q are mathematical rationals, while
`actual_delta_pf` and the physical recipe bind the binary64 kernel step. The
L_D=0 endpoint retains the one-body matrix in the deterministic head.
"""
import ast
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from . import pr_rpe_rmse as A
from .rpe_scope_map import MAX_WIDTH_MAPPER, exact, encode

B_SOURCE='437ba4acdb82708b6fc98e3cd119a2a734f7cf64'
ADAPTER='A_PR_DF_REQUESTED_MAX_WIDTH_V1'
TOTAL_ADAPTER=ADAPTER+'_TOTAL_DECODER_VIEW'
SCHEMA='PR_RPE_RMSE_V2'


def selected(config):
    return config.get('spec',config).get('schedule_adapter') in (ADAPTER,TOTAL_ADAPTER)


def _load(root,path,names,scope):
    text=subprocess.check_output(['git','show',B_SOURCE+':'+path],cwd=root)
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Fixed B metadata functions absent')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),B_SOURCE+':'+path,'exec'),scope)


@lru_cache(maxsize=8)
def _api(root, source):
    # The selected scientific source is fixed by the controller/worker release,
    # not determined by arbitrary repository HEAD at execution time.
    root=Path(root);common=A.common
    ns=dict(p=common,deepcopy=deepcopy,hashlib=hashlib,json=json,SCHEMA=SCHEMA)
    _load(root,'src/trottertracks/pr_rpe_rmse_controller.py',('_id','comparison_method','_assemble'),ns)
    b=SimpleNamespace(p=common,a=SimpleNamespace(exact_value=encode,exact_scalar=exact),_assemble=ns['_assemble'])
    native=dict(A=A,B=b,p=common,deepcopy=deepcopy,Fraction=Fraction,ADAPTER=ADAPTER,A_RELEASE=source)
    _load(root,'src/trottertracks/pr_rpe_rmse_integration.py',('_exact','_schedule','a_spec','generate_from_a_spec'),native)
    old_spec=native['a_spec']
    def spec(inputs,method='PR'):
        d=inputs['design']
        if (method!='PR' or d['pf_grid_policy']!=MAX_WIDTH_MAPPER or d.get('early_rounds')
                or type(d['prefix']) is not int or not 0<=d['prefix']<d['binding']['rank']
                or d['PF']['method']!='B0' or d['PF']['prefix']!=d['prefix']):
            raise ValueError('Explicit PR-DF max-width, nonempty tail and unaltered deterministic head required')
        s=old_spec(inputs,method)
        s['delta_pf']=encode(d['requested_delta_pf'])
        s['event_mapping']=dict(kind=MAX_WIDTH_MAPPER,source=dict(commit=source,path='src/trottertracks/pr_rpe_rmse.py'),
                                pf_grid='EXACT_CEIL_REQUESTED_MAX_WIDTH_BINARY64_KERNEL_STEP')
        if d['prefix']==0:s['split']['one_body_policy']='DETERMINISTIC'
        return s
    native['a_spec']=spec
    return SimpleNamespace(a_spec=spec,generate=native['generate_from_a_spec'],assemble=b._assemble)


def build(root, base, design, *, tail, tail_source, source):
    args=deepcopy(base['spec']['A_input']);d=args['design'];ld=design['L_D']
    requested=exact(design['delta_pf']) if design.get('delta_pf') is not None else exact(d['base_time'])*exact(design['delta_over_h'])
    d.update(prefix=ld,formula=design.get('formula',d['formula']),K=design.get('K',d['K']),
             delta=float(requested),pf_grid_policy=MAX_WIDTH_MAPPER,requested_delta_pf=encode(requested))
    d['PF']['prefix']=ld
    args['kappa_paper']=str(exact(design['kappa']));args['tail_norm'].update(prefix=ld,lambda_R=tail['lambda_R'],source=deepcopy(tail_source))
    api=_api(str(Path(root).resolve()),source)
    return api.generate(api.a_spec(args))


def generate(root,spec):
    if spec.get('schedule_adapter')!=ADAPTER:raise ValueError('New PR-DF adapter required')
    return _api(str(Path(root).resolve()),spec['A_release']).generate(spec)


def validate(root,config):
    if config!=generate(root,config['spec']):raise ValueError('Max-width configuration differs from its canonical mapper')


def actual_step_matches(row):
    """A rounding exception is legal only for the declared exact ceil grid."""
    detail=row.get('mapping') or {}
    if detail.get('pf_grid_policy')!=MAX_WIDTH_MAPPER:return False
    t=exact(row['T']);step=exact(row['actual_delta_pf']);q=row['q_PF'];w=exact(detail['requested_delta_pf'])
    return (q==max(1,math.ceil(t/w)) and exact(detail['actual_delta_pf_exact'])==t/q
            and step==Fraction(float(t/q)) and detail['actual_delta_binary64']==float(step).hex())


def jobs(config, physical_by_round, *, b_validate,b_exact_scalar,numerical_recipe,saved_inventory=()):
    """The existing signal contract, including the newly supported one-body endpoint."""
    from . import pr_rpe_signal_plan as planner
    return planner.plan(config,physical_by_round,b_validate=b_validate,b_exact_scalar=b_exact_scalar,
                        numerical_recipe=numerical_recipe,saved_inventory=saved_inventory)


def wrap_b(root,old):
    """Only the new configuration invokes the new canonical validator."""
    def check(config):
        return validate(root,config) if selected(config) else old.validate(config)
    return SimpleNamespace(**dict(vars(old),validate=check))


def enable_endpoint(provider):
    """Build the exact same context with prefix 0 and a nonempty DF tail.

    This is the only change to the fixed numerical provider. Its expm head loop
    already includes matrix 0 even at prefix 0; its tail loop starts at 1.
    """
    from types import SimpleNamespace
    import inspect
    import textwrap
    old=provider.make_context
    text=textwrap.dedent(inspect.getsource(old))
    token='not 1<=prefix<binding[\'rank\']'
    if text.count(token)!=1:raise ValueError('Endpoint provider guard differs from fixed source')
    ns=dict(old.__globals__)
    exec(compile(text.replace(token,"not 0<=prefix<binding['rank']"), 'explicit-DF-one-body-endpoint', 'exec'),ns)
    return SimpleNamespace(**dict(vars(provider),make_context=ns['make_context']))


def install_q3(route, views=()):
    """Bind only a dedicated max-width Q3 invocation to the existing B/C engine.

    The fixed total-estimator and statistical code is unchanged. Regenerate the
    new A/B origin and its decoder-only view; point C at the new physical plan.
    The wrapper is installed in the request process, not on frozen disk files.
    """
    total=route.U.T;B=total.B;root=Path(route.U.ROOT)
    if getattr(total,'_max_width_installed',False):return
    old_generate,old_validate=B.generate,B._validate
    total.FIXED_GRID_ORIGIN=ADAPTER;total.FIXED_GRID_TOTAL=TOTAL_ADAPTER
    def gen(spec,*,event_mapper=None):
        adapter=spec.get('schedule_adapter')
        if adapter in (ADAPTER,TOTAL_ADAPTER):
            if event_mapper is not None:raise ValueError('Maximum-width schedule has no external event mapper')
            return generate(root,spec) if adapter==ADAPTER else total.generate_fixed_grid_view(spec)
        return old_generate(spec,event_mapper=event_mapper)
    def check(config):
        if selected(config):
            if config!=gen(config['spec']):raise ValueError('Max-width origin/total configuration mutated')
        else:old_validate(config)
    B.generate=gen;B._validate=check
    for directory in {str(root),*(str(v)for v in views)}:
        pipeline=total.C.interfaces(directory).pipeline
        metadata=pipeline.providers(directory).A.metadata
        old_plan=metadata.plan
        def plan(config,physical_by_round,*,_old=old_plan,**kwargs):
            return jobs(config,physical_by_round,**kwargs) if selected(config) else _old(config,physical_by_round,**kwargs)
        metadata.plan=plan
    total._max_width_installed=True
