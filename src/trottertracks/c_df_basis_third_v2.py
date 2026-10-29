"""Metadata-only repair around the frozen third-DF numerical engine.

The component record fields match the existing P4/B1 producer. No law,
probability, objective, numerical gate or native implementation is replaced.
Dedicated process only; temporary engine bindings are restored on every exit.
"""
from contextlib import contextmanager
from pathlib import Path
from . import c_df_basis_third as engine

VERSION='third_DF_existing_components_metadata_v2'
SCOPE='C_DF_BASIS_SWITCH_THIRD_GAMMA0_REPAIR_V2'
FRESH='artifacts/research_c_df_basis_third/2026-10-29/run02'
LIMITS,MAX_CALLS=engine.LIMITS,engine.MAX_CALLS
BASE_REFERENCE=engine.stage_reference
SCIENCE_KEYS=('policy','target','allowances','sampling','compiler','scenarios',
    'stage_call_limits','call_limits','maximum_logical_units','input',
    'original_input_sha256','input_snapshot_sha256','export','export_sha256','fit_SOURCE',
    'records','training_records','training_rows','packages','transfer_models','transfer_policy',
    'transfer','kernel_scope','resource_model','statistics','new_fit','optimizer','rank_search',
    'retry','resume','quantum_shots','new_native_trajectories_max','new_kernel_build_max',
    'new_compile_max','QPY_max','physical_DF_error_recomputation','PF_rounds','signal_model',
    'resources','unknown','future_cost_aware_fit')


def with_components(report,tail):
    """Serialize existing component values; never regenerate/renormalize them."""
    records=[dict(component_id=x.component_id,probability=x.probability,
        coefficient_abs=x.coefficient_abs,coefficient_sign=x.coefficient_sign,
        basis_id=x.basis_id,df_fragment_id=x.df_fragment_id) for x in tail.components]
    if 'components' in report and report['components']!=records:
        raise ValueError('EXISTING_COMPONENT_METADATA_CONFLICT')
    return dict(report,components=records)


def repaired_reference(root,c,out,call):
    """Augment the existing tail_gate result inside that same logical unit."""
    state={}
    def invoke(kind,label,function):
        if kind=='law_prepare':
            def prepare():
                value=function();state['tail']=value[0];return value
            return call(kind,label,prepare)
        if kind=='tail_gate':
            return call(kind,label,lambda:with_components(function(),state['tail']))
        return call(kind,label,function)
    return BASE_REFERENCE(root,c,out,invoke)


@contextmanager
def bindings():
    previous=(engine.VERSION,engine.SCOPE,engine.FRESH,engine.stage_reference)
    try:
        engine.VERSION,engine.SCOPE,engine.FRESH=VERSION,SCOPE,FRESH
        engine.stage_reference=repaired_reference
        yield
    finally:
        engine.VERSION,engine.SCOPE,engine.FRESH,engine.stage_reference=previous


def check_revision(root,path):
    root=Path(root);c=engine.io.read(path)
    if c['implementation']!=VERSION or c['scope']!=SCOPE or c['technical_relaunch']!=1:
        raise ValueError('THIRD_REPAIR_SCOPE')
    old_path=root/c['previous_contract']
    if engine.io.sha(old_path)!=c['previous_contract_sha256']:
        raise ValueError('THIRD_REPAIR_OLD_CONTRACT')
    old=engine.io.read(old_path)
    for key in SCIENCE_KEYS:
        if c[key]!=old[key]:raise ValueError('THIRD_REPAIR_CHANGED_SCIENCE:'+key)
    if any(c['saved_inputs'].get(p)!=h for p,h in old['saved_inputs'].items()):
        raise ValueError('THIRD_REPAIR_CHANGED_INPUTS')
    return c


def preflight(root,path):
    check_revision(root,path)
    with bindings():return engine.preflight(root,path)


def run(root,path):
    check_revision(root,path)
    with bindings():return engine.run(root,path)
