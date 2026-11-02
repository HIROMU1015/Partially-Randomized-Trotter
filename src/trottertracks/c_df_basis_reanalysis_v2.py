"""Task-local v2 binding: old science engine/statistics/source remain intact."""
from contextlib import contextmanager
from pathlib import Path
from . import c_df_basis_reanalysis as engine
from . import c_df_basis_surrogate_v2 as features

SCOPE='C_DF_BASIS_NATIVE_SAVED64_REPAIR_V2'
FRESH='artifacts/research_c_df_basis_surrogate/2026-10-29/run02'
LIMITS,MAX_CALLS=engine.LIMITS,engine.MAX_CALLS


@contextmanager
def bindings():
    """Single dedicated process only; restore old bindings even on failure."""
    previous=(engine.features,engine.SCOPE,engine.FRESH)
    try:
        engine.features,engine.SCOPE,engine.FRESH=features,SCOPE,FRESH
        yield
    finally:engine.features,engine.SCOPE,engine.FRESH=previous


def check_revision(root,path):
    c=engine.read(path)
    if c['implementation']!=features.VERSION or c['technical_relaunch']!=1:
        raise ValueError('BASIS_REPAIR_REVISION')
    if engine.sha(Path(root)/c['previous_contract'])!=c['previous_contract_sha256']:
        raise ValueError('BASIS_REPAIR_PREVIOUS_CONTRACT')
    old=engine.read(Path(root)/c['previous_contract'])
    # Scientific hypotheses, data, feature set and statistical rules unchanged.
    for key in ('cohorts','models','folds','descriptive_plan','features','numerical_checks',
                'DF_clusters','samples_per_cohort','event_count','call_limits','maximum_logical_units','packages'):
        if c[key]!=old[key]:raise ValueError('BASIS_REPAIR_CHANGED_SCIENCE:'+key)
    return c


def preflight(root,path):
    check_revision(root,path)
    with bindings():return engine.preflight(root,path)


def run(root,path):
    check_revision(root,path)
    with bindings():return engine.run(root,path)
