"""Adapt only price-request metadata for the explicit PR-DF max-width mapper.

The fixed circuit preparation, sampling, compilation, and receipt validation
remain unchanged. The one-body endpoint has an empty DF head but retains the
same deterministic one-body orbitals and a nonempty randomized DF tail.
"""
import ast
from pathlib import Path
import subprocess

from . import pr_df_max_width as maximum

PRICE_ADAPTER_SOURCE='e827367629ca6c2920d9d908fc1631314399db86'
PRICE_ADAPTER_PATH='src/trottertracks/candidate_price_adapter.py'


def install(root, adapter, mapper_source):
    if getattr(adapter,'_max_width_price_source',None)==mapper_source:return
    if getattr(adapter,'_max_width_price_source',None):raise ValueError('Different max-width mapper already selected')
    raw=subprocess.check_output(['git','show',PRICE_ADAPTER_SOURCE+':'+PRICE_ADAPTER_PATH],cwd=root).decode()
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='from_b')
    text=ast.get_source_segment(raw,node)
    edits={
        'from . import pr_rpe_rmse_controller as B\n    B._validate(config)':'_validate_maximum(config)',
        "not 1 <= c['prefix'] < rank":"not 0 <= c['prefix'] < rank",
        "one_body is None and s.get('schedule_adapter') == 'A_FIXED_GRID_510FBACD_V1'":
            "one_body is None and s.get('schedule_adapter') == 'A_PR_DF_REQUESTED_MAX_WIDTH_V1' and c['prefix'] > 0",
    }
    for old,new in edits.items():
        if text.count(old)!=1:raise ValueError('Fixed price adapter metadata guard differs')
        text=text.replace(old,new)
    def validate(config):
        if config['spec'].get('A_release')!=mapper_source or not maximum.selected(config):
            raise ValueError('Price candidate differs from selected max-width mapper')
        maximum.validate(Path(root),config)
    namespace=dict(vars(adapter),_validate_maximum=validate)
    exec(compile(text,PRICE_ADAPTER_SOURCE+':'+PRICE_ADAPTER_PATH,'exec'),namespace)
    original=adapter.from_b
    def from_b(config,*args,**kwargs):
        return namespace['from_b'](config,*args,**kwargs) if maximum.selected(config) else original(config,*args,**kwargs)
    adapter.from_b=from_b
    adapter._max_width_price_source=mapper_source
