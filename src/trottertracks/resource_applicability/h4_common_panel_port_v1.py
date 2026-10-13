"""Fixed H4 input adapter for the frozen H6 native-cost implementation.

Reuse the existing numerical and compiler functions, changing only the loader
boundary and model binding. Function globals are copied, never patched in the
old module. H6 scientific/runtime source bytes remain unchanged.
"""
import json
from pathlib import Path
import types
from copy import deepcopy
from .h48_preflight_contract_v1 import PREP,task_caps
from .ax2b_h4_contract_v5 import SNAPSHOT,SNAPSHOT_SHA,snapshot_binding,file_hash

def h4_parent(root):
    root=Path(root)
    snap=root/PREP/'h4_snapshot_receipt.json'
    df=root/PREP/'h4_df_receipt.json'
    return dict(snapshot_path=SNAPSHOT,snapshot_receipt_path=str(snap.relative_to(root)),
        df_receipt_path=str(df.relative_to(root)),snapshot_sha256=file_hash(root/SNAPSHOT),
        snapshot_receipt_sha256=file_hash(snap),df_receipt_sha256=file_hash(df))

def h4_loader(path,snapshot_receipt,df_receipt):
    from trotterlib.pr2_new_series_validation import _load_snapshot_once
    if file_hash(path)!=SNAPSHOT_SHA or snapshot_receipt['sha256']!=SNAPSHOT_SHA or df_receipt['sha256']!=SNAPSHOT_SHA:
        raise ValueError('H4_INPUT_BYTE_BINDING')
    result=_load_snapshot_once(path)
    ham,sector,full,state,metadata,layout=result
    if (ham.n_qubits,ham.n_blocks,sector.dimension)!=(8,12,36) or metadata!=snapshot_receipt['metadata']:
        raise ValueError('H4_FIXED_MODEL_BINDING')
    return result

def bound_function(function,changes):
    namespace=dict(function.__globals__)
    if not set(changes)<=set(namespace):raise ValueError('UNKNOWN_ADAPTER_BOUNDARY')
    namespace.update(changes)
    return types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)

def signal_port(root,manifest,writer,progress):
    from . import h6_matched_port_v1 as frozen
    class H4Port(frozen.MatchedPort):pass
    H4Port.setup=bound_function(frozen.MatchedPort.setup,
                    {'verify_parent':h4_parent,'load_h6_snapshot':h4_loader})
    port=H4Port(root,manifest,writer,progress)
    port.setup();port.correctness()
    writer.write('signal_result.json',{'signal_count':port.completed,
        'calls_attempted':port.calls.used,'state_or_hamiltonian_regenerated':False,
        'H8_full_campaign_authorized':False,'population_mean_certified':False})

def cost_port(root,manifest,task,writer,progress):
    from . import h6_matched_port_v1 as frozen
    from .h6_hash_reuse_v1 import FingerprintReuse
    cache=FingerprintReuse()
    local=deepcopy(manifest)
    local['plan']['caps_proposed']=task_caps(task,manifest['plan']['caps_proposed'])
    f=bound_function(frozen.cost_task,{'load_h6_snapshot':h4_loader,
                                     'canonical_qiskit_circuit_fingerprint':cache.fingerprint})
    try:
        f(Path(root),local,task,writer,progress)
        writer.write('hash_reuse_receipt.json',cache.snapshot())
    finally:cache.clear()
