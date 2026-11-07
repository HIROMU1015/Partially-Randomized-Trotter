"""Public scientific-input replay without publishing the legacy private pickle.

Only the nine numerical/preparation fields used by this study are exported.
This stores existing arrays and defined RZ/XXPlusYY operations, not a new fit,
Hamiltonian partition, eigensolve, or approval record.
"""
from dataclasses import fields,is_dataclass
import hashlib
import gzip
import importlib
import json
from pathlib import Path
import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit.library import RZGate,XXPlusYYGate
from trotterlib.df_rte_tail import DFBasisRegistry

KEYS=('sector','state','energy','physical_H','preps','spectra','block_matrices','tails','tail_spectra')


def encode_model(c):
    arrays={}
    def enc(v):
        if isinstance(v,np.ndarray):
            key='a'+str(len(arrays));arrays[key]=v
            return {'array':key,'dtype':v.dtype.str,'shape':list(v.shape),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
        if isinstance(v,np.generic):return enc(v.item())
        if isinstance(v,Gate):
            if v.name not in ('rz','xx_plus_yy'):raise ValueError('unexpected scientific basis gate')
            return {'gate':v.name,'params':[float(x) for x in v.params]}
        if isinstance(v,DFBasisRegistry):return {'registry':[enc(d) for d in v._definitions.values()]}
        if is_dataclass(v):return {'dataclass':type(v).__module__+'.'+type(v).__name__,'fields':{f.name:enc(getattr(v,f.name)) for f in fields(v)}}
        if isinstance(v,tuple):return {'tuple':[enc(x) for x in v]}
        if isinstance(v,list):return [enc(x) for x in v]
        if isinstance(v,dict):return {'mapping':[[enc(k),enc(x)] for k,x in v.items()]}
        if isinstance(v,complex):return {'complex':[v.real.hex(),v.imag.hex()]}
        if v is None or isinstance(v,(str,int,float,bool)):return v
        raise TypeError(type(v))
    data=enc({k:c[k] for k in KEYS});text=json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    return text,arrays


def model_identity(c):return hashlib.sha256(encode_model(c)[0].encode()).hexdigest()


def export_model(c,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);text,arrays=encode_model(c)
    (directory/'MODEL.json.gz').write_bytes(gzip.compress(text.encode(),mtime=0));np.savez_compressed(directory/'MODEL_ARRAYS.npz',**arrays)
    return hashlib.sha256(text.encode()).hexdigest()


def load_model(directory):
    directory=Path(directory);text=gzip.decompress((directory/'MODEL.json.gz').read_bytes()).decode() if (directory/'MODEL.json.gz').exists() else (directory/'MODEL.json').read_text();data=json.loads(text)
    with np.load(directory/'MODEL_ARRAYS.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    def dec(v):
        if isinstance(v,list):return [dec(x) for x in v]
        if not isinstance(v,dict):return v
        if 'array' in v:
            a=arrays[v['array']];assert a.dtype.str==v['dtype'] and list(a.shape)==v['shape'] and hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==v['sha256'];return a
        if 'gate' in v:return (RZGate if v['gate']=='rz' else XXPlusYYGate)(*v['params'])
        if 'registry' in v:
            r=DFBasisRegistry();ds=[dec(d) for d in v['registry']];r._definitions={d.basis_id:d for d in ds};return r
        if 'dataclass' in v:
            module,name=v['dataclass'].rsplit('.',1)
            if not module.startswith('trotterlib.'):raise ValueError('unexpected replay class')
            return getattr(importlib.import_module(module),name)(**{k:dec(x) for k,x in v['fields'].items()})
        if 'tuple' in v:return tuple(dec(x) for x in v['tuple'])
        if 'mapping' in v:return {dec(k):dec(x) for k,x in v['mapping']}
        if 'complex' in v:return complex(*map(float.fromhex,v['complex']))
        raise ValueError('unexpected replay object')
    c=dec(data);assert model_identity(c)==hashlib.sha256(text.encode()).hexdigest();return c
