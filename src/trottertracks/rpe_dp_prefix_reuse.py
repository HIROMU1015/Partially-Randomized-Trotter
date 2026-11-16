"""Opt-in exact prefix checkpoints around the fixed total-state accelerator.

The original kernel and accelerator AST guards still execute. Only the outer
round-loop entry/exit is instrumented; transition, pruning, residual and final
moment equations are not replaced. No cache can change the evaluation scope.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

SCHEMA='RPE_TOTAL_DP_PREFIX_V1'
# Filled from the fixed accelerator method, not from the user's approval.
ACCELERATOR_SOURCE_SHA256='d8bf773bf699ab84f3ec2d00216d15f692b9825c3a453346827bfcf79d8b9286'


def _encode(x):
    if x is None or isinstance(x,(str,bool)):return x
    if type(x)is int:return {'int':hex(x)}
    if type(x)is float:return {'float':x.hex()}
    if isinstance(x,Fraction):return {'fraction':[hex(x.numerator),hex(x.denominator)]}
    if isinstance(x,tuple)and hasattr(x,'numerator')and hasattr(x,'denominator'):
        return {'mass':[hex(x.numerator),hex(x.denominator)]}
    if isinstance(x,(list,tuple)):return [_encode(v)for v in x]
    if isinstance(x,dict):return {k:_encode(v)for k,v in x.items()}
    raise TypeError('Unsupported cache value '+str(type(x)))


def _decode(x,mass_type):
    if isinstance(x,list):return [_decode(v,mass_type)for v in x]
    if not isinstance(x,dict):return x
    if set(x)=={'int'}:return int(x['int'],16)
    if set(x)=={'float'}:return float.fromhex(x['float'])
    if set(x)=={'fraction'}:return Fraction(*(int(v,16)for v in x['fraction']))
    if set(x)=={'mass'}:return mass_type(*(int(v,16)for v in x['mass']))
    return {k:_decode(v,mass_type)for k,v in x.items()}


def _bytes(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _digest(value):return hashlib.sha256(_bytes(_encode(value))).hexdigest()


class PrefixStore:
    """Owned-run cache; atomic JSON snapshots, no executable deserialization."""
    def __init__(self,directory=None):
        self.directory=None if directory is None else Path(directory)
        if self.directory is not None:self.directory.mkdir(parents=True,exist_ok=True)
        self.memory={}  # memory mode only; disk mode does not retain duplicate states

    def get(self,key,mass_type):
        if self.directory is None:return deepcopy(self.memory.get(key))
        p=self.directory/(key+'.json')
        if not p.exists():return None
        envelope=json.loads(p.read_bytes());body=envelope['body']
        if envelope['sha256']!=hashlib.sha256(_bytes(body)).hexdigest():raise ValueError('Cache payload checksum mismatch')
        if body['schema']!=SCHEMA or body['key']!=key:raise ValueError('Cache identity mismatch')
        return _decode(body['snapshot'],mass_type)

    def put(self,key,snapshot):
        if self.directory is None:self.memory[key]=deepcopy(snapshot);return
        body=dict(schema=SCHEMA,key=key,snapshot=_encode(snapshot))
        data=_bytes(dict(body=body,sha256=hashlib.sha256(_bytes(body)).hexdigest()))
        name=None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory,prefix='.writing-',delete=False)as f:
                name=f.name;f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(name,self.directory/(key+'.json'))
        finally:
            if name is not None:Path(name).unlink(missing_ok=True)


def _instrument(tree):
    """Add a restore before the round loop and save after its mass gate."""
    fn=tree.body[0]
    loops=[(i,n)for i,n in enumerate(fn.body)if isinstance(n,ast.For)
           and isinstance(n.iter,ast.Call)and getattr(n.iter.func,'id',None)=='enumerate'
           and len(n.iter.args)==1 and getattr(n.iter.args[0],'id',None)=='tables']
    if len(loops)!=1:raise ValueError('Outer DP round loop changed')
    i,loop=loops[0]
    if not(isinstance(loop.body[-1],ast.Expr)and isinstance(loop.body[-1].value,ast.Call)
           and getattr(loop.body[-1].value.func,'attr',None)=='append'):
        raise ValueError('Round mass gate/checkpoint boundary changed')
    restore=ast.parse('''
_prefix_start, _prefix_state = _prefix_restore(tables,h,target,scope)
if _prefix_state is not None:
    states = {(flag,phase):mass for flag,phase,mass in _prefix_state['states']}
    denominator = _prefix_state['denominator']
    pairs = _prefix_state['pairs']
    worst = _prefix_state['worst']
    res = _prefix_state['res']
    out['rounds'] = _prefix_state['rounds']
''').body
    loop.body[:0]=ast.parse('if m < _prefix_start: continue').body
    loop.body+=ast.parse("_prefix_save(m,states,denominator,pairs,worst,res,out['rounds'])").body
    fn.body[i:i]=restore
    return tree


def accelerator_class(P):
    """Extend this pinned module's class, keeping both original SOURCE guards."""
    source=inspect.getsource(P.Accelerator._kernel)
    if hashlib.sha256(source.encode()).hexdigest()!=ACCELERATOR_SOURCE_SHA256:
        raise ValueError('Unsupported accelerator SOURCE; prefix adaptation requires review')
    tree=ast.parse(source.lstrip());fn=tree.body[0]
    indices=[i for i,n in enumerate(fn.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
             and getattr(n.value.func,'id',None)=='exec']
    if len(indices)!=1:raise ValueError('Accelerator compile boundary changed')
    fn.body[indices[0]:indices[0]]=ast.parse('''
tree = _prefix_instrument(tree)
scope.update(_prefix_restore=self._restore, _prefix_save=self._save)
''').body
    namespace=dict(P.__dict__,_prefix_instrument=_instrument)
    exec(compile(ast.fix_missing_locations(tree),'fixed_accelerator_with_prefix_hooks','exec'),namespace)
    build_kernel=namespace['_kernel']
    import numpy as np
    runtime=dict(prefix_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version,numpy=np.__version__,source_files={
        str(m.__file__):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()for m in (P,P.R,P.R.T)})

    class PrefixAccelerator(P.Accelerator):
        _kernel=build_kernel
        def __init__(self,*,store=None,cache_namespace,checkpoint_rounds=None,**kwargs):
            if not cache_namespace:raise ValueError('Explicit configuration/signal/evaluation cache_namespace required')
            self.store=store if store is not None else PrefixStore()
            self.cache_namespace=deepcopy(cache_namespace)
            self.checkpoint_rounds=checkpoint_rounds
            self.cache_events=[];self.cache_sink=None
            super().__init__(**kwargs)

        def _event(self,value):
            self.cache_events.append(value)
            if self.cache_sink is not None:self.cache_sink(value)

        def _restore(self,tables,h,target,scope):
            M=len(tables)-1
            # SOURCE is provenance; limits and every other scope field remain in key.
            scientific_scope={k:v for k,v in scope.items()if k!='source'}
            key=_digest(dict(schema=SCHEMA,accelerator_ast=ACCELERATOR_SOURCE_SHA256,
                reference_ast=P.REFERENCE_DP_AST_SHA256,runtime=runtime,namespace=self.cache_namespace,
                h=h,target=target,M=M,scope=scientific_scope))
            self.keys=[]
            for t in tables:
                key=_digest([key,t]);self.keys.append(key)
            self.boundaries={M-1}if self.checkpoint_rounds is None else set(self.checkpoint_rounds)
            if any(type(m)is not int or not 0<=m<M for m in self.boundaries):
                if M==0 and self.checkpoint_rounds is None:self.boundaries=set()
                else:raise ValueError('Checkpoint must be a nonterminal round')
            for m in sorted(self.boundaries,reverse=True):
                try:
                    snapshot=self.store.get(self.keys[m],P.Mass)
                    if snapshot is None:continue
                    if snapshot['completed']!=m or len(snapshot['rounds'])!=m+1:
                        raise ValueError('Checkpoint boundary mismatch')
                    if snapshot['denominator']<=0:raise ValueError('Checkpoint denominator invalid')
                    self._event(dict(event='HIT',completed=m,reused_rounds=m+1,key=self.keys[m]))
                    self.observations=deepcopy(snapshot['observations'])
                    return m+1,snapshot
                except (OSError,ValueError,KeyError,TypeError)as exc:
                    self._event(dict(event='READ_REJECTED',completed=m,reason=str(exc)))
            self._event(dict(event='MISS',reused_rounds=0))
            return 0,None

        def _save(self,m,states,denominator,pairs,worst,res,rounds):
            if m not in self.boundaries:return
            snapshot=dict(completed=m,states=[(f,p,w)for(f,p),w in states.items()],
                denominator=denominator,pairs=pairs,worst=worst,
                res={k:v for k,v in res.items()if k!='_cached'},
                rounds=rounds,observations=self.observations)
            try:
                self.store.put(self.keys[m],snapshot)
                self._event(dict(event='SAVE',completed=m,key=self.keys[m],states=len(states)))
            except (OSError,ValueError,TypeError)as exc:
                # Caching is optional. A write failure never converts a numerical result to PASS.
                self._event(dict(event='WRITE_FAILED',completed=m,reason=str(exc)))

        def __call__(self,*args,**kwargs):
            self.cache_events=[]
            return super().__call__(*args,**kwargs)
    return PrefixAccelerator


@contextmanager
def installed(P,*,directory,cache_namespace,checkpoint_rounds=None,event_sink=None):
    """Patch only this worker's accelerator constructor for its existing entry."""
    cls=accelerator_class(P);store=PrefixStore(directory)
    # Existing Observed subclasses resolve super() to the old base. Replace its
    # _kernel and add hook methods for the context rather than changing its MRO.
    original=P.Accelerator.__init__
    def init(self,**kwargs):
        self.store=store;self.cache_namespace=deepcopy(cache_namespace)
        self.checkpoint_rounds=checkpoint_rounds;self.cache_events=[];self.cache_sink=event_sink
        original(self,**kwargs)
    with patch.object(P.Accelerator,'__init__',init),patch.object(P.Accelerator,'_kernel',cls._kernel),\
         patch.object(P.Accelerator,'_event',cls._event,create=True),\
         patch.object(P.Accelerator,'_restore',cls._restore,create=True),\
         patch.object(P.Accelerator,'_save',cls._save,create=True):
        yield
