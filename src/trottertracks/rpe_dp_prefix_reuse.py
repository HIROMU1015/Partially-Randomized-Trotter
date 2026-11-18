"""Opt-in checkpoints and exact residual aggregation around the fixed total-state DP.

The reference and accelerator SOURCE guards remain active. Transition selection,
pruning geometry, loss bounds and final-moment equations stay in the fixed kernel.
"""
import ast
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import fcntl
from fractions import Fraction
import gzip
import hashlib
import inspect
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import tempfile
import textwrap
import time
import sys
from unittest.mock import patch
import zlib

SCHEMA='RPE_TOTAL_DP_PREFIX_V1'
DEFAULT_CACHE_MAX_BYTES = 40 * 2**30
# The former V1 helper hash is the checkpoint semantics identity. Update this
# only when changing the checkpoint content or the round boundaries of the hooks.
CHECKPOINT_SEMANTICS_ID = '554460755ab6fcd38ad5b9f1a680b543aed7e26906579b26470d73fa659148f2'
_CACHE_FILE = re.compile(r'[0-9a-f]{64}\.json(?:\.gz)?')
_STAGING_PREFIX = '.prefix-writing-'
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
    """Bound the recoverable cache per directory, including legacy V1 files."""

    def __init__(self, directory=None, *, max_bytes=DEFAULT_CACHE_MAX_BYTES):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Positive integer prefix_cache_max_bytes required')
        self.directory = None if directory is None else Path(directory)
        self.max_bytes = max_bytes
        self.memory = {}  # memory mode only

    def get(self, key, mass_type):
        if self.directory is None:
            return deepcopy(self.memory.get(key))
        for suffix in ('.json.gz', '.json'):
            path = self.directory / (key + suffix)
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                continue  # also covers a concurrent eviction
            self.last_read_bytes = len(data)
            if suffix == '.json.gz':
                try:
                    data = gzip.decompress(data)
                except (EOFError, zlib.error) as exc:
                    raise ValueError('Incomplete or corrupt compressed checkpoint') from exc
            envelope = json.loads(data)
            body = envelope['body']
            if envelope['sha256'] != hashlib.sha256(_bytes(body)).hexdigest():
                raise ValueError('Cache payload checksum mismatch')
            if body['schema'] != SCHEMA or body['key'] != key:
                raise ValueError('Cache identity mismatch')
            snapshot = _decode(body['snapshot'], mass_type)
            try:
                os.utime(path, None)  # LRU mark after verifying the checkpoint
            except OSError:
                pass  # eviction or a read-only cache can still supply this hit
            return snapshot
        return None

    def _make_room(self, incoming_bytes):
        """Called under the writer lock, before writing an atomic staging file."""
        entries = []
        with os.scandir(self.directory) as files:
            for entry in files:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name.startswith(_STAGING_PREFIX):
                    os.unlink(entry.path)  # remains only after an interrupted writer
                elif _CACHE_FILE.fullmatch(entry.name):
                    info = entry.stat(follow_symlinks=False)
                    entries.append((info.st_mtime_ns, entry.name, info.st_size))
        total = sum(size for _, _, size in entries)
        if total + incoming_bytes <= self.max_bytes:
            return
        evicted_files = evicted_bytes = 0
        for _, name, size in sorted(entries):
            if total + incoming_bytes <= self.max_bytes:
                break
            (self.directory / name).unlink()
            total -= size
            evicted_files += 1
            evicted_bytes += size
        self.last_evicted_files = evicted_files
        self.last_evicted_bytes = evicted_bytes

    def put(self, key, snapshot):
        self.last_evicted_files = self.last_evicted_bytes = 0
        if self.directory is None:
            self.memory[key] = deepcopy(snapshot)
            return True
        destination = key + '.json.gz'
        if not _CACHE_FILE.fullmatch(destination):
            raise ValueError('Invalid checkpoint cache key')
        body = _bytes(dict(schema=SCHEMA, key=key, snapshot=_encode(snapshot)))
        checksum = hashlib.sha256(body).hexdigest().encode()
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb', compresslevel=1, mtime=0) as stream:
            stream.write(b'{"body":')
            stream.write(body)
            stream.write(b',"sha256":"' + checksum + b'"}')
        del body
        data = buffer.getbuffer()
        self.last_write_bytes = len(data)
        if len(data) > self.max_bytes:
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / '.prefix-storage.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._make_room(len(data))
            name = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, prefix=_STAGING_PREFIX, delete=False) as file:
                    name = file.name
                    file.write(data)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(name, self.directory / destination)
            finally:
                if name is not None:
                    Path(name).unlink(missing_ok=True)
        return True


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


PROPAGATE_SOURCE_SHA256 = 'd84467c758e23e4fdd3e9b56e1c1dd5d0101cb90118c16ce397bc39c276c8321'
MIN_DROPS_PER_WORKER = 1024
_drop_context = None  # inherited by the existing Linux fork worker model


def _grouped_sum(original, residual, key, mass, coefficient):
    if 'groups' not in residual:
        return original(residual, key, mass, coefficient)
    if coefficient and mass.numerator:
        coefficient = Fraction(coefficient)
        terms = residual['groups'].setdefault(key, {})
        denominator = mass.denominator * coefficient.denominator
        terms[denominator] = terms.get(denominator, 0) + mass.numerator * coefficient.numerator


def _drop_batch(bounds, context=None):
    ordered, denominator, completed, drop, mass_type = _drop_context if context is None else context
    partial = dict(events=0, groups={})
    for index in range(*bounds):
        (flag, phase), count = ordered[index]
        drop(flag, phase, mass_type(count, denominator), completed, partial)
    return partial


def _drop_many(P, worker_count, ordered, denominator, completed, drop):
    if not ordered:
        return
    context = ordered, denominator, completed, drop, P.Mass
    available = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    workers = min(worker_count or available, max(1, len(ordered)//MIN_DROPS_PER_WORKER))
    if 'fork' not in mp.get_all_start_methods():
        workers = 1

    residual = drop.residual()
    def merge(partial):
        residual['events'] += partial['events']
        for key, terms in partial['groups'].items():
            for den, num in terms.items():
                P._accumulate(residual, key, P.Mass(num, den), 1)

    if workers == 1:
        merge(_drop_batch((0,len(ordered)),context))
        return
    global _drop_context
    _drop_context = context
    chunk = (len(ordered) + workers - 1)//workers
    batches = [(i,min(i+chunk,len(ordered))) for i in range(0,len(ordered),chunk)]
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
            for partial in pool.map(_drop_batch,batches):
                merge(partial)
    finally:
        _drop_context = None


def _drop_loop(body, expression):
    indices = [i for i,n in enumerate(body) if isinstance(n,ast.For) and len(n.body)==1
               and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Call)
               and getattr(n.body[0].value.func,'id',None)=='drop']
    if len(indices)!=1:
        raise ValueError('Fixed DP omission loop changed')
    body[indices[0]:indices[0]+1] = ast.parse(expression).body


def _instrument_residual(tree):
    fn = tree.body[0]
    drop_index, drop = next((i,n) for i,n in enumerate(fn.body) if isinstance(n,ast.FunctionDef) and n.name=='drop')
    # Use the original closure by default and explicitly accept a batch-local
    # sink. A getter follows the live closure even after prefix restoration.
    for n in ast.walk(drop):
        if isinstance(n,ast.Name) and n.id=='res':
            n.id='sink'
    drop.args.args.append(ast.arg(arg='sink'))
    drop.args.defaults.append(ast.Constant(None))
    drop.body[:0] = ast.parse('if sink is None: sink = res').body
    fn.body[drop_index+1:drop_index+1] = ast.parse('drop.residual = lambda: res').body
    loop = next(n for n in fn.body if isinstance(n,ast.For) and getattr(n.iter,'func',None)
                and getattr(n.iter.func,'id',None)=='enumerate')
    prune = next(n for n in loop.body if isinstance(n,ast.If) and
                 any(isinstance(t,ast.For) and getattr(t.body[0],'value',None)
                     and getattr(getattr(t.body[0].value,'func',None),'id',None)=='drop' for t in n.body))
    _drop_loop(prune.body,'_drop_many(ordered[maximum_states:],denominator,m,drop)')
    return tree


def _fixed_propagate(P):
    source = inspect.getsource(P.Accelerator.propagate)
    if hashlib.sha256(source.encode()).hexdigest()!=PROPAGATE_SOURCE_SHA256:
        raise ValueError('Fixed accelerator propagation SOURCE changed')
    tree=ast.parse(textwrap.dedent(source))
    _drop_loop(tree.body[0].body,'self._drop_many(ordered[count:],previous,m-1,drop)')
    scope=dict(P.__dict__)
    exec(compile(ast.fix_missing_locations(tree),'fixed_accelerator_with_grouped_omissions','exec'),scope)
    return scope['propagate']


def accelerator_class(P, *, parallel_residual=False):
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
    if parallel_residual:
        fn.body[indices[0]:indices[0]]=ast.parse('''
tree = _instrument_residual(tree)
scope.update(_drop_many=self._drop_many, _accumulate=self._grouped_sum)
''').body
    namespace=dict(P.__dict__,_prefix_instrument=_instrument,_instrument_residual=_instrument_residual)
    exec(compile(ast.fix_missing_locations(tree),'fixed_accelerator_with_prefix_hooks','exec'),namespace)
    build_kernel=namespace['_kernel']
    import numpy as np
    runtime=dict(prefix_source=CHECKPOINT_SEMANTICS_ID,python=sys.version,numpy=np.__version__,source_files={
        str(m.__file__):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()for m in (P,P.R,P.R.T)})
    if parallel_residual:
        runtime['residual_source']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

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
            start=time.perf_counter();read_seconds=0.
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
            hash_seconds=time.perf_counter()-start
            for m in sorted(self.boundaries,reverse=True):
                read_start=time.perf_counter()
                try:
                    snapshot=self.store.get(self.keys[m],P.Mass)
                    read_seconds+=time.perf_counter()-read_start
                    if snapshot is None:continue
                    if snapshot['completed']!=m or len(snapshot['rounds'])!=m+1:
                        raise ValueError('Checkpoint boundary mismatch')
                    if snapshot['denominator']<=0:raise ValueError('Checkpoint denominator invalid')
                    self._event(dict(event='HIT',completed=m,reused_rounds=m+1,key=self.keys[m],hash_seconds=hash_seconds,
                        read_seconds=read_seconds,serialized_bytes=getattr(self.store,'last_read_bytes',None)))
                    self.observations=deepcopy(snapshot['observations'])
                    return m+1,snapshot
                except (OSError,ValueError,KeyError,TypeError)as exc:
                    self._event(dict(event='READ_REJECTED',completed=m,reason=str(exc)))
            self._event(dict(event='MISS',reused_rounds=0,hash_seconds=hash_seconds,read_seconds=read_seconds))
            return 0,None

        def _save(self,m,states,denominator,pairs,worst,res,rounds):
            if m not in self.boundaries:return
            snapshot=dict(completed=m,states=[(f,p,w)for(f,p),w in states.items()],
                denominator=denominator,pairs=pairs,worst=worst,
                res={k:v for k,v in res.items()if k!='_cached'},
                rounds=rounds,observations=self.observations)
            try:
                write_start=time.perf_counter()
                stored=self.store.put(self.keys[m],snapshot)
                event=dict(event='SAVE' if stored else 'SAVE_SKIPPED',completed=m,key=self.keys[m],states=len(states),
                    write_seconds=time.perf_counter()-write_start,serialized_bytes=getattr(self.store,'last_write_bytes',None),
                    evicted_files=self.store.last_evicted_files,evicted_bytes=self.store.last_evicted_bytes)
                if not stored:event['reason']='SNAPSHOT_EXCEEDS_BYTE_LIMIT'
                self._event(event)
            except (OSError,ValueError,TypeError)as exc:
                # Caching is optional. A write failure never converts a numerical result to PASS.
                self._event(dict(event='WRITE_FAILED',completed=m,reason=str(exc)))

        def __call__(self,*args,**kwargs):
            self.cache_events=[]
            return super().__call__(*args,**kwargs)
    if parallel_residual:
        PrefixAccelerator.propagate=_fixed_propagate(P)
        PrefixAccelerator._drop_many=lambda self,*args:_drop_many(P,self.workers,*args)
        PrefixAccelerator._grouped_sum=lambda self,*args:_grouped_sum(P._accumulate,*args)
    return PrefixAccelerator


@contextmanager
def installed(P,*,directory,cache_namespace,checkpoint_rounds=None,event_sink=None,max_bytes=DEFAULT_CACHE_MAX_BYTES,
              parallel_residual=False):
    """Patch only this worker's accelerator constructor for its existing entry."""
    cls=accelerator_class(P,parallel_residual=parallel_residual);store=PrefixStore(directory,max_bytes=max_bytes)
    # Existing Observed subclasses resolve super() to the old base. Replace its
    # _kernel and add hook methods for the context rather than changing its MRO.
    original=P.Accelerator.__init__
    def init(self,**kwargs):
        self.store=store;self.cache_namespace=deepcopy(cache_namespace)
        self.checkpoint_rounds=checkpoint_rounds;self.cache_events=[];self.cache_sink=event_sink
        original(self,**kwargs)
    hooks={'__init__':init,'_kernel':cls._kernel,'_event':cls._event,'_restore':cls._restore,'_save':cls._save}
    if parallel_residual:
        hooks.update(propagate=cls.propagate,_drop_many=cls._drop_many,_grouped_sum=cls._grouped_sum)
    with ExitStack() as stack:
        for name,fn in hooks.items():
            stack.enter_context(patch.object(P.Accelerator,name,fn,create=True))
        yield
