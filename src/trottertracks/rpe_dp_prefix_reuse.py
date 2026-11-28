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
_ADMISSION_FILE = '.prefix-admissions.json'
_ADMISSION_MAX_KEYS = 4096
DEFAULT_SELECTIVE_WINDOW_SECONDS = 600
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
        self._admission_memory = {}

    def _admissions(self, entries, keys, window, observe):
        now = time.time()
        entries = {k: v for k, v in entries.items() if 0 <= now-v[0] <= window}
        if observe:
            for key in keys:
                _, count = entries.pop(key, (now, 0))
                entries[key] = [now, min(2, count+1)]
        if len(entries) > _ADMISSION_MAX_KEYS:
            entries = dict(list(entries.items())[-_ADMISSION_MAX_KEYS:])
        return entries

    def observe(self, keys, window):
        """Record small demand hints at lookup, before a candidate reaches a boundary."""
        if not keys:
            return
        self._selective_admissions(keys, window, observe=True)

    def admit(self, key, window):
        entries = self._selective_admissions((), window, observe=False)
        return key in entries and entries[key][1] >= 2

    def _selective_admissions(self, keys, window, *, observe):
        if self.directory is None:
            self._admission_memory = self._admissions(self._admission_memory, keys, window, observe)
            return self._admission_memory
        if observe:
            self.directory.mkdir(parents=True, exist_ok=True)
        elif not self.directory.is_dir():
            return {}
        path = self.directory / _ADMISSION_FILE
        with (self.directory / '.prefix-storage.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                entries = json.loads(path.read_bytes())
                entries = self._admissions(entries, keys, window, observe)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, FileNotFoundError):
                entries = self._admissions({}, keys, window, observe)
            if observe:
                data=json.dumps(entries,separators=(',',':')).encode()
                if not self._make_room(len(data)):
                    return {}
                # Use one atomic file for metadata. Interrupted writes use the
                # same staging prefix cleaned by the checkpoint writer.
                with tempfile.NamedTemporaryFile(dir=self.directory, prefix=_STAGING_PREFIX, delete=False) as temporary:
                    try:
                        temporary.write(data)
                        temporary.flush()
                        os.replace(temporary.name,path)
                    finally:
                        Path(temporary.name).unlink(missing_ok=True)
            return entries

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
        metadata_bytes=0
        with os.scandir(self.directory) as files:
            for entry in files:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name.startswith(_STAGING_PREFIX):
                    os.unlink(entry.path)  # remains only after an interrupted writer
                elif _CACHE_FILE.fullmatch(entry.name):
                    info = entry.stat(follow_symlinks=False)
                    entries.append((info.st_mtime_ns, entry.name, info.st_size))
                elif entry.name == _ADMISSION_FILE:
                    metadata_bytes=entry.stat(follow_symlinks=False).st_size
        total = metadata_bytes+sum(size for _, _, size in entries)
        if metadata_bytes+incoming_bytes > self.max_bytes:
            return False
        if total + incoming_bytes <= self.max_bytes:
            return True
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
        return True

    def reclaim(self, retained_bytes):
        """Trim checkpoints under the writer lock while retaining demand metadata."""
        if self.directory is None or not self.directory.is_dir():
            return 0
        with (self.directory / '.prefix-storage.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            metadata = self.directory / _ADMISSION_FILE
            floor = metadata.stat().st_size if metadata.is_file() else 0
            original_limit = self.max_bytes
            self.last_evicted_files = self.last_evicted_bytes = 0
            try:
                self.max_bytes = max(1, floor, min(retained_bytes, original_limit))
                self._make_room(0)
                return self.last_evicted_files
            finally:
                self.max_bytes = original_limit

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
            if not self._make_room(len(data)):
                return False
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
    ordered, denominator, completed, drop, mass_type, profile = _drop_context if context is None else context
    started = time.perf_counter() if profile else None
    partial = dict(events=0, groups={})
    for index in range(*bounds):
        (flag, phase), count = ordered[index]
        drop(flag, phase, mass_type(count, denominator), completed, partial)
    if started is not None:
        partial['worker_seconds'] = time.perf_counter()-started
        partial['worker_pid'] = os.getpid()
    return partial


_persistent_drop=None
_persistent_mass=None


def _init_persistent_drop(drop,mass_type):
    global _persistent_drop,_persistent_mass
    _persistent_drop,_persistent_mass=drop,mass_type


def _persistent_drop_batch(payload):
    ordered,denominator,completed,profile=payload
    return _drop_batch((0,len(ordered)),(ordered,denominator,completed,_persistent_drop,_persistent_mass,profile))


def _immutable_closure_value(value):
    if type(value) in (type(None),bool,int,float,str,bytes,Fraction):return True
    if type(value) in (tuple,frozenset):return all(_immutable_closure_value(item) for item in value)
    return False


class ResidualPool:
    """Opt-in per-evaluation worker pool for stable drop closures only.

    Changed callable identity, worker footprint, or mutable captured state
    falls back to the original per-call fork path. Only the explicit newly
    instrumented sink fallback may capture the parent residual dictionary;
    every worker receives its independent sink as in the existing path.
    """
    def __init__(self):self.pool=None;self.key=None;self.owner_pid=os.getpid()

    def close(self):
        if os.getpid()!=self.owner_pid:return
        if self.pool is not None:self.pool.shutdown(wait=True);self.pool=None
        self.key=None

    def prepare(self,drop,mass_type,workers):
        if os.getpid()!=self.owner_pid:return False,False
        if not hasattr(drop,'__code__') or not hasattr(drop,'__globals__'):
            self.close();return False,False
        closure=[]
        for name,cell in zip(drop.__code__.co_freevars,drop.__closure__ or ()):
            if name=='res' and getattr(drop,'_residual_pool_sink',False):continue
            try:value=cell.cell_contents
            except ValueError:self.close();return False,False
            if not _immutable_closure_value(value):self.close();return False,False
            closure.append((name,value))
        # The same fixed callback and the same bindings must survive across
        # rounds. Mutable globals are rejected rather than copied implicitly.
        import types
        globals_key=[]
        for name in drop.__code__.co_names:
            if name not in drop.__globals__:continue
            value=drop.__globals__[name]
            if not (_immutable_closure_value(value) or isinstance(value,(types.FunctionType,types.ModuleType,type))):
                self.close();return False,False
            globals_key.append((name,id(value)))
        key=(drop,mass_type,workers,tuple(closure),tuple(globals_key))
        if self.key==key and self.pool is not None:return True,True
        self.close()
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork'),
                                      initializer=_init_persistent_drop,initargs=(drop,mass_type))
        self.key=key
        return True,False


def _drop_many(P, worker_count, ordered, denominator, completed, drop, event_sink=None,residual_pool=None):
    if not ordered:
        return
    started = time.perf_counter() if event_sink is not None else None
    context = ordered, denominator, completed, drop, P.Mass, event_sink is not None
    available = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    workers = min(worker_count or available, available, max(1, len(ordered)//MIN_DROPS_PER_WORKER))
    if 'fork' not in mp.get_all_start_methods():
        workers = 1

    residual = drop.residual()
    merge_seconds=0.
    worker_seconds={}
    def merge(partial):
        nonlocal merge_seconds
        start=time.perf_counter() if started is not None else None
        residual['events'] += partial['events']
        for key, terms in partial['groups'].items():
            for den, num in terms.items():
                P._accumulate(residual, key, P.Mass(num, den), 1)
        if start is not None:
            merge_seconds+=time.perf_counter()-start
            pid=partial['worker_pid']
            worker_seconds[pid]=worker_seconds.get(pid,0.)+partial['worker_seconds']

    reused=False;persistent=False
    if workers == 1:
        merge(_drop_batch((0,len(ordered)),context))
    else:
        if residual_pool is not None:persistent,reused=residual_pool.prepare(drop,P.Mass,workers)
        global _drop_context
        _drop_context = context
        # Keep a useful minimum batch, but allow the existing fork pool
        # to distribute uneven state work dynamically. pool.map retains
        # deterministic merge order; grouped terms are exact integers.
        chunk = max(MIN_DROPS_PER_WORKER,(len(ordered)+4*workers-1)//(4*workers))
        batches = [(i,min(i+chunk,len(ordered))) for i in range(0,len(ordered),chunk)]
        try:
            if persistent:
                payloads=((ordered[start:end],denominator,completed,event_sink is not None) for start,end in batches)
                for partial in residual_pool.pool.map(_persistent_drop_batch,payloads):merge(partial)
            else:
                with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
                    for partial in pool.map(_drop_batch,batches):merge(partial)
        finally:
            _drop_context = None
    if started is not None:
        event_sink(dict(event='RESIDUAL_TIMING',completed=completed,states=len(ordered),workers=workers,
            wall_seconds=time.perf_counter()-started,parent_merge_seconds=merge_seconds,
            worker_max_seconds=max(worker_seconds.values()),worker_sum_seconds=sum(worker_seconds.values()),
            persistent_pool=persistent,pool_reused=reused,worker_pids=sorted(worker_seconds)))


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
    fn.body[drop_index+1:drop_index+1] = ast.parse('drop.residual = lambda: res\ndrop._residual_pool_sink = True').body
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
scope.update(_drop_many=self._drop_many, _accumulate=_grouped_accumulator)
''').body
    # The grouped accumulator is stateless. Keep the fixed function binding
    # rather than capturing a mutable Accelerator instance in the fork closure.
    def grouped_accumulator(*args, original=P._accumulate):
        return _grouped_sum(original, *args)
    namespace=dict(P.__dict__,_prefix_instrument=_instrument,_instrument_residual=_instrument_residual,
                   _grouped_accumulator=grouped_accumulator)
    exec(compile(ast.fix_missing_locations(tree),'fixed_accelerator_with_prefix_hooks','exec'),namespace)
    build_kernel=namespace['_kernel']
    import numpy as np
    runtime=dict(prefix_source=CHECKPOINT_SEMANTICS_ID,python=sys.version,numpy=np.__version__,source_files={
        str(m.__file__):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()for m in (P,P.R,P.R.T)})
    if parallel_residual:
        runtime['residual_source']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    class PrefixAccelerator(P.Accelerator):
        _kernel=build_kernel
        def __init__(self,*,store=None,cache_namespace,checkpoint_rounds=None,selective_rounds=(),
                     selective_window_seconds=DEFAULT_SELECTIVE_WINDOW_SECONDS,profile_residual=False,**kwargs):
            if not cache_namespace:raise ValueError('Explicit configuration/signal/evaluation cache_namespace required')
            self.store=store if store is not None else PrefixStore()
            self.cache_namespace=deepcopy(cache_namespace)
            self.checkpoint_rounds=checkpoint_rounds
            self.selective_rounds=set(selective_rounds)
            self.selective_window_seconds=selective_window_seconds
            self.profile_residual=profile_residual
            self.cache_events=[];self.cache_sink=None
            super().__init__(**kwargs)

        def _event(self,value):
            self.cache_events.append(value)
            if self.cache_sink is not None:self.cache_sink(value)

        def _observe_selective(self, restored):
            keys=[self.keys[m] for m in self.selective_rounds if m>restored]
            try:
                self.store.observe(keys,self.selective_window_seconds)
            except OSError as exc:
                self._event(dict(event='ADMISSION_UNAVAILABLE',reason=str(exc)))

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
            if not self.selective_rounds <= self.boundaries or self.selective_window_seconds <= 0:
                raise ValueError('Selective checkpoints need configured boundaries and a positive reuse window')
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
                    self._observe_selective(m)
                    return m+1,snapshot
                except (OSError,ValueError,KeyError,TypeError)as exc:
                    self._event(dict(event='READ_REJECTED',completed=m,reason=str(exc)))
            self._event(dict(event='MISS',reused_rounds=0,hash_seconds=hash_seconds,read_seconds=read_seconds))
            self._observe_selective(-1)
            return 0,None

        def _save(self,m,states,denominator,pairs,worst,res,rounds):
            if m not in self.boundaries:return
            try:
                if m in self.selective_rounds and not self.store.admit(self.keys[m],self.selective_window_seconds):
                    self._event(dict(event='SAVE_NOT_ADMITTED',completed=m,key=self.keys[m],reason='NO_RECENT_REPEAT'))
                    return
                snapshot=dict(completed=m,states=[(f,p,w)for(f,p),w in states.items()],
                    denominator=denominator,pairs=pairs,worst=worst,
                    res={k:v for k,v in res.items()if k!='_cached'},
                    rounds=rounds,observations=self.observations)
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
        PrefixAccelerator._drop_many=lambda self,*args:_drop_many(P,self.workers,*args,
            event_sink=self._event if self.profile_residual else None,residual_pool=getattr(self,'_residual_pool',None))
        PrefixAccelerator._grouped_sum=lambda self,*args:_grouped_sum(P._accumulate,*args)
    return PrefixAccelerator


@contextmanager
def installed(P,*,directory,cache_namespace,checkpoint_rounds=None,event_sink=None,max_bytes=DEFAULT_CACHE_MAX_BYTES,
              parallel_residual=False,selective_rounds=(),selective_window_seconds=DEFAULT_SELECTIVE_WINDOW_SECONDS,
              profile_residual=False,persistent_residual_pool=False):
    """Patch only this worker's accelerator constructor for its existing entry."""
    cls=accelerator_class(P,parallel_residual=parallel_residual);store=PrefixStore(directory,max_bytes=max_bytes)
    # Existing Observed subclasses resolve super() to the old base. Replace its
    # _kernel and add hook methods for the context rather than changing its MRO.
    if persistent_residual_pool and not parallel_residual:raise ValueError('Persistent residual pool requires parallel residuals')
    pool=ResidualPool() if persistent_residual_pool else None
    original=P.Accelerator.__init__
    def init(self,**kwargs):
        self.store=store;self.cache_namespace=deepcopy(cache_namespace)
        self.checkpoint_rounds=checkpoint_rounds;self.cache_events=[];self.cache_sink=event_sink
        self.selective_rounds=set(selective_rounds);self.selective_window_seconds=selective_window_seconds
        self.profile_residual=profile_residual;self._residual_pool=pool
        original(self,**kwargs)
    hooks={'__init__':init,'_kernel':cls._kernel,'_event':cls._event,'_restore':cls._restore,'_save':cls._save,
           '_observe_selective':cls._observe_selective}
    if parallel_residual:
        hooks.update(propagate=cls.propagate,_drop_many=cls._drop_many,_grouped_sum=cls._grouped_sum)
    with ExitStack() as stack:
        for name,fn in hooks.items():
            stack.enter_context(patch.object(P.Accelerator,name,fn,create=True))
        try:yield
        finally:
            if pool is not None:pool.close()
