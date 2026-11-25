"""Reuse the fixed DF provider's deterministic S2 flanks within one price task."""
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
import ast
import inspect
import textwrap
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


class Flanks:
    def __init__(self, native):
        self.native = native
        self.original = native.build_partial_native
        self.pieces = {}

    def templates(self, blocks, n, signed_time):
        key = id(blocks), n, signed_time.hex()
        if key not in self.pieces:
            native = self.native
            forward, reverse = native.QuantumCircuit(n + 1), native.QuantumCircuit(n + 1)
            half = signed_time / 2
            for block in blocks:
                native._append_block(forward, block, half, 'UNCONTROLLED', n)
            for block in reversed(blocks):
                native._append_block(reverse, block, half, 'DIRECTIONAL', n)
            # Keep a reference to the block tuple for the lifetime of its ID.
            self.pieces[key] = blocks, forward, reverse
        return self.pieces[key][1:]

    def build(self, blocks, *, num_system_qubits, T, q, scalar, controlled_tails,
              control_policy, max_instructions):
        if (type(blocks) is not tuple or type(q) is not int or q != 1
                or control_policy != 'symmetric_directional' or controlled_tails is None):
            return self.original(blocks, num_system_qubits=num_system_qubits,
                                 T=T, q=q, scalar=scalar, controlled_tails=controlled_tails,
                                 control_policy=control_policy, max_instructions=max_instructions)
        native = self.native
        n = native.validate_blocks(blocks, num_system_qubits)
        T, scalar = native._real(T, 'T'), native._real(scalar, 'scalar')
        cap = native._limit(max_instructions)
        if len(controlled_tails) != 1:
            raise ValueError('One explicit controlled tail per outer step is required.')
        tail = controlled_tails[0]
        native._gate_only(tail, n + 1)
        bound = sum(native.block_instruction_bound(b, mode)
                    for b in blocks for mode in ('UNCONTROLLED', 'DIRECTIONAL')) + len(tail.data) + 1
        if bound > cap:
            raise RuntimeError('PRE_BUILD_INSTRUCTION_BUDGET')
        forward, reverse = self.templates(blocks, n, T)
        circuit = forward.copy(name='ax2a_df_partial_s2')
        circuit.compose(tail, inplace=True)
        circuit.compose(reverse, inplace=True)
        if scalar*T != 0:
            circuit.p(-scalar*T, n)
        return native._finish(circuit, n, T, q, 'partial_2nd', control_policy, bound, cap)


@contextmanager
def reuse_df_price_events(rte):
    """Reuse immutable events after each original sampling decision."""
    make = rte._make_event
    active_components = active_distribution = None

    @lru_cache(maxsize=4096)
    def event(indices, order):
        return make(indices, active_components, active_distribution, order)

    def cached(indices, components, distribution, order):
        nonlocal active_components, active_distribution
        if components is not active_components or distribution is not active_distribution:
            event.cache_clear()
            active_components, active_distribution = components, distribution
        return event(tuple(map(int, indices)), order)

    with patch.object(rte, '_make_event', cached):
        yield



EVENT_DICTIONARY_SCHEMA = 'RPE_EVENT_DICTIONARY_V1'
EVENT_BLOCK_SIZE = 4096


class EventRow(Sequence):
    """Read-only view of one already sampled tail; never resample an event."""
    def __init__(self, events):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        if isinstance(index, slice):return EventRow(self.events[index])
        return self.events[index].to_dict()


class EventRows(Sequence):
    def __init__(self, requests):
        self.requests = requests

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        if isinstance(index, slice):return EventRows(self.requests[index])
        return EventRow(self.requests[index].rte_occurrence.events)


class FlatEvents:
    def __init__(self, rows):self.rows = rows
    def __len__(self):return sum(map(len, self.rows))
    def __iter__(self):
        for row in self.rows:yield from row


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _canonical_readonly_event(event, cache):
    """Reuse encoding only while holding the original read-only dictionary."""
    key = id(event)
    if key not in cache:
        cache[key] = (event, _canonical(event))
        if len(cache) > EVENT_BLOCK_SIZE:cache.popitem(last=False)
    else:
        cache.move_to_end(key)
    return cache[key][1]


def write_event_dictionary(path, document, *, block_size=EVENT_BLOCK_SIZE):
    """Lossless bounded-block dictionary with exact original row boundaries.

    Store actual sampled event dictionaries once per block, not RNG seeds
    alone. IDs use canonical JSON bytes (so -0.0, types, and all fields are
    preserved). The footer digest is the original expanded event-list digest.
    """
    if type(block_size) is not int or block_size < 1:raise ValueError('Invalid event block size')
    if 'events' not in document:raise ValueError('Missing event rows')
    metadata = {key: value for key, value in document.items() if key != 'events'}
    if set(metadata) != {'seeds', 'laws', 'conditional'}:raise ValueError('Unexpected DF event metadata')
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256();hasher.update(b'[')
    row_count = event_count = 0
    with path.open('x', encoding='utf-8') as stream:
        stream.write('{"schema":'+_canonical(EVENT_DICTIONARY_SCHEMA)+',"metadata":'+_canonical(metadata)+',"blocks":[\n')
        table = {}; templates = []; segments = []; used = 0; written = 0
        readonly = OrderedDict() if isinstance(document['events'], EventRows) else None
        def flush():
            nonlocal used, written
            if not segments:return
            if written:stream.write(',\n')
            # Exact canonical strings were produced once per occurrence; keep
            # only a bounded block of distinct strings and integer indices.
            stream.write('{"dictionary":[')
            for i,template in enumerate(templates):
                if i:stream.write(',')
                stream.write(template)
            stream.write('],"segments":')
            stream.write(_canonical(segments));stream.write('}')
            table.clear();templates.clear();segments.clear();used=0;written+=1
        for row in document['events']:
            if row_count:hasher.update(b',')
            hasher.update(b'[');indices=[];count=0
            for event in row:
                if not isinstance(event,dict):raise ValueError('Event row must contain dictionaries')
                if used == block_size:
                    if indices:segments.append([indices,False]);indices=[]
                    flush()
                raw = _canonical_readonly_event(event, readonly) if readonly is not None else _canonical(event)
                if count:hasher.update(b',')
                hasher.update(raw.encode());count+=1;event_count+=1
                token = table.get(raw)
                if token is None:
                    token = len(templates);table[raw]=token;templates.append(raw)
                indices.append(token);used+=1
            segments.append([indices,True]);hasher.update(b']');row_count+=1
            # Empty rows must also remain bounded.
            if len(segments) >= block_size or used == block_size:flush()
        flush();hasher.update(b']')
        footer = dict(rows=row_count, events=event_count, event_digest=hasher.hexdigest())
        stream.write('\n],"footer":'+_canonical(footer)+'}\n')
    return footer


def is_dictionary_event_file(path):
    with Path(path).open('rb') as stream:
        return stream.read(128).startswith(b'{"schema":"RPE_EVENT_DICTIONARY_V1"')


def _unique_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate event JSON key')
        result[key]=value
    return result


class _EventJSON:
    """Bounded forward decoder; one compact block, never a whole path."""
    def __init__(self, stream, chunk):
        if type(chunk) is not int or chunk < 1:raise ValueError('Invalid event JSON chunk size')
        self.stream=stream;self.chunk=chunk;self.buf='';self.pos=0;self.eof=False
        self.decoder=json.JSONDecoder(object_pairs_hook=_unique_pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite event JSON')))
    def more(self):
        self.buf=self.buf[self.pos:];self.pos=0
        value=self.stream.read(self.chunk);self.buf+=value;self.eof=not value
    def peek(self):
        while True:
            while self.pos<len(self.buf) and self.buf[self.pos].isspace():self.pos+=1
            if self.pos<len(self.buf):return self.buf[self.pos]
            if self.eof:return ''
            self.more()
    def take(self,char):
        if self.peek()!=char:raise ValueError('Invalid event dictionary syntax: '+char)
        self.pos+=1
    def value(self):
        self.peek()
        while True:
            try:value,end=self.decoder.raw_decode(self.buf,self.pos)
            except json.JSONDecodeError:
                if self.eof:raise ValueError('Truncated event dictionary')
                self.more();continue
            # A number at a chunk boundary is not necessarily complete.
            if end==len(self.buf) and not self.eof:
                self.more();continue
            self.pos=end;return value
    def field(self,name):
        if self.value()!=name:raise ValueError('Unexpected event dictionary field: '+name)
        self.take(':')


class DictionaryEvents:
    """Expand one row incrementally and verify the original receipt digest."""
    def __init__(self,path,expected_digest=None,*,chunk=65536):
        self.path=Path(path);self.expected=expected_digest;self.chunk=chunk
        self.metadata={};self.digest=None;self.counts=None
    def batches(self):
        hasher=hashlib.sha256();hasher.update(b'[');rows=events=0
        with self.path.open(encoding='utf-8') as stream:
            p=_EventJSON(stream,self.chunk);p.take('{');p.field('schema')
            if p.value()!=EVENT_DICTIONARY_SCHEMA:raise ValueError('Unknown event dictionary schema')
            p.take(',');p.field('metadata');self.metadata=p.value()
            if not isinstance(self.metadata,dict) or set(self.metadata)!={'seeds','laws','conditional'}:
                raise ValueError('Invalid event metadata')
            p.take(',');p.field('blocks');p.take('[')
            def segments():
                first=True
                while p.peek()!=']':
                    if not first:p.take(',')
                    first=False;p.take('{');p.field('dictionary');p.take('[')
                    dictionary=[];encoded=[]
                    while p.peek()!=']':
                        if dictionary:p.take(',')
                        item=p.value()
                        if not isinstance(item,dict):raise ValueError('Invalid event dictionary entry')
                        dictionary.append(item);encoded.append(_canonical(item).encode())
                    p.take(']');p.take(',');p.field('segments');p.take('[');part_count=0
                    while p.peek()!=']':
                        if part_count:p.take(',')
                        part=p.value();part_count+=1
                        if (not isinstance(part,list) or len(part)!=2 or not isinstance(part[0],list)
                                or type(part[1]) is not bool or any(type(i) is not int or i<0 or i>=len(dictionary) for i in part[0])):
                            raise ValueError('Invalid event indices or boundary')
                        yield dictionary,encoded,part[0],part[1]
                    if not part_count:raise ValueError('Invalid empty event block')
                    p.take(']');p.take('}')
                p.take(']')
            iterator=iter(segments())
            for first_part in iterator:
                if rows:hasher.update(b',')
                hasher.update(b'[');rows+=1;complete=False;count=0
                def row():
                    nonlocal complete,count,events
                    part=first_part
                    while True:
                        dictionary,encoded,ids,end=part
                        for index in ids:
                            event=dictionary[index]
                            if count:hasher.update(b',')
                            hasher.update(encoded[index]);count+=1;events+=1
                            # Readers may retain or mutate their own result;
                            # never expose the mutable dictionary template.
                            yield deepcopy(event)
                        if end:break
                        part=next(iterator,None)
                        if part is None:raise ValueError('Missing event row boundary')
                    hasher.update(b']');complete=True
                yield row()
                if not complete:raise ValueError('Event row not fully consumed')
            hasher.update(b']');p.take(',');p.field('footer');footer=p.value();p.take('}')
            self.digest=hasher.hexdigest();self.counts=dict(rows=rows,events=events)
            if (p.peek() or footer!=dict(self.counts,event_digest=self.digest)
                    or self.expected is not None and self.digest!=self.expected):
                raise ValueError('Event dictionary digest, count or trailing data differs')


def expand_event_dictionary(source, target, *, expected_digest=None):
    """Explicit exact legacy JSON export for a reader that needs it."""
    reader=DictionaryEvents(source,expected_digest)
    with Path(target).open('x',encoding='utf-8') as stream:
        stream.write('{"events":[')
        for i,row in enumerate(reader.batches()):
            if i:stream.write(',')
            stream.write('[')
            for j,event in enumerate(row):
                if j:stream.write(',')
                stream.write(_canonical(event))
            stream.write(']')
        stream.write(']')
        for key,value in reader.metadata.items():stream.write(','+_canonical(key)+':'+_canonical(value))
        stream.write('}\n')
    return reader.digest


def _event_materialization(tree):
    """Specialize only the two exact read-only comprehensions, on loaded SOURCE."""
    counts=[0,0]
    class Replace(ast.NodeTransformer):
        def visit_ListComp(self,node):
            rendered=ast.unparse(node)
            if rendered=='[[e.to_dict() for e in r.rte_occurrence.events] for r in requests]':
                counts[0]+=1;return ast.copy_location(ast.parse('_event_rows(requests)',mode='eval').body,node)
            if rendered=='[e for row in eventrows for e in row]':
                counts[1]+=1;return ast.copy_location(ast.parse('_flat_events(eventrows)',mode='eval').body,node)
            return self.generic_visit(node)
    return Replace().visit(tree),tuple(counts)


def _legacy_event_tree(tree):
    # The runner has already read its immutable legacy SOURCE. Keep the
    # original selection/validation AST unchanged except for event views.
    result=deepcopy(tree)
    fn=next((n for n in result.body if isinstance(n,ast.FunctionDef) and n.name=='acquire_path'),None)
    if fn is None:return tree
    _,counts=_event_materialization(fn)
    return ast.fix_missing_locations(result) if counts==(1,1) else tree


def prepare_lazy_price_runner(runner,provider,stack):
    """Install representation-only specializations when the pinned call sites match."""
    if not all(hasattr(provider,name) for name in ('blob','BASE','F')) or not hasattr(runner,'selected_acquire_path'):
        return False
    try:
        raw=provider.blob(provider.BASE,'src/trottertracks/compiled_price_fast.py')
        tree=ast.parse(raw);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_backend_class')
        _,counts=_event_materialization(fn)
        if counts!=(1,0):return False
        backend=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        backend_ns=dict(provider.F.__dict__,_event_rows=EventRows)
        exec(compile(backend,str(getattr(provider.F,'__file__','compiled_price_fast')),'exec'),backend_ns)
        selector=ast.parse(textwrap.dedent(inspect.getsource(runner.selected_acquire_path)))
        selected=selector.body[0]
        positions=[i for i,node in enumerate(selected.body) if isinstance(node,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id=='tree' for t in node.targets)]
        if len(positions)!=1:return False
        selected.body.insert(positions[0]+1,ast.parse('tree = _legacy_event_tree(tree)').body[0])
        ns=dict(runner.__dict__,_legacy_event_tree=_legacy_event_tree)
        exec(compile(ast.fix_missing_locations(selector),str(getattr(runner,'__file__','selected_price_runner')),'exec'),ns)
    except (OSError,TypeError,ValueError,KeyError,StopIteration,SyntaxError,AttributeError):
        return False
    stack.enter_context(patch.dict(provider.__dict__,_event_rows=EventRows,_flat_events=FlatEvents))
    stack.enter_context(patch.object(provider.F,'make_backend_class',backend_ns['make_backend_class']))
    stack.enter_context(patch.object(runner,'selected_acquire_path',ns['selected_acquire_path']))
    return True

def streaming_digest(value):
    """The fixed canonical JSON digest, without producing a giant string/bytes pair."""
    hasher = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    readonly = OrderedDict()
    def emit(value):
        if isinstance(value, (tuple, list, EventRows, EventRow)):
            hasher.update(b'[')
            # Event rows are lists of small dictionaries. Use the C encoder
            # on bounded batches; never encode the whole path in one string.
            if isinstance(value, EventRows):
                for position,item in enumerate(value):
                    if position:hasher.update(b',')
                    emit(item)
            elif isinstance(value, EventRow):
                for position,item in enumerate(value):
                    if position:hasher.update(b',')
                    hasher.update(_canonical_readonly_event(item, readonly).encode())
            elif all(not isinstance(item, (tuple, list, EventRows, EventRow)) for item in value):
                for start in range(0, len(value), 256):
                    if start:hasher.update(b',')
                    hasher.update(encoder.encode(value[start:start+256])[1:-1].encode())
            else:
                for position, item in enumerate(value):
                    if position:hasher.update(b',')
                    emit(item)
            hasher.update(b']')
        elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
            hasher.update(b'{')
            for position, key in enumerate(sorted(value)):
                if position:hasher.update(b',')
                hasher.update(encoder.encode(key).encode()+b':');emit(value[key])
            hasher.update(b'}')
        else:
            for piece in encoder.iterencode(value):hasher.update(piece.encode())
    emit(value)
    return hasher.hexdigest()


def compact_event_writer(original):
    """Same complete JSON document and path; avoid indentation overhead.

    Keep the existing writer for all other receipts, and retain exclusive
    creation, seed order, and the content digest. The existing split-axis
    writer still hard-links one common event document.
    """
    def write(path, value):
        path = Path(path)
        if path.name != 'events_PRIVATE.json':return original(path, value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
    return write


def dictionary_event_writer(original):
    def write(path,value):
        if Path(path).name != 'events_PRIVATE.json':return original(path,value)
        return write_event_dictionary(path,value)
    return write


def pair_pending_axis_jobs(original):
    """New paths share the original sampled trajectory and composition for X/Y.

    The fixed planner still checks all original STARTED/STOP/axis witnesses.
    A one-axis continuation is never expanded to a second compilation.
    """
    def plan(cell,index,out,evidence,parallel=False):
        jobs=original(cell,index,out,evidence,parallel=parallel)
        # A saved/held companion must keep its independent ownership markers.
        if parallel and len(jobs)==2 and [job[0]['request']['axes'] for job in jobs]==[['X'],['Y']]:
            return original(cell,index,out,evidence,parallel=False)
        return jobs
    return plan


@contextmanager
def shared_event_dicts(event_type):
    """Price-provider-only bounded cache; its consumers read event rows only."""
    original = event_type.to_dict
    cache = OrderedDict()
    def as_dict(event):
        key = id(event)
        if key in cache:
            cache.move_to_end(key)
            return cache[key][1]
        result = original(event)
        cache[key] = (event, result)
        if len(cache) > 4096:cache.popitem(last=False)
        return result
    with patch.object(event_type, 'to_dict', as_dict):yield


@contextmanager
def reuse_df_price_flanks(runner):
    """Warm templates after the existing input checks and before the workers fork."""
    load_runtime = runner.runtime
    with ExitStack() as stack:
        patched = set()
        caches = {}
        warmed = {}
        event_modules = set()
        science_modules = set()
        digest_modules = set()
        lazy_providers = set()
        if hasattr(runner,'pending_axis_jobs'):
            stack.enter_context(patch.object(runner,'pending_axis_jobs',pair_pending_axis_jobs(runner.pending_axis_jobs)))

        def runtime():
            provider = load_runtime()
            from trotterlib import rte
            if rte not in event_modules:
                stack.enter_context(reuse_df_price_events(rte))
                stack.enter_context(shared_event_dicts(rte.RTEEvent))
                event_modules.add(rte)
            runtime_type = provider.D.NativeRuntime
            if runtime_type in patched:
                return provider
            original_context = runtime_type.context

            def context(instance, request):
                result = original_context(instance, request)
                science = getattr(instance, 'science', None)
                if science is not None and science not in science_modules:
                    stack.enter_context(patch.object(science, 'write', dictionary_event_writer(science.write)))
                    science_modules.add(science)
                if science is not None and science.m not in digest_modules:
                    # Bind only when the fixed provider really uses this canonical
                    # representation; never change a different hash algorithm.
                    probe = {'events': [('X', -0.0, 'λ')], 'seed': 17}
                    if science.m.digest(probe) == streaming_digest(probe):
                        stack.enter_context(patch.object(science.m, 'digest', streaming_digest))
                        digest_modules.add(science.m)
                if science is not None and science.m in digest_modules and id(provider) not in lazy_providers:
                    prepare_lazy_price_runner(runner,provider,stack)
                    lazy_providers.add(id(provider))
                if id(result) in warmed:
                    return result
                native = result.native
                if native not in caches:
                    cache = caches[native] = Flanks(native)
                    stack.enter_context(patch.object(native, 'build_partial_native', cache.build))
                cache = caches[native]
                candidate = request['physical_context']['physical_candidate']
                blocks = result.prep.deterministic_blocks
                if type(blocks) is tuple:
                    steps = result.h['core'].schedule(candidate['formula'], candidate['q'],
                        'ABS_TIME_PER_OUTER', candidate['R_outer'], T=candidate['T'])
                    times = {native._real(step['signed_time'], 'T').hex() for step in steps}
                    for signed_time in sorted(times):
                        cache.templates(blocks, result.prep.num_system_qubits, float.fromhex(signed_time))
                warmed[id(result)] = result
                return result

            stack.enter_context(patch.object(runtime_type, 'context', context))
            patched.add(runtime_type)
            return provider

        stack.enter_context(patch.object(runner, 'runtime', runtime))
        yield
