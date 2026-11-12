"""S0a saved-path parity preparation. No scientific work at import/prepare.

The only numerical entry is execute(..., execute=True), under separate user
scientific permission. Four saved paths, two axes, no old S recount/fallback.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import r_price_phase1 as c1

BASE = 'fcf5b64fac24f3897ab3408b84fd8256f46e8513'
C1 = '2e6a0cdf5a647058cc7afeeb5741bfafa24c974b'
C1_SOURCE = '1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7'
B = 'f09af82482aad11460c4bb73d3e13d7ada40be6f'
RUN = 'docs/research/r_price_phase1_results_20261111/run01/'
CELLS = ('H4_q8_R8','H4_q16_R8','H6_q1_R32','H6_q2_R32')
SOURCE_FILES = ('src/trottertracks/s0a_saved_structure.py',
                'src/trottertracks/s0a_structure_context.py','scripts/run_s0a_saved_structure.py',
                'src/trottertracks/mean_compiled_price.py','src/trottertracks/r_price_phase1.py',
                'src/trottertracks/r_price_phase1_science.py','src/trottertracks/r_price_phase1_runner.py',
                'src/trottertracks/track_d_native_resources.py')


def saved(root, path):return json.loads(c1.blob(root,C1,RUN+path))


def selected(root):
    """Read fixed public metadata; select original index 0 without reanalysis."""
    plan=saved(root,'PLAN.json');c1.validate_plan(plan)
    if plan['source']!=C1_SOURCE:raise ValueError('C1_SOURCE')
    cal=saved(root,'CALIBRATION.json');val=saved(root,'VALIDATION.json')['data']
    rows=[]
    for name in CELLS:
        cell=next(c for c in plan['cells'] if c['id']==name)
        records=(cal if cell['role']=='CALIBRATION' else val)[name]
        matches=[r for r in records if r['index']==0]
        if len(matches)!=1:raise ValueError('SAVED_PATH_SELECTOR')
        record=matches[0]
        if (record['cell_id']!=name or record['candidate_id']!=cell['candidate_id']
            or record['cohort']!=cell['cohort'] or record['conditional'] is not False
            or record['source']['execution_source']!=C1_SOURCE
            or record['compiler']!=cell['request']['identity']['profile']['compiler']):raise ValueError('SAVED_PATH_BINDING')
        raw=Path(record['source']['private_result']).with_name('events_PRIVATE.json')
        rows.append(dict(cell=cell,record=record,raw=str(raw)))
    return plan,rows


def prepare(root, source):
    plan,rows=selected(root)
    # S and C are deliberately absent from this preparation artifact.
    return dict(status='PREPARED_NOT_EXECUTED',source=source,C1_commit=C1,B_commit=B,
        C1_plan_digest=c1.digest(plan),paths=[dict(cell_id=r['cell']['id'],
            candidate_id=r['cell']['candidate_id'],original_role=r['cell']['role'],index=0,
            cohort=r['cell']['cohort'],identity=deepcopy(r['cell']['request']['identity']),
            law_binding=deepcopy(r['cell']['request']['law_binding']),profile=deepcopy(r['cell']['request']['identity']['profile']),
            raw=r['raw'],event_digest=r['record']['event_digest'],raw_exists=Path(r['raw']).is_file(),
            saved_value_ref=RUN+('CALIBRATION.json' if r['cell']['role']=='CALIBRATION' else 'VALIDATION.json'),
            S_stream=None,parity=None) for r in rows],runtime=deepcopy(plan['runtime']),
        axes=list(c1.AXES),new_stream_S=8,new_legacy_S=0,new_compile_sampling_fit_mean_N_G=0,
        physical_state_closure=None,mean_compiled_price=None,qualified_for_candidate=False,
        development_diagnostic=True,fresh_holdout=False,mandatory_stop=True)


def _pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('DUPLICATE_JSON_KEY')
        out[k]=v
    return out


class _JSON:
    """Stdlib decoder, one event/metadata value at a time; bounded read calls."""
    def __init__(self, file, chunk):
        if type(chunk) is not int or chunk<1:raise ValueError('CHUNK_SIZE')
        self.file=file;self.chunk=chunk;self.buf='';self.pos=0;self.eof=False
        self.decoder=json.JSONDecoder(object_pairs_hook=_pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NONFINITE_JSON')))

    def more(self):
        self.buf=self.buf[self.pos:];self.pos=0
        value=self.file.read(self.chunk)
        self.buf+=value;self.eof=not value

    def peek(self):
        while True:
            while self.pos<len(self.buf) and self.buf[self.pos].isspace():self.pos+=1
            if self.pos<len(self.buf):return self.buf[self.pos]
            if self.eof:return ''
            self.more()

    def take(self, char):
        if self.peek()!=char:raise ValueError('JSON_SYNTAX_EXPECTED_'+char)
        self.pos+=1

    def value(self):
        self.peek()
        while True:
            try:value,end=self.decoder.raw_decode(self.buf,self.pos)
            except json.JSONDecodeError:
                if self.eof:raise ValueError('TRUNCATED_OR_INVALID_JSON')
                self.more();continue
            self.pos=end;return value


class SavedEvents:
    """Forward-only C1 JSON reader with canonical nested-event SHA256.

    Preserves stdlib JSON binary64 values, original event and legal-tail order.
    Each yielded tail must be exhausted before requesting the next one.
    """
    def __init__(self,path,q,allocation,expected_digest,validator,*,chunk=65536):
        self.path=Path(path);self.q=q;self.allocation=tuple(allocation)
        self.expected=expected_digest;self.validator=validator;self.chunk=chunk
        self.metadata={};self.digest=None

    def batches(self):
        h=hashlib.sha256();tails=0;seen=set()
        with self.path.open(encoding='utf-8') as file:
            p=_JSON(file,self.chunk);p.take('{')
            while p.peek()!='}':
                if seen:p.take(',')
                key=p.value();p.take(':')
                if key in seen or key not in ('events','seeds','laws','conditional'):raise ValueError('RAW_SCHEMA')
                seen.add(key)
                if key!='events':self.metadata[key]=p.value();continue
                p.take('[');h.update(b'[')
                while p.peek()!=']':
                    if tails:p.take(',');h.update(b',')
                    j=tails;tails+=1
                    if tails>self.q*len(self.allocation):raise ValueError('EXTRA_TAIL')
                    p.take('[');h.update(b'[');complete=[False]
                    def tail(j=j):
                        count=0
                        while p.peek()!=']':
                            if count:p.take(',');h.update(b',')
                            event=p.value();count+=1
                            if not isinstance(event,dict) or count>self.allocation[j%len(self.allocation)]:raise ValueError('EXTRA_OR_INVALID_EVENT')
                            self.validator(j,event)
                            h.update(json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
                            yield event
                        p.take(']');h.update(b']')
                        if count!=self.allocation[j%len(self.allocation)]:raise ValueError('MISSING_EVENT')
                        complete[0]=True
                    yield tail()
                    if not complete[0]:raise ValueError('TAIL_NOT_FULLY_CONSUMED')
                p.take(']');h.update(b']')
            p.take('}')
            if p.peek() or seen!={'events','seeds','laws','conditional'}:raise ValueError('RAW_SCHEMA_OR_TRAILING_DATA')
        if tails!=self.q*len(self.allocation):raise ValueError('MISSING_TAIL')
        self.digest=h.hexdigest()
        if self.digest!=self.expected:raise ValueError('EVENT_DIGEST_MISMATCH')


def check_metadata(reader,cell,expected_laws):
    if (reader.metadata.get('conditional') is not False or reader.metadata.get('seeds')!=cell['seeds'][0]
            or reader.metadata.get('laws')!=expected_laws):raise ValueError('RAW_LAW_SEED_OR_CONDITIONAL')


def parity_one(context,cell,record,raw):
    """Scientific leaf. Caller must have permission; never run in preparation."""
    from .mean_compiled_price import stream_structure
    candidate=cell['request']['identity']['candidate']
    allocation=cell['request']['identity']['structure']['allocation']
    laws,validator=context.event_validator(cell)
    def reader():return SavedEvents(raw,candidate['q'],allocation,record['event_digest'],validator)
    # Check complete identity/law/digest before the first Accounting instance.
    preflight=reader()
    for batch in preflight.batches():
        for _ in batch:pass
    check_metadata(preflight,cell,laws)
    values=[]
    for axis in c1.AXES:
        r=reader()
        value=stream_structure(context.Accounting,context.blocks,candidate,context.scalar,axis,
            r.batches(),allocation=allocation)['accounting']['structural_rz']
        check_metadata(r,cell,laws);values.append(value)
    if values!=record['S']:raise ValueError('SAVED_S_PARITY_MISMATCH')
    return dict(cell_id=cell['id'],candidate_id=cell['candidate_id'],index=0,original_role=cell['role'],
        cohort=cell['cohort'],axes=list(c1.AXES),event_digest=record['event_digest'],S_stream=values,
        S_saved=record['S'],parity=True,unit='FULL_WRAPPER_STRUCTURAL_RZ_SINGLE_PATH',
        mean_compiled_price=None,qualified_for_candidate=False)


def execute(root,plan,out,*,execute=False,context_factory=None):
    if not execute:raise ValueError('EXPLICIT_SCIENTIFIC_EXECUTION_REQUIRED')
    expected=prepare(root,plan['source'])
    if plan!=expected:raise ValueError('FIXED_S0A_PLAN_CHANGED')
    for name in SOURCE_FILES:
        if (Path(root)/name).read_bytes()!=c1.blob(root,plan['source'],name):raise ValueError('S0A_SOURCE_CHANGED:'+name)
    if any(not p['raw_exists'] for p in plan['paths']):raise FileNotFoundError('SAVED_EVENTS_MISSING_NO_SAMPLING_FALLBACK')
    from .r_price_phase1_runner import write
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write(out/'PLAN.json',plan)
    try:
        old,rows=selected(root)
        if context_factory is None:
            from .s0a_structure_context import Context
            context_factory=lambda cell:Context(root,old,cell['system'])
        results=[];contexts={}
        for row in rows:
            cell=row['cell'];system=cell['system']
            if system not in contexts:contexts[system]=context_factory(cell)
            results.append(parity_one(contexts[system],cell,row['record'],row['raw']))
        write(out/'RESULTS.json',dict(paths=results,new_stream_S=8,new_legacy_S=0,mean_compiled_price=None,
            physical_state_closure=None,qualified_for_candidate=False,full_rpe_resources=None))
        write(out/'STOP.json',dict(status='S0A_PARITY_COMPLETE_MANDATORY_STOP',mandatory_stop=True))
    except BaseException as error:
        write(out/'STOP.json',dict(status='S0A_FAILED_NO_AUTOMATIC_RETRY',error=type(error).__name__+': '+str(error),mandatory_stop=True))
        raise
    return dict(status='S0A_PARITY_COMPLETE_MANDATORY_STOP',new_stream_S=8)
