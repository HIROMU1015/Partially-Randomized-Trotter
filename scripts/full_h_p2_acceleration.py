"""Process-local acceleration of the fixed Full-H circuit recipe.

No sampler/lowering/compiler-profile changes. Controlled gate synthesis is
memoized by its exact angle; X/Y compilation uses separate forked processes.
"""
from contextlib import contextmanager
from pathlib import Path
import gzip
import hashlib
import json
import multiprocessing as mp
import time
import traceback


@contextmanager
def fast_legacy_builds(native):
    """Keep the existing identical-byte writer active for legacy replay as well."""
    from h46_b3_fast_fingerprint import use_fast_fingerprint, legacy, _ALIASES
    import importlib
    module = importlib.import_module('.'+_ALIASES[0],'trottertracks.resource_applicability')
    original = native.compose;timings=[]
    def compose(*args,**kwargs):
        started=time.monotonic()
        activate=module.canonical_qiskit_circuit_fingerprint is legacy.canonical_qiskit_circuit_fingerprint
        try:
            if activate:
                with use_fast_fingerprint():return original(*args,**kwargs)
            return original(*args,**kwargs)
        finally:
            timings.append(dict(seconds=time.monotonic()-started,fast_writer_activated_here=activate))
    native.compose = compose
    try:
        yield timings
    finally:
        native.compose = original


@contextmanager
def memoized_controlled_rotations(provider):
    from qiskit.circuit.library import RZGate, RZZGate
    base = provider.QiskitDFRTEEventCircuitBuilder
    original = base._append_central_application
    cache = {}; stats = dict(constructed=0, reused=0)

    def append(circuit, event, application, *, controlled, ancilla_qubit):
        support = application.diagonal_pauli_support
        if controlled and application.role == 'rotation' and support is not None and len(support) in (1, 2):
            theta = 2.0 * event.unsigned_rotation_angle
            key = len(support), theta.hex()
            if key not in cache:
                cache[key] = (RZGate(theta) if len(support) == 1 else RZZGate(theta)).control(1)
                stats['constructed'] += 1
            else:
                stats['reused'] += 1
            circuit.append(cache[key].copy(), [ancilla_qubit, *support])
            return 0.0
        return original(circuit, event, application, controlled=controlled, ancilla_qubit=ancilla_qubit)

    base._append_central_application = staticmethod(append)
    try:
        yield stats
    finally:
        base._append_central_application = staticmethod(original)


def sine_from_cosine(wrapper):
    """The fixed wrapper differs only by sdg before its final h/measure."""
    from qiskit.circuit import CircuitInstruction
    from qiskit.circuit.library import SdgGate
    result = wrapper.copy()
    if [i.operation.name for i in result.data[-2:]] != ['h', 'measure']:
        raise ValueError('Expected measured cosine wrapper')
    ancilla = result.data[-2].qubits[0]
    result.data.insert(len(result.data)-2, CircuitInstruction(SdgGate(), (ancilla,), ()))
    return result


class ParallelAxesCompiler:
    """Compile each axis once; the consumer still sees its original X/Y calls.

    The predicted Y wrapper must match the actual later Y fingerprint exactly.
    Child results are small; representative compiled circuits use native QPY.
    """
    def __init__(self, native, fingerprint, *, compiler=None):
        self.native = native; self.fingerprint = fingerprint
        self.compiler = compiler or native.compile_cached
        self.workers = {}; self.receipts = {}; self.expected = {}; self.timings = {}

    def _start(self, axis, wrapper, fp, representative):
        receive, send = mp.get_context('fork').Pipe(duplex=False)
        def work():
            started = time.monotonic()
            try:
                record, qc, reused = self.compiler(wrapper, fp, representative)
                send.send(dict(status='PASS',record=record,reused=reused,has_qc=qc is not None,
                               seconds=time.monotonic()-started))
            except BaseException:
                send.send(dict(status='STOP',error=traceback.format_exc()))
            finally:
                send.close()
        child = mp.get_context('fork').Process(target=work, name='FullH_compile_'+axis)
        child.start(); send.close()
        self.workers[axis] = child, receive
        self.expected[axis] = fp

    def _collect(self, axis, representative):
        child, pipe = self.workers[axis]
        try:
            result = pipe.recv()
        except EOFError:
            result = dict(status='STOP',error='Compiler process exited without a record')
        child.join(); pipe.close()
        self.receipts[axis] = result
        if child.exitcode != 0 or result['status'] != 'PASS':
            raise ValueError('Parallel '+axis+' compiler STOP: '+result.get('error','nonzero exit'))
        self.timings[axis] = result['seconds']
        qc = None
        if representative and result['has_qc']:
            from qiskit import qpy
            path = self.native.PRIVATE/'compiled_cache'/(self.expected[axis]+'_compiled.qpy.gz')
            with gzip.open(path,'rb') as stream:
                qc = qpy.load(stream)[0]
        return result['record'], qc, result['reused']

    def __call__(self, wrapper, fp, representative):
        if not self.workers:
            self._start('cosine',wrapper,fp,representative)
            sine = sine_from_cosine(wrapper); sine_fp = self.fingerprint(sine)
            if sine_fp == fp:
                raise ValueError('X/Y wrappers have identical fingerprints')
            self._start('sine',sine,sine_fp,representative)
            return self._collect('cosine',representative)
        if fp != self.expected['sine']:
            raise ValueError('Actual Y differs from the fixed predicted wrapper')
        return self._collect('sine',representative)

    def close(self):
        # Already-started compiles finish and retain their native records; no retries.
        for axis in self.workers:
            if axis not in self.receipts:
                try:
                    self._collect(axis,False)
                except ValueError:
                    pass


def accelerated_path(P, full, row, cohort, index, seeds, *, parents=None, compiler=None):
    """The existing P2/P1/native path, with local memoization and parallel axes."""
    native = full.setup()
    provider = __import__('trotterlib.df_rte_qiskit',fromlist=['x'])
    from h46_b3_fast_fingerprint import FingerprintSession
    fp = FingerprintSession()
    with memoized_controlled_rotations(provider) as cache, fast_legacy_builds(native) as builds:
        parallel = ParallelAxesCompiler(native,fp.fingerprint,compiler=compiler)
        try:
            result = P.acquire_path(full,row,cohort,index,seeds,compile_cached=parallel,parents=parents)
        finally:
            parallel.close()
        result['acceleration'] = dict(gate_cache=dict(cache),parallel_axes=True,compile_wall_seconds=parallel.timings,
                                      build_timings=builds)
        return result


def restore_saved_requests(row,prep,payload,laws):
    from h46_same_event import decode_event
    from h46_m5_contract import core
    from h46_pr_pilot_repair import canonical_tau
    from trotterlib import rte
    from trotterlib.df_partial_s2 import DFPartialS2StepRequest
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    schedule=core.schedule(row['formula'],row['q'],'ABS_TIME_PER_OUTER',row['R_outer'],T=row['T'])
    if len(payload['events'])!=len(schedule) or len(payload['seeds'])!=len(schedule):
        raise ValueError('Saved event group allocation differs')
    requests=[]
    for step,raw,seed,law in zip(schedule,payload['events'],payload['seeds'],laws,strict=True):
        if len(raw)!=step['rte_steps']:raise ValueError('Saved event count differs')
        events=tuple(decode_event(rte,e) for e in raw)
        tau=canonical_tau(prep,step['signed_time'],step['rte_steps'])
        config,dist=rte.make_rte_config(prep.rte_preparation.symbolic_tail,evolution_time=step['signed_time'],
            rte_steps=step['rte_steps'],finite_taylor_order=row['K'],
            truncation_tolerance=_explicit_cutoff_tolerance(tau,row['K']),seed=seed)
        if law!=dict(steps=step['rte_steps'],orders=list(dist.orders),probabilities=list(dist.order_probabilities)):
            raise ValueError('Saved finite law differs')
        for event in events:prep.rte_preparation.resolve_event_basis_definitions(event)
        occurrence=DFRTEEventSequenceCircuitRequest(events=events,component_specs=prep.rte_preparation.component_specs,
            controlled=True,ancilla_qubit=prep.num_system_qubits,tail_id=config.tail_id,tail_hash=config.tail_hash,
            occurrence_rte_steps=step['rte_steps'])
        requests.append(DFPartialS2StepRequest(preparation=prep,step_time=step['signed_time'],rte_config=config,
            rte_distribution=dist,rte_occurrence=occurrence,controlled=True,ancilla_qubit=prep.num_system_qubits,seed=seed))
    return requests


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def resume_saved_path(P,full,row,cohort,index,seeds,checkpoint):
    """Finish the original parity/phase checks; reuse sampling/compile/numeric gates."""
    started=time.monotonic()
    for path,digest in checkpoint['files'].items():
        if file_hash(path)!=digest:raise ValueError('Saved checkpoint bytes changed: '+path)
    trial=json.loads(Path(checkpoint['trial']).read_text())
    payload=json.loads(Path(checkpoint['events']).read_text())
    if (trial['status']!='PASS' or trial['candidate']!=row or trial['cohort']!=cohort or trial['index']!=index
            or payload['cohort']!=cohort or payload['seeds']!=seeds or payload['conditional'] is not False
            or any(not g['error']<g['tolerance'] or g['phase_alignment'] for g in trial['checks'])):
        raise ValueError('Saved trial/seed/correctness binding differs')
    expected={'low_control','high_control_leakage','independent_occupation_vector',
        'Hadamard_cosine','Hadamard_sine','compiled_vector_cosine','compiled_vector_sine'} if index==0 else set()
    if {g['check'] for g in trial['checks']}!=expected:
        raise ValueError('Saved numerical correctness gates incomplete')
    native=full.setup();ctx=full.context(row['system']);prep=ctx.preparations[row['prefix']]
    scalar=ctx.ham.constant+prep.extracted_identity_coefficient
    if scalar!=trial['scalar'] or prep.preparation_hash!=trial['preparation_hash']:
        raise ValueError('Saved scalar/preparation differs')
    requests=restore_saved_requests(row,prep,payload,trial['laws'])
    provider=__import__('trotterlib.df_rte_qiskit',fromlist=['x'])
    from h46_b3_fast_fingerprint import FingerprintSession
    from trottertracks.resource_applicability import ax2b_native_df_v5 as wrapper_api
    fp=FingerprintSession().fingerprint
    with memoized_controlled_rotations(provider) as cache,fast_legacy_builds(native) as builds:
        with P.P1.zz_module().efficient_controlled_zz(provider):
            builder_lowering=getattr(provider.QiskitDFRTEEventCircuitBuilder,'lowering_id',None)
            evolution=native.compose(row,prep,requests,scalar,fp)
        legacy=native.compose(row,prep,requests,scalar,lambda qc:'LEGACY_REPLAY')
        parity=P.P1.zz_module().instruction_parity(legacy.circuit,evolution.circuit)
    n=prep.num_system_qubits;counts=P.P1.op_counts(evolution.circuit)
    one_ids={c.component_id for c in prep.rte_preparation.symbolic_tail.components if c.df_fragment_id=='full-h-one-body'}
    events=P.event_projection([r.rte_occurrence.events for r in requests])
    if json.dumps(events,sort_keys=True)!=json.dumps(payload['events'],sort_keys=True):
        raise ValueError('Saved events changed during restoration')
    facts=dict(trial_gates=[g['check'] for g in trial['checks']],builder_lowering_id=builder_lowering,
        legacy_rzz_control_instructions=counts.get('crzz',0),new_crz=counts.get('crz',0),parity=parity,
        one_body_applications=sum(a.component_id in one_ids for r in requests for e in r.rte_occurrence.events for a in e.application_sequence),
        scalar=scalar,scalar_expected=scalar,
        scalar_event_phases=P.P1.scalar_event_phase_gate(evolution.circuit,n,events,scalar,scalar,row['T'],provider=provider),
        scalar_once_legacy_diagnostic=P.P1.scalar_once(evolution.circuit,n,-scalar*row['T']),provider_file=str(provider.__file__))
    wrappers={};fps={};receipts={}
    for axis in ('cosine','sine'):
        w=wrapper_api.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,
            max_instructions=len(evolution.circuit.data)+4)
        digest=fp(w);path=Path(checkpoint['private'])/'compiled_cache'/(digest+'.json')
        if str(path) not in checkpoint['files']:raise ValueError('Wrapper not in checkpoint compile set')
        rec=json.loads(path.read_text())
        if rec['fingerprint']!=digest or rec['compiler']!=native.COMPILER_HASH or rec['metrics']!=trial['prices'][axis]:
            raise ValueError('Saved wrapper/compile/price binding differs')
        wrappers[axis]=w;fps[axis]=digest
        receipts[axis]=dict(record=rec,fingerprint=digest,reused=True,locator=dict(path=str(path),sha256=file_hash(path)),
            producer_commit=checkpoint['source_commit'],wall_seconds=0.,checkpoint_reused=True)
    xy=P.P1.xy_phase_gate(wrappers,n,events,scalar,scalar,row['T'],provider=provider)
    if (builder_lowering!=P.P1.LOWERING_ID or parity['status']!='PASS' or parity['replaced_controlled_zz']<=0 or facts['legacy_rzz_control_instructions']!=0
            or facts['one_body_applications']<=0 or facts['scalar_event_phases']['status']!='PASS' or xy['status']!='PASS'):
        raise ValueError('Restored checkpoint parity/phase/XY correctness failed')
    return dict(trial=trial,checks=facts,xy_phase_gate=xy,events=events,seeds=seeds,fingerprints=fps,
        compile_receipts=receipts,index=index,cohort=cohort,row=row,sampling='SAVED_NORMAL_EVENTS_NO_DRAW',parents=[],
        checkpoint=checkpoint,acceleration=dict(gate_cache=dict(cache),verification_seconds=time.monotonic()-started,
            new_sampling=0,new_compiles=0,numerical_gates_reused_from=checkpoint['trial'],build_timings=builds))
