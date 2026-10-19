"""Unadopted parallel host v3: a separate entrypoint, never an old-v2 upgrade.

Production requires an independently verified approval AND an adopted release.
This SOURCE is intentionally not such a release. Artificial API accepts only a
private fake port and synthetic factories; it cannot emit a production receipt.
"""
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
import resource
import threading
import time

from . import g6_w2c_b3_parallel_admission as A
from . import g6_w2c_b3_parallel_driver as D
from . import g6_w2c_b3_parallel_scheduler as P

_HOST_LOCK = threading.Lock()


@contextmanager
def _execution_context(contract):
    # RUSAGE_CHILDREN and the driver codec are process-scoped. Overlapping
    # bundles in this same parent cannot share their CPU accounting window.
    A.require(_HOST_LOCK.acquire(blocking=False), 'PARENT_EXECUTION_ALREADY_ACTIVE')
    previous = D.SERIALIZATION_MODE
    try:
        codec = contract.get('settings',{}).get('codec')
        A.require(codec in ('reference','fused'), 'TECHNICAL_VARIANT')
        D.SERIALIZATION_MODE = codec
        yield
    finally:
        D.SERIALIZATION_MODE = previous
        _HOST_LOCK.release()


class ProductionFactory:
    """Minimal spawn payload; contains no authority, ledger, receipt or claim."""
    artificial = False

    def __init__(self, root, candidate, runtime_manifest):
        self.root, self.candidate, self.runtime_manifest = root, candidate, runtime_manifest

    def __call__(self, task, settings):
        A.require(False, 'TECHNICAL_CANDIDATE_NOT_SCIENCE_RELEASE')
        A.check_runtime(self.root, self.runtime_manifest)
        modules = D.load_modules(self.root, self.candidate['source_pins'])
        # A parent-issued task capability delegates this one already admitted
        # pair. It issues no claim and has no admission/approval API.
        permit = D.Boundary(); permit.state = 'ENTERED'
        return D.LegacyEndpoint(modules, task['arm'], task['layer'], self.candidate['decisions']['angle'],
                               lambda:D.check_pins(self.root,self.candidate['source_pins']), permit=permit)


class ProductionBits:
    def __init__(self, root, candidate):
        self.root, self.candidate = root, candidate

    def __call__(self, arm, axis):
        A.require(False, 'TECHNICAL_CANDIDATE_NOT_SCIENCE_RELEASE')
        modules = D.load_modules(self.root,self.candidate['source_pins'])
        return modules['g7'].DeterministicBits(D.seed_for(self.candidate,arm,axis))


def execute_approved_v3(contract, *, root, ledger, authority=None):
    with _execution_context(contract):
        port = A.ProductionAdmission(contract, Path(root), ledger, authority)
        with port:
            port._check(D.identity(port.contract['candidate']))
            return _execute(port, root, ProductionFactory(root,port.contract['candidate'],port.contract['runtime_manifest']),
                            ProductionBits(root,port.contract['candidate']))


def execute_artificial_v3(contract, *, root, ledger, authority, endpoint_factory, bits_factory,
                          admission_fault=lambda point:None, watch=None):
    """Private /tmp fake authority only; unavailable through production API."""
    with _execution_context(contract):
        port = A.ArtificialAdmission(contract, Path(root), ledger, authority, fault=admission_fault)
        with port:
            A.require(endpoint_factory.artificial is True, 'ARTIFICIAL_ENDPOINT_REQUIRED')
            code_path=Path(endpoint_factory.__call__.__code__.co_filename)
            expected=Path(contract['runtime_manifest']['runtime_root'])/contract['runtime_manifest']['artificial_fixture']['path']
            A.require(code_path.resolve() == expected and Path(bits_factory.__code__.co_filename).resolve() == expected and
                      A.A.digest(code_path.read_bytes()) == contract['input_pins'].get('artificial-fixture.py'),
                      'ARTIFICIAL_FACTORY_SOURCE_BINDING')
            return _execute(port, root, endpoint_factory, bits_factory, watch)


def _execute(port, root, factory, bits, watch=None):
    before, child_before, io_before = D.clock(), resource.getrusage(resource.RUSAGE_CHILDREN), P.own_io()
    c, root, control = port.contract, Path(root), None
    candidate, runtime = c['candidate'], c['runtime_manifest']
    settings = P.Settings(**c['settings'])
    result = None
    try:
        binding, _ = port._check(D.identity(candidate))
        receipt = port.admit_artificial(D.identity(candidate)) if port.artificial else port.admit_once(D.identity(candidate))
        A.check_runtime(root,runtime)
        D.need(factory.runtime_manifest == runtime, 'WORKER_RUNTIME_BINDING')
        D.SERIALIZATION_MODE = settings.codec
        output = A.A.exact_path(root,candidate['output_relative'])
        D.need(not output.exists(),'EXCLUSIVE_OUTPUT_NAMESPACE')
        if settings.parent_fault == 'before_output': raise P.Halt('ARTIFICIAL_OUTPUT_BEFORE')
        output.parent.mkdir(parents=True,exist_ok=True); output.mkdir()
        tasks, planned = P.plan_for(settings.profile_n if port.artificial else D.SAVED_N)
        control = P.Coordinator(settings,output,tasks,planned)
        control.log(dict(event='BUNDLE_BINDING', schema=A.SCHEMA, formal_digest=binding['formal_digest'],
                         driver_digest=binding['driver_contract_digest'], runtime_manifest_sha256=D.identity(runtime),
                         source_pins_sha256=D.identity(c['source_pins']), artificial_only=port.artificial))
        for phase in range(3):
            if control.stopped.value: break
            control.run_phase([t for t in tasks if t['phase_index']==phase],root,c['source_pins'],factory,bits,watch)
            try:
                P.validate_success(control.results)
                A.check_runtime(root,runtime)
            except BaseException as exc: control.stop(dict(phase=phase,validation_error=repr(exc)))
        result = control.report()
        if not result['complete'] or len(result['pairs']) != 24: raise P.BundleFailure(result)
        meter, finalize_before = D.Meter(D.clock), D.clock()
        stream = D.Stream(output/'summary.jsonl',meter,byte_limit=settings.result_bytes,flush_rows=1)
        stream.file = P.ReservedFile(stream.file,lambda n:control.budget.reserve('result',n),
                                    lambda n:control.budget.wrote('result',n))
        summary = dict(result, binding=dict(formal_digest=binding['formal_digest'],
                       runtime_manifest_sha256=D.identity(runtime), source_pins_sha256=D.identity(c['source_pins'])),
                       measurement_condition=c['measurement_condition'], artificial_only=port.artificial)
        try: stream.emit(summary)
        finally: stream.close()
        completion = dict(schema='b3_parallel_receipt_v3', event='COMPLETION_RECEIPT',
                          formal_digest=binding['formal_digest'], summary_chain=stream.previous,
                          summary_sha=stream.file_hash.hexdigest(), all_workers_joined=True,
                          source_pins_sha256=D.identity(c['source_pins']), runtime_manifest_sha256=D.identity(runtime))
        control.log(completion)
        result.update(binding=summary['binding'], measurement_condition=c['measurement_condition'],
                      runtime_manifest=runtime, completion_receipt=completion,
                      finalization=dict(coarse=meter.rows,elapsed_ns=D.delta(finalize_before,D.clock())),
                      output=dict(limits=control.budget.limit,reserved=dict(control.budget.reserved),
                                  observed=dict(control.budget.observed)))
        return result
    except BaseException as exc:
        if control is not None:
            if not isinstance(exc,P.BundleFailure): control.stop(dict(parent_failure=repr(exc)))
            if control.active: control.run_phase([],root,c['source_pins'],factory,bits,watch,drain_only=True)
            try:
                if settings.parent_fault == 'stop_save': raise OSError('ARTIFICIAL_STOP_SAVE_FAILURE')
                control.log(dict(event='MANDATORY_STOP',complete=False,states=dict(control.states.counts)))
            except BaseException as secondary: exc.secondary_stop_error = repr(secondary)
            result=control.report();result['binding']=dict(formal_digest=D.identity(c),runtime_manifest_sha256=D.identity(runtime))
            if hasattr(exc,'secondary_stop_error'): result['secondary_stop_error']=exc.secondary_stop_error
            if isinstance(exc,P.BundleFailure): exc.report=result
            else:
                wrapped=P.BundleFailure(result);wrapped.primary=repr(exc);raise wrapped from exc
        raise
    finally:
        if control is not None:
            control.states.close();control.journal.file.close()
        after, children = D.clock(),resource.getrusage(resource.RUSAGE_CHILDREN)
        if isinstance(result,dict):
            result['execution_outer']=dict(parent_cpu_ns=after[0]-before[0],wall_ns=after[1]-before[1],
                worker_cpu_ns=round((children.ru_utime+children.ru_stime-child_before.ru_utime-child_before.ru_stime)*1e9),
                io={k:v-io_before.get(k,0)for k,v in P.own_io().items()})
