"""Bounded stage/memory diagnostics; no scientific libraries or computations."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import resource
import traceback

_ACTIVE_TRACE = ContextVar('ax2b_h4_v4_resource_trace', default=None)


def memory_snapshot():
    result = {'cumulative_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmSize', 'VmHWM'):
                result[key + '_bytes'] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        result['proc_status_available'] = False
    return result


class ResourceTrace:
    """Write small exclusive records through the pilot's aggregate-cap writer."""
    def __init__(self, writer, elapsed):
        self.writer, self.elapsed = writer, elapsed
        self.context = {}
        self.last_failure_context = None
        self.sequence = 0

    def event(self, kind, **fields):
        if self.sequence >= 1024:
            raise RuntimeError('DIAGNOSTIC_RECORD_CAP')
        record = {'kind': kind, 'elapsed_seconds': self.elapsed(),
                  'context': dict(self.context), 'memory': memory_snapshot(), **fields}
        self.writer(f'diagnostic_{self.sequence:04d}.json', record)
        self.sequence += 1

    @contextmanager
    def bind(self):
        token = _ACTIVE_TRACE.set(self)
        try:
            yield
        finally:
            _ACTIVE_TRACE.reset(token)

    @contextmanager
    def stage(self, stage, **context):
        previous = self.context
        self.context = {**previous, **context, 'stage': stage}
        try:
            self.event('begin')
            yield
            self.event('end')
        except Exception:
            # Keep the deepest context. Avoid new diagnostic allocations while
            # unwinding a MemoryError; the outer handler releases its reserve.
            if self.last_failure_context is None:
                self.last_failure_context = self.context
            raise
        finally:
            self.context = previous

    def failure(self, error):
        frames = []
        tb = error.__traceback__
        while tb is not None and len(frames) < 32:
            code = tb.tb_frame.f_code
            frames.append({'file': code.co_filename[-512:],
                           'function': code.co_name[:128], 'line': tb.tb_lineno})
            tb = tb.tb_next
        # Do not retain locals or source text. In particular, failed circuit
        # builders may otherwise remain alive through the exception traceback.
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
        record = {'exception_type': type(error).__name__, 'message': str(error)[:1000],
                  'context': self.last_failure_context or self.context,
                  'traceback_frames': frames, 'traceback_frame_limit': 32,
                  'locals_recorded': False, 'memory_after_unwind': memory_snapshot(),
                  'scope': 'diagnostic snapshot; allocation failure peak not inferred'}
        self.writer('failure_diagnostics.json', record)
        return {'artifact': 'failure_diagnostics.json', 'context': record['context'],
                'last_frame': frames[-1] if frames else None}


@contextmanager
def fingerprint_stage(circuit):
    trace = _ACTIVE_TRACE.get()
    if trace is None:
        yield
    else:
        with trace.stage('numeric_fingerprint', instructions=len(circuit.data)):
            yield
