"""Atomic exclusive records and bounded progress; stdlib only."""
from __future__ import annotations
import json
import os
from pathlib import Path
import resource
import time
from .ax2b_h6_controller import BoundedWriter
from .ax2b_limits import output_size


class AtomicWriter(BoundedWriter):
    observer = None

    def write(self, name, value, *, diagnostic=False, terminal=False):
        if Path(name).name != name or not name.endswith('.json'):
            raise ValueError('OUTPUT_NAME')
        payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
        limit = self.byte_cap if terminal else self.byte_cap-self.reserve
        if output_size(self.output) + len(payload) > limit:
            raise RuntimeError('OUTPUT_WRITE_CAP')
        if diagnostic:
            self.diagnostics.take('records')
        temporary = self.output/('.pending_'+name)
        created = False
        try:
            with temporary.open('xb') as stream:
                created = True
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic publication without replacing an existing record.
            os.link(temporary, self.output/name)
        finally:
            if created:
                temporary.unlink(missing_ok=True)
        if self.observer is not None and not name.startswith('progress_') and not terminal:
            self.observer(name, value)


class Progress:
    def __init__(self, writer, unit, *, cap=1024, started=None):
        if type(cap) is not int or cap < 1:
            raise ValueError('PROGRESS_CAP')
        self.writer, self.cap = writer, cap
        self.started = time.monotonic() if started is None else started
        self.row = {'unit': unit, 'phase': None, 'cell': None, 'dps': None, 'last_completed_record': None,
                    'correctness_attempted': 0, 'correctness_completed': 0, 'mp_attempted': 0, 'mp_completed': 0,
                    'event_attempted': 0, 'event_completed': 0, 'primitive_attempted': 0, 'primitive_completed': 0,
                    'control_attempted': 0, 'control_completed': 0, 'oracle_work': None}
        self.count = 0
        writer.observer = self.record

    def update(self, **values):
        if self.count >= self.cap:
            raise RuntimeError('PROGRESS_RECORD_CAP')
        self.row.update(values)
        self.row['elapsed_seconds'] = time.monotonic()-self.started
        self.row['worker_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        self.row['progress_sequence'] = self.count
        self.writer.write('progress_%04d.json'%self.count, self.row, diagnostic=True)
        self.count += 1

    def record(self, name, value):
        if name.startswith('phase_'):
            self.update(phase=value['phase'])
        elif name.endswith('_correctness.json'):
            self.update(correctness_completed=self.row['correctness_completed']+1, last_completed_record=name)
        elif name.endswith(('_mp80.json', '_mp120.json')):
            self.update(mp_completed=self.row['mp_completed']+1, last_completed_record=name)
        elif '_explicit_order' in name:
            self.update(event_completed=self.row['event_completed']+1, last_completed_record=name)
        elif name in ('input_reference.json', 'primitive_validation.json'):
            self.update(last_completed_record=name)


def latest_progress(output):
    """Last atomically published snapshot only; no inference for killed work."""
    paths = sorted(Path(output).glob('progress_[0-9][0-9][0-9][0-9].json'))
    if not paths:
        return None
    if len(paths) > 1024 or paths[-1].stat().st_size > 16384:
        raise ValueError('PROGRESS_READ_CAP')
    return json.loads(paths[-1].read_text())
