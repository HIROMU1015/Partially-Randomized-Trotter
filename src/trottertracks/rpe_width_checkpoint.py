"""Small replayable point deltas for an active width-search controller.

The planner snapshot still owns scientific results and selection decisions.
This journal records preparation transitions between full checkpoints; no
scientific result is synthesized from a queue entry.
"""
import json
import threading
from pathlib import Path

FIELDS = ('status', 'reason', 'configuration_id', 'evaluation_configuration_id',
          'input_dispatch_cycle', 'input_dispatches', 'previous_hold')
COMPLETION_FIELDS = ('status', 'key', 'configuration_id', 'reason', 'failure_class',
                     'retry_protected', 'preparation_recovery', 'readiness')


def identity(point):
    return point['method'], point['family'], str(point['kappa'])


def replay(state, path):
    """Apply only events newer than the atomically written planner snapshot."""
    sequence = int(state.get('point_checkpoint_seq', 0))
    points = {identity(p): p for p in state.get('points', ())}
    path = Path(path)
    if not path.is_file():
        return sequence
    with path.open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'):
                break  # incomplete final write
            event = json.loads(raw)
            if event['seq'] <= sequence:
                continue
            sequence = event['seq']
            point = points.get(tuple(event['point']))
            if point is None:
                # The referenced decision must already be in the full
                # checkpoint; do not invent a new coordinate on replay.
                continue
            point.update(event['fields'])
            for field in event['removed']:
                point.pop(field, None)
            if 'completion' in event:
                point['completion'] = dict(point.get('completion', {}), **event['completion'])
    return sequence


class PointJournal:
    def __init__(self, path, state):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.sequence = replay(state, self.path)
        # A crashed append may leave an incomplete final line. Remove only
        # that disposable journal fragment, never a prior completed event.
        if self.path.is_file():
            with self.path.open('r+b') as stream:
                stream.seek(0, 2)
                size = stream.tell()
                if size:
                    stream.seek(-1, 2)
                    if stream.read(1) != b'\n':
                        start = max(0, size - 1024)
                        while True:
                            stream.seek(start)
                            tail = stream.read(size - start)
                            end = tail.rfind(b'\n')
                            if end >= 0 or start == 0:
                                stream.truncate(start + end + 1)
                                break
                            start = max(0, start - 4096)
        self.last = {}

    def record(self, point):
        fields = {k: point[k] for k in FIELDS if k in point}
        completion = point.get('completion')
        event = dict(point=identity(point), fields=fields,
                     removed=[k for k in FIELDS if k not in point])
        if completion is not None:
            event['completion'] = {k: completion[k] for k in COMPLETION_FIELDS if k in completion}
        with self.lock:
            # Comparing a small, JSON-compatible payload avoids repeated writes
            # when the same completed input is revisited.
            encoded = json.dumps(event, sort_keys=True, default=str)
            if self.last.get(event['point']) == encoded:
                return
            self.sequence += 1
            event['seq'] = self.sequence
            with self.path.open('a') as stream:
                stream.write(json.dumps(event, default=str, separators=(',', ':'))+'\n')
            self.last[event['point']] = encoded

    def checkpoint(self, state, write):
        with self.lock:
            state['point_checkpoint_seq'] = self.sequence
            write()
