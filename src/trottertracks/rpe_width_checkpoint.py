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
    path = Path(path)
    if not path.is_file():
        return sequence
    points = {identity(p): p for p in state.get('points', ())
              if all(k in p for k in ('method','family','kappa'))}
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


class HistoryJournal:
    """Keep complete decision history outside the frequently saved snapshot.

    Each checkpoint references a committed journal head. An entry appended
    just before a crash can remain in the journal without becoming a decision
    of the older snapshot. Branches from interrupted writes are retained.
    """
    def __init__(self, path, state):
        import hashlib
        self.path = Path(path)
        self.entries = {}
        self._hash = lambda raw: hashlib.sha256(raw).hexdigest()
        if self.path.is_file():
            with self.path.open('r+b') as stream:
                while raw := stream.readline():
                    if not raw.endswith(b'\n'):
                        stream.seek(-len(raw), 1)
                        stream.truncate()
                        break
                    event = json.loads(raw)
                    self.entries[event['id']] = event
        if 'history_archive' in state:
            self.full_history(state)

    @staticmethod
    def _encode(value):
        return json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode()

    def full_history(self, state):
        archive = state.get('history_archive')
        if archive is None:
            return list(state.get('history', ()))
        if archive.get('file') != self.path.name:
            raise ValueError('Decision history journal differs')
        head, count = archive['head'], archive['count']
        reversed_history = []
        while head is not None:
            event = self.entries.get(head)
            if event is None or event['count'] != count:
                raise ValueError('Committed decision history is incomplete')
            payload = {k:event[k] for k in ('previous','count','entry')}
            if self._hash(self._encode(payload)) != head:
                raise ValueError('Committed decision history content differs')
            reversed_history.append(event['entry'])
            head, count = event['previous'], count-1
        if count != 0:
            raise ValueError('Committed decision history count differs')
        history = list(reversed(reversed_history))
        tail = state.get('history', ())
        if history and (not tail or tail[0] != history[-1]):
            raise ValueError('Decision history snapshot tail differs')
        return history + list(tail[1:]) if history else list(tail)

    def checkpoint(self, state, write):
        import os
        archive = state.get('history_archive') or {}
        count, head = archive.get('count',0), archive.get('head')
        pending = state.get('history', ())
        if count:
            pending = pending[1:]  # snapshot retains the last committed decision
        additions = []
        for entry in pending:
            count += 1
            payload = dict(previous=head,count=count,entry=entry)
            head = self._hash(self._encode(payload))
            if head not in self.entries:
                additions.append(dict(id=head,**payload))
        if additions:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('ab') as stream:
                for event in additions:
                    stream.write(self._encode(event)+b'\n')
                stream.flush(); os.fsync(stream.fileno())
            # Ensure first creation of the journal is durable before the
            # snapshot can point at its contents.
            directory = os.open(self.path.parent,os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
            self.entries.update((event['id'],event) for event in additions)
        snapshot = dict(state)
        if count:
            snapshot['history_archive'] = dict(file=self.path.name,head=head,count=count)
            snapshot['history'] = list(state['history'][-1:])
        write(snapshot)
        state.update(snapshot)
