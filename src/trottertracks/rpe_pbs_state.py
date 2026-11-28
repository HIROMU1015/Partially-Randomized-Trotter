"""Durable, bounded per-allocation deltas for new opt-in PBS manager scopes."""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading


SEQUENCE = 'pbs_state_checkpoint_seq'


def _journal(path):
    return Path(path).with_name('STATE_DELTAS.jsonl')


def _read(path):
    path=Path(path)
    while True:
        with path.open('r') as stream:
            before=os.fstat(stream.fileno())
            state=json.load(stream)
        sequence=int(state.pop(SEQUENCE,0))
        gap=False
        try:
            with _journal(path).open('rb') as stream:
                for raw in stream:
                    if not raw.endswith(b'\n'):
                        break
                    event=json.loads(raw)
                    if event['seq']<=sequence:continue
                    if event['seq']!=sequence+1:
                        gap=True;break
                    state.setdefault('allocations',{}).update(event['allocations'])
                    for token in event['removed_allocations']:
                        state['allocations'].pop(token,None)
                    state.update(event['fields'])
                    for key in event['removed_fields']:
                        state.pop(key,None)
                    sequence=event['seq']
        except FileNotFoundError:
            pass
        after=path.stat()
        if (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino):
            continue
        if gap:raise ValueError('PBS state journal sequence gap')
        return state,sequence


def read_pbs_state(path):
    return _read(path)[0]


def is_pbs_state(path):
    path=Path(path)
    return path.name=='STATE.json' and path.parent.name=='pbs' and path.parent.parent.name=='distributed'


class StateJournal:
    """One existing manager lock owns this writer; a reader never repairs."""
    def __init__(self,path,atomic):
        self.path=Path(path)
        self.delta=_journal(path)
        self.atomic=atomic
        self.lock=threading.RLock()
        self.last,self.sequence=_read(self.path) if self.path.is_file() else (None,0)
        if self.delta.is_file():
            with self.delta.open('r+b') as stream:
                while raw:=stream.readline():
                    if not raw.endswith(b'\n'):
                        stream.seek(-len(raw),1)
                        stream.truncate()
                        break

    def save(self,state,*,checkpoint=False):
        with self.lock:
            state=deepcopy(state)
            if SEQUENCE in state:
                raise ValueError('Reserved PBS state sequence key')
            old=self.last
            if old==state and not checkpoint:return
            seq=self.sequence+(old!=state)
            if old is not None and not checkpoint:
                before,after=old.get('allocations',{}),state.get('allocations',{})
                event=dict(seq=seq,allocations={k:v for k,v in after.items() if before.get(k)!=v},
                    removed_allocations=sorted(before.keys()-after.keys()),
                    fields={k:v for k,v in state.items() if k!='allocations' and (k not in old or old[k]!=v)},
                    removed_fields=sorted(old.keys()-state.keys()-{'allocations'}))
                raw=(json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                previous=self.delta.stat().st_size if self.delta.is_file() else 0
                # Bound retained journal by the actual checkpoint size, not
                # by a wall-time or a fixed allocation count.
                if previous+len(raw)<self.path.stat().st_size:
                    fresh=not self.delta.is_file()
                    with self.delta.open('ab') as stream:
                        stream.write(raw);stream.flush();os.fsync(stream.fileno())
                    if fresh:
                        fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
                        try:os.fsync(fd)
                        finally:os.close(fd)
                    self.last=state;self.sequence=seq
                    return
            self.atomic(self.path,dict(state,**{SEQUENCE:seq}))
            self.last=state;self.sequence=seq
            if self.delta.is_file():
                with self.delta.open('wb') as stream:
                    stream.flush();os.fsync(stream.fileno())
