"""Bounded grouped dispatch; deterministic science independent of completion order."""
from collections import deque
from ..h4_geometry.identity import require


class GroupQueue:
    def __init__(self,tasks):
        require(len(tasks)==240 and len({t['task_id'] for t in tasks})==240,'unique closed map')
        grouped={}
        for task in tasks:grouped.setdefault((task['geometry'],task['prefix']),[]).append(task)
        # Most expensive groups first; no configuration is omitted based on cost.
        self.groups=deque(deque(sorted(group,key=lambda t:(-t['template']['q'],t['axis'])))
            for _,group in sorted(grouped.items(),key=lambda item:(-item[0][1],item[0][0])))
        self.assigned={};self.active={};self.done=set()
    def next(self,slot):
        require(slot not in self.active,'one live job per owned worker')
        group=self.assigned.get(slot)
        if not group:
            if not self.groups:return None
            group=self.groups.popleft();self.assigned[slot]=group
        task=group.popleft();require(task['task_id'] not in self.done,'no repeated group job')
        self.active[slot]=task;return task
    def complete(self,slot,task_id):
        require(slot in self.active and self.active[slot]['task_id']==task_id and task_id not in self.done,'completion belongs to assigned worker/task')
        self.done.add(task_id);del self.active[slot]
    def complete_map(self):
        return len(self.done)==240 and not self.active and not self.groups and not any(self.assigned.values())


class FrameReader:
    def __init__(self):self.buffer=bytearray();self.length=None
    def wanted(self):return (8 if self.length is None else self.length)-len(self.buffer)
    def feed(self,piece):
        require(bool(piece),'worker EOF before complete frame');self.buffer.extend(piece)
        if self.length is None and len(self.buffer)==8:
            self.length=int.from_bytes(self.buffer,'big');require(0<self.length<=8192,'bounded owned reply');self.buffer.clear()
        elif self.length is not None and len(self.buffer)==self.length:
            result=bytes(self.buffer);self.buffer.clear();self.length=None;return result
        return None
