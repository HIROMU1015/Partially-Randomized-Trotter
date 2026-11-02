"""Independent chained decision reader and state witness, bounded by 24 cursors."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_parallel_verifier_core as G


class Decisions:
    def __init__(self,path,tasks,run_id,state_path=None,prefix=None):
        self.path=Path(path);self.tasks={t['key']:t for t in tasks};self.run_id=run_id
        self.state_path=state_path;self.prefix=prefix;self.stats={};self.counts=Counter()
        self.cursors=Counter();self.current={};self.witness_counts=Counter();self.prefix_checked=False
    def __iter__(self):
        previous=V.ZERO;file_sha=sha256();size=0;seq=0
        with self.path.open('rb') as f:
            for line in f:
                V.require(line.endswith(b'\n'),'TRACE_PARTIAL_LINE')
                e=V.plain(line);V.require(set(e)=={'payload','previous','sha256'} and type(e['payload']) is str,'TRACE_ENVELOPE')
                raw=e['payload'].encode();V.require(e['previous']==previous,'TRACE_PREVIOUS')
                previous=sha256(previous.encode()+raw).hexdigest();V.require(e['sha256']==previous,'TRACE_CHAIN')
                r=V.plain(raw);V.require(json.dumps(r,separators=(',',':'),ensure_ascii=True).encode()==raw,'TRACE_CANONICAL_PAYLOAD')
                V.require(type(r['seq']) is int and r['seq']==seq and r['run_id']==self.run_id,'TRACE_SEQ_RUN_ID')
                op=r['op'];self.counts[op]+=1;seq+=1;size+=len(line);file_sha.update(line)
                if op in ('PHASE_BEGIN','PHASE_END'):
                    V.require(type(r['phase']) is int and 0<=r['phase']<3,'TRACE_PHASE_RANGE')
                if self.prefix and size==self.prefix['bytes']:
                    V.require(seq==self.prefix['rows'] and previous==self.prefix['chain'] and file_sha.hexdigest()==self.prefix['sha256'],'TRACE_PREFIX_BINDING')
                    self.prefix_checked=True
                if op in ('RESERVE','ENTER','COMPLETE','FAIL','CANCEL','UNKNOWN'):
                    key=r['task_key'];V.require(key in self.tasks,'TRACE_TASK');t=self.tasks[key]
                    if op=='RESERVE':
                        i=self.cursors[key];V.require(i<2*t['n'] and key not in self.current,'DUPLICATE_RESERVE')
                        axis='X' if i<t['n'] else 'Y';local=i%t['n']
                        V.require(r['trial_id']==V.trial(dict(t,N=t['n']),axis,local),'TRACE_TRIAL_ID_ORDER')
                        self.current[key]=[r['trial_id'],t['offset']+i,'START_RESERVED']
                    else:
                        V.require(key in self.current and self.current[key][0]==r['trial_id'],'TRACE_CURRENT_ID')
                        ticket=self.current[key]
                        if op=='ENTER':
                            V.require(ticket[2]=='START_RESERVED','DOUBLE_ENTER');ticket[2]='IN_FLIGHT'
                        else:
                            if op in ('COMPLETE','FAIL'):V.require(ticket[2]=='IN_FLIGHT','TERMINAL_WITHOUT_ENTER')
                            if op=='CANCEL':V.require(ticket[2]=='START_RESERVED','CANCEL_ENTERED')
                            expected={'COMPLETE':'COMPLETED','FAIL':'FAILED','CANCEL':'CANCELLED_BEFORE_START','UNKNOWN':'TERMINATED_WITH_UNKNOWN_RESULT'}[op]
                            observed=r.get('observed_source_state',expected)
                            V.require(observed==expected or op=='UNKNOWN' and observed=='FAILED','TRACE_SOURCE_STATE')
                            if self.state_path:
                                with Path(self.state_path).open('rb') as s:s.seek(ticket[1]);code=s.read(1)
                                V.require(code==bytes([G.STATES.index(observed)]),'TRACE_FINAL_STATE_WITNESS')
                            self.witness_counts[observed]+=1;self.current.pop(key);self.cursors[key]+=1
                    if 'global_offset' in r:V.require(type(r['global_offset']) is int and r['global_offset']==t['offset']+self.cursors[key]-(op in ('COMPLETE','FAIL','CANCEL','UNKNOWN')),'TRACE_GLOBAL_OFFSET')
                yield r
        if self.prefix:V.require(self.prefix_checked,'TRACE_PREFIX_NOT_ON_ROW_BOUNDARY')
        self.stats=dict(bytes=size,rows=seq,sha256=file_sha.hexdigest(),chain=previous,op_counts=dict(self.counts),
                        terminal_state_witness=dict(self.witness_counts),outstanding=dict(self.current),
                        prefix_checked=self.prefix_checked,global_offset='derived from pinned task offsets; optional explicit field checked')
