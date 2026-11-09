"""Same-round native 1Q/CX bridge, with streaming and exact-run reuse.

No new rewrite rule or matrix synthesis. The frozen oneq_runs routine solves
each cache miss; the frozen CX feed performs each cancellation. No full P0.
"""
import math
import struct
import time
from types import SimpleNamespace

from .track_d_native_resources import NativeCounter,count_native
from .track_d_p0_cx_bridge import CXBoundaryBridge
from . import track_d_p0_1q_bridge as original


class ObservedCX(CXBoundaryBridge):
    """Same feed, optional removal of redundant counts on validated input."""
    def __init__(self,*a,recount=True,**kw):
        super().__init__(*a,**kw)
        if not recount:self._input=SimpleNamespace(feed=lambda op:None)
        self.processed=0;self.refs=0;self.peak_refs=0;self.peak_wire=0
        self.frontier_bytes=sum(x.__sizeof__() for x in self._frontier)
        self.peak_frontier_bytes=self.frontier_bytes

    def feed(self,op):
        wires=op.qubits+tuple(self.num_qubits+c for c in op.clbits)
        before_bytes=sum(self._frontier[w].__sizeof__() for w in wires)
        before=self.cancelled_pairs
        result=super().feed(op)
        self.processed+=1
        self.refs+=(-2 if self.cancelled_pairs!=before else len(wires))
        self.peak_refs=max(self.peak_refs,self.refs)
        self.peak_wire=max(self.peak_wire,max((len(self._frontier[w]) for w in wires),default=0))
        self.frontier_bytes+=sum(self._frontier[w].__sizeof__() for w in wires)-before_bytes
        self.peak_frontier_bytes=max(self.peak_frontier_bytes,self.frontier_bytes)
        return result

    def statistics(self):
        return dict(processed=self.processed,peak_live_nodes=self.peak_live_nodes,
            retained_node_slots=len(self._nodes),node_list_bytes=self._nodes.__sizeof__(),
            peak_frontier_references=self.peak_refs,peak_single_wire_stack=self.peak_wire,
            peak_frontier_lists_bytes=self.peak_frontier_bytes)


class OneQStream:
    """Preserve original flush order, including final ascending wire order."""
    def __init__(self,nq,previous):
        self.nq=nq;self.previous=previous;self.cache={};self.phase=[]
        self.replaced=0;self.scanned=0;self.hits=0;self.calls=0
        self.peak_pending=0;self.peak_run=0;self.synthesis_seconds=0.

    def run(self,operations):
        pending=[[] for _ in range(self.nq)];pending_count=0
        def flush(wire):
            nonlocal pending_count
            run=pending[wire]
            if not run:return ()
            # No approximate matching; retain signed-zero/parameter bits.
            key=(wire,tuple((op.name,tuple(struct.pack('>d',v) for v in op.params)) for op in run))
            entry=self.cache.get(key)
            if entry is None:entry=self.previous.get(key)
            if entry is None:
                start=time.perf_counter()
                output,phase,replaced=original.oneq_runs(iter(run),self.nq)
                self.synthesis_seconds+=time.perf_counter()-start;self.calls+=1
                entry=(tuple(output) if replaced else None,phase,replaced)
            else:self.hits+=1
            self.cache[key]=entry
            output,phase,replaced=entry
            self.replaced+=replaced
            if replaced:self.phase.append(phase)
            pending_count-=len(run);pending[wire]=[]
            return output if replaced else run
        for op in operations:
            self.scanned+=1
            if op.name in ('rz','sx','x'):
                run=pending[op.qubits[0]];run.append(op);pending_count+=1
                self.peak_pending=max(self.peak_pending,pending_count)
                self.peak_run=max(self.peak_run,len(run))
            else:
                for wire in op.qubits:yield from flush(wire)
                yield op
        for wire in range(self.nq):yield from flush(wire)


def _consume_owned(word):
    # Only our intermediate list; never modify a caller's input container.
    for i in range(len(word)):
        op=word[i];word[i]=None;yield op


def optimize_native_stream(operations,num_qubits,num_clbits=0,*,global_phase=0.):
    import qiskit
    if qiskit.__version__!='1.3.0':raise ValueError('fixed Qiskit1.3.0 required')
    initial=NativeCounter(num_qubits,num_clbits,global_phase=global_phase)
    def validated():
        for op in operations:initial.feed(op);yield op
    stream=validated();phase=float(global_phase);rounds=[];details=[];previous={};additions=[]
    peak_word=0;peak_word_bytes=0
    while True:
        start=time.perf_counter();oneq=OneQStream(num_qubits,previous)
        cx=ObservedCX(num_qubits,num_clbits,recount=False).extend(oneq.run(stream))
        addition=math.fsum(oneq.phase);additions.append(addition)
        phase=math.fsum((phase,addition))
        before=oneq.scanned
        word=list(cx.operations());peak_word=max(peak_word,len(word));peak_word_bytes=max(peak_word_bytes,word.__sizeof__())
        rounds.append(dict(input_size=before,output_size=len(word),oneq_runs_replaced=oneq.replaced,
                           cx_pairs_removed=cx.cancelled_pairs))
        details.append(dict(seconds=time.perf_counter()-start,cx=cx.statistics(),
            cache_hits=oneq.hits,synthesis_calls=oneq.calls,synthesis_seconds=oneq.synthesis_seconds,
            cache_slots=len(previous)+len(oneq.cache),peak_pending_1q=oneq.peak_pending,peak_1q_run=oneq.peak_run,
            oneq_scanned=oneq.scanned))
        previous=oneq.cache
        del cx,oneq
        if len(word)==before:break
        if len(word)>before:raise ValueError('same rule must strictly decrease size')
        stream=_consume_owned(word)
    output=count_native(iter(word),num_qubits,num_clbits,global_phase=phase)
    return dict(input=initial.result(),output=output,operations=word,global_phase=phase,rounds=rounds,
        phase_additions=additions,performance=dict(rounds=details,peak_word_length=peak_word,
            peak_word_list_bytes=peak_word_bytes,full_folded_word_materialized=False,
            original_input_word_materialized=False),p0_qualification='P0_UNQUALIFIED')


def iter_raw_native(circuit,ledger,statistics):
    """Bounded per-gate lowering, using the unchanged official emitter."""
    from .track_d_native_emitter import NativeLowerer
    lowerer=NativeLowerer()
    for item in circuit.data:
        q=tuple(circuit.find_bit(b).index for b in item.qubits)
        c=tuple(circuit.find_bit(b).index for b in item.clbits)
        block=[];ledger.add(lowerer.emit(item.operation,q,c,block.append))
        statistics['peak_gate_chunk']=max(statistics.get('peak_gate_chunk',0),len(block))
        yield from block
