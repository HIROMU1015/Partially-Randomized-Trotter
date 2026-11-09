"""Opt-in phase ledger; original lowering and optimization rules stay unchanged.

Accumulate the binary64 phase terms returned by existing emitters in MPFR160.
Local gate decompositions and 2x2 synthesis still use their original precision.
"""
import math
from unittest.mock import patch

from .c1_h6_compiler_precision import PHASE_BITS
from .track_d_native_emitter import NativeLowerer


class PhaseLedger:
    def __init__(self,values=()):
        import symengine as se
        self.se=se;self.total=se.RealMPFR(0,PHASE_BITS);self.values=[];self.running=0.
        for value in values:self.add(value)

    def add(self,value):
        value=float(value)
        if not math.isfinite(value):raise ValueError('finite phase term required')
        # RealMPFR(float, bits) uses a decimal conversion; retain exact binary64.
        numerator,denominator=value.as_integer_ratio()
        self.total+=self.se.RealMPFR(numerator,PHASE_BITS)/self.se.RealMPFR(denominator,PHASE_BITS)
        self.values.append(value);self.running=math.fsum((self.running,value))

    def result(self):
        period=2*self.se.pi.evalf(PHASE_BITS)
        reduced=self.total-int(self.se.floor(self.total/period))*period
        return dict(bits=PHASE_BITS,terms=len(self.values),legacy_running=self.running,
            binary64_full_fsum=math.fsum(self.values),mpfr_total=str(self.total),
            normalized=float(reduced),period='2*pi evaluated at MPFR160',
            local_terms_precision='unchanged binary64 gate/1Q outputs')


def lower_with_ledger(circuit,consumer):
    """Same raw gate order/wires/definitions as circuit_chunk, new sum only."""
    lowerer=NativeLowerer();ledger=PhaseLedger([float(circuit.global_phase)])
    for item in circuit.data:
        q=tuple(circuit.find_bit(b).index for b in item.qubits)
        c=tuple(circuit.find_bit(b).index for b in item.clbits)
        ledger.add(lowerer.emit(item.operation,q,c,consumer))
    return ledger


def optimize_with_ledger(word,nq,nc,ledger,observed_optimize):
    from . import track_d_p0_1q_bridge as bridge
    original=bridge.oneq_runs;additions=[]
    def capture(*a,**kw):
        result=original(*a,**kw);additions.append(float(result[1]));return result
    # Existing optimizer sees its original running phase, preserving old output.
    with patch.object(bridge,'oneq_runs',capture):
        result,observer=observed_optimize(word,nq,nc,ledger.running)
    final=PhaseLedger(ledger.values)
    for value in additions:final.add(value)
    return result,observer,final,additions
