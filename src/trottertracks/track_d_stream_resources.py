"""Connect a phase-producing native iterator to the unchanged-rule bridge."""
import math

from .track_d_native_phase160 import PhaseLedger
from .track_d_stream_bridge_fast import optimize_native_stream,iter_raw_native


def evaluate_native_stream(operations,num_qubits,num_clbits,ledger):
    """The producer may finish its phase ledger only when input is exhausted.

    Rewrite decisions do not inspect total circuit phase. Reconstruct the old
    scalar addition order afterwards; use the full logical round additions,
    not the subset of actual synthesis calls (which is cache-dependent).
    """
    result=optimize_native_stream(operations,num_qubits,num_clbits,global_phase=0.)
    legacy=ledger.running;precise=PhaseLedger(ledger.values)
    for addition in result['phase_additions']:
        legacy=math.fsum((legacy,addition));precise.add(addition)
    result['input']['metrics']['global_phase']=ledger.running
    result['output']['metrics']['global_phase']=legacy
    result['global_phase']=legacy;result['precise_phase']=precise.result()
    return result


def evaluate_saved_raw(circuit):
    """Saved raw QC remains owned by the caller; no native/full-wrapper QC."""
    ledger=PhaseLedger([float(circuit.global_phase)]);statistics={}
    result=evaluate_native_stream(iter_raw_native(circuit,ledger,statistics),
                                  circuit.num_qubits,circuit.num_clbits,ledger)
    result['lowering_statistics']=statistics
    return result
