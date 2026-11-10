"""Exact source-instruction expectation and pricing feasibility, not extrapolated compiled cost."""
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ART='artifacts/calpha_h4_step1/2026-11-10/'


def evaluate():
    from trotterlib.calpha_sampling_law import finite_cdf_masses
    from trotterlib.rte_compiled_cost import _circuit_payload
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZGate,RZZGate,XXPlusYYGate,PhaseGate
    profile=json.loads((ROOT/(ART+'run09_sampling_law/gate_profile.json')).read_text())
    account=json.loads((ROOT/(ART+'run10_rpe_accounting/result.json')).read_text())
    law=json.loads((ROOT/(ART+'run09_sampling_law/result.json')).read_text())
    # Use the actual sampler CDF order. Do not use the lexicographic mean
    # projection or a short-circuit empirical price to infer a long price.
    total=sum(x['probability'] for x in profile['components'])
    import math
    total=math.fsum(x['probability'] for x in profile['components'])
    masses,_=finite_cdf_masses([float(x['probability']/total) for x in profile['components']])
    groups={};lengths={}
    for component,mass in zip(profile['components'],masses):
        f=component['fragment'];groups[f]=groups.get(f,Fraction(0))+mass
    for frame in profile['frames']:
        if frame['role']=='tail':lengths[frame['fragment']]=len(frame['operations'])
    expected_tail=Fraction(4)+2*sum((lengths[f]*(p+3*p*(1-p)) for f,p in groups.items()),Fraction(0))
    # Exact deterministic head emission, using the fixed saved gate parameters.
    half=QuantumCircuit(9)
    for frame in profile['frames'][:4]:
        gates=[]
        for op in frame['operations']:
            gate=(RZGate(op['parameters'][0]) if op['name']=='rz'
                  else XXPlusYYGate(*op['parameters']))
            gates.append((gate,op['qubits']))
        for gate,qubits in reversed(gates):half.append(gate.inverse(),qubits)
        diagonal=frame['diagonal_primitives']
        for p,q,angle in diagonal['rzz']:
            if angle!=0:half.append(RZZGate(angle).control(1),[8,p,q])
        for q,angle in diagonal['rz']:half.append(RZGate(angle).control(1),[8,q])
        if diagonal['global_phase']!=0:half.append(PhaseGate(diagonal['global_phase']),[8])
        for gate,qubits in gates:half.append(gate,qubits)
    head_instructions=2*len(half.data)
    expected_step=head_instructions+expected_tail
    min_tail=4+2*min(lengths.values())
    max_tail=4+8*max(lengths.values())
    # The materialized v2 hasher expands each instruction definition at every
    # occurrence. Its encoded deterministic-head bytes give a genuine lower
    # bound for the full m17 JSON allocation; no build or price extrapolation.
    payload=_circuit_payload(half)
    half_serialized_instruction_bytes=sum(len(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
                                           for x in payload['instructions'])
    q17=1<<17
    encoded_head_lower=2*q17*half_serialized_instruction_bytes
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemTotal:','MemAvailable:')):
            k,v,_=line.split();memory[k.rstrip(':')]=int(v)*1024
    rows=[];total_native=Fraction(0)
    for row in account['rounds']:
        q=row['q']
        x=q*expected_step+4 # one scalar plus H,H,measure
        y=q*expected_step+5 # one additional Sdg
        aggregate=row['N_X']*x+row['N_Y']*y
        total_native+=aggregate
        rows.append(dict(m=row['m'],q=q,expected_native_wrapper_X_exact=str(x),
            expected_native_wrapper_Y_exact=str(y),shot_weighted_native_instructions_exact=str(aggregate),
            uncompiled_native_lower=q*(head_instructions+min_tail)+1,
            uncompiled_native_upper=q*(head_instructions+max_tail)+1,
            compiled_expected_price=None))
    return dict(status='SOURCE_INSTRUCTION_ACCOUNTING_COMPLETE_COMPILED_TOTAL_UNAVAILABLE',
        expected_per_step_native_instructions_exact=str(expected_step),
        deterministic_head_instructions_per_step=head_instructions,
        expected_tail_instructions_per_step_exact=str(expected_tail),
        total_shot_weighted_native_instructions_exact=str(total_native),
        coordinate='Uncompiled source-level instruction count; high-level Gaussian and controlled gates each count as one. Not RZ/CX/T/depth or a compiled-cost bound.',
        expectation_method='Exact CDF masses and IID adjacent-basis-reuse indicator expectations; all q and both measured wrappers included.',
        same_known_K0='same physical program and same source-instruction expectation',
        rows=rows,m17_pricing=dict(
            materialized_v2_fingerprint_deterministic_head_JSON_bytes_lower=encoded_head_lower,
            lower_exceeds_available_RAM=encoded_head_lower>memory['MemAvailable'],
            resources_at_check=memory,
            streamed_fingerprint='Existing same-policy streaming removes this allocation; it does not remove flat circuit, wrapper copies or whole compiler DAG.',
            whole_DAG_price_feasibility='UNESTABLISHED_FOR_THIS_K0_M17; no huge job launched or impossible claim made',
            price=None,expected_price=None,total_compiled_RPE_cost=None),
        limitations=['One m0 trajectory does not estimate expected prices.',
            'Short compiled price is never multiplied by q.',
            'Streaming/fingerprint evidence for short K2 does not certify all-round K0 expected prices.',
            'Partitioned compilation changes whole-compiler prices in existing tests; not substituted.',
            'No fair full/truncated PF compiled total exists in this output.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(started_UTC=datetime.now(timezone.utc).isoformat(),SOURCE_SHA256=source,status='STARTED')
    try:result.update(evaluate())
    except Exception as exc:result.update(status='TECHNICAL_EXCEPTION',reason=str(exc));raise
    finally:(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])
