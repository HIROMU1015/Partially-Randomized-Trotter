#!/usr/bin/env python3
"""Saved-only cost completion; explicitly preserve the strict-phase failure."""
import argparse
import csv
import json
from pathlib import Path
import resource
import signal
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate, RZGate, RZZGate, PhaseGate
from qiskit.quantum_info import Operator

from scripts.diagnostics.rte_basis_cost_attribution import dump, guarded_process, summarize, METRICS, ROLES


def restore(ir):
    circuit = QuantumCircuit(ir['qubits'])
    circuit.global_phase = ir['global_phase']
    for op in ir['operations']:
        name, wires, params = op['name'], op['qubits'], op['parameters']
        if name in ('rz', 'cx', 'sx', 'x', 'cz'):
            getattr(circuit, name)(*params, *wires)
        elif name == 'xx_plus_yy':
            circuit.append(XXPlusYYGate(*params), wires)
        elif name == 'crz':
            circuit.append(RZGate(*params).control(1), wires)
        elif name == 'crzz':
            circuit.append(RZZGate(*params).control(1), wires)
        elif name == 'p':
            circuit.append(PhaseGate(*params), wires)
        else:
            raise ValueError('UNSUPPORTED_SAVED_GATE:' + name)
    return circuit


def phase_diagnostic(raw, native):
    """One common phase for the entire operator; never align branches separately."""
    inner = complex(np.vdot(raw, native) / raw.shape[0])
    phase = inner / abs(inner) if abs(inner) > .5 else 1.+0.j
    half = raw.shape[0] // 2
    return {
        'strict_max_abs_difference': float(np.max(np.abs(native - raw))),
        'common_phase': {'real': phase.real, 'imag': phase.imag},
        'common_phase_max_abs_residual': float(np.max(np.abs(native - phase * raw))),
        'ancilla_zero_block_residual': float(np.max(np.abs(native[:half,:half] - phase * raw[:half,:half]))),
        'ancilla_one_block_residual': float(np.max(np.abs(native[half:,half:] - phase * raw[half:,half:]))),
        'ancilla_offdiagonal_residual': float(max(np.max(np.abs(native[:half,half:] - phase * raw[:half,half:])),
                                                np.max(np.abs(native[half:,:half] - phase * raw[half:,:half])))),
        'branch_specific_phase_alignment': False,
    }


def run(saved, out):
    start, cpu = time.monotonic(), time.process_time()
    guarded_process()
    if json.loads((saved/'failure.json').read_text())['error'] != "RuntimeError('STRICT_COMPILED_PHASE_VALIDATION_FAILED')":
        raise RuntimeError('UNEXPECTED_FAILURE_REASON')
    inventory = json.loads((saved/'compile_inventory.json').read_text())
    records = {p.stem:json.loads(p.read_text()) for p in (saved/'compiled').glob('*.json')}
    if set(records) != set(inventory['keys']):
        raise RuntimeError('INCOMPLETE_COMPILE_INVENTORY')
    rows=[]
    for p in sorted((saved/'inputs').glob('*.json')):
        info=json.loads(p.read_text())
        raw=Operator(restore(info['raw_circuit'])).data
        native=Operator(restore(records[info['full_key']]['native'])).data
        check=phase_diagnostic(raw,native)
        if check['common_phase_max_abs_residual'] > 1e-10:
            raise RuntimeError('COMPILED_RELATIVE_PHASE_OR_OPERATOR_MISMATCH')
        additive={r:{m:0 for m in METRICS} for r in ROLES}
        for s in info['segments']:
            record=records[s['key']]
            for m in METRICS:
                actual=sum(op['name'] == {'rz_count':'rz','cx_count':'cx'}.get(m,'')
                           for op in record['native']['operations']) if m!='circuit_size' else len(record['native']['operations'])
                if actual!=record['metrics'][m]:raise RuntimeError('NATIVE_COUNT_MISMATCH')
                additive[s['role']][m]+=actual
        full=records[info['full_key']]['metrics']
        for m in METRICS:
            actual=sum(op['name']=={'rz_count':'rz','cx_count':'cx'}.get(m,'')
                       for op in records[info['full_key']]['native']['operations']) if m!='circuit_size' else len(records[info['full_key']]['native']['operations'])
            if actual!=full[m]:raise RuntimeError('WHOLE_COUNT_MISMATCH')
        row={k:info[k] for k in ('id','length','index','seed','event_digest',
                               'cancelled_basis_change_pairs','emitted_basis_change_count','basis_switch_count')}
        row.update(additive=additive,full=full,operator_phase_diagnostic=check)
        row['summary']=summarize([row]);rows.append(row)
        print(p.stem, 'operator_residual', check['common_phase_max_abs_residual'], flush=True)
    if len(rows)!=8:raise RuntimeError('WRONG_SELECTED_ROWS')
    dump(out/'rows.json',rows)
    summary={
        'status':'SAVED_COST_ATTRIBUTION_COMPLETE_WITH_COMMON_GLOBAL_PHASE_CAVEAT',
        'original_run_status':'TECHNICAL_INCONCLUSIVE', 'original_strict_phase_failure_preserved':True,
        'diagnostic_scope':'Small fixed saved H4 RTE columns; additive RZ/CX basis costs, not a new algorithm or final PR resource result',
        'source_commit':json.loads((saved/'run_start.json').read_text())['source_commit'],
        'input_identity':json.loads((saved/'input_identity.json').read_text())['physical_instance'],
        'compiler':inventory['compiler'],'basis_policy':'registered_full_basis_with_existing_reuse',
        'pooled_unweighted_selected_rows':summarize(rows),
        'by_length':{str(n):summarize([r for r in rows if r['length']==n]) for n in (3,6)},
        'operator_scope':'Full 512x512 operators for saved already-controlled 9-qubit circuits; one common whole-circuit phase; both ancilla blocks and offdiagonal blocks checked together',
        'strict_phase_failed_rows':sum(r['operator_phase_diagnostic']['strict_max_abs_difference']>1e-10 for r in rows),
        'maximum_common_phase_operator_residual':max(r['operator_phase_diagnostic']['common_phase_max_abs_residual'] for r in rows),
        'circuits_or_compiler_modified':False,'branch_specific_phase_alignment':False,
        'implication':'A common global phase of the entire already-controlled RTE circuit leaves its density operator and Hadamard observables unchanged. This does not permit dropping a relative phase before control or adding another control to the compiled circuit.',
        'original_native_compile_calls':len(records),'additional_native_compile_calls':0,'new_random_seeds':0,
        'original_wall_seconds_before_failure':'See original outer-process receipt',
        'saved_audit_wall_seconds':time.monotonic()-start,'saved_audit_cpu_seconds':time.process_time()-cpu,
        'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'T_synthesis':0,'molecule_generation':0,'GPU':0,'solver':0,'shots':0,
        'limitations':['Additive separately compiled segments; not unique optimized cost attribution.',
                       'Eight selected short old columns, not an unconditional RTE cost mean.',
                       'No T count, QPE/RPE, full PR or new-method advantage claim.',
                       'The strict original source condition failed; cause inside compiler is not localized.'],
        'mandatory_STOP':True,
    }
    dump(out/'summary.json',summary)
    with (out/'resource_rows.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','length','metric','basis','central','other','additive_total',
            'whole_compiled','basis_percent_additive','gap_percent_of_whole'])
        for r in rows:
            for m in METRICS:
                s=r['summary'];writer.writerow([r['id'],r['length'],m,*[r['additive'][role][m] for role in ROLES],
                    s['additive_total'][m],r['full'][m],s['basis_percent_additive'][m],s['gap_percent_of_whole'][m]])
    print(json.dumps(summary['pooled_unweighted_selected_rows'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--saved',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:run(args.saved,args.output)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
