#!/usr/bin/env python3
"""One additional invented common-basis case; explicit strong-baseline fusion."""
import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/tracks/algorithm_codesign'))
from vacuum_cswap_artificial_benchmark import (
    CompilerProfile, CouplingMap, QuantumCircuit, Statevector, COMPILE_TOL,
    body, compile_audit, compiled_wrapper, compile_cache_key, initial, known_se,
    matrix_gate, metrics, np, recorded, transpile, v, bit_probability,
)
from trotterlib.research_direction_structure_pilot import build_conjugated_pauli_sequence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'artifacts/o2_vacuum_reference_cswap/2026-11-08/artificial_run05'
    assert json.loads((prior/'TEST_RESULTS.json').read_text())['status']=='PASS'
    source=json.loads((prior/'SOURCE.json').read_text())
    for name,digest in source['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    own=str(Path(__file__).relative_to(ROOT))
    source.update(scope='ONE_INVENTED_2_QUBIT_COMMON_BASIS_FUSED_CASE',
        reused_correctness='artificial_run05: 14 PASS',
        finite_benchmark='X/Y x 4 methods x 2 topologies; explicit fusion and diagonal angle sums')
    source['sha256'][own]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    start=time.perf_counter();rows=[];status='PASS';calls=dict(native_compile=0,native_cache_hits=0,routing_compile=0,routing_cache_hits=0)
    try:
        # Artificial real one-particle rotation: no stored geometry or basis.
        t=.23;O=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        supports=[(0,),(1,),(0,1)]*4
        angles=[.11+.017*(i%5) for i in range(12)]
        def grouped(lo,hi,controlled):
            ss=sorted(set(supports[lo:hi]))
            aa=[float(sum((F(angles[j]) for j in range(lo,hi) if supports[j]==s),F(0))) for s in ss]
            return build_conjugated_pauli_sequence([O]*len(ss),supports=ss,angles=aa,
                controlled=controlled,scalar_phase=.03*(hi-lo),fuse_shared_basis=True)
        full=grouped(0,12,False);R=grouped(0,6,False);L=grouped(6,12,False)
        from qiskit.quantum_info import Operator
        unfused=build_conjugated_pauli_sequence([O]*12,supports=supports,angles=angles,
            controlled=False,scalar_phase=.36,fuse_shared_basis=False)
        fusion_audit=matrix_gate(Operator(full).data,Operator(unfused).data,COMPILE_TOL,label='EXPLICIT_FUSION_NO_PHASE_FIT')
        matrix_gate(Operator(R.compose(L)).data,Operator(full).data,COMPILE_TOL,label='SPLIT_IDENTITY')
        inputs=dict(case='artificial_n2_shared_basis_12_blocks',one_particle_basis=O.tolist(),
            supports=supports,angles_hex=[x.hex() for x in angles],scalar_per_block=.03,
            raw_blocks=12,optimized_baseline_central_rotations=3,
            fusion='existing shared-basis builder + exact dyadic sum by commuting support',
            L=recorded(L),R=recorded(R),fusion_audit=fusion_audit)
        (args.output/'ARTIFICIAL_INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n')
        profile=CompilerProfile();cache={};route_cache={}
        for axis in ('X','Y'):
            ctrl=QuantumCircuit(3);ctrl.h(2);ctrl.compose(grouped(0,12,True),inplace=True)
            if axis=='Y':ctrl.sdg(2)
            ctrl.h(2)
            from qiskit import ClassicalRegister
            bit=ClassicalRegister(1,'controlled_signal');ctrl.add_register(bit);ctrl.measure(2,bit[0])
            candidate=v.build_instrument(L,R,axis=axis,measure=True,initialize_reference=True)
            circuits=dict(optimized_controlled=ctrl,known_SE=known_se(L,R,axis,initialize_reference=True),
                PR_pathwise=candidate.circuit,known_cat_SE=v.build_instrument(L,R,axis=axis,
                    measure=True,initialize_reference=True,fanout=True).circuit)
            context=dict(scope='invented_common_basis_fusion',axis=axis)
            keys={m:compile_cache_key(c,profile,context) for m,c in circuits.items()}
            assert keys['known_SE']==keys['PR_pathwise']
            psi=np.array([1.,2j,3.,-4j])/np.sqrt(30.)
            p0=bit_probability(Statevector(initial(psi,3,range(2))).evolve(body(ctrl)).data,2)
            for method,raw in circuits.items():
                a=2 if method=='optimized_controlled' else 4
                val=bit_probability(Statevector(initial(psi,raw.num_qubits,range(2))).evolve(body(raw)).data,a)
                assert abs(val-p0)<COMPILE_TOL
                key=keys[method];warm=key in cache;tic=time.perf_counter()
                native=compiled_wrapper(raw,profile,cache=cache,identity_context=context)
                native_time=time.perf_counter()-tic;calls['native_cache_hits' if warm else 'native_compile']+=1
                for topology in ('all_to_all','line_fixed_layout'):
                    tic=time.perf_counter();routing_warm=False
                    if topology=='all_to_all':compiled=native
                    else:
                        routing_warm=key in route_cache
                        if not routing_warm:
                            route_cache[key]=transpile(native,basis_gates=list(profile.basis_gates),
                                optimization_level=profile.optimization_level,seed_transpiler=profile.seed_transpiler,
                                approximation_degree=profile.approximation_degree,qubits_initially_zero=False,
                                coupling_map=CouplingMap.from_line(raw.num_qubits),initial_layout=list(range(raw.num_qubits)),
                                layout_method='trivial',routing_method='sabre')
                        compiled=route_cache[key].copy();calls['routing_cache_hits' if routing_warm else 'routing_compile']+=1
                    row=dict(case=inputs['case'],axis=axis,method=method,topology=topology,
                        metrics=metrics(compiled),audit=compile_audit(raw,compiled,2),
                        native_compile_seconds=native_time,native_cache_hit=warm,
                        routing_compile_seconds=time.perf_counter()-tic,routing_cache_hit=routing_warm,
                        signal_probability_zero=val,cache_key=key)
                    rows.append(row)
        report=dict(status=status,profile=asdict(profile),rows=rows,calls=calls,cells=len(rows),
            elapsed_seconds=time.perf_counter()-start,scientific_input_files=[])
        (args.output/'BENCHMARK_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    except Exception:
        status='TECHNICAL_FAILURE';(args.output/'RUN.txt').write_text(traceback.format_exc())
        (args.output/'BENCHMARK_RESULTS.json').write_text(json.dumps(dict(status=status,rows=rows,calls=calls),indent=2)+'\n')
    (args.output/'STOP.json').write_text(json.dumps(dict(status='MANDATORY_STOP',reason=status,
        automatic_next_stage=False,elapsed_seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        H4_calls=0,shot_calls=0,RNG_sampling_calls=0),indent=2)+'\n')
    print(json.dumps(dict(status=status,rows=len(rows),calls=calls)))
    return 0 if status=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
