"""Small artificial circuit benchmark; no chemistry, saved input or pricing.

Output timings describe this invented gate sequence only. They are not H6
scientific-kernel timings or a prediction for the proposed molecular run.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def artificial(norb, layers):
    import numpy as np
    from qiskit import QuantumCircuit
    modes = 2*norb; q = QuantumCircuit(modes+1); q.p(-.137, modes)
    indices = np.array([sum(1 << (2*j) for j in a)+sum(1 << (2*j+1) for j in b)
        for a in itertools.combinations(range(norb), norb//2)
        for b in itertools.combinations(range(norb), norb//2)])
    bits = (indices[:, None] >> np.arange(modes)) & 1
    z = 1-2*bits; exponent = np.full(len(indices), -.137)
    for layer in range(layers):
        for j in range(modes):
            theta = .013*(j+1)/(layer+1); k = (j+2) % modes
            q.crz(theta, modes, j); exponent -= theta*z[:, j]/2
            q.cx(j, k); q.crz(-.007, modes, k); q.cx(j, k)
            exponent += .007*z[:, j]*z[:, k]/2
    return q, indices, np.diag(np.exp(1j*exponent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norb', type=int, choices=(2, 4, 6), default=6)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-columns', type=int, default=16)
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.layers < 1: parser.error('positive layers required')
    import numpy as np
    from trottertracks import c_h6_validation_preparation as m
    from trottertracks.c1_h6_compiler_precision import compiled_wrapper
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    q, indices, w = artificial(args.norb, args.layers)
    tick = time.perf_counter(); compiled = compiled_wrapper(q, CompilerProfile())
    compile_time = time.perf_counter()-tick
    tick = time.perf_counter()
    result = m.check_kernel(q, compiled, w, indices,
        batch_columns=args.batch_columns, threads=args.threads)
    elapsed = time.perf_counter()-tick
    dense = None
    if args.dense:
        from trottertracks.c_full_unitary import decode
        from trottertracks.c1_stage2_native import actual_adjoint
        tick = time.perf_counter()
        matrices = [decode(circuit, threads=args.threads) for circuit in
                    (q, compiled, actual_adjoint(q), actual_adjoint(compiled))]
        dense = dict(seconds=time.perf_counter()-tick,
                     four_matrices_bytes=sum(x.nbytes for x in matrices),
                     compiled_raw_max_entry=float(np.max(abs(matrices[0]-matrices[1]))))
    payload = dict(kind='ARTIFICIAL_DIAGONAL_CIRCUIT_ONLY', norb=args.norb,
        layers=args.layers, raw_operations=len(q.data), compiled_operations=len(compiled.data),
        threads=args.threads, batch_columns=args.batch_columns,
        input_columns=result['input_dimension'], output_amplitudes=result['output_dimension'],
        compile_seconds=compile_time, complete_sector_check_seconds=elapsed,
        dense_four_operators=dense, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        affinity_cpus=len(os.sched_getaffinity(0)), loadavg=os.getloadavg(),
        maximum_component_residual=max(v['max_entry'] for v in result['checks'].values()),
        correctness=result['status'], full_Fock_input_coverage=result['full_Fock_input_coverage'],
        real_molecular_evaluations=0, actual_CA_achieved=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload))


if __name__ == '__main__': main()
