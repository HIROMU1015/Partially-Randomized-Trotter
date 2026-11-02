#!/usr/bin/env python3
"""Artificial long native sequence, not an H4 benchmark or pilot."""
import argparse
from fractions import Fraction as Q
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('arb_toy_fixture', ROOT/'tests/test_h4_arb_bridge.py')
    T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
    # Rz(a) [SX SX X = I] Rz(-a) [CX CX = I], exact identity for encoded a.
    block = [T.gate('rz',2,theta=.125), T.gate('sx',2), T.gate('sx',2), T.gate('x',2),
             T.gate('rz',2,theta=-.125), T.gate('cx',2,0), T.gate('cx',2,0)]
    p = T.program(3, block * 7000, phase=.3)
    raw = [[.1,.2],[.3,-.25],[-.5,.125],[1.,-.75]]
    records = []
    for bits in T.B.PRECISIONS:
        start, cpu = time.perf_counter(), time.process_time()
        signal, record = T.evaluate(p, raw, bits)
        record.update(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
            analytic_signal_exact='1', contains_analytic_identity=bool(signal.contains(1)),
            width_target_met=Q(record['interval_width_exact']) <= Q(1,10**8))
        assert record['contains_analytic_identity'] and record['width_target_met']
        records.append(record)
    with args.output.open('x') as f:
        json.dump(dict(status='PASS_ARTIFICIAL_ONLY', total_qubits=3, native_instructions=len(p['instructions']),
            generated_from_H4=False, new_compile=0, real_H4_state_actions=0, records=records), f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(dict(status='PASS_ARTIFICIAL_ONLY', instructions=len(p['instructions']),
        widths=[float(Q(r['interval_width_exact'])) for r in records], wall_seconds=[r['wall_seconds'] for r in records])))


if __name__ == '__main__': main()
