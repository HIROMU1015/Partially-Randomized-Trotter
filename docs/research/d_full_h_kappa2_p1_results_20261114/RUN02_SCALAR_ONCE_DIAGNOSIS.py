"""Structure-only diagnosis of the run02 `scalar_once=false` STOP (after the STOP; no retry).

Reads the two saved uncompiled X/Y wrappers (qpy) and the saved index-0 events of run02.
No simulation, no new circuit, no compile, no sampling. It lists every single-qubit
`p` on the ancilla and compares them with what the pinned provider builder emits:
per request, the event phases (+1/-1) are accumulated and appended as at most ONE
ancilla PhaseGate (c31ef424 `trotterlib/df_rte_qiskit.py`), and compose() appends the
scalar `p(-scalar*T)` once at the end.

Run with the h6 venv (qiskit 1.3.0):  python -B -I RUN02_SCALAR_ONCE_DIAGNOSIS.py > RUN02_SCALAR_ONCE_DIAGNOSIS.json
"""
import gzip
import io
import json
import math
from collections import Counter
from pathlib import Path

from qiskit import qpy

PRIVATE = Path('/home/AbeHiromu/projects/d-full-h-kappa-20261114-private/P1-KAPPA2-M3/run02')
CELL = PRIVATE / 'cost/D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1/FULL_H_4de6130f0d3a66e5d695/0'
SCALAR, T = -0.33147781341681615, .8  # saved Full-H scalar (child result) and T of m3

events = json.loads((CELL / 'events_PRIVATE.json').read_text())['events']
per_request = []
for r, batch in enumerate(events):
    minus = sum(e['phase'] == {'real': -1.0, 'imag': 0.0} for e in batch)
    plus = sum(e['phase'] == {'real': 1.0, 'imag': 0.0} for e in batch)
    angle = math.pi * minus
    per_request.append(dict(request=r, events=len(batch), phase_minus_one=minus, phase_plus_one=plus,
                            orders=dict(Counter(str(e['taylor_order']) for e in batch)),
                            accumulated_ancilla_phase=angle))
expected = sorted([x['accumulated_ancilla_phase'] for x in per_request if x['accumulated_ancilla_phase'] != 0.]
                  + [-SCALAR * T])

wrappers = {}
for f in sorted((PRIVATE / 'compiled_cache').glob('*_wrapper.qpy.gz')):
    qc = qpy.load(io.BytesIO(gzip.decompress(f.read_bytes())))[0]
    n = qc.num_qubits - 1
    ancilla = [(i, it.operation.name, [float(p) for p in it.operation.params]) for i, it in enumerate(qc.data)
               if [qc.find_bit(q).index for q in it.qubits] == [n]]
    p = [(i, prm[0]) for i, name, prm in ancilla if name == 'p']
    axis = 'Y(sine)' if any(name == 'sdg' for _, name, _ in ancilla) else 'X(cosine)'
    wrappers[f.name.split('_')[0]] = dict(
        axis=axis, qubits=qc.num_qubits, instructions=len(qc.data),
        single_qubit_ops_on_ancilla=dict(Counter(name for _, name, _ in ancilla)),
        ancilla_p=[dict(instruction=i, angle=a) for i, a in p],
        ancilla_p_with_scalar_angle=sum(a == -SCALAR * T for _, a in p),
        last_ancilla_p_is_scalar=bool(p) and p[-1][1] == -SCALAR * T,
        ancilla_p_multiset_equals_builder_expectation=sorted(a for _, a in p) == expected,
        crzz=sum(it.operation.name == 'crzz' for it in qc.data), crz=sum(it.operation.name == 'crz' for it in qc.data))

print(json.dumps(dict(scope='STRUCTURE_ONLY_NO_SIMULATION_NO_COMPILE', scalar=SCALAR, T=T, scalar_angle=-SCALAR * T,
                      per_request=per_request, expected_ancilla_p_angles=expected, wrappers=wrappers), indent=1))
