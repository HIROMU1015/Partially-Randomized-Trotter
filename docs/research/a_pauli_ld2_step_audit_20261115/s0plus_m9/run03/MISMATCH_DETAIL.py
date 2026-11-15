"""Record of the S0+ reference mismatch (reads only run03/work token arrays and REFERENCE_RUN02.json; no QPY).

For M9_ROUND/p1 step 1, wire 3: compares the frozen run02 first-step entry F_3 (375 tokens) with the
m9 segment around the first differing position, and reports token kinds (rz / multi-qubit).

    python -B docs/research/a_pauli_ld2_step_audit_20261115/s0plus_m9/run03/MISMATCH_DETAIL.py
"""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'scripts'))
import pauli_ld2_step_audit as A
HERE = Path(__file__).resolve().parent
A.WORK = HERE / 'work'
ref = json.loads((HERE.parent / 'REFERENCE_RUN02.json').read_text())['wires'][3]
F = np.array([int(x, 16) for x in ref['F']], dtype=np.uint64)
out = {}
for pid in ('M9_ROUND/p0', 'M9_ROUND/p1'):
    arr = A.load_x_arrays(pid, 512, 9)
    h, rz, mq = arr[3]['segs'][0]
    neq = np.flatnonzero(h[:len(F)] != F)
    k = int(neq[0]) if neq.size else None
    nxt = int(np.flatnonzero(mq[len(F) - 1:])[0]) + len(F) - 1 if mq[len(F) - 1:].any() else None
    out[pid] = dict(first_step_wire3_first_difference_from_F=k, F_length=len(F),
                    token_at_difference_is_rz=bool(rz[k]) if k is not None else None,
                    token_at_difference_is_multi_qubit=bool(mq[k]) if k is not None else None,
                    first_multi_qubit_position_at_or_after_F_end=nxt,
                    later_steps_matching_P_and_S_on_all_wires=None)
# do the later steps of p1 match P and S on all wires? (informational, beyond the fixed first-mismatch stop)
refs = json.loads((HERE.parent / 'REFERENCE_RUN02.json').read_text())['wires']
arr = A.load_x_arrays('M9_ROUND/p1', 512, 9)
bad = []
for s in range(1, 512):
    for w, r in enumerate(refs):
        h = arr[w]['segs'][s][0]
        P = np.array([int(x, 16) for x in r['P']], dtype=np.uint64)
        S = np.array([int(x, 16) for x in r['S']], dtype=np.uint64)
        if not (np.array_equal(h[:len(P)], P) and np.array_equal(h[len(h) - len(S):], S)):
            bad.append((s + 1, w))
out['M9_ROUND/p1']['later_steps_matching_P_and_S_on_all_wires'] = dict(steps_checked=511, mismatching=bad[:20], count=len(bad))
print(json.dumps(out, indent=1))
