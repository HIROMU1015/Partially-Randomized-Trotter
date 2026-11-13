"""Audit check (artificial circuits only): is the X/Y Hadamard-wrapper compiled-RZ difference local?

Fixed research compiler profile: Qiskit 1.3.0, basis rz/sx/x/cx, optimization_level 1, seed 17.
Wrapper as ax2b_native_df_v5.build_native_hadamard_wrapper: h(anc) . U . [sdg(anc)] . h(anc) . measure.
Hypothesis tested directly on the compiled circuits: after removing the trailing one-qubit run on
the ancilla (the gates after its last two-qubit gate), the X and Y compiled circuits are identical
wire by wire. Then |C_Y - C_X| <= RZ count of one ZSX run (at most 3) and all CX metrics agree.
Random fictional circuits; no chemistry, no saved inputs.
"""
import math, sys, collections
import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, transpile
assert qiskit.__version__ == '1.3.0'
PROFILE = dict(basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1, seed_transpiler=17)
rng = np.random.default_rng(1114)

def body(n, gates, final_phase):
    qc = QuantumCircuit(n + 1); a = n
    special = [0.0, math.pi / 2, math.pi, -math.pi / 2, math.pi / 4]
    for _ in range(gates):
        k = rng.integers(0, 9); q = int(rng.integers(0, n)); r = int(rng.integers(0, n))
        th = float(rng.choice(special)) if rng.random() < .25 else float(rng.normal())
        if k == 0: qc.h(q)
        elif k == 1 and q != r: qc.cx(q, r)
        elif k == 2: qc.rz(th, q)
        elif k == 3: qc.crz(th, a, q)
        elif k == 4: qc.cz(a, q)
        elif k == 5 and q != r: qc.cx(q, r); qc.crz(th, a, r); qc.cx(q, r)
        elif k == 6: qc.sx(q)
        elif k == 7: qc.cx(a, q)
        elif k == 8: qc.rz(th, a)
    qc.p(final_phase, a)       # scalar phase on the ancilla, applied once (as in compose)
    return qc

def wrap(u, axis):
    n = u.num_qubits - 1; qc = QuantumCircuit(n + 1); qc.h(n); qc.compose(u, inplace=True)
    if axis == 'sine': qc.sdg(n)
    qc.h(n); qc.add_register(ClassicalRegister(1)); qc.measure(n, 0); return qc

def compiled(qc):
    t = transpile(qc, **PROFILE); ops = t.count_ops()
    return t, dict(rz=ops.get('rz', 0), cx=ops.get('cx', 0), size=t.size(), depth=t.depth())

def wires_without_final_ancilla_run(t):
    """Per-wire ordered operation lists; the ancilla list is cut after its last two-qubit gate."""
    a = t.num_qubits - 1; wires = [[] for _ in range(t.num_qubits)]; tail = []
    for inst in t.data:
        if inst.operation.name == 'measure': continue
        idx = [t.find_bit(q).index for q in inst.qubits]
        rec = (inst.operation.name, tuple(idx), tuple(round(float(x), 12) for x in inst.operation.params))
        for q in idx: wires[q].append(rec)
        if a in idx: tail = [] if len(idx) > 1 else tail + [rec]
    if tail: wires[a] = wires[a][:-len(tail)]
    return wires, sum(r[0] == 'rz' for r in tail)

stats = collections.Counter(); diffs = collections.Counter(); bad = []
trials = int(sys.argv[1]) if len(sys.argv) > 1 else 400
for i in range(trials):
    n = int(rng.integers(2, 6)); gates = int(rng.integers(5, 120))
    phase = float(rng.choice([0.0, math.pi / 2, -math.pi / 2, math.pi, math.pi / 4])) if rng.random() < .4 else float(rng.normal())
    u = body(n, gates, phase); x, y = wrap(u, 'cosine'), wrap(u, 'sine')
    (tx, mx), (ty, my) = compiled(x), compiled(y)
    (wx, rx), (wy, ry) = wires_without_final_ancilla_run(tx), wires_without_final_ancilla_run(ty)
    d = my['rz'] - mx['rz']; diffs[d] += 1
    stats['equal_rz'] += d == 0; stats['equal_cx'] += my['cx'] == mx['cx']
    stats['max_final_run_rz'] = max(stats['max_final_run_rz'], rx, ry)
    ok = wx == wy and d == ry - rx; stats['identical_outside_final_ancilla_run'] += ok
    if not ok: bad.append((i, n, gates, phase, mx, my, rx, ry))
print('trials', trials, dict(stats)); print('C_Y - C_X distribution', dict(sorted(diffs.items())))
print('counterexamples to locality:', len(bad)); [print('  ', b) for b in bad[:5]]
sys.exit(0 if not bad else 1)
