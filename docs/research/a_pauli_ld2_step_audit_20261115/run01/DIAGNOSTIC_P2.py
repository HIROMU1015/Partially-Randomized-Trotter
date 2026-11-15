"""Post-hoc diagnostic of the run01 P2 FAIL (not a verdict; rules unchanged).

Re-reads one already-audited path (S66/p0, axis X: its wrapper and compiled QPY, both among the
52 files read by run01; no new file) with the frozen functions of SOURCE 2f3bf20ce, and locates
the expected ancilla CX entries that are absent from the compiled circuit.

    PYTHONDONTWRITEBYTECODE=1 $VENV/bin/python -B docs/research/a_pauli_ld2_step_audit_20261115/run01/DIAGNOSTIC_P2.py
"""
import json, sys, difflib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import pauli_ld2_step_audit as A
data = json.loads((A.OUT / 'INPUTS.json').read_text())
pid = sys.argv[1] if len(sys.argv) > 1 else 'S66/p0'
p = next(x for x in data['paths'] if x['path_id'] == pid)
e = p['axes']['X']
wr = A.load_qpy(e['files']['wrapper']['path'])
st = A.uncompiled_structure(wr, p['q'])
tok = A.tokenize(A.load_qpy(e['files']['compiled']['path']))
anc = tok['n'] - 1
got = [tok['instrs'][j][1][1] for j in tok['wires'][anc] if len(tok['instrs'][j][1]) > 1]
exp = st['expected']
sm = difflib.SequenceMatcher(a=[x[0] for x in exp], b=got, autojunk=False)
missing = []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag != 'equal':
        missing.append((tag, i1, i2, j1, j2))
print('opcodes(non-equal):', missing[:20])
zero_pairs = []
for s, stp in enumerate(st['steps'], 1):
    for k, pr in enumerate(stp['pairs']):
        if float.fromhex(pr['angle']) == 0.0:
            zero_pairs.append((s, k, pr['target'], pr['angle'], pr['block']))
print('zero-angle D pairs:', len(zero_pairs), zero_pairs[:12])
dels = [(exp[i][0], exp[i][1], exp[i][2], exp[i][3]) for tag, i1, i2, j1, j2 in missing if tag == 'delete' for i in range(i1, i2)]
print('deleted expected entries:', len(dels), dels[:12])
print('blocks per step:', [len(b) for b in st['steps'][0]['blocks']], 'gps:', st['steps'][0]['gps'])
b0 = st['steps'][0]['pairs']
print('pairs 24..33:', [(k, b0[k]['target'], float.fromhex(b0[k]['angle']), b0[k]['block']) for k in range(24, 34)])
small = [(s, k, pr['target'], float.fromhex(pr['angle'])) for s, stp in enumerate(st['steps'], 1) for k, pr in enumerate(stp['pairs']) if abs(float.fromhex(pr['angle'])) < 1e-9]
print('pairs with |angle|<1e-9:', len(small), small[:10])
