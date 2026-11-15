"""Location of the P4 RZ cross-check difference in run02 (reads only run02/work token arrays; no QPY).

P4 compares P2R66 step 2 with S66 path1 step 1 per wire: c = common suffix, dx/dy = non-common
leading lengths, px/py = later/first deterministic entry prefixes (P_w, F_w). The RZ cross-check
uses the full prefixes; it is exact only if RZ(x[dx:px]) == RZ(y[dy:py]) on every wire.

    python -B docs/research/a_pauli_ld2_step_audit_20261115/run02/P4_RZ_CHECK.py
"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import pauli_ld2_step_audit as A

edges = json.loads((A.RUN / 'EDGES.json').read_text())['per_wire']
n = len(edges)
child = A.load_x_arrays('P2R66/p0', 2, n)
parent = A.load_x_arrays('S66/p1', 1, n)
rows = []
for w in range(n):
    x, y = child[w]['segs'][1], parent[w]['segs'][0]
    c = A.lcs_len([x[0], y[0]])
    dx, dy = len(x[0]) - c, len(y[0]) - c
    px, py = edges[w]['later_prefix_len'], edges[w]['first_prefix_len']
    rows.append(dict(wire=w, common_suffix=c, dx=dx, dy=dy, P_w=px, F_w=py,
                     rz_noncommon_child=int(x[1][:dx].sum()), rz_noncommon_parent=int(y[1][:dy].sum()),
                     rz_P_minus_F=int(x[1][:px].sum()) - int(y[1][:py].sum()),
                     rz_actual_diff=int(x[1].sum()) - int(y[1].sum()),
                     body_len_child=len(x[0]) - px - edges[w]['suffix_len'],
                     body_len_parent=len(y[0]) - py - edges[w]['suffix_len']))
for r in rows:
    print(json.dumps(r))
