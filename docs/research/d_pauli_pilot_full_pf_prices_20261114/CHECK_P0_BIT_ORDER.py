"""Representation check of A's saved P0 LD1/LD2 operators (no evolution, no signal).

Reads the pinned dictionary JSON and saved full-2^n arrays (4220f300) and checks:
labels are reversed between Qiskit and saved order; sum c P(saved label) equals the
saved Pauli_nonidentity; A's reverse_bits maps it to sum c P(qiskit label); saved
head/tail equal the block sums; tail = Pauli + identity (pauli_identity and saved s_R).
"""
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import pauli_tail_price_connection as P  # noqa: E402

D = 'docs/research/df_head_pauli_tail_p0_results_20261114/run01/'
COMMIT = '4220f3006d0d8d413e195ed948e250578513b557'


def blob(path):
    return subprocess.check_output(['git', 'show', COMMIT + ':' + D + path], cwd=ROOT)


def main():
    import numpy as np
    common = np.load(io.BytesIO(blob('COMMON_OPERATORS.npz')))
    out = dict(source=dict(commit=COMMIT, directory=D), blocks=int(common['blocks'].shape[0]),
               dimension=int(common['blocks'].shape[1]), prefixes={})
    for prefix in (1, 2):
        d = json.loads(blob('LD%d_DICTIONARY.json' % prefix))
        ops = np.load(io.BytesIO(blob('LD%d_OPERATORS.npz' % prefix)))
        r = P.bit_order_checks(d, {k: ops[k] for k in ops.files}, common['blocks'])
        r.update(terms=len(d['terms']), pauli_identity_minus_saved_sR=d['pauli_identity'] - d['saved_DF_scalar'])
        out['prefixes'][str(prefix)] = r
    out['pass'] = all(v['pass'] for v in out['prefixes'].values())
    return out


if __name__ == '__main__':
    result = main()
    (Path(__file__).parent / 'P0_BIT_ORDER_CHECK.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))
