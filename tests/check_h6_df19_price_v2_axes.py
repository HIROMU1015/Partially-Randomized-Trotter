"""Artificial parallel-axis regression; no molecular data or RTE draws."""
import sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_h6_df19_price_v2_q4 as m
import qiskit
from qiskit import QuantumCircuit
from functools import partial
import trotterlib.rte_compiled_cost as costs
from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution
costs.transpile=partial(qiskit.transpile,num_processes=1)
qc=QuantumCircuit(2);qc.crz(.37,1,0);qc.p(.23,1)
with tempfile.TemporaryDirectory(prefix='h6_v2_artificial_axes_') as root:
    a=Path(root)/'sequential';b=Path(root)/'parallel';a.mkdir();b.mkdir()
    with use_fingerprint_reuse() as fp:
        evolution=NativeEvolution(qc,1,.5,1,'2nd','symmetric_directional',4,fp.fingerprint(qc))
        # Fork before compiling in the parent, matching the production lifetime.
        parallel=m.parallel_axes(b,evolution,fp)
        sequential={axis:m.price_axis(axis,a,evolution,fp) for axis in m.p.AXES}
    for axis in m.p.AXES:
        assert parallel[axis]['metrics']==sequential[axis]['metrics']
        assert parallel[axis]['fingerprint']==sequential[axis]['fingerprint']
print('Artificial X/Y fork vs sequential: all six metrics and canonical wrapper fingerprints PASS (4 synthetic compiles)')
