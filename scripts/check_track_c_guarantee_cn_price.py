"""Technical checks of the new certificate, with no price or old suite rerun."""
import os
from pathlib import Path
import sys
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import track_c_source_guarantee as a
from trottertracks.track_c_h4 import common,save_json
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

a.mp.iv.prec=a.BITS
c=common();qb=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices])
checks=[]
for index in (0,4):
    operations=c['preps'][12].deterministic_blocks[index].runtime_basis_operations
    actual,error=a.basis(a.opskey(operations))
    q=QuantumCircuit(8)
    for gate,qs in operations:q.append(gate,list(qs))
    expected=Operator(q).data[np.ix_(qb,qb)]
    difference=float(np.linalg.norm(actual-expected,2))
    assert difference<1e-10
    checks.append(dict(block=index,sector_matrix_point_difference=difference,
        SOURCE_basis_evaluation_error_upper=error,scope='implementation cross-check; not the physical bound'))

left=np.array([[.1+2.3j,-.7+1.1j],[1.0000000000000002-1.3j,.2-.4j]])
right=np.array([[.3-.8j,.9+.11j],[-.5+2.1j,-1.7+.6j]])
value,bound=a.product(left,right)
def exact(z):return Fraction.from_float(float(z.real)),Fraction.from_float(float(z.imag))
def cmul(z,w):return z[0]*w[0]-z[1]*w[1],z[0]*w[1]+z[1]*w[0]
squared=Fraction(0)
for i in range(2):
    for j in range(2):
        terms=[cmul(exact(left[i,k]),exact(right[k,j])) for k in range(2)]
        real=sum(x[0] for x in terms);imag=sum(x[1] for x in terms);v=exact(value[i,j])
        squared+=(real-v[0])**2+(imag-v[1])**2
assert Fraction.from_float(bound)**2>=squared
inertia=a.inertia(np.diag([-1.,2.]).astype(complex),0.)
assert inertia['negative_count']==1
save_json(a.OUT/'TECHNICAL_CHECKS.json',dict(SOURCE_basis=checks,
    complex_BLAS_error_encloses_exact_rational_product=True,known_inertia=inertia,
    old_44_cases_repeated=False,old_H4_q1_wrapper_repeated=False,new_price_cells=0))
print('New certificate technical checks passed',flush=True)
