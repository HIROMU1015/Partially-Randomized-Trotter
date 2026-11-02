# Adapted from qiskit-nature 0.7.2, bogoliubov_transform.py and utils/linalg.py.
# (C) Copyright IBM 2022, 2023. Licensed under Apache License, Version 2.0.
# See LICENSES/qiskit-nature-Apache-2.0.txt. SPDX-License-Identifier: Apache-2.0
# Modifications: preserve nonzero entries; stable atan2 angle binding; project-local
# numeric number-conserving routine only. No shared package modification.
"""Stable version of the existing RZ/XXPlusYY number-conserving native lowering."""
import numpy as np
from qiskit.circuit.library import RZGate,XXPlusYYGate
from qiskit_nature.utils import apply_matrix_to_slices


def exact_zero_givens(a,b):
    if a==0:
        cosine=0.;sine=1.
    elif b==0:
        cosine=1.;sine=0.
    else:
        hypotenuse=np.hypot(abs(a),abs(b))
        cosine=abs(a)/hypotenuse
        sine=(a/abs(a))*b.conjugate()/hypotenuse
    return np.array([[cosine,sine],[-np.conjugate(sine),cosine]])


def stable_givens_ops(u):
    """Same U.T, elimination order and native gate family as the original path.

    Only mathematically zero entries are skipped. atan2(|s|,c) avoids losing a
    small nonzero angle when the Givens cosine rounds to one in binary64.
    """
    u=np.asarray(u)
    if u.ndim!=2 or len(u)!=u.shape[1] or not np.isfinite(u).all():
        raise ValueError('FINITE_SQUARE_GAUSSIAN_MATRIX')
    n=len(u);current=u.T.copy();left=[];right=[]
    for i in range(n-1):
        if i%2==0:
            for j in range(i+1):
                target=i-j;row=n-j-1
                if current[row,target]!=0:
                    g=exact_zero_givens(current[row,target+1],current[row,target])
                    right.append((g,(target+1,target)))
                    current=apply_matrix_to_slices(current,g,
                        [(Ellipsis,target+1),(Ellipsis,target)])
        else:
            for j in range(i+1):
                target=n-i+j-1;col=j
                if current[target,col]!=0:
                    g=exact_zero_givens(current[target-1,col],current[target,col])
                    left.append((g,(target-1,target)))
                    current=apply_matrix_to_slices(current,g,[target-1,target])
    for g,(i,j) in reversed(left):
        g=g.T.conj().astype(complex,copy=True)
        g[:,0]*=current[i,i];g[:,1]*=current[j,j]
        new=exact_zero_givens(g[1,1],g[1,0]);right.append((new.T,(i,j)))
        phase=g@new;current[i,i]=phase[0,0];current[j,j]=phase[1,1]
    operations=[(RZGate(float(np.angle(current[i,i]))),(i,)) for i in range(n)]
    for g,(i,j) in reversed(right):
        theta=float(np.arctan2(abs(g[0,1]),np.real(g[0,0])))
        phi=float(np.angle(g[0,1]))
        operations.append((XXPlusYYGate(2*theta,phi-np.pi/2),(j,i)))
    return operations
