"""Task-local real-Schur codec for ordinary-number real DF tensors.

Uses existing SciPy Schur/expm and ffsim parameter packing/unpacking. No package
patch, decomposition, optimizer or fallback to the old complex-log projection.
"""
import math
from .c1_h6_initialization import ATOL, reconstruct

VERSION = 'real_schur_column_gauge_v1'


def _maxabs(value):
    import numpy as np
    return float(np.max(np.abs(value)))


def real_skew_log(rotation):
    """Return A, V, column signs, diagnostics with exp(A)=V=U D.

    D flips the last column only for negative determinant. The exact real-Schur
    block structure identifies rotation planes; separate -1 real blocks are
    paired into pi planes. Every numerical reduction is checked by expm.
    """
    import numpy as np
    from scipy.linalg import schur, expm
    U=np.asarray(rotation)
    if (U.dtype!=np.float64 or np.iscomplexobj(U) or U.ndim!=2
            or not U.shape[0] or U.shape[0]!=U.shape[1] or not np.isfinite(U).all()):
        raise ValueError('REAL_FLOAT64_SQUARE_ROTATION_REQUIRED')
    n=len(U);identity=np.eye(n);orthogonal=_maxabs(U.T@U-identity)
    if orthogonal>ATOL:raise ValueError('NONORTHOGONAL_ROTATION')
    determinant=float(np.linalg.det(U))
    if abs(abs(determinant)-1)>n*ATOL:raise ValueError('INVALID_ORTHOGONAL_DETERMINANT')
    signs=np.ones(n)
    if determinant<0:signs[-1]=-1
    V=U*signs
    T,Q=schur(V,output='real',check_finite=True)
    B=np.zeros_like(T);negative=[];planes=[];i=0
    while i<n:
        # LAPACK returns exact zero at the boundary of real Schur blocks.
        # Avoid a user-chosen eigenvalue cutoff at the pi branch.
        if i+1<n and T[i+1,i]!=0:
            cosine=.5*(T[i,i]+T[i+1,i+1])
            sine=.5*(T[i,i+1]-T[i+1,i])
            theta=math.atan2(float(sine),float(cosine))
            B[i,i+1]=theta;B[i+1,i]=-theta
            planes.append(dict(indices=[i,i+1],angle=theta));i+=2
        else:
            sign=1 if T[i,i]>=0 else -1
            if abs(T[i,i]-sign)>ATOL:raise ValueError('NONORTHOGONAL_REAL_SCHUR_BLOCK')
            if sign<0:negative.append(i)
            i+=1
    if len(negative)%2:raise ValueError('UNPAIRED_MINUS_ONE_SCHUR_BLOCK')
    pairs=[]
    for i,j in zip(negative[::2],negative[1::2],strict=True):
        B[i,j]=math.pi;B[j,i]=-math.pi;pairs.append([i,j])
    raw=Q@B@Q.T;A=.5*(raw-raw.T)
    report=dict(det_original=determinant,det_representative=float(np.linalg.det(V)),
        column_signs=signs.tolist(),orthogonality_maxabs=orthogonal,
        schur_reconstruction_maxabs=_maxabs(Q@T@Q.T-V),
        schur_block_exponential_maxabs=_maxabs(expm(B)-T),
        skew_roundoff_projection_maxabs=_maxabs(A-raw),
        skew_maxabs=_maxabs(A+A.T),representative_exponential_maxabs=_maxabs(expm(A)-V),
        rotation_planes=planes,minus_one_pairs=pairs)
    if max(report[k] for k in ('schur_reconstruction_maxabs','schur_block_exponential_maxabs',
            'skew_roundoff_projection_maxabs','skew_maxabs','representative_exponential_maxabs'))>ATOL:
        raise ValueError('REAL_SCHUR_ROUNDTRIP_FAIL: '+str(report))
    return A,V,signs,report


def roundtrip(J,U):
    """Return J_out,U_out,x,diagnostics,raw arrays in locked ffsim layout.

    Unchanged J, each original column density projector and reconstructed tensor
    are required. U_out matches the sign-gauged representative, not a forbidden
    exponential representation of an original det=-1 matrix.
    """
    import numpy as np
    import ffsim.linalg.util as util
    J,U=np.asarray(J),np.asarray(U)
    if (J.dtype!=np.float64 or U.dtype!=np.float64 or J.ndim!=3 or U.shape!=J.shape
            or not len(J) or J.shape[1]!=J.shape[2] or not J.shape[1]
            or not np.isfinite(J).all() or not np.isfinite(U).all()):
        raise ValueError('REAL_DF_BATCH_SCHEMA')
    if not np.array_equal(J,J.transpose(0,2,1)):raise ValueError('EXACT_SYMMETRIC_J_REQUIRED')
    generators=[];representatives=[];reports=[]
    for frame,rotation in enumerate(U):
        A,V,_,report=real_skew_log(rotation)
        generators.append(A);representatives.append(V);reports.append(dict(frame=frame,**report))
    generators=np.asarray(generators);representatives=np.asarray(representatives)
    x=np.concatenate([util.antihermitians_to_parameters(generators,real=True),
                      util.real_symmetrics_to_parameters(J,None)])
    J2,U2=util.df_tensors_from_params(x,len(J),J.shape[-1],None,real=True)
    if not np.array_equal(J2,J):raise ValueError('FFSIM_J_ROUNDTRIP_FAIL')
    for frame,report in enumerate(reports):
        original=np.einsum('pi,qi->ipq',U[frame],U[frame])
        restored=np.einsum('pi,qi->ipq',U2[frame],U2[frame])
        report.update(restored_representative_maxabs=_maxabs(U2[frame]-representatives[frame]),
            original_rotation_difference_maxabs=_maxabs(U2[frame]-U[frame]),
            restored_orthogonality_maxabs=_maxabs(U2[frame].T@U2[frame]-np.eye(U.shape[-1])),
            density_projector_maxabs=_maxabs(restored-original))
        if max(report[k] for k in ('restored_representative_maxabs',
                'restored_orthogonality_maxabs','density_projector_maxabs'))>ATOL:
            raise ValueError('FRAME_PROJECTOR_ROUNDTRIP_FAIL: '+str(report))
    before=reconstruct(J,U);difference=reconstruct(J2,U2)-before
    allowance=ATOL*max(1.,float(np.linalg.norm(before.ravel())))
    result=dict(version=VERSION,actual_rank=len(J),norb=J.shape[-1],real=True,
        parameter_count=len(x),dtype=str(x.dtype),J_maxabs=0.,frames=reports,
        tensor_maxabs=_maxabs(difference),tensor_frobenius=float(np.linalg.norm(difference.ravel())),
        tensor_allowance=allowance,component_allowance=ATOL,rigorous_enclosure=None)
    if max(result['tensor_maxabs'],result['tensor_frobenius'])>allowance:
        raise ValueError('TENSOR_ROUNDTRIP_FAIL: '+str(result))
    return J2,U2,x,result,dict(generators=generators,representatives=representatives)
