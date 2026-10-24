"""Stage 1 array glue and independent operators; no work runs on import.

Only the separately authorized runner may supply molecular arrays. Tests use
artificial arrays. No fitter, JW algebra, Gaussian synthesis or solver is
implemented here. Optional numerical imports stay inside explicit entries.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib
import math
import os
from pathlib import Path
import sys
from types import ModuleType

from . import c1_general_k_adapter as adapter
from .c1_ffsim_compat import VERSION as COMPAT_VERSION

FFSIM_COMMIT = adapter.FFSIM_SOURCE
EXPORT_SCHEMA = "C1_STAGE1_FFSIM_NUMBER_EXPORT_v1"
FIT_SETTINGS = dict(optimize=True, max_vecs=8, tol=1e-8, method="L-BFGS-B",
                    options={}, callback=None, diag_coulomb_indices=None,
                    cholesky=True, return_optimize_result=True)
ALLOWANCE_VERSION = "binary64_operation_scale_guard8_v1"
ROUNDING_GUARD = 8


class Calls:
    """Logical calls, not BLAS operations; callback journals before execution."""
    def __init__(self, callback=None):
        self.counts = Counter()
        self.callback = callback

    def mark(self, kind, label="", count=1):
        self.counts[kind] += count
        if self.callback:
            self.callback(kind, label, count)


def expected_evaluation_calls(rank, dimension):
    return dict(adapter_import=1, physical_build=1, xdf_build=1,
                xdf_matvec=dimension, base_build=2, frame_build=2*rank,
                full_sum=2, sector_restrict=2, spectral_norm=rank+4,
                structural_frobenius=2*rank+11, allowance=3*rank+15)


def corrected_kappa(h, g):
    import numpy as np
    return h - 0.5*np.einsum("prrq->pq", g)


def finite_arrays(*arrays):
    import numpy as np
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("NONFINITE_ARRAY")


def fit_once(h, g, constant, calls, *, fit_function=None, raw_callback=None):
    """One lower-level call; factory-equivalent original-g kappa, no refit.

    A test may inject a mock; production uses the fixed ffsim implementation.
    Unsuccessful finite outputs are returned as partial evidence, never retried.
    """
    import numpy as np
    if fit_function is None:
        from .c1_ffsim_compat import double_factorized
        import ffsim  # initializes the fixed ffsim/JAX configuration
        import jax
        if not jax.config.jax_enable_x64:
            raise ValueError("JAX_X64_REQUIRED")
        fit_function = double_factorized
    finite_arrays(h, g, constant)
    if h.dtype != np.float64 or g.dtype != np.float64:
        raise ValueError("FLOAT64_INPUT_REQUIRED")
    calls.mark("fit", "one ffsim lower-level fit")
    J, U, result = fit_function(g, **FIT_SETTINGS)
    if raw_callback:
        raw_callback(J,U,result)
    arrays = dict(kappa=corrected_kappa(h, g), diag_coulomb_mats=np.asarray(J),
                  orbital_rotations=np.asarray(U), constant=np.asarray(constant),
                  optimizer_x=np.asarray(result.x), optimizer_jac=np.asarray(result.jac))
    calls.mark("fit_qualification")
    finite_arrays(*arrays.values(), result.fun)
    n = len(h)
    rank = len(J)
    if (not 0 < rank <= FIT_SETTINGS['max_vecs'] or J.shape != (rank,n,n)
            or U.shape != J.shape or any(a.dtype != np.float64 for a in arrays.values())
            or not np.array_equal(J, J.transpose(0,2,1))):
        raise ValueError("FIT_SCHEMA_OR_SYMMETRY")
    calls.mark("tensor_reconstruct")
    reconstructed = np.einsum("lij,lpi,lqi,lrj,lsj->pqrs", J,U,U,U,U,optimize=True)
    residual = g-reconstructed
    calls.mark("tensor_maxabs")
    maximum = float(np.max(np.abs(residual)))
    calls.mark("tensor_frobenius")
    frobenius = float(np.linalg.norm(residual.ravel()))
    metadata = dict(schema=EXPORT_SCHEMA, ffsim_commit=FFSIM_COMMIT,
                    compatibility_version=COMPAT_VERSION,
                    z_representation=False, one_body_role="ffsim_corrected_kappa",
                    actual_rank=rank, settings=FIT_SETTINGS,
                    optimizer=dict(success=bool(result.success), status=int(result.status),
                                   message=str(result.message), fun=float(result.fun),
                                   nit=int(result.nit), nfev=int(result.nfev),
                                   njev=int(result.njev)),
                    tensor_residual=dict(maxabs=maximum, frobenius=frobenius,
                         objective_from_export=0.5*frobenius**2,
                         objective_difference=abs(float(result.fun)-0.5*frobenius**2)))
    return arrays, metadata


def legacy_df_module():
    """Load the existing DF module without trotterlib's eager circuit imports.

    A private package namespace retains the real relative imports and functions;
    no package/function doubles are used in production or numeric tests.
    """
    namespace = "trottertracks._c1_stage1_legacy"
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(Path(__file__).resolve().parents[1]/"trotterlib")]
        sys.modules[namespace] = package
    return importlib.import_module(namespace+".df_hamiltonian")


def sector_indices(n, nalpha, nbeta):
    import numpy as np
    return np.array([s for s in range(1 << (2*n))
                     if sum((s>>(2*p))&1 for p in range(n)) == nalpha
                     and sum((s>>(2*p+1))&1 for p in range(n)) == nbeta], dtype=int)


def bit_reverse_indices(modes):
    import numpy as np
    return np.array([int(f"{s:0{modes}b}"[::-1],2) for s in range(1<<modes)])


def spin_lift_unthresholded(one, eri):
    """Pure helper copied from 0c329b30:h4_molecular_inputs.py; no SCF imports."""
    import numpy as np
    n=len(one);spin_one=np.zeros((2*n,2*n));spin_two=np.zeros((2*n,)*4)
    for a in range(2):
        spin_one[a::2,a::2]=one
        for b in range(2):
            spin_two[a::2,b::2,b::2,a::2]=eri.transpose(0,2,3,1)
    return spin_one,spin_two


def unthresholded_sparse(terms, modes):
    """Use OF's JW evaluator while bypassing symbolic += EQ_TOLERANCE."""
    from openfermion import FermionOperator, get_sparse_operator
    operator = FermionOperator()
    # Terms here are unique monomials, supplied in their original action order.
    # Assignment is intentional: SymbolicOperator.__iadd__ removes tiny values.
    operator.terms = dict(terms)
    return get_sparse_operator(operator, n_qubits=modes)


def physical_sector(h, g, constant, indices):
    import numpy as np
    from openfermion import InteractionOperator
    one, two = spin_lift_unthresholded(h,g)
    interaction = InteractionOperator(constant,one,two/2)
    sparse = unthresholded_sparse(((t,interaction[t]) for t in interaction),len(one))
    permutation = bit_reverse_indices(len(one))
    selected = permutation[indices]
    matrix = sparse[selected,:][:,selected].toarray()
    scale = abs(constant)+float(np.sum(np.abs(one)))+float(np.sum(np.abs(two/2)))
    return matrix, scale


def car_bilinears(modes):
    """Existing generic CAR bit action; sparse products avoid dense zero work."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from .pr_dictionary_validation.reference import car_annihilator
    class Context:
        @staticmethod
        def matrix(rows, columns):
            return np.zeros((rows,columns),dtype=np.complex128)
    annihilators = [csr_matrix(car_annihilator(Context,modes,p)) for p in range(modes)]
    return np.array([[(p.conj().T@q).toarray() for q in annihilators]
                     for p in annihilators])


def raw_base(kappa, constant, bilinears):
    import numpy as np
    n = len(kappa)
    lifted = np.kron(kappa,np.eye(2))
    return constant*np.eye(1<<(2*n))+np.einsum("pq,pqab->ab",lifted,bilinears)


def candidate_base(model):
    import numpy as np
    one=np.asarray(model.one_body_tensor); modes=len(one)
    terms=[((),model.constant)]+[(((p,1),(q,0)),one[p,q])
                                for p in range(modes) for q in range(modes)]
    matrix=unthresholded_sparse(terms,modes).toarray()
    p=bit_reverse_indices(modes)
    return matrix[np.ix_(p,p)]


def raw_frame(J, U, bilinears):
    import numpy as np
    n=len(J)
    # Raw spatial J/U only. No K, V, coefficient expansion or candidate helper.
    b=bilinears[0::2,0::2]+bilinears[1::2,1::2]
    densities=np.einsum("pi,qi,pqab->iab",U,U.conj(),b,optimize=True)
    weighted=np.einsum("ij,jab->iab",J,densities,optimize=True)
    matrix=0.5*np.einsum("iab,ibc->ac",densities,weighted,optimize=True)
    s=np.sum(np.abs(U),axis=0)**2
    scale=2*float(np.sum(np.abs(J)*s[:,None]*s[None,:]))
    return matrix,scale


def candidate_frame(frame):
    import numpy as np
    V=np.asarray(frame.orbital_rotation); modes=len(V); d=1<<modes
    eye=np.eye(d,dtype=complex); p=bit_reverse_indices(modes)
    zs=[]
    for a in range(modes):
        terms=[(((q,1),(r,0)),V[q,a]*V[r,a].conjugate())
               for q in range(modes) for r in range(modes)]
        density=unthresholded_sparse(terms,modes).toarray()[np.ix_(p,p)]
        zs.append(eye-2*density)
    b=1+2*np.sum(np.abs(V),axis=0)**2
    result=np.zeros((d,d),complex); scale=0.
    for support,coefficient in adapter.general_k_coefficients(frame.K):
        if not support:
            term=eye; size=1.
        elif len(support)==1:
            term=zs[support[0]]; size=b[support[0]]
        else:
            a,c=support; term=zs[a]@zs[c]; size=b[a]*b[c]
        result+=coefficient*term
        scale+=abs(coefficient)*size
    return result,float(scale)


def saved_xdf_sector(one, lambdas, matrices, constant, nalpha, nbeta, indices, calls):
    import numpy as np
    df=legacy_df_module(); modes=len(one)
    model=df.DFHamiltonian(float(constant),one,lambdas,tuple(matrices),{})
    sector=df.PhysicalSector.spin_sector(n_qubits=modes,nelec_alpha=nalpha,nelec_beta=nbeta)
    # At d=36 Python avoids numba compilation and needs no thread/global changes.
    op,counter=df.df_linear_operator(model,sector,backend="python")
    calls.mark("xdf_matvec",count=sector.dimension)
    matrix=df._linear_operator_to_dense(op)
    if counter['count']!=sector.dimension:
        raise RuntimeError('XDF_MATVEC_INVENTORY')
    reverse=bit_reverse_indices(modes)
    positions={int(reverse[s]):i for i,s in enumerate(sector.basis_indices)}
    perm=[positions[int(s)] for s in indices]
    scale=abs(constant)+float(np.sum(np.abs(one)))
    scale+=sum(abs(float(l))*float(np.sum(np.abs(G)))**2 for l,G in zip(lambdas,matrices))
    return matrix[np.ix_(perm,perm)],float(scale)


def gamma(operations):
    t=operations*2.0**-53
    if not 0 <= t < 1:
        raise ValueError("ROUNDING_OPERATION_DOMAIN")
    return t/(1-t)


def operation_lengths(n, d, rank):
    """Conservative path lengths, complex multiply/add charged at <=8 steps.

    One GEMM dot: 8*d; density coefficient sums <=n² (spin-lift
    candidate <=m²); ordered spin-density sum <=4*n²; K coefficients and
    I/Z/ZZ sum <=m²+m+1. The 32 term covers scalar/diagonal operations.
    These engineering envelopes are not interval certificates.
    """
    m=2*n
    return dict(raw=8*d+8*n*n+32*n*n+32,
                candidate=8*d+8*m*m+8*(1+m+m*(m-1)//2)+2*m*m+32,
                base=8*m*m+32,
                physical=8*(m**4+m*m+1),
                xdf=16*m*m+8*rank+32,
                unitary=8*n+8*n*n+16)


def allowance(scale_left, scale_right, operations_left, operations_right, d, difference):
    """Absolute roundoff proposal, fixed before physical data; no tuning."""
    import numpy as np
    formation=gamma(operations_left)*scale_left+gamma(operations_right)*scale_right
    # SVD/Frobenius low-order dimension allowance, separate from formation.
    norm_term=32*d*2.0**-53*float(np.linalg.norm(difference.ravel()))
    return ROUNDING_GUARD*(formation+norm_term)


class _GateFailure(Exception):
    def __init__(self,status,records,calls):
        self.result=dict(status=status,records=list(records),calls=dict(calls.counts))


def evaluate(*args,**kwargs):
    try:
        return _evaluate(*args,**kwargs)
    except _GateFailure as failure:
        return failure.result


def _evaluate(h,g,c,kappa,J,U,saved_one,saved_lambdas,saved_G,*,
             nalpha,nbeta,source_identity,input_identity,calls=None,operator_sink=None):
    """Independent Stage 1 routes; production dispatch must be authorized first."""
    import numpy as np
    from scipy.linalg import svd
    from threadpoolctl import threadpool_limits
    calls=Calls() if calls is None else calls
    n=len(h); modes=2*n; d=1<<modes; R=len(J)
    if R < 1 or R > 8:
        raise ValueError("CDF_RANK_RANGE")
    finite_arrays(h,g,c,kappa,J,U,saved_one,saved_lambdas,saved_G)
    indices=sector_indices(n,nalpha,nbeta); ds=len(indices)
    calls.mark("adapter_import")
    model=adapter.from_ffsim_number_arrays(one_body_tensor=kappa,
        diag_coulomb_mats=J,orbital_rotations=U,constant=float(c),
        z_representation=False,one_body_role="ffsim_corrected_kappa",
        frame_ids=[f"F{i}" for i in range(R)],group_ids=[f"G{i}" for i in range(R)],
        source_identity=source_identity,input_identity=input_identity,enable_general_k=True)
    lengths=operation_lengths(n,d,R)
    lengths['xdf']=16*modes*modes+8*len(saved_lambdas)+32
    records=[]
    phase='structure'
    def keep(name,matrix):
        if operator_sink:
            operator_sink(name,matrix)
    def check(label,delta,bl,br,ml,mr,*,spectral=False,gate=True):
        calls.mark("spectral_norm" if spectral else "structural_frobenius",label)
        error=float(svd(delta,compute_uv=False,lapack_driver="gesdd")[0]) if spectral else float(np.linalg.norm(delta.ravel()))
        calls.mark("allowance",label)
        # Frobenius propagation may be sqrt(d) larger than spectral envelope.
        scale_factor=1 if spectral else math.sqrt(len(delta))
        limit=scale_factor*allowance(bl,br,ml,mr,len(delta),delta)
        finite_arrays(error,limit)
        row=dict(id=label,error=error,allowance=limit,gate=gate,
                 passed=bool(error<=limit) if gate else None)
        records.append(row)
        if gate and not row['passed']:
            raise _GateFailure('SCIENTIFIC_STOP_STRUCTURE' if phase=='structure' else 'SCIENTIFIC_STOP_EQUALITY',records,calls)
        return row
    def herm(label,H,B,m):
        return check(label,H-H.conj().T,B,B,m,m)
    base_scale=abs(c)+2*float(np.sum(np.abs(kappa)))
    k0=corrected_kappa(h,g); correction_scale=float(np.sum(np.abs(h)))+0.5*float(np.sum(np.abs(g)))
    herm("kappa_Hermitian",kappa,base_scale,lengths['base'])
    check("kappa_original_g",kappa-k0,base_scale,correction_scale,lengths['base'],8*n+16)
    check("saved_XDF_kappa",saved_one-np.kron(k0,np.eye(2)),
          float(np.sum(np.abs(saved_one))),2*correction_scale,lengths['base'],8*n+16)
    for i in range(R):
        check(f"J_symmetric_{i}",J[i]-J[i].T,float(np.sum(np.abs(J[i]))),
              float(np.sum(np.abs(J[i]))),8,8)
        check(f"U_unitary_{i}",U[i].conj().T@U[i]-np.eye(n),
              float(np.linalg.norm(U[i])**2),1.,lengths['unitary'],0)
    phase='operator'
    calls.mark("physical_build"); hp,bp=physical_sector(h,g,c,indices)
    keep('physical_sector',hp)
    calls.mark("xdf_build")
    hx,bx=saved_xdf_sector(saved_one,saved_lambdas,saved_G,c,nalpha,nbeta,indices,calls)
    keep('XDF_sector',hx)
    bilinears=car_bilinears(modes)
    calls.mark("base_build","raw"); raw=raw_base(kappa,c,bilinears)
    calls.mark("base_build","K"); candidate=candidate_base(model)
    keep('raw_base',raw);keep('K_base',candidate)
    check("full_base",raw-candidate,base_scale,base_scale,lengths['base'],lengths['base'],spectral=True)
    workers=min(R,len(os.sched_getaffinity(0))) if hasattr(os,'sched_getaffinity') else min(R,os.cpu_count() or 1)
    threads=max(1,(len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1)//workers)
    def pair(i):
        return raw_frame(J[i],U[i],bilinears),candidate_frame(model.frames[i])
    br=base_scale; bk=base_scale
    for i in range(R):
        calls.mark("frame_build",f"raw_{i}"); calls.mark("frame_build",f"K_{i}")
    with threadpool_limits(limits=threads), ThreadPoolExecutor(max_workers=workers) as executor:
        for i,((a,ba),(b,bb)) in enumerate(executor.map(pair,range(R))):
            keep(f'raw_frame_{i}',a);keep(f'K_frame_{i}',b)
            check(f"full_frame_{i}",a-b,ba,bb,lengths['raw'],lengths['candidate'],spectral=True)
            raw+=a; candidate+=b; br+=ba;bk+=bb
    calls.mark("full_sum",count=2)
    keep('raw_total',raw);keep('K_total',candidate)
    for name,H,B,m in [('raw',raw,br,lengths['raw']),('K',candidate,bk,lengths['candidate'])]:
        for sigma in (0,1):
            occupations=np.array([sum((s>>(2*p+sigma))&1 for p in range(n)) for s in range(d)])
            comm=H*(occupations[None,:]-occupations[:,None])
            check(f"{name}_N{sigma}",comm,2*n*B,0,m+8*R,0)
    calls.mark("sector_restrict",count=2)
    hr=raw[np.ix_(indices,indices)]; hk=candidate[np.ix_(indices,indices)]
    keep('CDF_sector',hr);keep('K_sector',hk)
    mr=lengths['raw']+8*R; mk=lengths['candidate']+8*R
    for name,H,B,m in [('physical',hp,bp,lengths['physical']),('CDF',hr,br,mr),
                        ('K',hk,bk,mk),('XDF',hx,bx,lengths['xdf'])]:
        herm(name+"_Hermitian",H,B,m)
    check("sector_adapter",hr-hk,br,bk,mr,mk,spectral=True)
    check("sector_DF_CDF",hp-hr,bp,br,lengths['physical'],mr,spectral=True,gate=False)
    check("sector_DF_XDF",hp-hx,bp,bx,lengths['physical'],lengths['xdf'],spectral=True,gate=False)
    if dict(calls.counts)!=expected_evaluation_calls(R,ds):
        raise RuntimeError("CALL_INVENTORY_MISMATCH: "+str(dict(calls.counts)))
    return dict(status="STAGE1_REPRESENTATION_PASS" if all(r['passed'] is not False for r in records) else "SCIENTIFIC_STOP_EQUALITY",
                records=records,calls=dict(calls.counts),actual_rank=R,sector_dimension=ds,
                full_dimension=d,workers=workers,blas_threads=threads,
                allowance_version=ALLOWANCE_VERSION,operation_lengths=lengths,
                approximation_error_policy="Report only; Stage2 eligibility undecided")
