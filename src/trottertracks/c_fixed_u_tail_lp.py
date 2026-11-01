"""Fixed-U tail-tensor preserving exact Pauli L1 LP; passive import.

No rotation codec, DF fitter, optimizer restart or native circuit entry.
Real saved arrays require a separately permitted scientific execution.
"""
from dataclasses import dataclass
from . import c_pr_aware_df as objective
from .c1_h6_initialization import reconstruct

VERSION='fixed_U_full_tensor_exact_tail_L1_highs_ds_v1'
OPTIONS=dict(presolve=True,primal_feasibility_tolerance=1e-10,
             dual_feasibility_tolerance=1e-10)
ALLOWANCES=dict(orthogonality=1e-10,tensor_absolute=1e-10,tensor_relative=1e-10,
    primal_absolute=1e-9,dual_absolute=1e-9,objective_absolute=1e-9,
    objective_relative=1e-9,operator_component=1e-10,identity=1e-10,
    kernel_component=1e-10,improvement_absolute=1e-8,improvement_relative=1e-8,
    improvement_gap_multiplier=10.,retained_condition_limit=1e12,
    attenuation_absolute=1e-9,rigorous_enclosure=None)


def _max(x):
    import numpy as np
    return float(np.max(np.abs(x),initial=0.))


def pack(J):
    """Same frame-major upper triangle as ffsim real_symmetrics_to_parameters."""
    import numpy as np
    J=np.asarray(J)
    if (J.dtype!=np.float64 or J.ndim!=3 or not len(J) or J.shape[1]!=J.shape[2]
            or not J.shape[1] or not np.isfinite(J).all()
            or not np.array_equal(J,J.transpose(0,2,1))):raise ValueError('REAL_SYMMETRIC_J_REQUIRED')
    return J[:,*np.triu_indices(J.shape[-1])].reshape(-1).copy()


def unpack(j,frames,n):
    import numpy as np
    j=np.asarray(j)
    if j.shape!=(frames*n*(n+1)//2,) or np.iscomplexobj(j) or not np.isfinite(j).all():raise ValueError('PACKED_J_SCHEMA')
    rows,cols=np.triu_indices(n);out=np.zeros((frames,n,n),dtype=np.float64)
    out[:,rows,cols]=j.reshape(frames,-1);out[:,cols,rows]=out[:,rows,cols]
    return out


class FixedU:
    """Contraction and its adjoint are independent of explicit LP assembly."""
    def __init__(self,U):
        import numpy as np
        U=np.asarray(U)
        if (U.dtype!=np.float64 or U.ndim!=3 or not len(U) or U.shape[1]!=U.shape[2]
                or not U.shape[1] or not np.isfinite(U).all()):raise ValueError('REAL_U_SCHEMA')
        self.U=U.copy();self.frames,self.n=U.shape[:2];self.q=self.frames*self.n*(self.n+1)//2
        self.orthogonality=_max(U.transpose(0,2,1)@U-np.eye(self.n))
        if self.orthogonality>ALLOWANCES['orthogonality']:raise ValueError('NONORTHOGONAL_U_NO_POLAR_FALLBACK')
        self.U.setflags(write=False)

    def apply(self,j):return reconstruct(unpack(j,self.frames,self.n),self.U)

    def adjoint(self,tensor):
        import numpy as np
        tensor=np.asarray(tensor)
        if tensor.shape!=(self.n,)*4 or np.iscomplexobj(tensor) or not np.isfinite(tensor).all():raise ValueError('REAL_TENSOR_SCHEMA')
        values=np.einsum('tpi,tqi,pqrs,trj,tsj->tij',self.U,self.U,tensor,self.U,self.U,optimize=True)
        i,j=np.triu_indices(self.n)
        return (values[:,i,j]+np.where(i==j,0,values[:,j,i])).reshape(-1)

    def matrix(self):
        import numpy as np
        # H4 only needs 256x60; no dense map is formed for apply/adjoint.
        columns=[]
        for U in self.U:
            P=np.einsum('pi,qi->ipq',U,U).reshape(self.n,-1)
            for i,j in zip(*np.triu_indices(self.n)):
                column=np.outer(P[i],P[j])
                if i!=j:column=column+np.outer(P[j],P[i])
                columns.append(column.reshape(-1))
        return np.column_stack(columns)

    def coefficients(self,j):
        import numpy as np
        J=unpack(j,self.frames,self.n);c=objective.coefficients(J,np)
        if np.any(np.all(c==0,axis=1)&np.any(J!=0,axis=(1,2))):raise ValueError('COEFFICIENT_UNDERFLOW_NOT_ZERO_TAIL')
        return c.reshape(-1)

    def coefficient_matrix(self):
        import numpy as np
        from scipy.sparse import block_diag
        q=self.n*(self.n+1)//2
        basis=np.array([unpack(v,1,self.n)[0] for v in np.eye(q)])
        local=objective.coefficients(basis,np).T
        return block_diag([local]*self.frames,format='csr')


@dataclass
class Problem:
    mapping: object
    j0: object
    A: object
    T: object
    b: object
    conditioning: dict


def problem(J,U):
    import numpy as np
    mapping=FixedU(U);j0=pack(J)
    if len(j0)!=mapping.q:raise ValueError('J_U_BATCH_MISMATCH')
    A=mapping.matrix();T=mapping.coefficient_matrix();b=mapping.apply(j0).reshape(-1)
    limit=ALLOWANCES['tensor_absolute']+ALLOWANCES['tensor_relative']*_max(b)
    if _max(A@j0-b)>limit:raise ValueError('ORIGINAL_CANDIDATE_NOT_NUMERICALLY_FEASIBLE')
    singular=np.linalg.svd(A,compute_uv=False)
    cutoff=float(np.finfo(float).eps*max(A.shape)*singular[0]) if len(singular) else 0.
    rank=int(np.sum(singular>cutoff))
    conditioning=dict(singular_values=singular.tolist(),diagnostic_cutoff=cutoff,effective_rank=rank,
        columns=A.shape[1],rows=A.shape[0],near_null_columns=A.shape[1]-rank,
        retained_condition=float(singular[0]/singular[rank-1]) if rank else None,
        full_condition=None if rank<A.shape[1] else float(singular[0]/singular[-1]),
        equality_rows_dropped=False,near_null_directions_used_as_exact=False,
        input_orthogonality=mapping.orthogonality,
        statement='SVD is diagnostic only. All original full-tensor rows go to HiGHS; presolve is internal. No truncated-nullspace or polar correction.')
    return Problem(mapping,j0,A,T,b,conditioning)


def solve_once(p,*,solver=None):
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix,hstack,vstack,eye
    q=p.mapping.q;m=p.T.shape[0]
    G=vstack((hstack((p.T,-eye(m))),hstack((-p.T,-eye(m)))),format='csr')
    E=hstack((csr_matrix(p.A),csr_matrix((len(p.b),m))),format='csr')
    return (solver or linprog)(np.r_[np.zeros(q),np.ones(m)],A_ub=G,b_ub=np.zeros(2*m),
        A_eq=E,b_eq=p.b,bounds=[(None,None)]*q+[(0,None)]*m,
        method='highs-ds',options=dict(OPTIONS))


def diagnose(p,result):
    """SciPy marginals: y=eqlin; z=minus-inequality minus plus-inequality.

    Then A.T@y=T.T@z, |z|<=1 and the dual objective is b@y.
    These are numerical diagnostics, not rounded validated lower bounds.
    """
    import numpy as np
    import math
    mass0=float(np.sum(abs(p.mapping.coefficients(p.j0))))
    initial=_max(p.A@p.j0-p.b);eq_limit=ALLOWANCES['tensor_absolute']+ALLOWANCES['tensor_relative']*_max(p.b)
    if initial>eq_limit:raise ValueError('ORIGINAL_CANDIDATE_NOT_NUMERICALLY_FEASIBLE')
    report=dict(status='UNRESOLVED_SOLVER',success=bool(result.success),solver_status=int(result.status),
        message=str(result.message),nit=int(result.nit or 0),original_lambda=mass0,
        original_equality_residual=initial,equality_limit=eq_limit,conditioning=p.conditioning,
        numerical_optimality=False,rigorous_certificate=None,strict_improvement=None)
    if not result.success or result.status!=0:return report
    q=p.mapping.q;m=p.T.shape[0]
    x=np.asarray(result.x);ineq=np.asarray(result.ineqlin.marginals);y=np.asarray(result.eqlin.marginals)
    if (x.shape!=(q+m,) or ineq.shape!=(2*m,) or y.shape!=p.b.shape
            or not all(np.isfinite(v).all() for v in (x,ineq,y))
            or not np.isfinite(result.fun)):raise ValueError('MALFORMED_SUCCESSFUL_LP_OUTPUT')
    j,u=x[:q],x[q:];z=ineq[m:]-ineq[:m];c=p.T@j
    primal=math.fsum(float(v) for v in u);dual=math.fsum(float(a)*float(v) for a,v in zip(p.b,y));mass=math.fsum(abs(float(v)) for v in c);gap=primal-dual
    dual_dot_scale=math.fsum(abs(float(a)*float(v)) for a,v in zip(p.b,y))
    objective_guard=float(8*np.finfo(float).eps*(2*len(p.b)+2*q+m)*max(1.,abs(primal)+mass+dual_dot_scale))
    obj_limit=ALLOWANCES['objective_absolute']+ALLOWANCES['objective_relative']*max(abs(primal),abs(dual),mass0)
    residuals=dict(equality=_max(p.A@j-p.b),epigraph=max(0.,float(np.max(abs(c)-u))),
        nonnegative_u=max(0.,float(-np.min(u))),dual_equality=_max(p.A.T@y-p.T.T@z),
        dual_box=max(0.,_max(z)-1),dual_inequality_sign=max(0.,float(np.max(ineq))),
        epigraph_dual=max(0.,float(-np.min(1+ineq[:m]+ineq[m:]))),
        complementarity=_max(ineq*np.r_[u-c,u+c]),
        fun_consistency=abs(float(result.fun)-primal),epigraph_surplus=abs(primal-mass))
    qualified=(residuals['equality']<=eq_limit
        and max(residuals[k] for k in ('epigraph','nonnegative_u'))<=ALLOWANCES['primal_absolute']
        and max(residuals[k] for k in ('dual_equality','dual_box','dual_inequality_sign','epigraph_dual'))<=ALLOWANCES['dual_absolute']
        and max(abs(gap),objective_guard,residuals['fun_consistency'],residuals['epigraph_surplus'],residuals['complementarity'])<=obj_limit
        and p.conditioning['retained_condition'] is not None
        and p.conditioning['retained_condition']<=ALLOWANCES['retained_condition_limit'])
    threshold=(ALLOWANCES['improvement_absolute']+ALLOWANCES['improvement_relative']*max(mass0,mass)
        +ALLOWANCES['improvement_gap_multiplier']*(abs(gap)+objective_guard))
    difference=mass0-mass
    status=('NUMERICAL_LP_IMPROVEMENT' if difference>threshold else
            'NUMERICAL_LP_NO_CLEAR_IMPROVEMENT' if difference>=-threshold else 'UNRESOLVED_OBJECTIVE_INCREASE') if qualified else 'UNRESOLVED_LP_DIAGNOSTICS'
    report.update(status=status,residuals=residuals,new_lambda=mass,epigraph_objective=primal,
        dual_objective_pilot=dual,primal_dual_gap=gap,numerical_optimality=qualified,
        dual_dot_absolute_scale=dual_dot_scale,objective_arithmetic_guard=objective_guard,
        old_minus_new_lambda=difference,improvement_threshold=threshold,
        strict_improvement=difference>threshold if qualified and status!='UNRESOLVED_OBJECTIVE_INCREASE' else None,
        y=y.tolist(),z=z.tolist(),qualification='full-row primal/dual numerical pilot only; no exact nullspace, certified dual bound or mathematical optimality certificate')
    return report


def replace_tail(J,U,new_tail,ld):
    import numpy as np
    pack(J);FixedU(U)
    if type(ld)is not int or not 0<=ld<len(J) or new_tail.shape!=J[ld:].shape:raise ValueError('FIXED_TAIL_PARTITION')
    pack(new_tail);out=J.copy();out[ld:]=new_tail
    return out,U.copy()


def uniform_shift(J,shifts):
    """Artificial reference family only; no extra molecular LP execution."""
    import numpy as np
    pack(J);a=np.asarray(shifts,dtype=float)
    if a.shape!=(len(J),) or not np.isfinite(a).all() or abs(float(np.sum(a)))>1e-14:
        raise ValueError('TAIL_ZERO_SUM_SHIFT_REQUIRED')
    return J+a[:,None,None]
