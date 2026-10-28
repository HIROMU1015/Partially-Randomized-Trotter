"""Fixed-support general-J CDF: a parameter restriction, not a new optimizer.

Imports are passive. Actual molecule fitting/evaluation requires separate user
permission. The existing ffsim codec, objective and SciPy solver stay unchanged.
"""
from contextlib import contextmanager
from dataclasses import dataclass

VERSION='tail_J_path_support_compact_parameters_v1'

@dataclass(frozen=True)
class Support:
    rank:int=8
    norb:int=4
    ld:int=2

    def __post_init__(self):
        if any(type(x)is not int for x in (self.rank,self.norb,self.ld)) or self.rank<1 or self.norb<2 or not 0<=self.ld<self.rank:
            raise ValueError('SUPPORT_DIMENSIONS')

    @property
    def pairs(self):return tuple((i,j) for i in range(self.norb) for j in range(i,self.norb))
    @property
    def rotations(self):return self.rank*self.norb*(self.norb-1)//2
    @property
    def full_size(self):return self.rank*self.norb**2
    @property
    def kept(self):
        nJ=len(self.pairs)
        return tuple(range(self.rotations))+tuple(self.rotations+l*nJ+k for l in range(self.rank) for k,(i,j) in enumerate(self.pairs) if l<self.ld or j-i<=1)
    @property
    def size(self):return len(self.kept)

    def project(self,x):
        """Explicit initialization projection; this changes the DF tensor."""
        import numpy as np
        x=np.asarray(x)
        if x.shape!=(self.full_size,) or x.dtype!=np.float64 or not np.isfinite(x).all():raise ValueError('SUPPORT_FULL_PARAMETERS')
        out=np.zeros_like(x);out[list(self.kept)]=x[list(self.kept)];return out

    def reduce(self,x):
        import numpy as np
        y=self.project(x)
        if not np.array_equal(y,x):raise ValueError('NONZERO_FORBIDDEN_J')
        return y[list(self.kept)]

    def expand(self,y):
        import numpy as np
        y=np.asarray(y)
        if y.shape!=(self.size,) or y.dtype!=np.float64 or not np.isfinite(y).all():raise ValueError('SUPPORT_COMPACT_PARAMETERS')
        x=np.zeros(self.full_size);x[list(self.kept)]=y;return x

    def assess(self,J):
        import numpy as np
        J=np.asarray(J)
        if J.shape!=(self.rank,self.norb,self.norb) or J.dtype!=np.float64 or not np.isfinite(J).all() or not np.array_equal(J,J.transpose(0,2,1)):
            raise ValueError('SUPPORT_J_SCHEMA')
        if any(J[l,i,j]!=0 for l in range(self.ld,self.rank) for i,j in self.pairs if j-i>1):raise ValueError('NONZERO_FORBIDDEN_J')
        from .c_pr_aware_df import coefficients
        coeff=coefficients(J,np)
        return dict(status='SUPPORT_PASS',rank=self.rank,full_parameters=self.full_size,compact_parameters=self.size,
            J_nonzero_upper=[int(np.count_nonzero(a[np.triu_indices(self.norb)])) for a in J],
            nonidentity_Pauli_support=[int(np.count_nonzero(a)) for a in coeff],
            numerical_inner_rank=[int(np.linalg.matrix_rank(a)) for a in J],inner_rank_policy='NumPy default SVD rank;diagnostic only,no truncation',
            kept_tail_pairs=[list(p) for p in self.pairs if p[1]-p[0]<=1],frame_order=list(range(self.rank)),grouping_changed=False,
            meaning='hard parameter support;not same-H equivalence or a compiled-cost estimate')


def pullback(fun,support):
    """Chain rule under a fixed linear injection; tensor/penalty AD is reused."""
    import numpy as np
    def compact(y):
        value,gradient=fun(support.expand(y));gradient=np.asarray(gradient,dtype=np.float64)
        if gradient.shape!=(support.full_size,) or not np.isfinite(gradient).all() or not np.isfinite(value):raise ValueError('SUPPORT_VALUE_GRADIENT')
        return float(value),gradient[list(support.kept)]
    return compact


@contextmanager
def compact_optimizer(support):
    """Process-local adapter for the existing SciPy call; never iterate here.

The actual raw solver x/jac are kept as restricted_optimizer_x/jac. The
ffsim-facing x and jac are lifted to full layout; jac is the projected gradient,
NOT the unrestricted gradient. The old diagnostic hook evaluates the latter.
"""
    import numpy as np
    import scipy.optimize
    original=scipy.optimize.minimize
    def minimize(fun,x0,**kwargs):
        if kwargs!=dict(method='L-BFGS-B',jac=True,callback=None,options=None):raise ValueError('SUPPORT_SOLVER_POLICY')
        y0=support.reduce(x0);result=original(pullback(fun,support),y0,**kwargs)
        y=np.asarray(result.x,dtype=np.float64).copy();jac=np.asarray(result.jac,dtype=np.float64).copy()
        if jac.shape!=y.shape or not np.isfinite(jac).all():raise ValueError('SUPPORT_SOLVER_RESULT')
        result.restricted_optimizer_x=y;result.restricted_optimizer_jac=jac
        result.x=support.expand(y);result.jac=support.expand(jac)
        result.gradient_convention='full-layout projected gradient;raw optimizer gradient is compact'
        return result
    try:
        scipy.optimize.minimize=minimize;yield
    finally:scipy.optimize.minimize=original


def fit(g,J,U,x0,spec,support,*,before_optimizer,after_optimizer):
    from . import c_pr_aware_df as existing
    if (spec.rank,spec.norb,spec.ld)!=(support.rank,support.norb,support.ld):raise ValueError('SUPPORT_OBJECTIVE_DIMENSIONS')
    support.assess(J);support.reduce(x0)
    with compact_optimizer(support):
        a,u,result=existing.fit(g,J,U,x0,spec,before_optimizer=before_optimizer,after_optimizer=after_optimizer)
    support.assess(a)
    return a,u,result
