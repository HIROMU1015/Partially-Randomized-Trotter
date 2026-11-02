"""Opt-in same-array PR coefficient chart. No optimizer or molecular I/O.

Reuse the original invertible metric, DF tensor map/adjoint, Pauli map and
analytic Hessian. The stored binary64 coefficient point is decoded ONCE.
Both derivative paths use that point, with extended residual accumulation.
"""
from types import SimpleNamespace
from . import c_j_u_precision as precision
from . import c_separated_j_u_consistent as legacy
from .c_j_u_consistency_repair import compare

VERSION='decoded_array_extended_coefficient_chart_v1'


class CoefficientChart:
    def __init__(self,p,J,policy=legacy.DEFAULT,*,original=None):
        import numpy as np
        if not p.is_pr and not getattr(p,'is_entry',False):raise ValueError('TAIL_SMOOTH_MASS_REQUIRED')
        self.ld=precision.require_extended();self.p=p;self.policy=policy
        self.original=original or (p.metric_chart(J,policy) if getattr(p,'is_entry',False) else legacy.CoefficientChart(p,J,policy))
        self.strength=p.spec.strength if getattr(p,'is_entry',False) else p.spec.gamma
        self.epsilon=p.penalty_epsilon if getattr(p,'is_entry',False) else p.spec.epsilon
        if self.original.p is not p or not np.array_equal(self.original.x0,p.pack(J)):
            raise ValueError('ORIGINAL_METRIC_POINT_MISMATCH')
        self.x0=self.original.x0.copy();self.W=self.original.W.copy()
        self.singular=self.original.singular.copy()
        self.U=p.mapping.U.astype(self.ld);self.w=p.w.astype(self.ld)
        # Existing matrix assembly with extended U, not a new DF map.
        view=SimpleNamespace(U=self.U,n=p.spec.norb)
        self.physical_A=legacy.linear.FixedU.matrix(view)/self.w
        # Pauli map is U-independent; retain the original coefficient convention.
        self.physical_T=(p.T.astype(self.ld) if getattr(p,'is_entry',False) else p.mapping.coefficient_matrix()[p.spec.ld*p.spec.norb*(2*p.spec.norb+1):].toarray().astype(self.ld)/self.w)
        self.A=self.physical_A@self.W.astype(self.ld)
        self.T=self.physical_T@self.W.astype(self.ld)
        self.Q=self.A.T@self.A/self.ld(p.spec.loss_scale)

    def point(self,z):
        import numpy as np
        if (z.dtype!=np.float64 or z.shape!=self.x0.shape or not np.isfinite(z).all()):
            raise ValueError('FINITE_BINARY64_CHART_POINT')
        x=self.x0+self.W@z
        J=self.p.unpack(x)  # one binary64 decode, same as an actual endpoint
        return x,J

    def _primal(self,z):
        import numpy as np
        x,J=self.point(z);j=J.astype(self.ld);p=self.p
        residual=legacy.linear.reconstruct(j,self.U)-p.g.astype(self.ld)
        c=(p.penalty_coefficients(j) if getattr(p,'is_entry',False) else legacy.pr.coefficients(j[p.spec.ld:],np).ravel())
        h=np.hypot(c,np.asarray(self.epsilon,dtype=self.ld));mass=(p.penalty_mass(j) if getattr(p,'is_entry',False) else np.sum(h,dtype=self.ld))
        value=np.sum(residual*residual,dtype=self.ld)/(2*self.ld(p.spec.loss_scale))+self.ld(self.strength)*(mass/self.ld(p.spec.mass_scale))**2
        return x,J,residual,c,h,mass,value

    def vg(self,z):
        import numpy as np
        _,_,res,c,h,mass,value=self._primal(z);p=self.p
        grad=self.A.T@res.ravel()/self.ld(p.spec.loss_scale)+2*self.ld(self.strength)*mass/self.ld(p.spec.mass_scale)**2*(self.T.T@(c/h))
        if not np.isfinite(value) or not np.isfinite(grad).all():raise ValueError('NONFINITE_POINT_VG')
        return float(value),np.asarray(grad,dtype=float)

    def hess(self,z):
        import numpy as np
        _,_,_,c,h,mass,_=self._primal(z);p=self.p
        v=self.T.T@(c/h)
        H=self.Q+2*self.ld(self.strength)/self.ld(p.spec.mass_scale)**2*(np.outer(v,v)+mass*(
            self.T.T@(((np.asarray(self.epsilon,dtype=self.ld)/h)**2/h)[:,None]*self.T)))
        if not np.isfinite(H).all():raise ValueError('NONFINITE_POINT_HESSIAN')
        return np.asarray(H,dtype=float)

    def independent(self,z):
        """Original-coordinate tensor contraction/adjoint, then W pullback.

        Keep the physical gradient extended until AFTER the pullback. Casting
        before a cancelling coordinate contraction is not the same operation.
        """
        import numpy as np
        _,_,res,c,h,mass,value=self._primal(z);p=self.p
        view=SimpleNamespace(U=self.U,n=p.spec.norb)
        gx=legacy.linear.FixedU.adjoint(view,res)/self.w/self.ld(p.spec.loss_scale)
        gx+=2*self.ld(self.strength)*mass/self.ld(p.spec.mass_scale)**2*(self.physical_T.T@(c/h))
        return float(value),gx,np.asarray(self.W.astype(self.ld).T@gx,dtype=float)

    def assess(self,z,*,vectors=False):
        import numpy as np
        value,gz=self.vg(z);independent,gx,pullback=self.independent(z);policy=self.policy
        checks=dict(value=compare(value,independent,atol=policy.value_atol,rtol=policy.value_rtol),
            chart_vs_physical_gradient=compare(gz,pullback,atol=policy.gradient_atol,rtol=policy.gradient_rtol))
        if not all(v['PASS'] for v in checks.values()):raise ValueError('NEW_POINT_CONSISTENCY_MISMATCH')
        row=dict(status='NUMERICAL_POINT_CONSISTENCY_PASS',version=VERSION,objective=value,checks=checks,
            longdouble_significant_bits=np.finfo(self.ld).nmant+1,
            objective_unchanged=True,thresholds_changed=False,directions_dropped=0,
            evaluation_point='one actual binary64 x=x0+Wz and J=unpack(x);no affine-residual reassociation',
            independent_path='existing tensor-adjoint before extended W pullback',
            rigorous_gradient_error_bound=None)
        return (row,dict(gradient_chart=gz,gradient_physical_extended=gx,gradient_pullback=pullback)) if vectors else row
