"""Opt-in entry-L1 backend for the existing stable reduced/Armijo solver.

The oracle definition/weights/smoothing come from c_near_objective. Only its
linear smooth-norm derivatives and the explicit backend wiring are added here.
No module monkeypatch, molecular I/O, fallback, or second optimizer.
"""
from functools import partial
from types import SimpleNamespace
from . import c_near_objective as near
from . import c_pr_inner_point_solver as inner
from . import c_pr_guarded_step as guarded
from . import c_u_step_control as control

stable = inner.stable
VERSION = 'entry_L1_same_array_stable_reduced_backend_v1'


def validate(spec):
    if type(spec) is not near.Objective:
        raise ValueError('ENTRY_OBJECTIVE_IDENTITY_REQUIRED')
    if spec.strength <= 0:
        raise ValueError('ENTRY_POSITIVE_STRENGTH_REQUIRED_FOR_ADMISSION')


class Coefficients(stable.Coefficients):
    """Reuse isometric packing, tensor map/adjoint; replace tail linear map.

    Unique-entry c=(5/4 Jii, 3 Jij for i<j). Vector smoothing is w*eta,
    hence sum hypot(c,w*eta)=B_eta, including the ordered off-diagonal pairs.
    """
    is_entry = True

    def __init__(self, g, U, spec):
        import numpy as np
        validate(spec)
        super().__init__(g,U,near.pr.Objective(spec.rank,spec.norb,spec.ld,
            spec.loss_scale,spec.mass_scale,0.,spec.epsilon))
        self.spec,self.is_pr=spec,False
        self.i,self.k=np.triu_indices(spec.norb)
        self.entry_weights=near.entry_weights(spec.norb,np)[self.i,self.k]*np.where(self.i==self.k,1.,2.)
        self.penalty_epsilon=np.tile(self.entry_weights*spec.entry_epsilon,spec.rank-spec.ld)
        start=spec.ld*len(self.i)
        self.T=np.zeros((len(self.penalty_epsilon),len(self.w)))
        self.T[:,start:]=np.diag(np.tile(self.entry_weights,spec.rank-spec.ld)/self.w[start:])

    def penalty_coefficients(self,J):
        return (J[self.spec.ld:,self.i,self.k]*self.entry_weights).ravel()

    def penalty_mass(self,J):
        import numpy as np
        return near.mass(J,self.spec,np)  # original B oracle, also in LD

    def metric_chart(self,J,policy):
        import numpy as np
        W,singular=stable.blocks.metric(np.vstack((self.A/np.sqrt(self.spec.loss_scale),
            np.sqrt(2*self.spec.strength)*self.T/self.spec.mass_scale)),policy.metric_singular_floor)
        return SimpleNamespace(p=self,x0=self.pack(J),W=W,singular=singular)

    def parts(self,x,*,independent=False):
        return parts_gradient_J(self.g,self.unpack(x),self.mapping.U,self.spec,extended=False)[0]

    def vg(self,x,*,independent=False):
        p,g,_=parts_gradient_J(self.g,self.unpack(x),self.mapping.U,self.spec,extended=False)
        return p['objective'],g


def parts_gradient_J(g,J,U,spec,*,extended=True):
    """Same supplied J/U for extended primal, analytic entry adjoint and JAX U.

    Entry-penalty AD remains independent in near.make_value_and_grad/tests.
    Two agreeing residual paths are not an error enclosure.
    """
    import numpy as np
    validate(spec)
    ld=stable.precision.require_extended() if extended else np.float64
    for a in (g,J,U):
        if a.dtype!=np.float64 or not np.isfinite(a).all():
            raise ValueError('FINITE_BINARY64_ARRAY_POINT')
    j,u=J.astype(ld),U.astype(ld)
    residual=stable.linear.reconstruct(j,u)-g.astype(ld)
    view=SimpleNamespace(U=u,n=spec.norb)
    w=stable.old.weights(spec.rank,spec.norb).astype(ld)
    grad=stable.linear.FixedU.adjoint(view,residual)/w/ld(spec.loss_scale)
    mass=near.mass(j,spec,np)
    weights=near.entry_weights(spec.norb,np).astype(ld)
    derivative=np.zeros_like(j)
    derivative[spec.ld:]=weights*j[spec.ld:]/np.hypot(j[spec.ld:],ld(spec.entry_epsilon))
    i,k=np.triu_indices(spec.norb)
    fold=np.where(i==k,1,2).astype(ld)
    grad+=2*ld(spec.strength)*mass/ld(spec.mass_scale)**2*(derivative[:,i,k]*fold).ravel()/w
    tensor=np.sum(residual*residual,dtype=ld)/2
    penalty=ld(spec.strength)*(mass/ld(spec.mass_scale))**2
    return dict(objective=float(tensor/ld(spec.loss_scale)+penalty),
        tensor_objective=float(tensor),normalized_tensor_objective=float(tensor/ld(spec.loss_scale)),
        penalty=float(penalty),smooth_entry_mass_Ha=float(mass),
        smooth_tail_mass_Ha=float(near.pr.smooth_mass(j,spec,np)),
        exact_tail_mass_Ha=float(np.sum(abs(near.pr.coefficients(j[spec.ld:],np)),dtype=ld)),
        full_ordered_Z_l2_Ha_squared=float(np.sum(j*j,dtype=ld)),
        objective_identity=near.VERSION),np.asarray(grad,dtype=float),residual


def state(*args,**kwargs):
    return stable.state(*args,parts_evaluator=parts_gradient_J,**kwargs)


def solve_j(*args,**kwargs):
    recorder=kwargs.pop('record',lambda *a:None)
    def record(info,arrays):
        recorder(dict(info,objective_identity=near.VERSION,backend=VERSION),arrays)
    J,info=inner.solve_j(*args,parts_evaluator=parts_gradient_J,record=record,**kwargs)
    return J,dict(info,objective_identity=near.VERSION,backend=VERSION)


def applicability(p,policy=stable.DEFAULT):
    return stable.previous.applicability(p,policy)


class Backend:
    validate=staticmethod(validate)
    Coefficients=Coefficients
    state=staticmethod(state)
    solve_j=staticmethod(solve_j)
    applicability=staticmethod(applicability)


BACKEND=Backend()


def trajectory(g,J,U,spec,physical,**kwargs):
    """Dispatch explicitly. PR default still uses its frozen evaluation path."""
    if type(spec) is near.pr.Objective:
        return control.trajectory(g,J,U,spec,physical,**kwargs)
    validate(spec)
    return control.trajectory(g,J,U,spec,physical,
        oracle_factory=partial(guarded.Oracle,backend=BACKEND),
        terminal_assessor=partial(control.small.assess,state_evaluator=state),**kwargs)
