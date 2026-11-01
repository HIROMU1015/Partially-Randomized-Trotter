"""Alternating coefficient/orbital DF blocks; passive import.

No tail-tensor equality, new penalty, fit restart, rotation gauge or optimizer
implementation. SciPy owns both optimizers and the augmented least-squares
solve; existing DF maps/codec/ffsim conventions define the objective.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import time
from . import c_fixed_u_tail_lp as linear
from . import c_pr_aware_df as pr
from . import c_rcdf_comparator as rc2

VERSION = 'alternating_isometric_J_local_orthogonal_U_v1'


@dataclass(frozen=True)
class Policy:
    max_cycles: int = 64
    j_maxiter: int = 1000
    j_solver_gtol: float = 1e-7
    j_metric_singular_floor: float = 1.
    u_maxiter: int = 25
    u_maxfun: int = 500
    u_solver_gtol: float = 1e-7
    u_ftol: float = 1e-14
    u_maxls: int = 40
    j_gradient_infinity: float = 1e-5
    u_tangent_l2: float = 1e-5
    gradient_atol: float = 1e-8
    gradient_rtol: float = 1e-7
    value_atol: float = 1e-10
    value_rtol: float = 1e-8
    stall_atol: float = 1e-12
    stall_rtol: float = 1e-10
    stall_cycles: int = 3
    orthogonality: float = 1e-10

    def __post_init__(self):
        import math
        for key, value in asdict(self).items():
            if key in ('max_cycles', 'j_maxiter', 'u_maxiter', 'u_maxfun',
                       'u_maxls', 'stall_cycles'):
                if type(value) is not int or value < 1:
                    raise ValueError('POSITIVE_INTEGER_POLICY:'+key)
            elif not math.isfinite(value) or value <= 0:
                raise ValueError('POSITIVE_FINITE_POLICY:'+key)


DEFAULT = Policy()


def weights(rank, norb):
    import numpy as np
    i, j = np.triu_indices(norb)
    return np.tile(np.where(i == j, 1., np.sqrt(2.)), rank)


class Coefficients:
    """x=D*upper(J) is isometric: ||x||2=||J||F, including head J.

    Reuse the full fixed-U map, its independent contraction/adjoint and the
    existing nonidentity Pauli map. Unlike the old LP, Ax is NOT constrained.
    """
    def __init__(self, g, U, spec):
        import numpy as np
        self.spec = spec
        self.mapping = linear.FixedU(U)
        if (self.mapping.frames, self.mapping.n) != (spec.rank, spec.norb):
            raise ValueError('SPEC_AND_FRAME_SHAPE')
        self.g = np.asarray(g)
        if (self.g.dtype != np.float64 or self.g.shape != (spec.norb,)*4
                or not np.isfinite(self.g).all()):
            raise ValueError('FINITE_REAL_TARGET_REQUIRED')
        self.w = weights(spec.rank, spec.norb)
        self.A = self.mapping.matrix()/self.w
        slots = spec.norb*(2*spec.norb+1)
        self.T = self.mapping.coefficient_matrix()[spec.ld*slots:].toarray()/self.w
        self.Q = self.A.T@self.A/spec.loss_scale
        self.is_pr = isinstance(spec, pr.Objective)
        if not self.is_pr and not isinstance(spec, rc2.Objective):
            raise ValueError('REGISTERED_OBJECTIVE_REQUIRED')

    def pack(self, J):
        if J.shape != self.mapping.U.shape:
            raise ValueError('J_FRAME_SHAPE')
        return linear.pack(J)*self.w

    def unpack(self, x):
        return linear.unpack(x/self.w, self.spec.rank, self.spec.norb)

    def parts(self, x, *, independent=False):
        import numpy as np
        J = self.unpack(x)
        residual = (self.mapping.apply(x/self.w)-self.g if independent else
                    (self.A@x-self.g.ravel()).reshape(self.g.shape))
        tensor = .5*float(np.vdot(residual, residual))
        if self.is_pr:
            smooth = float(pr.smooth_mass(J, self.spec, np))
            penalty = self.spec.gamma*(smooth/self.spec.mass_scale)**2
        else:
            smooth = float(pr.smooth_mass(J, self.spec, np))
            penalty = self.spec.strength*float(rc2.full_l2(J, np))/self.spec.l2_scale
        return dict(objective=tensor/self.spec.loss_scale+penalty,
            tensor_objective=tensor, normalized_tensor_objective=tensor/self.spec.loss_scale,
            penalty=penalty, smooth_tail_mass_Ha=smooth,
            exact_tail_mass_Ha=float(np.sum(abs(pr.coefficients(J[self.spec.ld:], np)))),
            full_ordered_Z_l2_Ha_squared=float(rc2.full_l2(J, np)))

    def vg(self, x, *, independent=False):
        import numpy as np
        parts = self.parts(x, independent=independent)
        if independent:
            residual = self.mapping.apply(x/self.w)-self.g
            grad = self.mapping.adjoint(residual)/self.w/self.spec.loss_scale
        else:
            grad = self.A.T@(self.A@x-self.g.ravel())/self.spec.loss_scale
        if self.is_pr:
            c = self.T@x
            h = np.hypot(c, self.spec.epsilon)
            grad += (2*self.spec.gamma*float(np.sum(h))/self.spec.mass_scale**2)*(
                self.T.T@(c/h))
        else:
            grad += 2*self.spec.strength*x/self.spec.l2_scale
        if not np.isfinite(parts['objective']) or not np.isfinite(grad).all():
            raise ValueError('NONFINITE_J_VALUE_GRADIENT')
        return parts['objective'], grad

    def hess(self, x):
        import numpy as np
        if not self.is_pr:
            return self.Q+2*self.spec.strength/self.spec.l2_scale*np.eye(len(x))
        c = self.T@x
        h = np.hypot(c, self.spec.epsilon)
        v = self.T.T@(c/h)
        # Stable epsilon^2/h^3; no smoothing change or added ridge.
        curvature = (self.spec.epsilon/h)**2/h
        result = self.Q+2*self.spec.gamma/self.spec.mass_scale**2*(
            np.outer(v, v)+np.sum(h)*(self.T.T@(curvature[:, None]*self.T)))
        if not np.isfinite(result).all():
            raise ValueError('NONFINITE_J_HESSIAN')
        return result


def gradient_gate(value, limit, policy):
    margin = policy.gradient_atol+policy.gradient_rtol*max(value, limit)
    return dict(value=float(value), threshold=limit, engineering_margin=margin,
        status='MET' if value+margin <= limit else
               'NOT_MET' if value-margin > limit else 'UNRESOLVED_MARGIN')


def nonincrease(before, after, policy):
    allowance = policy.value_atol+policy.value_rtol*max(abs(before), abs(after))
    if after-before > allowance:
        raise ValueError('BLOCK_OBJECTIVE_INCREASE')
    return dict(before=float(before), after=float(after), decrease=float(before-after),
                allowance=allowance, status='PASS_WITHIN_ENGINEERING_ALLOWANCE')


def _raw(result):
    import numpy as np
    value = getattr(result, 'fun', None)
    return dict(success=bool(result.success), status=int(result.status),
        message=str(result.message), fun=float(value) if value is not None and np.isfinite(value) else None,
        raw_fun_repr=repr(value), nit=int(getattr(result, 'nit', 0)),
        nfev=int(getattr(result, 'nfev', 0)), njev=int(getattr(result, 'njev', 0)),
        nhev=int(getattr(result, 'nhev', 0)))


def solve_j(problem, J, policy=DEFAULT, *, solver=None, record=lambda *args: None):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import lstsq
    evaluations = 0
    def vg(x):
        nonlocal evaluations
        evaluations += 1
        return problem.vg(x)
    x0 = problem.pack(J)
    before, gradient0 = vg(x0)
    tick, cpu = time.monotonic(), time.process_time()
    trace = []
    inverse = np.empty((0, 0))
    if problem.is_pr:
        from scipy.linalg import svd
        def callback(intermediate_result):
            trace.append(float(intermediate_result.fun))
        # This matrix defines ONLY an invertible optimizer metric. Its second
        # block is NOT added to F. Keep every singular direction, including
        # flat head directions, with a finite positive metric weight.
        B = np.vstack((problem.A/np.sqrt(problem.spec.loss_scale),
            np.sqrt(2*problem.spec.gamma)/problem.spec.mass_scale*problem.T))
        # A very wide artificial map also needs the complete right-null basis.
        _, singular, Vh = svd(B, full_matrices=B.shape[0] < B.shape[1],
                              lapack_driver='gesdd')
        singular = np.pad(singular, (0, len(x0)-len(singular)))
        scale = 1/np.maximum(singular, policy.j_metric_singular_floor)**2
        inverse = Vh.T@(scale[:, None]*Vh)
        inverse = .5*(inverse+inverse.T)
        if not np.isfinite(inverse).all():
            raise ValueError('NONFINITE_J_METRIC_NO_FALLBACK')
        result = (solver or minimize)(vg, x0, method='BFGS', jac=True,
            callback=callback, options=dict(gtol=policy.j_solver_gtol,
                maxiter=policy.j_maxiter, hess_inv0=inverse))
        x = np.asarray(result.x)
        metadata = dict(method='BFGS_full_direction_SVD_initial_metric', **_raw(result),
            metric_singular_values=singular.tolist(),
            metric_singular_floor=policy.j_metric_singular_floor,
            metric_directions_dropped=0, objective_regularization_added=False)
        raw_gradient = np.asarray(getattr(result, 'jac', np.empty(0)))
    else:
        ridge = np.sqrt(2*problem.spec.strength/problem.spec.l2_scale)
        A = np.vstack((problem.A/np.sqrt(problem.spec.loss_scale), ridge*np.eye(len(x0))))
        rhs = np.r_[problem.g.ravel()/np.sqrt(problem.spec.loss_scale), np.zeros(len(x0))]
        cond = np.finfo(float).eps*max(A.shape)
        x, _, rank, singular = (solver or lstsq)(A, rhs, cond=cond,
                                                lapack_driver='gelsd')
        x = np.asarray(x)
        metadata = dict(method='augmented_lstsq_gelsd', success=True, status=0,
            message='LAPACK solve returned; independent stationarity decides qualification',
            rank=int(rank), columns=len(x0), singular_values=np.asarray(singular).tolist(),
            cond_cutoff=cond, normal_equations=False, linear_solves=1,
            nfev=0, njev=0, nhev=0, nit=0)
        raw_gradient = np.empty(0)
    metadata.update(wall_seconds=time.monotonic()-tick, CPU_seconds=time.process_time()-cpu,
                    iteration_objectives=trace, initial_objective=before,
                    initial_J_gradient_infinity=float(np.max(abs(gradient0))))
    record(dict(metadata), dict(x0=x0, x=x, gradient0=gradient0,
                                solver_jac=raw_gradient, initial_inverse_metric=inverse))
    if x.dtype != np.float64 or x.shape != x0.shape or not np.isfinite(x).all():
        raise ValueError('NONFINITE_OR_MALFORMED_J_ENDPOINT')
    value, gradient = vg(x)
    metadata.update(independent_objective=value,
        stationarity=gradient_gate(float(np.max(abs(gradient))), policy.j_gradient_infinity, policy),
        progress=nonincrease(before, value, policy))
    if problem.is_pr:
        if (metadata['fun'] is None or abs(metadata['fun']-value) >
                policy.value_atol+policy.value_rtol*abs(value)
                or raw_gradient.dtype != np.float64 or raw_gradient.shape != x.shape
                or not np.isfinite(raw_gradient).all()
                or not np.allclose(raw_gradient, gradient,
                    atol=policy.gradient_atol, rtol=policy.gradient_rtol)):
            raise ValueError('J_SOLVER_INDEPENDENT_ENDPOINT_MISMATCH')
    metadata['value_gradient_calls_in_block'] = evaluations
    return problem.unpack(x), metadata


@lru_cache(None)
def _u_factory(rank, norb):
    """Runtime U/J/g, one shape-cached JAX graph; existing ffsim rotations.

    Its contraction is the same as ffsim's tensor loss. A local product chart
    is necessary to measure all physical tangent directions, including at pi.
    No differentiation through an inner argmin or matrix logarithm occurs.
    """
    import jax
    import jax.numpy as jnp
    from ffsim.linalg.util import unitaries_from_parameters_jax
    if not jax.config.jax_enable_x64:
        raise ValueError('JAX_X64_REQUIRED')
    def loss(eta, g, J, U, scale, penalty):
        W = unitaries_from_parameters_jax(eta/jnp.sqrt(2.), dim=norb,
                                         n_mats=rank, real=True)
        V = U@W
        reconstructed = jnp.einsum('kpi,kqi,kij,krj,ksj->pqrs', V, V, J, V, V,
                                   optimize=True)
        return .5*jnp.sum((g-reconstructed)**2)/scale+penalty
    return jax.jit(jax.value_and_grad(loss))


def u_value_gradient(eta, g, J, U, spec, penalty):
    import numpy as np
    value, grad = _u_factory(spec.rank, spec.norb)(eta, g, J, U, spec.loss_scale, penalty)
    value, grad = float(value), np.asarray(grad)
    if not np.isfinite(value) or grad.dtype != np.float64 or not np.isfinite(grad).all():
        raise ValueError('NONFINITE_U_VALUE_GRADIENT')
    return value, grad


def retract(U, eta):
    import numpy as np
    from ffsim.linalg.util import unitaries_from_parameters
    return U@unitaries_from_parameters(np.asarray(eta)/np.sqrt(2.), dim=U.shape[-1],
                                      n_mats=len(U), real=True)


def solve_u(g, J, U, spec, policy=DEFAULT, *, solver=None, record=lambda *args: None):
    import numpy as np
    from scipy.optimize import minimize
    total_tick, total_cpu = time.monotonic(), time.process_time()
    problem = Coefficients(g, U, spec)
    parts = problem.parts(problem.pack(J))
    zero = np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    evaluations = 0
    def fun(eta):
        nonlocal evaluations
        evaluations += 1
        return u_value_gradient(eta, g, J, U, spec, parts['penalty'])
    _, gradient0 = fun(zero)
    tick, cpu = time.monotonic(), time.process_time()
    trace = []
    def callback(intermediate_result):
        trace.append(float(intermediate_result.fun))
    result = (solver or minimize)(fun, zero, method='L-BFGS-B', jac=True,
        callback=callback, options=dict(ftol=policy.u_ftol, gtol=policy.u_solver_gtol,
            maxiter=policy.u_maxiter, maxfun=policy.u_maxfun,
            maxls=policy.u_maxls, maxcor=10))
    eta = np.asarray(result.x)
    metadata = dict(method='local_right_exponential_L-BFGS-B', **_raw(result),
        initial_objective=parts['objective'], wall_seconds=time.monotonic()-tick,
        CPU_seconds=time.process_time()-cpu, iteration_objectives=trace,
        initial_U_tangent_l2=float(np.linalg.norm(gradient0)))
    record(dict(metadata), dict(eta0=zero, eta=eta, gradient0=gradient0,
                                solver_jac=np.asarray(result.jac)))
    if eta.dtype != np.float64 or eta.shape != zero.shape or not np.isfinite(eta).all():
        raise ValueError('NONFINITE_OR_MALFORMED_U_ENDPOINT')
    value, grad = fun(eta)
    if (metadata['fun'] is None or abs(value-metadata['fun']) >
            policy.value_atol+policy.value_rtol*abs(value)
            or np.asarray(result.jac).shape != eta.shape
            or np.asarray(result.jac).dtype != np.float64
            or not np.allclose(grad, result.jac, atol=policy.gradient_atol, rtol=policy.gradient_rtol)):
        raise ValueError('U_SOLVER_INDEPENDENT_ENDPOINT_MISMATCH')
    new_U = retract(U, eta)
    if not np.isfinite(new_U).all() or np.max(abs(new_U.transpose(0, 2, 1)@new_U-np.eye(spec.norb))) > policy.orthogonality:
        raise ValueError('U_ORTHOGONALITY_FAIL_NO_PROJECTION')
    after = Coefficients(g, new_U, spec).parts(problem.pack(J))['objective']
    if abs(after-value) > policy.value_atol+policy.value_rtol*abs(value):
        raise ValueError('U_RETRACTION_VALUE_MISMATCH')
    _, physical_gradient = u_value_gradient(zero, g, J, new_U, spec, parts['penalty'])
    evaluations += 1
    metadata.update(progress=nonincrease(parts['objective'], after, policy),
        chart_gradient_l2=float(np.linalg.norm(grad)), independent_objective=after,
        final_U_tangent_l2_at_fixed_J=float(np.linalg.norm(physical_gradient)),
        value_gradient_calls_in_block=evaluations,
        total_block_wall_seconds=time.monotonic()-total_tick,
        total_block_CPU_seconds=time.process_time()-total_cpu)
    return new_U, metadata


def state(g, J, U, spec, policy=DEFAULT):
    import numpy as np
    p = Coefficients(g, U, spec)
    x = p.pack(J)
    parts = p.parts(x, independent=True)
    _, gj = p.vg(x, independent=True)
    zero = np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
    value, gu = u_value_gradient(zero, g, J, U, spec, parts['penalty'])
    if abs(value-parts['objective']) > policy.value_atol+policy.value_rtol*abs(value):
        raise ValueError('J_U_OBJECTIVE_MISMATCH')
    a = gradient_gate(float(np.max(abs(gj))), policy.j_gradient_infinity, policy)
    b = gradient_gate(float(np.linalg.norm(gu)), policy.u_tangent_l2, policy)
    return dict(components=parts, J_stationarity=a, U_stationarity=b,
        first_order_met=a['status'] == b['status'] == 'MET',
        orthogonality_maxabs=float(np.max(abs(U.transpose(0, 2, 1)@U-np.eye(spec.norb)))),
        rigorous_stationarity=False), gj, gu


def optimize(g, J, U, spec, policy=DEFAULT, *, call=lambda kind, label, fn: fn(),
             record=lambda *args: None):
    """One initial J solve, then U -> J pairs. Never keep stale post-U J.

    Every block is a separately counted logical unit. Finite inexact U steps
    may continue after status1 (registered iteration/evaluation limits). Other
    U solver stops terminate after the mandatory J refit. A J solve that fails
    independent stationarity terminates the arm without another attempt.
    """
    J, U = J.copy(), U.copy()
    states, blocks = [], []
    def assess(label):
        result, gj, gu = call('state_diagnostic', label, lambda: state(g, J, U, spec, policy))
        states.append(dict(label=label, **result))
        record('state', label, result, dict(J=J, U=U, gradient_J=gj, gradient_U=gu))
        return result
    def coefficient(label):
        def solve():
            tick, cpu = time.monotonic(), time.process_time()
            p = Coefficients(g, U, spec)
            value, info = solve_j(p, J, policy,
                record=lambda data, arrays: record('J_raw', label, data, arrays))
            info.update(total_block_wall_seconds=time.monotonic()-tick,
                        total_block_CPU_seconds=time.process_time()-cpu)
            return value, info
        value, info = call('J_solve', label, solve)
        blocks.append(dict(block='J', label=label, **info))
        record('J', label, info, dict(J=value, U=U))
        return value, info
    assess('initial')
    J, _ = coefficient('0')
    current = assess('0')
    status, completed, stalls = 'OUTER_LIMIT_UNQUALIFIED', 0, 0
    if current['J_stationarity']['status'] != 'MET':
        status = 'J_STATIONARITY_UNQUALIFIED'
    elif current['first_order_met']:
        status = 'FIRST_ORDER_MET'
    else:
        for cycle in range(1, policy.max_cycles+1):
            label, before = str(cycle), current['components']['objective']
            new_U, info = call('U_solve', label, lambda: solve_u(g, J, U, spec, policy,
                record=lambda data, arrays: record('U_raw', label, data, arrays)))
            blocks.append(dict(block='U', label=label, **info))
            U = new_U
            record('U', label, info, dict(J=J, U=U))
            J, _ = coefficient(label)
            current = assess(label)
            completed = cycle
            progress = nonincrease(before, current['components']['objective'], policy)
            current['cycle_progress'] = progress
            change = abs(progress['decrease'])
            stalls = stalls+1 if change <= policy.stall_atol+policy.stall_rtol*abs(before) else 0
            if current['J_stationarity']['status'] != 'MET':
                status = 'J_STATIONARITY_UNQUALIFIED'; break
            if current['first_order_met']:
                status = 'FIRST_ORDER_MET'; break
            if not info['success'] and info['status'] != 1:
                status = 'U_SOLVER_STOP_UNQUALIFIED'; break
            if stalls >= policy.stall_cycles:
                status = 'STALLED_NOT_STATIONARY'; break
    return J, U, dict(status=status, completed_cycles=completed, states=states,
        blocks=blocks, terminal=current, first_order_qualified=current['first_order_met'],
        solver_flags=[dict(block=b['block'], label=b['label'], success=b['success'],
                           status=b['status']) for b in blocks],
        policy=asdict(policy), retry=0, resume=0, rigorous_stationarity=False)
