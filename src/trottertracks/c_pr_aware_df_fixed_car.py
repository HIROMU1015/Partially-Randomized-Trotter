"""Opt-in validated CAR reference, independent of native gates and QPY.

Reuse SectorCAR, signed atom_operators, event_unitary and head scheduling.
Only their arithmetic backend changes. No molecular work happens on import.
The literal dyadic B2 map need not be unitary. A separate analytic polar-basis
bridge encloses its distance to the explicitly declared ideal unitary target.
"""
from fractions import Fraction as F
from types import SimpleNamespace

from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z

VERSION = 'fixed_CAR_Arb192_polar_bridge_v1'
TARGET = 'CAR_polar_of_exact_saved_U_exact_J_kappa_phase_v1'


class Matrix:
    """Tiny arithmetic adapter for the existing @/toarray CAR routines."""
    def __init__(self, value): self.ball = value
    def __len__(self): return self.ball.nrows()
    def __add__(self, other):
        if isinstance(other, int) and other == 0: return self
        return Matrix(self.ball + other.ball)
    __radd__ = __add__
    def __sub__(self, other): return Matrix(self.ball - other.ball)
    def __rsub__(self, other): return Matrix(other.ball - self.ball)
    def __matmul__(self, other): return Matrix(self.ball * other.ball)
    def __mul__(self, scalar): return Matrix(z._scale(self.ball, scalar))
    __rmul__ = __mul__
    def toarray(self): return self


class Arithmetic:
    @staticmethod
    def number(value):
        from flint import acb
        if isinstance(value, acb): return value
        if isinstance(value, F): return acb(z._real(value))
        value = complex(value)
        return acb(z._real(float(value.real)), z._real(float(value.imag)))
    def array(self, values):
        import numpy as np
        return np.array([[self.number(x) for x in row] for row in values], dtype=object)
    @staticmethod
    def eye(d):
        from flint import acb_mat
        return Matrix(acb_mat([[int(i == j) for j in range(d)] for i in range(d)]))
    @staticmethod
    def zeros(d):
        from flint import acb_mat
        return Matrix(acb_mat(d, d))
    @staticmethod
    def exponential(time, generator):
        from flint import acb
        return Matrix(z._scale(generator.ball, -acb(0, 1) * z._real(F(time))).exp())
    @staticmethod
    def cos(angle):
        from flint import acb
        return acb(z._real(float(angle)).cos())
    @staticmethod
    def sin(angle):
        from flint import acb
        return acb(z._real(float(angle)).sin())


def pack(matrix):
    """Exact endpoint serialization, never decimal approximations."""
    return [[[str(z._fraction(x.lower())), str(z._fraction(x.upper()))]
             for x in (matrix[i, j].real, matrix[i, j].imag)]
            for i in range(matrix.nrows()) for j in range(matrix.ncols())]


def unpack(entries, d):
    from flint import arb, acb, acb_mat
    def part(pair):
        lo, hi = map(F, pair)
        if lo > hi: raise ValueError('BALL_ENDPOINT_ORDER')
        return arb(z._real((lo + hi)/2), z._real((hi - lo)/2))
    if len(entries) != d*d: raise ValueError('BALL_MATRIX_SHAPE')
    return acb_mat(d, d, [acb(part(x[0]), part(x[1])) for x in entries])


def full_partition(modes):
    """All spin-count invariant blocks; covers every Fock basis state once."""
    if type(modes) is not int or modes < 2 or modes % 2: raise ValueError('SPIN_MODES')
    groups = {}
    for index in range(1 << modes):
        key = tuple(sum((index >> p) & 1 for p in range(s, modes, 2)) for s in (0, 1))
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _hermitian_input(M):
    import numpy as np
    a = np.asarray(M)
    if not np.isfinite(a).all() or not np.array_equal(a, a.conj().T):
        raise ValueError('EXACT_HERMITIAN_INPUT_REQUIRED')


def polar_bridge(arm, h, events):
    """Analytic model bridge, not a fit or a normalization of saved input.

    eta >= ||U*U-I||2 < 1, Q=U(U*U)^(-1/2) uniquely exists,
    ||U-Q||2 <= eta/(1+sqrt(1-eta)). Fock annihilator norm is ||u||2.
    Thus density error dn <= (sqrt(1+eta)+1)||U-Q||2. Use product
    telescoping for N_i N_j, signed Z/ZZ, and the complete trajectory.
    Hermitian head exponential error <= |t| ||H_U-H_Q||2.
    """
    from flint import acb_mat, arb
    frames = {}; errors = []
    for i, (J, U) in enumerate(zip(arm.J, arm.U, strict=True)):
        _hermitian_input(J)
        A = z._matrix(U); n = A.nrows()
        eye = acb_mat([[int(p == q) for q in range(n)] for p in range(n)])
        eta = F(z._norm(A.conjugate().transpose()*A-eye)['upper_exact'])
        if eta >= 1: raise ValueError('POLAR_BASIS_NOT_QUALIFIED')
        e = z._real(eta)
        distance = e/(1+(1-e).sqrt())
        dn = ((1+e).sqrt()+1)*distance
        jmass = sum((abs(F(float(x))) for row in J for x in row), F())
        frame_error = 2*z._real(jmass)*dn*(2+e)
        frames[i] = dict(gram_operator_upper_exact=str(eta),
            polar_distance_upper_exact=str(z._fraction(distance.upper())),
            density_upper_exact=str(z._fraction(dn.upper())),
            head_generator_upper_exact=str(z._fraction(frame_error.upper())))
        if i < arm.plan.ld:
            errors.extend([z._real(F(h)/2)*frame_error]*2)
    atom_map = {a.atom_id:a for a in arm.plan.atoms}
    for event in events:
        if event.taylor_order != 0 or event.phase != 1 or event.product_component_ids:
            raise ValueError('REGISTERED_K0_ONLY')
        atom = atom_map[event.rotation_component_id]
        frame = next(f.original_index for f in arm.general.frames
                     if arm.model.label+':'+f.frame_id == atom.frame_id)
        row = frames[frame]; eta = z._real(F(row['gram_operator_upper_exact']))
        dn = z._real(F(row['density_upper_exact']))
        delta = 2*dn*sum(((1+2*eta)**k for k in range(len(atom.support))), arb(0))
        errors.append(abs(z._real(float(event.rotation_angle)).sin())*delta)
    product = arb(1)
    for e in errors: product *= 1+e
    bound = z._fraction((product-1).upper())
    return dict(target=TARGET, upper_exact=str(bound), frames=frames,
        input_U_replaced=False, ideal_basis='unique right polar factor of each exact dyadic spatial U',
        input_nonunitarity_included=True,
        literal_map_unitarity_defect_upper_exact=str(2*bound+bound*bound),
        meaning='literal exact CAR event polynomial to ideal polar CAR unitary;not physical-integral/DF error')


def _block_task(task):
    """Existing CAR algebra, head order and event definitions with Arb numbers."""
    from flint import ctx, acb
    from .c1_molecule_extension import SectorCAR
    from .c1_stage3_reference import _ordered_exponentials
    from .c_pr_aware_df_stage_b import atom_operators
    from trotterlib.rte import event_unitary
    arm, kappa, constant, events, h, indices, bilinears = task
    with z._BALL_LOCK, ctx.workprec(z.PRECISION):
        a = Arithmetic(); d = len(indices)
        car = SimpleNamespace(n=len(kappa), d=d,
            bilinears={key:Matrix(z._matrix(E)) for key,E in bilinears.items()})
        car.E = [[car.bilinears[2*p,2*q]+car.bilinears[2*p+1,2*q+1]
                  for q in range(car.n)] for p in range(car.n)]
        base = SectorCAR.one(car,kappa,arithmetic=a)+a.number(constant)*a.eye(d)
        generators = [base]+[SectorCAR.raw_frame(car,J,U,arithmetic=a)
                             for J,U in zip(arm.J[:arm.plan.ld],arm.U[:arm.plan.ld],strict=True)]
        selected = {e.rotation_component_id for e in events}
        atom_arm = SimpleNamespace(model=arm.model,plan=SimpleNamespace(
            atoms=tuple(x for x in arm.plan.atoms if x.atom_id in selected),frames=arm.plan.frames))
        atoms = atom_operators(car,atom_arm,arithmetic=a)
        half = tuple((i,F(h)/2) for i in range(arm.plan.ld+1)); cache = {}
        left = _ordered_exponentials(generators,half,cache,arithmetic=a)
        right = _ordered_exponentials(generators,tuple(reversed(half)),cache,arithmetic=a)
        middle = a.eye(d)
        for event in events: middle = event_unitary(event,atoms,arithmetic=a) @ middle
        phase = (-acb(0,1)*z._real(F(h))*z._real(float(arm.plan.extracted_identity))).exp()
        result = (phase*(right @ middle @ left)).ball
        z._norm(result)  # rejects nonfinite enclosure
        return dict(indices=indices,entries=pack(result))


def build_reference(arm, kappa, constant, events, h=F(1,40), *, parallel=True):
    """Never reads QPY. All spin blocks, largest36 for H4, replace dense exp256."""
    import numpy as np
    from flint import ctx
    from .c1_stage1 import car_bilinears
    _hermitian_input(kappa)
    if not np.isfinite(constant): raise ValueError('NONFINITE_SCALAR')
    if arm.plan.ld != 2: raise ValueError('REGISTERED_LD2')
    full = car_bilinears(arm.model.modes)
    groups = full_partition(arm.model.modes)
    tasks = [(arm,kappa,constant,events,h,idx,
              {(p,q):full[p,q][np.ix_(idx,idx)] for p in range(arm.model.modes)
               for q in range(arm.model.modes) if p%2 == q%2}) for idx in groups]
    plan = fixed.resource_plan(len(groups))
    if parallel and arm.model.modes >= 6 and plan['workers'] > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            blocks = list(pool.map(_block_task,tasks))
    else: blocks = [_block_task(t) for t in tasks]
    with z._BALL_LOCK, ctx.workprec(z.PRECISION): bridge = polar_bridge(arm,h,events)
    return dict(status='VALIDATED_LITERAL_CAR_WITH_ANALYTIC_POLAR_BRIDGE',target=TARGET,
        system_modes=arm.model.modes,system_dimension=1<<arm.model.modes,
        blocks=blocks,bridge=bridge,resource_plan=plan,
        attenuation_applied=False,reference_independent_of_QPY=True,
        reference_arithmetic='acb matrix exponential/products;all interval radii retained',
        physical_Hamiltonian_bound=None,whole_law_bound=None)


_WORK = None
def _init_compare(record, reference):
    global _WORK
    from flint import ctx
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        blocks=[(b['indices'],unpack(b['entries'],len(b['indices']))) for b in reference['blocks']]
    indices=[i for idx,_ in blocks for i in idx]
    if len(indices)!=len(set(indices)): raise ValueError('DUPLICATE_FOCK_BASIS')
    lookup={col:(idx,A,j) for idx,A in blocks for j,col in enumerate(idx)}
    if sorted(lookup) != list(range(reference['system_dimension'])):
        raise ValueError('INCOMPLETE_FOCK_BASIS')
    _WORK=(record,reference,lookup,{})


def _compare_column(column):
    from flint import acb_mat,ctx
    r,ref,lookup,cache=_WORK; d=ref['system_dimension']
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        actual,radius=fixed._column(r,column,cache)
        target=acb_mat(2*d,1)
        if column < d: target[column,0]=1
        else:
            idx,A,j=lookup[column-d]
            for k,i in enumerate(idx):target[d+i,0]=A[k,j]
        diff=z._norm(actual-target)
        return dict(column=column,upper_exact=str(F(diff['upper_exact'])+radius),
            lower_exact=str(max(F(),F(diff['lower_exact'])-radius)),
            raw_evaluation_radius_exact=str(radius))


def compare(record, reference, *, parallel=True):
    """Full controlled basis, phase-sensitive. Reference radii counted once."""
    from flint import arb,ctx
    fixed._validate(record)
    d=2*reference['system_dimension']
    if (1<<record['qubits'])!=d or reference['target']!=TARGET:
        raise ValueError('CONTROLLED_REFERENCE_TARGET')
    plan=fixed.resource_plan(d)
    if parallel and d>=128 and plan['workers']>1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_compare,initargs=(record,reference)) as pool:
            rows=list(pool.map(_compare_column,range(d)))
    else:
        _init_compare(record,reference); rows=[_compare_column(i) for i in range(d)]
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        u=z._fraction(sum((z._real(F(x['upper_exact']))**2 for x in rows),arb(0)).sqrt().upper())
    bridge=F(reference['bridge']['upper_exact'])
    return dict(status='VALIDATED_FIXED_RAW_IDEAL_CAR',target=TARGET,norm='full_operator_2',
        dimension=d,columns_evaluated=d,phase_fit=False,columns=rows,
        literal_CAR_upper_exact=str(u),model_bridge_upper_exact=str(bridge),
        upper_exact=str(min(F(2),u+bridge)),
        lower_exact=str(max(F(),max(F(x['lower_exact']) for x in rows)-bridge)),
        reference_uncertainty='included in each acb target column;not added again',
        reference_independent_of_QPY=True,precision_bits=z.PRECISION,resource_plan=plan,
        whole_law_bound=None,physical_Hamiltonian_bound=None,CA_achieved=None,shots=None,total_resources=None)


def compare_saved_numeric(record, matrix, *, parallel=True):
    """A: certificate to fixed dyadic numbers; explicitly NOT ideal CAR.

    No norm<=2 clamp: a saved floating matrix need not be exactly unitary.
    No uncertainty about the mathematical CAR map is inferred from its bits.
    """
    from flint import ctx,arb
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        A=z._matrix(matrix);d=A.nrows()
        if A.ncols()!=d:raise ValueError('SAVED_MATRIX_SHAPE')
        entries=pack(A)
    reference=dict(system_dimension=d,blocks=[dict(indices=list(range(d)),entries=entries)])
    fixed._validate(record)
    if (1<<record['qubits'])!=2*d:raise ValueError('CONTROLLED_REFERENCE_TARGET')
    # Only a diagnostic helper for small invented matrices in this preparation.
    # The same independent full-column process path is used for a future caller.
    plan=fixed.resource_plan(2*d)
    if parallel and d>=64 and plan['workers']>1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_compare,initargs=(record,reference)) as pool:
            rows=list(pool.map(_compare_column,range(2*d)))
    else:
        _init_compare(record,reference);rows=[_compare_column(i) for i in range(2*d)]
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        upper=z._fraction(sum((z._real(F(x['upper_exact']))**2 for x in rows),arb(0)).sqrt().upper())
    return dict(status='VALIDATED_FIXED_SAVED_NUMERICAL_MATRIX_ONLY',norm='full_operator_2',phase_fit=False,
        upper_exact=str(upper),lower_exact=str(max(F(x['lower_exact']) for x in rows)),
        dimension=2*d,columns_evaluated=2*d,ideal_CAR_bound=None,reference_uncertainty=None,
        whole_law_bound=None,CA_achieved=None)


def compose(compiled_raw, raw_car, *, raw_identity, expected_raw_identity):
    """Triangle bound only for exactly the same raw operator and phase scope."""
    if raw_identity!=expected_raw_identity: raise ValueError('TRIANGLE_RAW_IDENTITY')
    if raw_car is None or compiled_raw is None: return dict(upper_exact=None,status='UNRESOLVED')
    if (compiled_raw.get('status') not in ('VALIDATED_FIXED_CIRCUIT_PAIR','VALIDATED_WITHIN_FIXED_ALLOWANCE')
            or raw_car.get('status')!='VALIDATED_FIXED_RAW_IDEAL_CAR'
            or compiled_raw.get('phase_fit') is not False or raw_car.get('phase_fit') is not False
            or compiled_raw.get('dimension')!=raw_car.get('dimension')):
        raise ValueError('TRIANGLE_QUALIFICATION')
    return dict(status='VALIDATED_FIXED_COMPILED_IDEAL_CAR',
        upper_exact=str(min(F(2),F(compiled_raw['upper_exact'])+F(raw_car['upper_exact']))),
        norm='full_operator_2',phase_fit=False,whole_law_bound=None,CA_achieved=None)
