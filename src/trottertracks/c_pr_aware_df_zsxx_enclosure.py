"""Validated *captured* ZSXX residuals, never an all-input libm certificate.

Actual fixed Rust output is frozen as binary64 angles and phase. FLINT/Arb
encloses the mathematical operator of those values;no output phase is fitted.
Regions compare reference matrices to that captured output, not unobserved
executions of the compiler on varying inputs. No molecular loader exists.
"""
from fractions import Fraction
import math
import threading

VERSION='ZSXX_captured_pair_Arb_enclosure_v1'
PRECISION=192
NEAR_UNITARY_DEFECT=Fraction(1,4)
_BALL_LOCK=threading.RLock()


def runtime_model():
    """Read process rounding and fixed packages;never change rounding mode."""
    import ctypes
    import flint
    import qiskit
    if (flint.__version__!='0.8.0' or flint.__FLINT_VERSION__!='3.3.1'
            or qiskit.__version__!='1.3.0'):
        raise ValueError('PINNED_VALIDATION_RUNTIME')
    libm=ctypes.CDLL('libm.so.6');libm.fegetround.restype=ctypes.c_int
    if libm.fegetround()!=0:raise ValueError('ROUND_TO_NEAREST_REQUIRED')
    return dict(basic_model='finite binary64 bit values;process FE_TONEAREST observed',
        FMA_build_policy='UNRESOLVED;not needed for captured-pair residual',
        gradual_underflow='probe only,not a universal libm accuracy proof',
        Rust_atan2_arg_accuracy=None,Rust_trig_accuracy=None,Rust_remainder_accuracy=None,
        recorded_output_evaluated_by='FLINT3.3.1 validated ball arithmetic',
        arbitrary_Rust_reexecution_covered=False)


def rational(x):
    if isinstance(x,Fraction):return x
    if type(x) is int:return Fraction(x)
    if type(x) is str:return Fraction(x)
    raise ValueError('EXACT_RATIONAL_REQUIRED')


def _real(x):
    """Exact float bit-value, or an outward-enclosed rational."""
    from flint import arb
    if isinstance(x,(float,int)):
        if isinstance(x,float) and not math.isfinite(x):raise ValueError('NONFINITE')
        q=Fraction(x)
    else:q=rational(x)
    if q.denominator & (q.denominator-1)==0:
        return arb((q.numerator,-(q.denominator.bit_length()-1)))
    return arb(q.numerator)/arb(q.denominator)


def _fraction(point):
    if not point.is_finite():raise ValueError('NONFINITE_BALL_BOUND')
    m,e=point.man_exp();m,e=int(m),int(e)
    return Fraction(m*(1<<e)) if e>=0 else Fraction(m,1<<(-e))


def _norm(A):
    """max-entry lower <= spectral norm <= Frobenius outward upper."""
    from flint import arb
    squared=arb(0);lower=Fraction()
    for i in range(A.nrows()):
        for j in range(A.ncols()):
            z=A[i,j]
            if not z.is_finite():raise ValueError('NONFINITE_BALL_MATRIX')
            squared+=z.abs_upper()**2
            lower=max(lower,_fraction(z.abs_lower()))
    return dict(lower_exact=str(lower),upper_exact=str(_fraction(squared.sqrt().upper())),
                norm='operator_2',upper_method='outward Frobenius;no numerical SVD')


def _matrix(M,radius=Fraction()):
    from flint import arb,acb,acb_mat
    radius=rational(radius)
    if radius<0:raise ValueError('NEGATIVE_REGION_RADIUS')
    pad=arb(0,_real(radius))
    return acb_mat([[acb(_real(float(z.real))+pad,_real(float(z.imag))+pad)
                     for z in row] for row in M])


def _scale(A,z):
    from flint import acb_mat
    return acb_mat([[A[i,j]*z for j in range(A.ncols())] for i in range(A.nrows())])


def gate_matrix(gate,angle=None):
    """Independent exact gate meaning via existing validated functions."""
    from flint import acb,acb_mat
    I=acb(0,1)
    if gate=='id':return acb_mat([[1,0],[0,1]])
    if gate=='x':return acb_mat([[0,1],[1,0]])
    if gate=='z':return acb_mat([[1,0],[0,-1]])
    if gate=='sx':return acb_mat([[(1+I)/2,(1-I)/2],[(1-I)/2,(1+I)/2]])
    if gate=='sxdg':return acb_mat([[(1-I)/2,(1+I)/2],[(1+I)/2,(1-I)/2]])
    if angle is None:raise ValueError('GATE_ANGLE_REQUIRED')
    t=angle if hasattr(angle,'is_finite') else _real(float(angle))
    if gate=='rz':return acb_mat([[(-I*t/2).exp(),0],[0,(I*t/2).exp()]])
    if gate=='p':return acb_mat([[1,0],[0,(I*t).exp()]])
    if gate=='ry':
        c,s=(t/2).cos(),(t/2).sin()
        return acb_mat([[c,-s],[s,c]])
    raise ValueError('UNSUPPORTED_EXACT_GATE:'+gate)


def _product(gates,phase=0.):
    from flint import acb
    # Balanced *reference* products avoid severe interval wrapping in long
    # left folds. Exact chronological meaning is unchanged;Arb does matmul.
    products=[gate_matrix(gate,angle) for gate,angle in gates]
    while len(products)>1:
        products=[products[i+1]*products[i] if i+1<len(products) else products[i]
                  for i in range(0,len(products),2)]
    result=products[0] if products else gate_matrix('id')
    return _scale(result,(acb(0,1)*_real(float(phase))).exp())


def _record(sequence):
    names={'StandardGate.RZGate':'rz','StandardGate.SXGate':'sx','StandardGate.XGate':'x'}
    gates=[]
    for name,params in sequence:
        if str(name) not in names:raise ValueError('NOT_ZSXX_OUTPUT')
        gate=names[str(name)]
        if len(params)!=(1 if gate=='rz' else 0):raise ValueError('OUTPUT_GATE_SCHEMA')
        angle=float(params[0]) if params else None
        if angle is not None and not math.isfinite(angle):raise ValueError('NONFINITE_OUTPUT')
        gates.append(dict(gate=gate,angle_hex=None if angle is None else angle.hex()))
    phase=float(sequence.global_phase)
    if not math.isfinite(phase):raise ValueError('NONFINITE_OUTPUT_PHASE')
    return dict(gates=gates,phase_hex=phase.hex(),phase_fit=False)


def _output(record):
    if record.get('phase_fit') is not False:raise ValueError('PHASE_FIT_FORBIDDEN')
    gates=[(x['gate'],None if x['angle_hex'] is None else float.fromhex(x['angle_hex']))
           for x in record['gates']]
    return _product(gates,float.fromhex(record['phase_hex']))


def _scope(kind):
    return dict(status='VALIDATED_REGION_ENCLOSURE',kind=kind,
        precision_bits=PRECISION,backend='python-flint0.8.0/FLINT3.3.1 Arb ball arithmetic',
        output_scope='this captured output;no arbitrary phase fitting',
        platform_libm_accuracy_assumed=False,global_branch_coverage=False,
        unobserved_reexecution_guaranteed=False,whole_law_bound=None,
        actual_H4_bound=None,guaranteed_shots=None,guaranteed_total=None)


def captured_matrix(M,*,reference_radius='0'):
    """Enclose ||X-V|| for all X in a box around fixed binary64 M.

    Rust is invoked only at M. For nonzero reference_radius this is a held-V
    reference uncertainty region, not compiler(X) coverage over that box.
    """
    import numpy as np
    from flint import ctx
    from qiskit._accelerate import euler_one_qubit_decomposer as rust
    runtime_model()
    M=np.array(M,dtype=np.complex128,order='C',copy=True)
    if M.shape!=(2,2) or not np.isfinite(M).all():raise ValueError('FINITE_2X2_REQUIRED')
    radius=rational(reference_radius)
    if radius<0:raise ValueError('NEGATIVE_REGION_RADIUS')
    with _BALL_LOCK,ctx.workprec(PRECISION):
        center=_matrix(M);defect=_norm(center.conjugate().transpose()*center-gate_matrix('id'))
        if Fraction(defect['upper_exact'])>NEAR_UNITARY_DEFECT:
            return dict(status='UNRESOLVED',reason='outside near-unitary input domain',
                        defect=defect,whole_law_bound=None)
        output=_record(rust.unitary_to_gate_sequence(M,['ZSXX'],0))
        V=_output(output);point=_norm(center-V);reference_box=_matrix(M,radius)
        region=_norm(reference_box-V)
        region_defect=_norm(reference_box.conjugate().transpose()*reference_box-gate_matrix('id'))
        return dict(_scope('captured_binary64_matrix_to_ZSXX'),
            input_hex=[[[float(z.real).hex(),float(z.imag).hex()] for z in row] for row in M],
            near_unitarity_defect=defect,reference_radius_exact=str(radius),output=output,
            reference_box_defect=region_defect,
            point_distance=point,region_distance=region,
            domain='all reference X with per-real/imag-entry radius around M;V is held fixed',
            compiler_inputs_covered='the single recorded binary64 M only')


def _angular_target(values,radius=Fraction()):
    """ZSXX parameter convention: e^{i(beta+(t+p+l)/2)} Rz(p)Ry(t)Rz(l)."""
    from flint import arb,acb
    t,p,l,b=values
    t=_real(t)+arb(0,_real(radius));p,l,b=map(_real,(p,l,b))
    target=gate_matrix('rz',p)*gate_matrix('ry',t)*gate_matrix('rz',l)
    return _scale(target,(acb(0,1)*(b+(t+p+l)/2)).exp())


def captured_angles(theta,phi,lam,phase,*,theta_reference_radius='0'):
    """Exercise actual shared Rust emitter, including simplification paths.

    This isolates the post-angle stage. It does not validate M->Euler angles.
    """
    from flint import ctx
    from qiskit._accelerate import euler_one_qubit_decomposer as rust
    runtime_model()
    values=tuple(map(float,(theta,phi,lam,phase)))
    if any(not math.isfinite(x) or abs(x)>32 for x in values):raise ValueError('FINITE_ARTIFICIAL_ANGLE_DOMAIN')
    radius=rational(theta_reference_radius)
    if radius<0:raise ValueError('NEGATIVE_REGION_RADIUS')
    output=_record(rust.generate_circuit(rust.EulerBasis('ZSXX'),*values,True,None))
    with _BALL_LOCK,ctx.workprec(PRECISION):
        V=_output(output)
        return dict(_scope('captured_Rust_generate_circuit_ZSXX'),
            input_hex=[x.hex() for x in values],output=output,
            point_distance=_norm(_angular_target(values)-V),
            region_distance=_norm(_angular_target(values,radius)-V),
            theta_reference_radius_exact=str(radius),
            domain='theta-reference interval against captured V;other angles fixed exact dyadics',
            compiler_inputs_covered='one recorded binary64 angle tuple;not all tuples in the reference interval',
            M_to_Euler_error=None)


def captured_run(gates):
    """Separate ideal gate product->chosen f64 input from f64 input->V.

    NumPy supplies a diagnostic floating product here. It is not claimed
    bitwise identical to the Rust matmul_1q fold or an FMA model.
    """
    import numpy as np
    from flint import ctx
    gates=tuple((name,None if angle is None else float(angle)) for name,angle in gates)
    with _BALL_LOCK,ctx.workprec(PRECISION):
        G=_product(gates);M=np.eye(2,dtype=complex)
        # Only artificial input construction;validation uses the ideal ball G.
        for name,angle in gates:
            B=gate_matrix(name,angle)
            point=np.array([[complex(float(B[i,j].real.mid()),float(B[i,j].imag.mid()))
                             for j in range(2)] for i in range(2)])
            M=point@M
        record=captured_matrix(M)
        if record['status']!='VALIDATED_REGION_ENCLOSURE':return record
        V=_output(record['output']);input_error=_norm(G-_matrix(M));direct=_norm(G-V)
        return dict(record,run_length=len(gates),input_product_error=input_error,
            ideal_product_to_output=direct,
            triangle_upper_exact=str(Fraction(input_error['upper_exact'])+Fraction(record['point_distance']['upper_exact'])),
            input_product_model='observed artificial NumPy product;not Rust fold equivalence',
            bound_to_use='direct ideal_product_to_output OR triangle,never their sum')


def verify_fixed_record(M,output):
    """Recheck a frozen output, also useful for phase-loss counterexamples."""
    from flint import ctx
    runtime_model()
    with _BALL_LOCK,ctx.workprec(PRECISION):return _norm(_matrix(M)-_output(output))


def captured_circuit(input_gates,compiled,*,input_phase=0.):
    """Validate an executed artificial1q whole-P0 circuit,including phase.

    Initial translation is covered for this captured instance only. This
    function does not build or compile, and does not cover unseen words.
    """
    from flint import ctx
    runtime_model()
    if compiled.num_qubits!=1 or compiled.num_clbits or compiled.parameters:
        raise ValueError('BOUND_ONE_QUBIT_CIRCUIT_ONLY')
    gates=[]
    for item in compiled.data:
        name=item.operation.name;params=item.operation.params
        if name not in ('rz','sx','x') or len(params)!=(1 if name=='rz' else 0):
            raise ValueError('FINAL_P0_BASIS')
        gates.append(dict(gate=name,angle_hex=float(params[0]).hex() if params else None))
    output=dict(gates=gates,phase_hex=float(compiled.global_phase).hex(),phase_fit=False)
    with _BALL_LOCK,ctx.workprec(PRECISION):
        direct=_norm(_product(input_gates,input_phase)-_output(output))
    return dict(_scope('captured_artificial1q_whole_P0'),output=output,
        direct_distance=direct,initial_translation='this executed circuit only',
        input_gates=[dict(gate=k,angle_hex=None if v is None else float(v).hex()) for k,v in input_gates],
        input_phase_hex=float(input_phase).hex())


def captured_inverse(output,adjoint):
    """Actual1q inverse output versus the exact adjoint of captured V."""
    from flint import ctx
    runtime_model()
    if adjoint.num_qubits!=1 or adjoint.num_clbits or adjoint.parameters:
        raise ValueError('BOUND_ONE_QUBIT_INVERSE_ONLY')
    gates=[]
    for item in adjoint.data:
        name=item.operation.name;params=item.operation.params
        if name not in ('rz','sx','sxdg','x') or len(params)!=(1 if name=='rz' else 0):
            raise ValueError('UNKNOWN_INVERSE_GATE')
        gates.append(dict(gate=name,angle_hex=float(params[0]).hex() if params else None))
    actual=dict(gates=gates,phase_hex=float(adjoint.global_phase).hex(),phase_fit=False)
    with _BALL_LOCK,ctx.workprec(PRECISION):
        distance=_norm(_output(output).conjugate().transpose()-_output(actual))
    return dict(_scope('captured_actual1q_adjoint'),output=actual,direct_distance=distance)


def accepted_word_budget(initial_nodes,records,*,conditions,initial_translation=None,phase=None):
    """Only accepted captured rewrites accrue error;not an all-law proof.

    Each direct residual already includes its input-product/reference error.
    Do not additionally multiply it by6N or add the triangle components again.
    """
    from .c_pr_aware_df_c2a import monotone_budget
    structural=monotone_budget(initial_nodes,**conditions)
    if structural['accepted_rewrites'] is None:
        return dict(status='UNRESOLVED',reason='structural conditions missing',whole_law_bound=None)
    if any(type(r.get('accepted')) is not bool or type(r.get('input_gates')) is not int
           or r['input_gates']<1 for r in records):raise ValueError('REWRITE_RECORD_SCHEMA')
    accepted=[r for r in records if r['accepted']]
    visits=sum(r['input_gates'] for r in accepted)
    if len(accepted)>initial_nodes or visits>structural['accepted_gate_visits']:
        raise ValueError('MONOTONE_COVERAGE_COUNT')
    total=Fraction()
    for r in accepted:
        if (r['status']!='VALIDATED_REGION_ENCLOSURE' or r.get('phase_fit') is not False
                or r.get('bound_scope')!='direct_ideal_run_to_captured_output'):
            return dict(status='UNRESOLVED',reason='no direct validated accepted-rewrite bound',whole_law_bound=None)
        value=rational(r['upper_exact'])
        if value<0:raise ValueError('NEGATIVE_BOUND')
        total+=value
    missing=[k for k,v in (('initial_translation',initial_translation),('phase',phase)) if v is None]
    if missing:return dict(status='UNRESOLVED',accepted_rewrites=len(accepted),accepted_gate_visits=visits,
        captured_rewrite_sum_exact=str(total),missing=missing,whole_law_bound=None)
    for value in (initial_translation,phase):
        value=rational(value)
        if value<0:raise ValueError('NEGATIVE_BOUND')
        total+=value
    return dict(status='CONDITIONAL_ENCLOSURE',accepted_rewrites=len(accepted),
        accepted_gate_visits=visits,captured_word_upper_exact=str(total),
        proof_inputs_verified_by_schema=False,whole_law_bound=None,
        all_unobserved_words_covered=False,actual_H4_bound=None)
