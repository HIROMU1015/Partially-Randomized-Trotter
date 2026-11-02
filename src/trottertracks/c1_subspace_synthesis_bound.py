"""Streaming full-column restricted bounds, numerical level B, never certificates.

No simulator, CAR, PF, compiler or RPE implementation is duplicated here.
Rows are input basis columns; every output amplitude must be retained.
"""
import math

VERSION='complete_basis_full_output_frobenius_v1'
LEVEL='B_NUMERICAL_OPERATOR_UPPER_BOUND'


def _positive_integer(value):
    if type(value) is not int or value<1:raise ValueError('POSITIVE_INTEGER_REQUIRED')
    return value


def row_norms(array):
    """Scaled existing NumPy reductions, avoiding squared tiny-value underflow."""
    import numpy as np
    a=np.asarray(array)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('FINITE_MATRIX_REQUIRED')
    scale=np.max(abs(a),axis=1)
    normalized=a/np.where(scale==0,1.,scale)[:,None]
    result=scale*np.sqrt(np.sum(abs(normalized)**2,axis=1))
    if not np.isfinite(result).all():raise ValueError('NORM_OVERFLOW')
    return result


class ColumnStream:
    """One complete orthonormal computational basis, not random probes.

    This is an upper bound for the represented column-difference matrix;
    state-action/reference/reduction rounding is NOT enclosed.
    """
    def __init__(self,input_dimension,output_dimension):
        self.inputs=_positive_integer(input_dimension)
        self.outputs=_positive_integer(output_dimension)
        self.seen=set();self.frobenius=0.;self.maximum=0.;self.columns=[]

    def add(self,column_ids,actual,expected):
        import numpy as np
        ids=list(column_ids);a=np.asarray(actual);b=np.asarray(expected)
        if (not ids or any(type(i) is not int or not 0<=i<self.inputs for i in ids)
                or len(set(ids))!=len(ids) or self.seen.intersection(ids)):
            raise ValueError('COLUMN_IDENTITY')
        if (a.shape!=b.shape or a.shape!=(len(ids),self.outputs)
                or a.dtype!=np.complex128 or b.dtype!=np.complex128
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError('FULL_OUTPUT_COLUMN_SCHEMA')
        error=a-b
        if not np.isfinite(error).all():raise ValueError('DIFFERENCE_OVERFLOW')
        norms=row_norms(error)
        self.frobenius=math.hypot(self.frobenius,*norms.tolist())
        self.maximum=max(self.maximum,float(np.max(abs(error))))
        if not math.isfinite(self.frobenius):raise ValueError('BOUND_OVERFLOW')
        self.seen.update(ids)
        self.columns.extend(dict(column=i,L2=float(v)) for i,v in zip(ids,norms,strict=True))

    def finish(self):
        complete=len(self.seen)==self.inputs
        return dict(status='COMPLETE' if complete else 'UNRESOLVED_INCOMPLETE_BASIS',
            evidence_level=LEVEL if complete else 'A_PARTIAL_COLUMNS_ONLY',
            input_dimension=self.inputs,output_dimension=self.outputs,
            covered_columns=len(self.seen),column_order=sorted(self.seen),
            represented_difference_frobenius=self.frobenius,
            restricted_operator_bound_pilot=(math.nextafter(self.frobenius,math.inf) if self.frobenius else 0.) if complete else None,
            max_entry=self.maximum,columns=sorted(self.columns,key=lambda x:x['column']),
            global_full_Fock_operator_bound=None,rigorous_operator_bound=None,
            floating_point_rounding_enclosed=False,
            missing_enclosures=['Aer state actions','saved CAR/PF reference','subtraction and norm reductions'])


def reference_info(kernel,component_allowance=1e-10):
    import numpy as np
    w=np.asarray(kernel)
    if (w.ndim!=2 or w.shape[0]!=w.shape[1] or len(w)==0 or w.dtype!=np.complex128
            or not np.isfinite(w).all()):raise ValueError('SQUARE_FINITE_COMPLEX128_REFERENCE')
    defect=w.conj().T@w-np.eye(len(w))
    maximum=float(np.max(abs(defect)))
    if not math.isfinite(maximum) or maximum>component_allowance:
        raise ValueError('REFERENCE_NONUNITARY')
    delta=math.hypot(*row_norms(defect).tolist())
    alpha=1. if delta==0 else math.nextafter(math.sqrt(math.nextafter(1.+delta,math.inf)),math.inf)
    return dict(evidence_level=LEVEL,reference_invariance='embedded square W maps S into S by construction',
        unitarity_defect_max_entry=maximum,unitarity_defect_frobenius=delta,
        reference_norm_upper_pilot=max(1.,alpha),
        reference_exactly_unitary_certified=False,rigorous_enclosure=None)


def basis_batch(indices,system_dimension,kernel,columns):
    """Same ordered sector embedding as existing H6 action.probes; no projection."""
    import numpy as np
    idx=np.asarray(indices);w=np.asarray(kernel);ids=list(columns);d=len(idx)
    if (idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or len(set(idx.tolist()))!=d
            or d<1 or np.min(idx)<0 or np.max(idx)>=system_dimension or w.shape!=(d,d)
            or not ids or any(type(i) is not int or not 0<=i<2*d for i in ids)):
        raise ValueError('BASIS_EMBEDDING_IDENTITY')
    initial=np.zeros((len(ids),2*system_dimension),complex)
    expected=np.zeros_like(initial)
    for row,column in enumerate(ids):
        control,j=divmod(column,d);initial[row,control*system_dimension+idx[j]]=1.
        if control:expected[row,system_dimension+idx]=w[:,j]
        else:expected[row]=initial[row]
    return initial,expected


def repeat_bound(epsilon,occurrences,*,reference_invariant,reference_norm_upper=1.):
    _positive_integer(occurrences)
    if (not math.isfinite(epsilon) or epsilon<0 or not math.isfinite(reference_norm_upper)
            or reference_norm_upper<1.):raise ValueError('FINITE_NONNEGATIVE_BOUND')
    if reference_invariant is not True:
        return dict(status='UNRESOLVED_REFERENCE_NOT_INVARIANT',operator_bound_pilot=None,
            rigorous_operator_bound=None)
    alpha=reference_norm_upper
    growth=alpha**occurrences
    factor=float(occurrences) if alpha==1. else math.expm1(occurrences*math.log(alpha))/(alpha-1.)
    bound=min(1.+growth,epsilon*factor)
    if not all(math.isfinite(x) for x in [growth,factor,bound]):raise ValueError('PROPAGATION_OVERFLOW')
    return dict(status='NUMERICAL_BOUND',operator_bound_pilot=bound,reference_power_norm_upper_pilot=growth,
        telescoping_factor=factor,occurrences=occurrences,rigorous_operator_bound=None,
        assumptions=['V structurally unitary on full output space','reference maps S into S',
                     'complete basis bound applies to (V-U)P_S'],evidence_level=LEVEL)


def round_bound(kernel,shell_distances,*,s,m,saved_pf_error):
    """Synthesis-only complex-signal bound + same-target PF phase-room bridge.

    Numerical W need not be set to exactly unitary: retain the norm-growth
    guard. No shot or CA function is called. No statistical error is added
    to an operator norm; only existing deterministic phase-room is reported.
    """
    if s not in (1,4) or type(m) is not int or not 0<=m<=12:raise ValueError('REGISTERED_ROUND')
    if kernel['evidence_level']!=LEVEL or kernel['restricted_operator_bound_pilot'] is None:
        return dict(m=m,status='UNRESOLVED_KERNEL_BOUND',complex_signal_bound_pilot=None,CA_achieved=None)
    if (set(shell_distances)!=set(['head','X','Y']) or saved_pf_error<0
            or not math.isfinite(saved_pf_error)
            or any(v is None or v<0 or not math.isfinite(v) for v in shell_distances.values())):
        raise ValueError('UNKNOWN_OR_INVALID_SHELL_PF_BOUND')
    repeated=repeat_bound(kernel['restricted_operator_bound_pilot'],s*(1<<m),reference_invariant=True,
        reference_norm_upper=kernel['reference_norm_upper_pilot'])
    a=repeated['reference_power_norm_upper_pilot']
    axis={name:math.fsum([shell_distances['head'],repeated['operator_bound_pilot'],a*shell_distances[name]]) for name in ('X','Y')}
    signal=(1.+a)*math.hypot(axis['X'],axis['Y'])
    # The saved E_PF is ||W^n-exp(-it H_DF)||_2. Its matrix element on a
    # normalized own-DF ground is a complex-signal error <= E_PF (factor1).
    combined=math.fsum([saved_pf_error,signal])
    phase=math.asin(combined) if combined<1. else None
    room=math.pi/12
    return dict(m=m,s=s,occurrences=s*(1<<m),evidence_level=LEVEL,**{k:v for k,v in repeated.items() if k not in ('occurrences','evidence_level','status')},
        synthesis_axis_state_bound_pilot=axis,complex_signal_bound_pilot=signal,
        saved_PF_operator_error=saved_pf_error,combined_complex_error_pilot=combined,
        phase_bias_bound_pilot=phase,registered_phase_room=room,
        status='POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL' if phase is not None and phase<=room else 'INFEASIBLE_REGISTERED_BOUND',
        rigorous_signal_error_bound=None,CA_achieved=None,shots=None,total_cost=None)
