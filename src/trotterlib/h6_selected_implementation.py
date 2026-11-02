"""Selected H6 trajectories only; no law mean, sampling, synthesis or compiler."""
from fractions import Fraction as F
import math
from types import SimpleNamespace as NS

import numpy as np
from qiskit.quantum_info import Statevector

from .h4_conditional_cng import extract


def require(ok, reason):
    if not ok:raise ValueError(reason)


# Existing pure numerical routines; no old H4 controller or flint import.
_pauli=dict(np=np)
mapping=extract('src/trotterlib/pauli_vector_enclosure.py','mapping',_pauli)
permute=extract('src/trotterlib/pauli_vector_enclosure.py','permute',_pauli)
propagate_saved=extract('src/trotterlib/h4_r0_empirical.py','propagate',
    dict(api=lambda:NS(np=np,Statevector=Statevector)))
compare_points=extract('src/trotterlib/h4_r0_empirical.py','compare_signals',
    dict(F=F,C=NS(require=require)))

INPUT_GATES={'h','s','sdg','cx','cy','cz','crz','p'}
COMPILED_GATES={'rz','sx','x','cx'}
EPS=2.**-52


def gamma(k):
    require(0<=k*EPS<1,'GAMMA_DOMAIN')
    return k*EPS/(1-k*EPS)


def norm_candidate(nu):return math.sqrt(float(nu))*(1+gamma(8))


def rotation_coefficients(angle):
    exact=F(angle);a=float(exact)
    require(math.isfinite(a),'FINITE_ROTATION')
    c,s=math.cos(a),math.sin(a)
    # Engineering libm/parameter allowance, not a rigorous transcendental enclosure.
    conversion=float(abs(F(a)-exact))
    coefficient=2*(conversion+gamma(16)*(1+abs(a)))
    arithmetic=gamma(8)*(abs(c)+abs(s))
    return c,s,coefficient+arithmetic


def selected_action(spec,raw,nu):
    require(len(raw)==1<<spec['n'] and len(spec['events'])==spec['q']*spec['r'],'SELECTED_DIMENSION')
    require(nu>0 and np.isfinite(raw).all(),'FINITE_RAW_STATE')
    cache={};coefficients={};out=raw.copy();error=0.;norm=norm_candidate(nu);counts=dict(rotations=0,products=0,scalar_phases=0)
    def pm(label,sign=1):
        key=(label,sign)
        if key not in cache:cache[key]=mapping(label,sign)
        return cache[key]
    def rotate(label,sign,angle):
        nonlocal out,error
        angle=F(angle)
        if angle not in coefficients:coefficients[angle]=rotation_coefficients(angle)
        c,s,fresh=coefficients[angle]
        out=c*out-1j*s*permute(out,pm(label,sign))
        error=(1+fresh)*error+fresh*norm
        counts['rotations']+=1
    retained=[(row['label'],F(row['coefficient'])*F(spec['delta'])/2) for row in spec['retained']]
    scalar=float(spec['scalar_step']);scalar_value=complex(math.cos(scalar),math.sin(scalar))
    scalar_fresh=gamma(16)*(1+abs(scalar))+gamma(8)
    for outer in range(spec['q']):
        for label,angle in retained:rotate(label,1,angle)
        for event in spec['events'][outer*spec['r']:(outer+1)*spec['r']]:
            for application in event.application_sequence:
                label=spec['labels'][application.component_id];sign=application.coefficient_sign
                if application.role=='product':
                    out=permute(out,pm(label,sign));counts['products']+=1
                else:
                    require(application.role=='rotation','SELECTED_EVENT_ROLE')
                    rotate(label,sign,F(event.rotation_angle))
            require(event.phase in (1.,-1.),'SIGNED_ORDER_PHASE')
            if event.phase==-1.:out=-out
        for label,angle in reversed(retained):rotate(label,1,angle)
        out=scalar_value*out;error=(1+scalar_fresh)*error+scalar_fresh*norm
        counts['scalar_phases']+=1
    z=complex(np.vdot(raw,out)/float(nu))
    nuf=float(nu);conversion=float(abs(F(nuf)-nu))/nuf
    u=(norm*error+gamma(8*len(raw))*norm*(norm+error))/nuf+abs(z)*(conversion+gamma(4))
    require(np.isfinite([z.real,z.imag,error,u]).all(),'FINITE_SELECTED_SIGNAL')
    return dict(real=z.real,imag=z.imag,engineering_signal_uncertainty=u,
        engineering_vector_error=error,operations=counts,state_actions=1,
        raw_norm_exact=str(nu),norm_divisions=1,B_multiplications=0,
        retained_angle_policy='exact encoded coefficient times exact saved delta/2; trig center binary64',
        event_angle_policy='exact saved binary64; signed products then rotation and order phase',
        scalar_policy='saved scalar_step_binary64 once per q',
        state_renormalized=False,law_mean_used=False,precision='complex128',
        numerical_error_upper=None,status='EMPIRICAL_WITH_ENGINEERING_RESOLUTION')


def validate_circuit(circuit,*,compiled,n):
    require(circuit.num_qubits==n+1 and circuit.num_clbits==1 and not circuit.parameters,'SAVED_PROGRAM_SHAPE')
    require(circuit.layout is None,'NO_NEW_LAYOUT')
    last=circuit.data[-1]
    require(last.operation.name=='measure' and [circuit.find_bit(q).index for q in last.qubits]==[n]
        and [circuit.find_bit(c).index for c in last.clbits]==[0],'TERMINAL_ANCILLA_MEASURE')
    for item in circuit.data[:-1]:
        op=item.operation
        require(not item.clbits and getattr(op,'condition',None) is None,'NO_CLASSICAL_CONTROL')
        require(op.name in (COMPILED_GATES if compiled else INPUT_GATES),'FIXED_GATE_FAMILY')
        require(1<=op.num_qubits<=2 and all(math.isfinite(float(x)) for x in op.params),'LOCAL_GATE_PARAMETERS')
        if op.name in ('cx','cy','cz','crz'):require(op.ctrl_state==1,'CLOSED_CONTROL')
    require(math.isfinite(float(circuit.global_phase)),'FINITE_PROGRAM_PHASE')


def program_resolution(circuit,nu,raw_signal):
    """Gate-count gamma envelope; engineering only, no library-internal proof."""
    norm=norm_candidate(nu);error=0.
    for item in circuit.data[:-1]:
        op=item.operation;width=1<<op.num_qubits
        angle=sum(abs(float(x)) for x in op.params)
        gate=gamma(32*width)*(1+angle)
        rounding=gamma(8*width)*math.sqrt(width)
        fresh=gate+rounding
        error=(1+fresh)*error+fresh*norm
    # Full-circuit phase is observable-invariant; retain its numerical application allowance.
    phase=gamma(16)*(1+abs(float(circuit.global_phase)))
    error=(1+phase)*error+phase*norm
    nuf=float(nu);d=1<<circuit.num_qubits
    u=(2*norm*error+error*error+gamma(8*d)*(norm+error)**2)/nuf
    u+=abs(raw_signal)*(float(abs(F(nuf)-nu))/nuf+gamma(8))
    require(math.isfinite(u),'FINITE_PROGRAM_RESOLUTION')
    return error,u


def saved_action(circuit,raw,nu,*,compiled):
    validate_circuit(circuit,compiled=compiled,n=int(math.log2(len(raw))))
    # Reuse the original statevector walk and raw p0-p1 observable, no probabilities() clipping.
    _,record=propagate_saved(circuit,raw,nu)
    error,u=program_resolution(circuit,nu,record['signal'])
    require(np.isfinite([record['signal'],record['output_norm_observed'],u]).all(),'FINITE_SAVED_SIGNAL')
    return dict(record,engineering_signal_uncertainty=u,engineering_vector_error=error,
        state_actions=1,precision='complex128',raw_norm_exact=str(nu),norm_divisions=1,B_multiplications=0,
        role='compiled' if compiled else 'native_input',program_instructions=len(circuit.data),
        resolution_policy='gamma_(32*local_width)*(1+angle)+gamma_(8*local_width)*sqrt(local_width), unitary correlated-error propagation',
        numerical_error_upper=None,status='EMPIRICAL_WITH_ENGINEERING_RESOLUTION')


def reduce_axis(selected,native,compiled,axis,B_exact,budget_exact):
    field='real' if axis=='cosine' else 'imag'
    require(axis in ('cosine','sine'),'MEASUREMENT_AXIS')
    row=compare_points(selected[field],native['signal'],compiled['signal'],F(B_exact),F(budget_exact))
    b=F(B_exact);g=F(budget_exact)
    uncertainty={key:value.get('engineering_signal_uncertainty') for key,value in
        [('selected',selected),('native',native),('compiled',compiled)]}
    if any(x is None for x in uncertainty.values()):
        return dict(row,engineering_comparison='UNRESOLVED_MISSING_RESOLUTION',engineering_intervals=None)
    require(min(uncertainty.values())>=0,'NONNEGATIVE_RESOLUTION')
    endpoints={}
    for name,roles in [('eta',('selected','native')),('delta',('native','compiled')),('d',('selected','compiled'))]:
        radius=b*sum((F(uncertainty[k]) for k in roles),F(0));point=F(row[name+'_exact'])
        lo,hi=max(F(0),point-radius),point+radius
        endpoints[name]=dict(lower_exact=str(lo),upper_candidate_exact=str(hi),upper_candidate=float(hi),
            resolution_exact=str(radius),direct_difference_resolved=point>radius)
    upper=F(endpoints['d']['upper_candidate_exact']);lower=F(endpoints['d']['lower_exact'])
    status=('ENGINEERING_UPPER_WITHIN_SAVED_LOCAL_BUDGET' if upper<=g else
            'ENGINEERING_LOWER_EXCEEDS_SAVED_LOCAL_BUDGET' if lower>g else 'UNRESOLVED_ENGINEERING')
    return dict(row,engineering_comparison=status,engineering_intervals=endpoints,
        actual_compiled_rho=None,whole_law_rho=None,local_trajectory_only=True,
        certified_upper=None,old_N_G_changed=False)
