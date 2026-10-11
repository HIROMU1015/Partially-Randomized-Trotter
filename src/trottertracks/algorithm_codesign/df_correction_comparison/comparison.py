"""Comparison worker mathematics. Registered recipes require a launch capability.

Signal values are diagnostics. Guarantees use scalar enclosures and conditional
IEEE/BLAS assembly guards; CD native/preparation/readout costs remain symbolic.
"""
from fractions import Fraction as F
import hashlib
import math
import numpy as np
import mpmath as mp
from .intervals import configure, iv, upper, bounds, record, midpoint, outward_upper, outward_lower
from .certification import Ball, primitive, comm, pauli_table, pauli_matrix, verified_rotation, frobenius_upper, gamma
from .digital import corrector_events, account, rounded_law, weight_point, sufficient_shots


def random_unitary(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n,n))+1j*rng.standard_normal((n,n)))
    return q*(np.diag(r)/abs(np.diag(r))).conjugate()


def cd_recipe(capability):
    from pathlib import Path
    from .launch import verify_child
    verify_child(Path(__file__).resolve().parents[4],capability)
    rng = np.random.default_rng(6121107)
    frames = [random_unitary(3,rng)[:,:r] for r in (1,2,1)]
    return frames


def e_recipe(capability):
    from pathlib import Path
    from .launch import verify_child
    verify_child(Path(__file__).resolve().parents[4],capability)
    rng = np.random.default_rng(7101107)
    random_unitary(4, rng)  # Required original draw, never skipped or reseeded.
    return {r:[random_unitary(r,rng)[:,:2] for _ in range(3)] for r in (3,6)}


def certified_primitives(frames, signs):
    qs, certificates = [], []
    for frame, sign in zip(frames, signs):
        q, certificate = primitive(frame, sign)
        qs.append(q); certificates.append(certificate)
    return qs, certificates


def tuples(qs, hs):
    result = {}
    for j in range(len(qs)-1):
        for a,b,c,den in [(j,j,k,24) for k in range(j+1,len(qs))]+[
            (a,j,c,12) for a in range(j+1,len(qs)) for c in range(j+1,len(qs))]:
            label = f'{j}:{den}:{a}:{b}:{c}'
            result[label] = {'coefficient':F(hs[a])*F(hs[b])*F(hs[c])/den,
                             'generator':comm(qs[a], comm(qs[b],qs[c])), 'indices':[a,b,c]}
    return result


def dyadic_midpoint(x, bits=80):
    q = midpoint(x); scaled = q*(1 << bits)
    value = F(scaled.numerator//scaled.denominator, 1 << bits)
    a,b = bounds(x)
    return value, max(abs(value-a),abs(value-b))


def rte_parameters(delta, bits=48):
    v = iv(F(delta)/4)
    weights = {str(k): v**k/math.factorial(k)*mp.iv.sqrt(1+(v/(k+1))**2) for k in (0,2,4)}
    norm = sum(weights.values(), iv(0))
    probabilities = {k:w/norm for k,w in weights.items()}
    law, distance = rounded_law(probabilities,bits)
    normalization, error = dyadic_midpoint(norm)
    return law, distance, normalization, error, norm


def slot_mean(rows, terms, u, arm, d):
    result = np.zeros((d,d),complex)
    radius, weight_radius = F(0), F(0)
    for row in rows:
        row['weight'] = F(1)
        if row['pauli'] is None or row['p'] == 0:
            mat, error = np.eye(d,dtype=complex), F(0)
        elif arm == 'A':
            x = verified_rotation(terms[row['tau']]['generator'],u)
            mat, error = x.value, x.error
        else:
            angle = mp.iv.atan2(iv(u)*iv(row['ell']),iv(1))
            sign = 1 if row['beta'] > 0 else -1
            # The executed angle is the exact binary64 radians used by native IR.
            theta = float(midpoint(angle))
            a,b = bounds(angle)
            angle_error = max(abs(F(theta)-a),abs(F(theta)-b))
            p = pauli_matrix(row['pauli'])
            mat = math.cos(theta)*np.eye(d)+1j*sign*math.sin(theta)*p
            row['weight'], wr = weight_point(u,row['ell'])
            error = angle_error+gamma(16)
            weight_radius += row['p']*wr
        result += float(row['p']*row['weight'])*mat
        radius += row['p']*abs(row['weight'])*error
    radius += gamma(4*len(rows))*d*max(abs(r['weight']) for r in rows)
    return Ball(result,radius), weight_radius


def cd_task(qs, terms, delta, n, arm, contract):
    configure()
    eta = min(F(1),F(contract['CD']['gamma'])*delta)
    mass = sum(t['coefficient'] for t in terms.values())
    u = mass*delta**3/(2*eta)
    q = {k:t['coefficient']/mass for k,t in terms.items()}
    tables, proxy_radius, pauli_records = {}, {}, {}
    if arm == 'B':
        for k,t in terms.items():
            table,_,radius,records = pauli_table(t['generator'], number_conserving=True, traceless=True)
            tables[k], proxy_radius[k],pauli_records[k] = table,radius,records
    zero_tuples={k for k,t in terms.items() if t['indices'][1]==t['indices'][2]}
    rows, law_l1 = corrector_events(eta,q,tables if arm=='B' else None,proved_zero_tuples=zero_tuples)
    corrector, wr = slot_mean(rows,terms,u,arm,len(qs[0].value))
    rte_law, rte_l1, rte_norm, rte_norm_error, ideal_norm = rte_parameters(delta)
    rte_angles={}
    for label in rte_law:
        k=int(label)
        interval=mp.iv.atan2(iv(delta/F(4*(k+1))),iv(1))
        lo,hi=bounds(interval);raw=F(float((lo+hi)/2))
        rte_angles[label]={'raw_binary64_radians':str(raw),'semantic_interval':record(interval),
                          'angle_error_upper':str(max(abs(raw-lo),abs(raw-hi))),
                          'paired_sign':(-1)**(k//2)}
    gamma_exec = rte_norm**n
    ideal_gamma = ideal_norm**n
    # Executed-law detailed moment, range, and SYMBOLIC expected prices.
    acc = account(rows)
    detailed_nu = gamma_exec**2 * acc['second_moment']**(2*n)
    detailed_range = gamma_exec * acc['range']**(2*n)
    if arm == 'B':
        # Keep ideal ell cap=32; computational ell has certified assembly radius.
        ell_cap_exec = F(32)+8*max(proxy_radius.values(),default=F(0))
        w_cap = upper(mp.iv.sqrt(1+(iv(u)*iv(ell_cap_exec))**2))+F(1,2**80)
        nu = max(detailed_nu,gamma_exec**2*(1-eta+eta*w_cap**2)**(2*n))
        width = max(detailed_range,gamma_exec*w_cap**(2*n))
    else:
        nu,width = detailed_nu,detailed_range
    nu,width=outward_upper(nu),outward_upper(width)
    digital = width*(2*n*law_l1+n*rte_l1)
    # Assembly proxy mismatch is separate from law rounding and weight rounding.
    proxy = (2*n*eta*abs(u)*sum((F(q[k])*proxy_radius.get(k,F(0)) for k in q),F(0)))*width
    normalization_radius = max(abs(gamma_exec-a) for a in bounds(ideal_gamma))
    numeric = (2*n*corrector.error+2*n*wr+normalization_radius+proxy)*max(F(1),width)
    # Add primitive perturbation to all deterministic and central factors.
    numeric += n*delta*sum((F(h)*q0.error for h,q0 in zip(contract['CD']['coefficients'],qs)),F(0))*max(F(1),width)
    numeric += n*max(F(a['angle_error_upper']) for a in rte_angles.values())*max(F(1),width)
    from .intervals import scalar_error_bounds
    bias = outward_upper(upper(scalar_error_bounds(delta,mass,eta,n)[arm]))
    maxima = {k:F(v) for k,v in contract['error']['axis_maxima'].items()}
    if bias > maxima['algorithmic_mean'] or digital > maxima['finite_bit'] or numeric > maxima['assembly_coefficient_arithmetic_angle']:
        raise ArithmeticError('CD bias/digital/numeric cap not certified; no parameter repair')
    stat = F(contract['confidence']['axis_epsilon'])-bias-maxima['finite_bit']-maxima['native_synthesis']-maxima['assembly_coefficient_arithmetic_angle']
    shots = sufficient_shots(nu,width,stat,F(contract['confidence']['axis_task_alpha']))
    # Signal is diagnostic only. Do not use it for confidence or native claims.
    from scipy.linalg import expm
    eye = np.eye(len(qs[0].value),dtype=complex)
    forward = expm(-1j*float(delta*F(9,40))*qs[0].value)@expm(-1j*float(delta*F(7,40))*qs[1].value)
    reverse = expm(-1j*float(delta*F(7,40))*qs[1].value)@expm(-1j*float(delta*F(9,40))*qs[0].value)
    central = np.zeros_like(eye)
    for label,p in rte_law.items():
        k=int(label)
        angle=float(F(rte_angles[label]['raw_binary64_radians']))
        central += float(rte_norm*p)*(-1)**(k//2)*expm(-1j*angle*qs[2].value)
    mean=corrector.value@forward@central@reverse@corrector.value
    state=np.zeros(len(eye),complex)
    state[[1,2,4,3,5,6]]=np.array([1,1j,1,1,-1j,1])/math.sqrt(6)
    ideal=expm(-1j*sum(float(F(h))*q0.value for h,q0 in zip(contract['CD']['coefficients'],qs)))
    signal=np.vdot(state,np.linalg.matrix_power(mean,n)@state)
    target=np.vdot(state,ideal@state)
    for row in rows:
        row['cost_variable'] = None if row['pauli'] is None else f'C_{arm}_{row["label"]}_x'
    return {'arm':arm,'delta':str(delta),'n':n,'eta':str(eta),'u':str(u),
            'algorithmic_certified_upper':str(bias),'finite_bit_bias_upper':str(digital),
            'numeric_guard_upper':str(numeric),'native_accuracy':'CONDITIONAL_ON_GLOBAL_NATIVE_LEDGER_AND_IDEAL_PREPARATION',
            'Gamma_ideal_interval':record(ideal_gamma),'Gamma_exec':str(gamma_exec),
            'RTE_executed_order_law':{k:str(v) for k,v in rte_law.items()},'RTE_angles':rte_angles,
            'executed_detailed_second_moment_interval':{'lo':str(outward_lower(detailed_nu)),'hi':str(outward_upper(detailed_nu))},
            'executed_detailed_range_interval':{'lo':str(outward_lower(detailed_range)),'hi':str(outward_upper(detailed_range))},
            'exact_moment_factorization':{'Gamma_exec':str(gamma_exec),'slot_second_moment':str(acc['second_moment']),'slot_count':2*n},
            'exact_range_factorization':{'Gamma_exec':str(gamma_exec),'slot_range':str(acc['range']),'slot_count':2*n},
            'primary_second_moment_upper':str(nu),'primary_range':str(width),
            'statistical_halfwidth':str(stat),'shots_per_axis':shots,
            'signal_diagnostic':[signal.real,signal.imag], 'target_diagnostic':[target.real,target.imag],
            'signal_certificate':'DIAGNOSTIC_FLOAT_ONLY_NO_SHOT_REDUCTION',
            'target_reference':'binary64 approximation to the SAME ideal ordered-Gram-Schmidt input; target point is not exact/certified',
            'executed_slot_events':rows,'symbolic_total_cost':'2N*(h_x+C_base_x+2n*sum(p_exec*C_event_x))',
            'PR_measured_total_cost':None,'pauli_coefficients':pauli_records}
