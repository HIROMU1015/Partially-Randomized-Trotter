"""Fixed-law moments and engineering signal errors; no sampling or circuits."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F
import math
from types import SimpleNamespace as NS

import numpy as np

from . import matched_partition_stage2 as K
from .matched_partition_stage4 import bias_inputs, outward
from .h4_conditional_cng import extract
from .h4_enclosure_prototype import Interval, sqrt_point, exponential_prototype

ORDERS = (0, 2, 4)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


# Reuse the existing pure functions without importing its unavailable flint backend.
_cdf_namespace = dict(require=require, rational=F, Q=F, math=math)
cdf_binary64 = extract('src/trotterlib/h4_ball_evidence.py', 'cdf_binary64', _cdf_namespace)
grid_masses = extract('src/trotterlib/h4_ball_evidence.py', 'grid_masses', _cdf_namespace)


def choice_law(vector, *, bits=53):
    vector = list(map(float, vector))
    cdf = cdf_binary64(vector)
    expected = np.asarray(vector).cumsum(); expected /= expected[-1]
    require(cdf == expected.tolist(), 'NUMPY_CDF_MAPPING')
    grid = grid_masses(cdf, bits)
    total = sum(map(F, vector), F(0))
    base = [F(x)/total for x in vector]
    counts = [int(p*(1 << bits)) for p in grid]
    require(sum(counts) == 1 << bits and min(counts) >= 0, 'INTEGER_CDF_COUNTS')
    info = dict(choice_input=vector, choice_input_hex=[x.hex() for x in vector],
        final_cdf=cdf, final_cdf_exact=list(map(lambda x:str(F(x)), cdf)),
        counts=counts, uniform_bits=bits, base_probabilities_exact=list(map(str, base)),
        grid_probabilities_exact=list(map(str, grid)), total_variation_exact=str(sum(abs(x-y) for x,y in zip(base,grid))/2),
        zero_grid_masses=counts.count(0), IID_uniform_words_assumed=True,
        PCG64_independence_proved=False, numpy_cdf_verified=True)
    return base, grid, info


def interval_error(value, interval):
    return max(abs(F(value)-interval.lo), abs(F(value)-interval.hi))


def angle_table(a):
    ideal, implemented, rows = [], [], []
    for k in ORDERS:
        x = F(a)/(k+1)
        root = sqrt_point(1+x*x, bits=96)
        ci = Interval(1)/root; si = ci*x
        theta = math.atan(a/(k+1))
        require(math.isfinite(theta), 'NONFINITE_ANGLE')
        rotation = exponential_prototype([[0, -F(theta)], [F(theta), 0]], degree=24)
        cg, sg = rotation[0][0], rotation[1][0]
        c0, s0 = float((ci.lo+ci.hi)/2), float((si.lo+si.hi)/2)
        c1, s1 = math.cos(theta), math.sin(theta)
        ideal.append((c0, s0, outward(interval_error(c0,ci),upper=True), outward(interval_error(s0,si),upper=True)))
        implemented.append((c1, s1, outward(interval_error(c1,cg),upper=True), outward(interval_error(s1,sg),upper=True)))
        dc, ds = cg-ci, sg-si
        rows.append(dict(order=k, a_exact=str(F(a)), ideal_x_exact=str(x),
            division_operand=a/(k+1), division_operand_exact=str(F(a/(k+1))),
            theta=theta, theta_hex=theta.hex(), theta_exact=str(F(theta)),
            double_angle_exact=str(F(2*theta)), double_angle_is_exact=F(2*theta)==2*F(theta),
            ideal_cos_interval_exact=[str(ci.lo),str(ci.hi)], ideal_sin_interval_exact=[str(si.lo),str(si.hi)],
            fixed_angle_cos_interval_exact=[str(cg.lo),str(cg.hi)], fixed_angle_sin_interval_exact=[str(sg.lo),str(sg.hi)],
            cos_difference_upper_exact=str(max(abs(dc.lo),abs(dc.hi))),
            sin_difference_upper_exact=str(max(abs(ds.lo),abs(ds.hi))),
            matrix_cos_ideal=c0, matrix_sin_ideal=s0, matrix_cos_grid=c1, matrix_sin_grid=s1,
            scalar_enclosure='rational sqrt96 and 2x2 exponential degree24; matrix signals remain complex128'))
        require(rows[-1]['double_angle_is_exact'], 'DOUBLE_ANGLE_BINDING')
    return ideal, implemented, rows


def bundle(matrix, num=0., bind=0.):
    require(np.isfinite(matrix).all() and min(num,bind) >= 0, 'INVALID_MATRIX_BUNDLE')
    return NS(matrix=matrix, num=float(num), bind=float(bind), norm=K.norm(matrix))


def multiply(a,b):
    rounding=K.product_roundoff(a.matrix,b.matrix)
    num=a.norm*b.num+b.norm*a.num+a.num*b.num+rounding
    bind=a.norm*b.bind+b.norm*a.bind+a.num*b.bind+a.bind*b.num+a.bind*b.bind
    return bundle(a.matrix@b.matrix,num,bind)


def moments(w):
    w2=multiply(w,w); w4=multiply(w2,w2)
    return (bundle(np.eye(len(w.matrix),dtype=complex)),w2,w4)


def event_mean(w,powers,order_mass,coefficients):
    ident=np.eye(len(w.matrix)); out=np.zeros_like(w.matrix,dtype=complex); en=eb=0.
    for k,power,pi,(c,s,ec,es) in zip(ORDERS,powers,order_mass,coefficients,strict=True):
        rm=c*ident-1j*s*w.matrix
        rn=abs(s)*w.num+ec*math.sqrt(len(ident))+es*(K.fro(w.matrix)+w.num)
        rn+=K.gamma(12)*(abs(c)*math.sqrt(len(ident))+abs(s)*K.fro(w.matrix))
        rot=bundle(rm,rn,(abs(s)+es)*w.bind)
        term=multiply(rot,power); p=float(pi); old=K.fro(out)
        out+=((-1)**(k//2))*p*term.matrix
        pe=float(abs(F(p)-pi))
        en+=abs(p)*term.num+pe*(term.norm+term.num)+K.gamma(4)*(old+abs(p)*K.fro(term.matrix))
        eb+=(abs(p)+pe)*term.bind
    return bundle(out,en,eb)


def component_moments(arm,descriptor,base,grid,n):
    """Stream components; saved DF U, no eigensolver or Gaussian gate lowering."""
    native=K.native_context(n) if arm=='DF' else None
    ident=np.eye(1<<n); zs={}; znorms={}; zerrors={}
    out=[np.zeros_like(ident,dtype=complex) for _ in range(2)]
    num=[0.,0.]; bind=[0.,0.]; weights=[0.,0.]; bounds=[]; maximum_error=0.
    for row,p0,p1 in zip(descriptor['components'],base,grid,strict=True):
        sign=1 if row['coefficient']>0 else -1
        if arm=='P':
            a=K.PauliHamiltonian.from_terms([(row['label'],1.)],num_qubits=n).sparse_operator().toarray()
            ea=0.; exact_norm=F(1)
        else:
            index=row['basis']
            if index not in zs:
                b=descriptor['basis'][index]; u=np.empty((n,n),dtype=complex)
                u.real=b['U_real'];u.imag=b['U_imag']
                zs[index]=[ident-2*native(np.outer(u[:,i],u[:,i].conj())).toarray() for i in range(n)]
                # Exact norms for Z=I-2 a†(u)a(u), also when encoded u is not normalized.
                un=[sum((F(float(z.real))**2+F(float(z.imag))**2 for z in u[:,i]),F(0)) for i in range(n)]
                znorms[index]=[max(F(1),abs(1-2*x)) for x in un]
                zerrors[index]=[K.gamma(32*n*n)*max(1.,K.fro(z)) for z in zs[index]]
            support=row['support']; i=support[0]
            a=zs[index][i]; ea=zerrors[index][i]; exact_norm=znorms[index][i]
            if len(support)==2:
                j=support[1]; b=zs[index][j]; eb=zerrors[index][j]
                ea=float(znorms[index][i])*eb+float(znorms[index][j])*ea+ea*eb+K.product_roundoff(a,b)
                a=a@b;exact_norm*=znorms[index][j]
        bounds.append(exact_norm);maximum_error=max(maximum_error,ea)
        a=sign*a; af=K.fro(a)
        for i,p in enumerate((p0,p1)):
            pf=float(p);out[i]+=pf*a
            num[i]+=float(abs(F(pf)-p))*float(exact_norm)
            bind[i]+=abs(pf)*ea
            weights[i]+=abs(pf)*af
    result=[]
    for i in range(2):
        num[i]+=K.gamma(4*len(bounds))*weights[i]
        result.append(bundle(out[i],num[i],bind[i]))
    return result,bounds,dict(components=len(bounds),maximum_component_assembly_error=maximum_error,
        exact_component_norm_bound_max=str(max(bounds)),DF_basis_count=len(zs),
        physical_operator_binding_difference_added=False)


def local_gap_bounds(base,grid,pi0,pi1,norms,angle_rows):
    """Rational coefficient perturbations; no subtraction of completed signals."""
    n0=sum((p*s for p,s in zip(base,norms)),F(0)); n1=sum((p*s for p,s in zip(grid,norms)),F(0))
    dw=sum((abs(p-q)*s for p,q,s in zip(base,grid,norms)),F(0))
    dcdf=F(0);dangle=F(0)
    for k,p0,p1,row in zip(ORDERS,pi0,pi1,angle_rows,strict=True):
        c=max(abs(F(x)) for x in row['ideal_cos_interval_exact'])
        s=max(abs(F(x)) for x in row['ideal_sin_interval_exact'])
        dpower=dw*sum((n1**j*n0**(k-1-j) for j in range(k)),F(0))
        dcdf+=abs(p1-p0)*(c+s*n1)*n1**k+p0*(s*dw*n1**k+(c+s*n0)*dpower)
        dangle+=p1*(F(row['cos_difference_upper_exact'])+F(row['sin_difference_upper_exact'])*n1)*n1**k
    return dict(CDF_one_event_upper_exact=str(dcdf),angle_one_event_upper_exact=str(dangle),
        component_moment_difference_upper_exact=str(dw),base_moment_norm_bound_exact=str(n0),
        grid_moment_norm_bound_exact=str(n1),scalar_coefficients_enclosed=True)


def full_gap_upper(one_event, left, right, et_norm, et_error, B, *, q=8,r=4):
    # Only the norm envelope uses engineering estimates, rather than certified endpoints.
    f=max(left.norm+left.num+left.bind,right.norm+right.num+right.bind)
    e=et_norm+et_error
    return float(B)*q*(e*e*f**r)**(q-1)*e*e*r*f**(r-1)*float(F(one_event))


def corrected_signal(et,et_error,tail,phi,nu,phase,phase_error,B,*,q=8,r=4):
    raw=K.signal_propagation(et,tail.matrix,phi,nu,phase,q=q,repetitions=r,
        et_error=et_error,tail_error=tail.num,phase_error=phase_error)
    # Separate the component-construction uncertainty from numerical signal error.
    amplitude=float(np.linalg.norm(phi)); bind=0.; ne=K.norm(et)+et_error; nf=tail.norm+tail.num
    for _ in range(q):
        amplitude*=ne;bind*=ne
        for _ in range(r):
            bind=(nf+tail.bind)*bind+tail.bind*amplitude;amplitude*=nf
        amplitude*=ne;bind*=ne
    zb=B*K.value(raw)
    return dict(real=zb.real,imag=zb.imag,u_num=B*raw['u_num']+K.gamma(4)*abs(zb),
        u_bind=B*float(np.linalg.norm(phi))*bind/float(nu),u_ref_transfer=0.,
        raw=raw,B_saved=B,B_multiplications=1,norm_divisions=1,scalar_phase_applications=1,
        precision='complex128; engineering uncertainty; not a full signal enclosure')


def difference(left,right):
    delta=K.value(left)-K.value(right)
    u=sum(float(row.get(key,0.)) for row in (left,right) for key in ('u_num','u_bind','u_ref_transfer'))
    return dict(real=delta.real,imag=delta.imag,absolute=abs(delta),uncertainty=u,
        axes={a:dict(value=x,upper_candidate=abs(x)+u,
            status='RESOLVED_ENGINEERING' if abs(x)>u else 'UNRESOLVED')
            for a,x in [('cosine',delta.real),('sine',delta.imag)]})


def headroom_preview(signals,stage2,stage4):
    biases={arm:bias_inputs(dict(signals={'nom_'+arm:signals[arm]['grid']},reference=stage2['reference']),arm)
            for arm in ('P','DF')}
    rows=[]
    for old in stage4['rows']:
        arm=old['arm']
        for axis in ('cosine','sine'):
            eps=old['axes'][axis].get('epsilon_operand_exact')
            available=biases[arm]['axes'] is not None and eps is not None
            h=F(eps)-F(biases[arm]['axes'][axis]['bias_upper_exact']) if available else None
            rows.append(dict(arm=arm,epsilon_sig=old['epsilon_sig'],axis=axis,epsilon_operand_exact=eps,
                bias_upper_exact=biases[arm]['axes'][axis]['bias_upper_exact'] if available else None,
                headroom_exact=str(h) if h is not None else None,
                headroom=float(h) if h is not None else None,
                status='POSITIVE_CONDITIONAL' if h is not None and h>0 else 'UNRESOLVED',
                native_bridge_assumed_zero=True,compiled_rho_assumed_zero=True,
                actual_implemented_headroom=None))
    return biases,rows


def evaluate_arm(arm,descriptor,law,normalization,et,et_error,phi,nu,phase,phase_error,nominal,n):
    original=[row['probability'] for row in law['components']]
    require([row['id'] for row in law['components']]==[row['id'] for row in descriptor['components']], 'COMPONENT_ORDER')
    require(original==[abs(row['coefficient'])/normalization['lambda_R'] for row in descriptor['components']], 'COMPONENT_PROBABILITY')
    total=math.fsum(original)
    p0,p1,pc=choice_law([x/total for x in original])
    pi0,pi1,oc=choice_law(normalization['order_probabilities'])
    coeff0,coeff1,angles=angle_table(normalization['dimensionless_step_time'])
    (w0,w1),norms,assembly=component_moments(arm,descriptor,p0,p1,n)
    power0,power1=moments(w0),moments(w1)
    tails=dict(base=event_mean(w0,power0,pi0,coeff0),cdf=event_mean(w1,power1,pi1,coeff0),
               grid=event_mean(w1,power1,pi1,coeff1))
    signals={name:corrected_signal(et,et_error,tail,phi,nu,phase,phase_error,normalization['B'])
             for name,tail in tails.items()}
    deltas={key:difference(signals[a],signals[b]) for key,a,b in [('CDF','cdf','base'),('angle','grid','cdf')]}
    deltas.update(bridge=difference(signals['base'],nominal),grid=difference(signals['grid'],nominal))
    bridge_ok=deltas['bridge']['absolute']<=deltas['bridge']['uncertainty']
    local=local_gap_bounds(p0,p1,pi0,pi1,norms,angles)
    cdf_up=full_gap_upper(local['CDF_one_event_upper_exact'],tails['base'],tails['cdf'],K.norm(et),et_error,normalization['B'])
    angle_up=full_gap_upper(local['angle_one_event_upper_exact'],tails['cdf'],tails['grid'],K.norm(et),et_error,normalization['B'])
    closure={name:dict(residual=deltas[name]['absolute'],upper_candidate=upper,
        numerical_resolution=deltas[name]['uncertainty'],
        status='PASS_ENGINEERING' if deltas[name]['absolute']<=upper+deltas[name]['uncertainty'] else 'MODEL_UNRESOLVED')
        for name,upper in [('CDF',cdf_up),('angle',angle_up)]}
    bridge_ok=bridge_ok and all(row['status']=='PASS_ENGINEERING' for row in closure.values())
    return dict(signals=signals,differences=deltas,coefficient_gap_closures=closure,
        bridge_status='CONSISTENT_ENGINEERING' if bridge_ok else 'MODEL_UNRESOLVED',
        law=dict(component=pc,order=oc),angles=angles,assembly=assembly,
        bounds=dict(local=local,CDF_signal_upper_candidate=cdf_up,angle_signal_upper_candidate=angle_up,
            law_angle_signal_upper_candidate=cdf_up+angle_up,
            grid_vs_saved_nominal_upper_candidate=cdf_up+angle_up+deltas['bridge']['absolute']+deltas['bridge']['uncertainty'],
            certified_gap_upper=None,norm_envelope='engineering spectral estimates and propagated numerical/binding errors'),
        event_means={name:dict(norm=t.norm,u_num=t.num,u_bind=t.bind) for name,t in tails.items()})


def evaluate(t,descriptors,laws,stage2,stage4,phi,nu,*,workers=2,checkpoint=None):
    n=int(math.log2(len(phi))); require(1<<n==len(phi),'STATE_DIMENSION')
    et,ee,et_info=K.exponential(t,float(K.DELTA)/2)
    angle=stage2['phase']['binary64_angle'];phase=np.exp(-1j*angle)
    phase_error=stage2['phase']['engineering_error']
    arms={}
    with ThreadPoolExecutor(max_workers=min(workers,2)) as pool:
        jobs={arm:pool.submit(evaluate_arm,arm,descriptors[arm],laws[arm],stage2['normalization'][arm],
            et,ee,phi,nu,phase,phase_error,stage2['signals']['nom_'+arm],n) for arm in ('P','DF')}
        for arm,job in jobs.items():
            arms[arm]=job.result()
            if checkpoint:checkpoint(arms,et_info)
    signals={a:v['signals'] for a,v in arms.items()}
    biases,headroom=headroom_preview(signals,stage2,stage4)
    good=all(a['bridge_status']=='CONSISTENT_ENGINEERING' for a in arms.values())
    if not good:
        for row in headroom:row['status']='MODEL_UNRESOLVED'
    return dict(arms=arms,common_T_half=et_info,bias_inputs=biases,headroom_preview=headroom,
        status='COMPLETE_ENGINEERING' if good else 'MODEL_UNRESOLVED',
        scope='Stage2 encoded operators; IID 53-bit law plus fixed binary64 event angles',
        native_bridge=dict(Gaussian_retained_uniform_upper=None,DF_basis_uniform_upper=None,
            phase_aggregation_uniform_upper=None,scalar_controlled_phase_uniform_upper=None,
            existing_evidence='Stage3 local native preflights only; not a whole-law bound'),
        compiled_rho_measured=False,implemented_accuracy_guaranteed=False,certified_gap_upper=None,
        new_N_G=None,population_price_tail_bound=None,finite_Taylor_bias_refined=False)
