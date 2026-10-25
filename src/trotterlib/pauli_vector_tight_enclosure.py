"""Precision refinement of the L2 engine: retain correlated input error.

The v1 interval remains valid but the separate cos/sin branches multiplied
the inherited radius by |cos|+|sin|. Apply the entire unitary (or mean of
unitaries) to that inherited error, and bound only fresh local arithmetic by
the existing gamma engine. Vector center operations and target are unchanged.
"""
from .pauli_vector_enclosure import *

def rotate(v,pm,angle):
    old=v.e;local=Vector(v.v,0.,add(v.norm,v.e));a=C.real(angle)
    out=linear([(C.acb(a.cos()),local),(C.acb(0,-a.sin()),local.pauli(pm))])
    out.e=add(old,out.e);out.norm=v.norm
    return out

def event(v,law,pms):
    old=v.e;norm=v.norm;local=Vector(v.v,0.,add(norm,old))
    ps=list(map(Q,law['component_mapping']['exact_masses']));pis=list(map(Q,law['order_mapping']['exact_masses']))
    def M(x):
        z=linear([(C.real(p),x.pauli(pm)) for p,pm in zip(ps,pms)]);z.norm=x.norm;return z
    powers=[local]
    for j in range(law['config']['K']+1):powers.append(M(powers[-1]))
    pieces=[]
    for k,pi,a in zip(law['orders'],pis,law['angles']):
        angle=C.real(a);w=C.real(pi*((-1)**(k//2)))
        pieces.extend([(C.acb(w*angle.cos()),powers[k]),(C.acb(0,-w*angle.sin()),powers[k+1])])
    out=linear(pieces);out.e=add(old,out.e);out.norm=norm
    return out

def evaluate(model,reference,cfg,law=None):
    from .h4_pauli_b0_execution import _interval,center_radius
    with C.ctx.workprec(128):
        raw=model['state'];v,nu=initial(raw);norm=v.norm;h=model['hamiltonian']
        retained=h['terms'][:cfg['L_D']];delta=Q(.8/cfg['q']);maps={s:mapping(s) for s,c in retained}
        if law:
            ps=list(map(Q,law['component_mapping']['exact_masses']));pis=list(map(Q,law['order_mapping']['exact_masses']))
            if sum(ps)!=1 or sum(pis)!=1 or min(ps+pis)<0:raise ValueError('NOT_CONVEX')
            pms=[mapping(c['label'],c['sign']) for c in law['components']]
            scalar=C.phase(C.real(law['scalar_step_binary64']))
        else:scalar=C.phase(C.real(-Q(h['constant'])*delta))
        for _ in range(cfg['q']):
            for reverse in (False,True):
                for s,c in (reversed(retained) if reverse else retained):v=rotate(v,maps[s],Q(c)*delta/2)
                if law and not reverse:
                    for _ in range(cfg['r']):v=event(v,law,pms)
            # A scalar unit phase has the same correlated-input rule.
            old=v.e;z=linear([(scalar,Vector(v.v,0.,add(v.norm,v.e)))]);z.e=add(old,z.e);z.norm=norm;v=z
        raw_mean=signal(raw,v,nu);B=Q(law['B_saved_binary64']) if law else Q(1)
        corrected=raw_mean*C.real(B)
        ref=C.acb(_interval(reference['reference']['real'],C),_interval(reference['reference']['imag'],C))
        bias=(corrected-ref)/C.real(nu)
        result=dict(state_actions=1,scalar_precision_bits=128,vector_center_precision_bits=53,
                    physical_corrected_mean=C.serialize(corrected/C.real(nu)),bias=C.serialize(bias),axes={},
                    raw_norm_exact=str(nu),state_renormalized=False,state_l2_error_upper=v.e,
                    B_saved_exact=str(B),B_applied_count=1,actual_rho=None,formal_certificate_accepted=False,
                    error_model='IEEE_BINARY64_GAMMA_L2_CORRELATED_UNITARY_INPUT_WITH_ARB_SCALARS',reference_method=reference['method'],
                    raw_dot_binary64=[float(np.sum(np.conjugate(np.array([complex(x,y) for x,y in raw]))*v.v).real),float(np.sum(np.conjugate(np.array([complex(x,y) for x,y in raw]))*v.v).imag)])
        for axis,key in [('cosine','real'),('sine','imag')]:
            center,radius=center_radius(result['bias'][key]);result['axes'][axis]=dict(signed_bias_exact=str(center),u_enc_exact=str(radius),interval_width_exact=str(2*radius),bias_interval=result['bias'][key])
        return result
