"""Arb effective finite-law mean using frozen C and Pauli/Arb primitives.

Preparation exposes only a 2/3-qubit artificial entrypoint. Production dispatch
is intentionally absent until the user approves the separately fixed study.
"""
from fractions import Fraction as Q


def _mean_enclosure(law, raw, reference, bits):
    from . import h4_arb_bridge as B
    from .h4_arb_pr0_preparation import _pauli_map, _pauli_action
    from .h4_pauli_b0_execution import _interval, center_radius
    C=B._backend()
    if bits not in (64,96,128):
        raise ValueError("PRECISION_SCHEDULE")
    n=law["n_system"];cfg=law["config"]
    if len(raw)!=1<<n or law["qubit_order"]!="qiskit_little_endian":
        raise ValueError("STATE_LAYOUT")
    ps=list(map(Q,law["component_mapping"]["exact_masses"]))
    pis=list(map(Q,law["order_mapping"]["exact_masses"]))
    if (sum(ps)!=1 or sum(pis)!=1 or min(ps+pis)<0
            or len(ps)!=len(law["components"]) or law["orders"]!=list(range(0,cfg["K"]+1,2))
            or len(pis)!=len(law["orders"]) or len(law["angles"])!=len(pis)):
        raise ValueError("CONVEX_LAW_DOMAIN")
    if any(c["sign"] not in (-1,1) for c in law["components"]):
        raise ValueError("INVOLUTION_SIGNS")
    with C.ctx.workprec(bits):
        nu=sum((C.rational(x)**2+C.rational(y)**2 for x,y in raw),Q(0))
        if nu<=0:
            raise ValueError("POSITIVE_RAW_NORM")
        if "state_norm" in reference and not Q(reference["state_norm"]["lower"])<=nu<=Q(reference["state_norm"]["upper"]):
            raise ValueError("REFERENCE_NORM")
        bra=[C.complex_ball(z) for z in raw];v=list(bra)
        if not all(z.real.is_exact() and z.imag.is_exact() for z in v):
            raise ValueError("EXACT_RAW_STATE")
        maps=[_pauli_map(c["label"],c["sign"]) for c in law["components"]]
        weights=list(map(C.real,ps));zero=[C.acb(0) for _ in v]
        def M(x):
            out=list(zero)
            for weight,pm in zip(weights,maps):
                px=_pauli_action(x,pm,C)
                out=[z+weight*p for z,p in zip(out,px)]
            return out
        delta=C.rational(law["step_time"])
        retained=[]
        for b in law["retained"]:
            angle=C.real(C.rational(b["coefficient"])*delta/2)
            retained.append((_pauli_map(b["label"],1),angle.cos(),C.acb(0,-angle.sin())))
        trig=[(C.real(a).cos(),C.acb(0,-C.real(a).sin())) for a in law["angles"]]
        scalar=C.phase(C.real(law["scalar_step_binary64"]))
        error=Q(0)
        for _ in range(cfg["q"]):
            for reverse in (False,True):
                for pm,cs,sn in (reversed(retained) if reverse else retained):
                    pv=_pauli_action(v,pm,C)
                    v,error=B._recenter([cs*z+sn*p for z,p in zip(v,pv)],error,C)
                if not reverse:
                    for _ in range(cfg["r"]):
                        powers=[v]
                        for degree in range(cfg["K"]+1):
                            powers.append(M(powers[-1]))
                        out=list(zero)
                        for k,pi,(cs,sn) in zip(law["orders"],pis,trig):
                            weight=C.real(pi*((-1)**(k//2)))
                            out=[z+weight*(cs*x+sn*y) for z,x,y in zip(out,powers[k],powers[k+1])]
                        # F = E[unitary event], hence ||F||<=1 (not unitary).
                        # For any contraction the identical L2 recenter triangle
                        # inequality e_new <= e_old + r remains valid. M powers
                        # are interval evaluated before this single recenter.
                        v,error=B._recenter(out,error,C)
            v,error=B._recenter([scalar*z for z in v],error,C)
        dot=sum((a.conjugate()*z for a,z in zip(bra,v)),C.acb(0))
        transfer=(C.real(nu).sqrt().upper()*C.real(error)).upper()
        raw_mean=C.acb(dot.real+C.arb(0,transfer),dot.imag+C.arb(0,transfer))
        corrected=raw_mean*C.real(law["B_saved_binary64"])
        ref=C.acb(_interval(reference["reference"]["real"],C),_interval(reference["reference"]["imag"],C))
        bias=(corrected-ref)/C.real(nu)
        if not bias.is_finite():
            raise ValueError("NONFINITE_MEAN_STOP")
        result=dict(bits=bits,state_actions=1,reference_state_actions=0,raw_norm_exact=str(nu),
            state_renormalized=False,raw_attenuated_mean=C.serialize(raw_mean),
            physical_corrected_mean=C.serialize(corrected/C.real(nu)),bias=C.serialize(bias),
            state_l2_error_upper_exact=str(error),B_applied_count=1,
            target_semantics=law["target_semantics"],actual_rho=None,formal_certificate_accepted=False,axes={})
        for axis,key in (("cosine","real"),("sine","imag")):
            center,radius=center_radius(result["bias"][key])
            result["axes"][axis]=dict(signed_bias_exact=str(center),u_enc_exact=str(radius),
                interval_width_exact=str(2*radius),bias_interval=result["bias"][key])
        return result


def artificial_mean(law,raw,reference,bits=64):
    if law.get("n_system") not in (2,3):
        raise ValueError("PREPARATION_NO_REAL_H4_ACTION")
    return _mean_enclosure(law,raw,reference,bits)
