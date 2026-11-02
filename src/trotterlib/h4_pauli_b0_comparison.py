"""Registered Pauli-prefix/q comparison using the existing B0/CNG primitives.

Family minima refer to the finite registered set. No molecular construction,
sampling, compiler change or formal certificate acceptance is provided here.
"""
from fractions import Fraction as Q

from . import h4_pauli_b0_execution as E
from . import h4_conditional_cng as A

AXES = E.AXES
EPSILONS = E.EPSILONS


def candidate_id(prefix, q):
    return f"p{prefix:03d}_q{q}"


def candidate_model(base, prefix, q):
    terms = base["hamiltonian"]["terms"]
    if type(prefix) is not int or not 1 <= prefix <= len(terms) or q not in (1,2,4,8):
        raise ValueError("REGISTERED_PAULI_PREFIX_Q")
    return dict(n_qubits=base["n_qubits"], qubit_order=base["qubit_order"],
                state=base["state"], hamiltonian=base["hamiltonian"], step_time=.8/q,
                cases=dict(PR=dict(config=dict(T=.8,q=q,r=4,K=4,method="PR"),
                                   blocks=[dict(label=label,coefficient=c) for label,c in terms[:prefix]])))


def commuting_prefix(base, prefix):
    return all(set(label) <= set("IZ") for label,_ in base["hamiltonian"]["terms"][:prefix])


def reuse_full_pf_target(C_result, raw_norm_exact):
    """Reuse the exact-operand nominal PF raw bias, not the rounded native phase.

    Only scalar division of saved interval endpoints by the same exact norm is
    needed. No new state action, midpoint reference or normalization occurs.
    """
    norm = Q(raw_norm_exact)
    result = dict(bits=C_result["bits"], state_actions=0, raw_norm_exact=str(norm),
                  state_renormalized=False, reference_state_actions=0,
                  reused_saved_full_PF_nominal_C=True, axes={}, bias={})
    for axis,key in zip(AXES,("real","imag")):
        old = C_result["nominal_law"]["PF"]["signed_bias"][key]
        bounds = dict(lower=str(Q(old["lower"])/norm),upper=str(Q(old["upper"])/norm))
        center,radius = E.center_radius(bounds)
        result["bias"][key] = bounds
        result["axes"][axis] = dict(bias_interval=bounds,signed_bias_exact=str(center),
                                   u_enc_exact=str(radius),interval_width_exact=str(2*radius))
    return result


def joint_status(target, epsilon):
    decisions = [d for d in E.precision_decisions(target) if d["epsilon_sig"]==epsilon]
    statuses = {d["status"] for d in decisions}
    if "BIAS_EXCEEDS_TOLERANCE" in statuses:
        return "BIAS_INELIGIBLE"
    return "NUMERICALLY_UNDETERMINED" if "WIDTH_AMBIGUITY" in statuses else "CONDITIONAL_GUARD_ELIGIBLE"


def account(target, prices, epsilon, *, scale=None):
    """Same outward sufficient condition/guard as saved H4; no new shot law."""
    scale = A.scale_bounds(1,0) if scale is None else scale
    axes = {}; n0 = {}; n1 = {}
    for axis in AXES:
        t=target["axes"][axis]
        a=A.base_axis(epsilon,t["signed_bias_exact"],t["u_enc_exact"],scale)
        zero=A.shot(a,0)
        edge=A.shot(a,a["rho_guard_exact"]) if a["domain_nonempty"] else dict(N=None,reason="EMPTY_GUARD_DOMAIN")
        axes[axis]=dict(a,rho_zero=zero,guard_face=edge)
        n0[axis]=zero["N"];n1[axis]=edge["N"]
    def total(counts):
        if any(counts[a] is None or prices[a] is None for a in AXES):
            return None
        return sum((Q(counts[a])*Q(prices[a]) for a in AXES),Q(0))
    low,high=total(n0),total(n1)
    return dict(epsilon_sig=str(epsilon),axes=axes,N_cos=n0["cosine"],N_sin=n0["sine"],
                N_total=None if None in n0.values() else sum(n0.values()),
                G_exact=None if low is None else str(low),
                G_guard_range_exact=None if low is None or high is None else [str(low),str(high)],
                hypothetical_rho="0",actual_rho=None,actual_N=None,actual_G=None,
                formal_certificate_accepted=False)


def finite_minimum(rows):
    eligible=[r for r in rows if r["G_exact"] is not None]
    if not eligible:
        return None
    minimum=min(Q(r["G_exact"]) for r in eligible)
    ties=[r for r in eligible if Q(r["G_exact"])==minimum]
    return dict(candidate_ids=[r["candidate_id"] for r in ties],G_exact=str(minimum),
                independent_guard_range_exact=[str(minimum),str(min(Q(r["G_guard_range_exact"][1]) for r in eligible))],
                selected=ties[0],selection_scope="FINITE_REGISTERED_FAMILY_MINIMUM")


def compare_ranges(left,right):
    if left is None or right is None:
        return dict(rho_zero_order="UNDETERMINED",independent_guard_order="UNDETERMINED",ratio_exact=None)
    l0,l1=map(Q,left);r0,r1=map(Q,right)
    return dict(rho_zero_order="DETERMINISTIC_LOWER" if l0<r0 else "PR_LOWER" if l0>r0 else "TIE",
                independent_guard_order="DETERMINISTIC_LOWER_ALL_GUARDS" if l1<r0 else "PR_LOWER_ALL_GUARDS" if l0>r1 else "GUARD_RANGES_OVERLAP",
                ratio_exact=str(l0/r0))
