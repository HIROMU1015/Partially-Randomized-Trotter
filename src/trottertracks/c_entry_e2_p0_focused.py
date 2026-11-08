"""Focused scalar proof-input adapters; no H4, compiler or interval engine.

The iteration lemma is conditional on successful deterministic native passes.
The numerical formulas separate exact semantics from unverified libm/model
contracts. No supplied number or artificial witness qualifies molecular work.
"""
from fractions import Fraction as F
from . import c_entry_e2_p0_feasibility as previous

FAMILY='artifacts/research_c_entry_e2_p0_focused/2026-11-08/technical'


def _q(x):
    return previous._q(x)


def epoch_bound(cx_count, *, deleting_iterations=None):
    """Conditional iteration bound 8K+7, not just total-node amortization.

    At most K iterations delete CX. In each intervening CX-free block the
    first pass reduces all 1Q runs to <=5, then each active run shrinks at
    most five times, and one unchanged pass confirms FixedPoint: <=7.
    """
    if type(cx_count) is not int or cx_count<0:raise ValueError('CX_COUNT')
    pairs=cx_count//2
    if deleting_iterations is not None:
        if type(deleting_iterations) is not int or not 0<=deleting_iterations<=pairs:
            raise ValueError('CX_DELETION_ITERATIONS')
        K=deleting_iterations
    else:K=pairs
    return dict(initial_CX_upper=cx_count,CX_deleting_iterations_upper=K,
        CX_free_block_upper=7,iterations_upper=8*K+7,
        below_guard_under_conditions=8*K+7<=1000,
        supplied_E2_epoch_bound_verified=False,all_E2_words_return_normally=False,
        conditions=['finite successful deterministic pass calls','native basis closed;target=None;ZSXX only',
            'no calibrations/control flow','translation idle in loop','fixed CX skeleton between deletion iterations',
            'no new 1Q run wakes without a CX deletion;global phase does not enter run product'],
        qualification='conditional structural lemma;not an E2 guard-success certificate')


def weighted_extraction(eta, rho, *, hypot_error=None, radial_atan2_error=None,
                        determinant_error=None, large_arg_phasor_error=None,
                        det_arg_phasor_error=None,parameter_arithmetic_error=None):
    """Source-specific uniform M->exact-reconstruction extraction preview.

    Arg at small entries needs no small ANGLE error. Their arbitrary phase
    error is weighted by a vanishing/near-vanishing amplitude instead.
    All contracts are supplied, not established by this helper.
    """
    base=dict(omega_extraction=None,contracts_verified=False,actual_Rust_omega=None,
        actual_H4_bound=None,phase_sensitive=True,compiler_image_enclosed=False)
    if eta is None or rho is None:return dict(base,status='UNRESOLVED_DOMAIN')
    eta,rho=map(_q,(eta,rho))
    if eta>=F(1,4) or not 0<rho<=1:raise ValueError('WEIGHTED_NEAR_UNITARY_DOMAIN')
    values=(hypot_error,radial_atan2_error,determinant_error,large_arg_phasor_error,
        det_arg_phasor_error,parameter_arithmetic_error)
    if any(x is None for x in values):return dict(base,status='UNRESOLVED_LIBM_OR_ARITHMETIC_CONTRACTS')
    h,t,d,g,gdet,e=map(_q,values)
    rmin=1-eta;detmin=rmin*rmin
    if 2*h>=rmin or d>=detmin or g>2 or gdet>2:
        raise ValueError('HYPOT_DET_OR_PHASOR_DOMAIN')
    theta_error=2*t+4*h/(rmin-2*h)
    kappa=(2*eta+eta*eta)/rmin
    weighted_c=min(F(2),max(2*rho/rmin,g))
    weighted_d=min(F(2),max(2*(rho+eta+kappa),g))
    det_phase=min(F(2),2*d/(detmin-d)+gdet)
    upper=theta_error/2+det_phase+2*weighted_c+2*weighted_d+e
    return dict(base,status='CONDITIONAL_WEIGHTED_EXTRACTION',omega_extraction=str(upper),
        theta_error_upper=str(theta_error),weighted_c_phasor=str(weighted_c),
        weighted_d_phasor=str(weighted_d),det_phasor=str(det_phase),
        arbitrary_small_entry_arg_allowed=True,
        domain=dict(eta=str(eta),small_entry_threshold=str(rho),radial_norm_lower=str(rmin),
            determinant_modulus_lower=str(detmin)),
        excludes=['ZSXX generation/simplification/modulo','DAG/MPFR final phase','RZ node matrix evaluation'])


def branch_budget(*, predicate_error=None,pi_center_error=None,
                  phase_mod_error=None,generation_arithmetic_error=None):
    """All ZSXX branch semantics, conditional on four explicit contracts.

    Flip is an exact identity for EVERY angle, so its modulo predicates need
    no accuracy contract. They still must return finite values without error.
    Three P emissions/drops + pi-near collapse + X selection cost <=5*atol.
    Four meaningful modulo uses have phase-gate, not raw-angle, error <=mu.
    """
    base=dict(actual_Rust_omega=None,omega_generation=None,contracts_verified=False,
        branch_coverage='zero/one/general SX;pi collapse;flip either way;X/SX pair;all P emit/drop;all mod endpoints',
        flip_predicate_accuracy_required=False,actual_H4_bound=None)
    values=(predicate_error,pi_center_error,phase_mod_error,generation_arithmetic_error)
    if any(x is None for x in values):return dict(base,status='UNRESOLVED_BRANCH_CONTRACTS')
    b,pi,mu,e=map(_q,values)
    atol=F.from_float(1e-12)
    return dict(base,status='CONDITIONAL_ALL_BRANCH_SEMANTICS',
        omega_generation=str(5*atol+2*b+2*pi+4*mu+e),
        semantic_threshold_part=str(5*atol),phase_modulus_uses_upper=4,
        arithmetic_excludes_mod_calls=True,DAG_phase_excluded=True,
        pi_center_excludes_pi_in_output_arithmetic=True)


def compose_split(identity,N,*,omega_extraction=None,omega_generation=None,
                  translation=None,phase=None,whole_zeta=None,**product_inputs):
    """Exact Euler+omega accounting; never add an already whole M->V zeta.

    Supplied bounds here must refer to phase-sensitive full512/all-word
    operator norms, and remain unverified proof inputs. The existing composer
    is used via previous.amortized_exact_euler_parts for the product/F leg.
    """
    if identity!=previous.binding.target_identity():raise ValueError('E2_P0_SOURCE_IDENTITY')
    if whole_zeta is not None:raise ValueError('CHOOSE_WHOLE_ZETA_OR_SPLIT_NOT_BOTH')
    parts=previous.amortized_exact_euler_parts(N,**product_inputs)
    values=(omega_extraction,omega_generation,translation,phase)
    conditional=None
    if parts['partial_upper'] is not None and all(x is not None for x in values):
        a,b,c,d=map(_q,values)
        conditional=str(min(F(2),F(parts['partial_upper'])+N*(a+b)+c+d))
    return dict(status='UNRESOLVED_INPUTS' if conditional is None else 'CONDITIONAL_UNVERIFIED_SPLIT_BOUND',
        conditional_operator_upper=conditional,exact_parts=parts,
        accounting='product+exact_F once;N*(extraction+generation);translation;DAG/final phase once',
        whole_zeta_also_charged=False,all_word_normal_return_proven=False,
        error_evidence=None,actual_Rust_omega=None,actual_H4_bound=None,
        actual_inverse_bound=None,raw_literal_bound=None,RPE_accuracy='UNQUALIFIED',
        scientific_calls=0,science_entry=None,mandatory_stop=True)
