"""E2-specific structural/mathematical previews, never scientific evidence.

Reuse the fixed P0 composer. New scalar lemmas keep literal CAR targets
distinct from exact gate-list semantics and do not assume literal U unitary.
No molecular loader, interval engine, compiler or RPE engine is implemented.
"""
from fractions import Fraction as F

from . import c_entry_e2_all_law_readiness as binding
from . import c_pr_aware_df_c2a as structural
from . import c_pr_aware_df_zsxx_uniform as uniform

COMPILER_PAIR = ('compiled_gate_list_exact_semantics', 'raw_gate_list_exact_semantics')
FAMILY = 'artifacts/research_c_entry_e2_p0_feasibility/2026-11-08/technical'


def _q(x):
    if type(x) not in (int, str, F):
        raise ValueError('EXACT_NONNEGATIVE_RATIONAL_INPUT')
    q=F(x)
    if q<0: raise ValueError('EXACT_NONNEGATIVE_RATIONAL_INPUT')
    return q


def euler_exact_map_bound(eta):
    """Analytic lemma for EXACT params_zyz reconstruction, including poles.

    If ||M-G||2<=eta<1/4 for some exact unitary G, the phase-sensitive exact
    Euler reconstruction F(M) satisfies ||F(M)-M||2 <=
    min(2+eta, 6eta/(1-eta)+8eta+2eta^2). See report for the row/phasor proof.
    This does NOT bound f64 Rust extraction, simplification, libm or modulo.
    """
    if eta is None:
        return dict(exact_map_to_input_upper=None, Rust_zeta=None,
            compiler_image_enclosed=False, actual_H4_bound=None)
    q=_q(eta)
    if q>=F(1,4): raise ValueError('EXACT_EULER_LEMMA_DOMAIN')
    upper=min(2+q,6*q/(1-q)+8*q+2*q*q)
    return dict(exact_map_to_input_upper=str(upper), norm='operator_2',
        domain='exists exact unitary G with ||M-G||2<=eta<1/4;zero entries and arg cuts included',
        phase_sensitive=True, exact_arithmetic_only=True, Rust_zeta=None,
        compiler_image_enclosed=False, actual_H4_bound=None)


def literal_zz_norm_model(gram_defect, *, sin_abs_upper='1', cos_abs_upper='1'):
    """Full-Fock CAR lemma, not a computation of the saved U's Gram defect.

    eps>=||U*U-I||2. Ordered P=+/-Za Zb need not be Hermitian/involutory.
    For E=cI-isP, ||E||<=1+(s^2*dP+|cs|*hP)/2 follows from E*E.
    The default |sin|,|cos|<=1 is valid without evaluating an angle.
    """
    if gram_defect is None:
        return dict(gram_defect=None,event_norm_upper=None,unitarity_inferred=False,
            actual_H4_bound=None)
    eps,s,c=map(_q,(gram_defect,sin_abs_upper,cos_abs_upper))
    if s>1 or c>1: raise ValueError('TRIG_ABSOLUTE_UPPER_IN_UNIT_INTERVAL')
    nu=1+2*eps
    dz=4*eps*(1+eps)
    dp=dz*(1+nu*nu)
    hp=8*eps*(1+eps)
    beta=1+(s*s*dp+c*s*hp)/2
    return dict(gram_defect=str(eps),Z_norm_upper=str(nu),Z_square_defect_upper=str(dz),
        P_adjoint_product_defect_upper=str(dp),P_Hermiticity_defect_upper=str(hp),
        event_norm_upper=str(beta),unitarity_inferred=False,
        qualification='conditional analytic CAR inequality;supplied Gram/trig bounds not verified',
        actual_H4_bound=None)


def amortized_exact_euler_parts(N, *, node_matrix_error=None, model=None):
    """Refine the existing visit sum, not a second whole-compiler engine.

    For eta_L=Lp/(1-Lp), B(eta_L)<=k(eta_N)*eta_L and sum L<=6N.
    Charge product plus exact reconstruction once: (1+k)*6Np/(1-Np).
    Actual Rust-versus-exact-F error, translation and phase remain unknown.
    """
    structural.monotone_budget(N,**uniform.CONDITIONS)
    r=uniform.run_bound(N,node_matrix_error=node_matrix_error,model=model)
    base=dict(actual_Rust_after_exact_F=None,translation=None,phase=None,
        whole_P0_bound=None,actual_H4_bound=None,accounting='product_plus_exact_reconstruction_only')
    if r['status']!='CONDITIONAL_ENCLOSURE':
        return dict(base,status='UNRESOLVED_INPUTS',partial_upper=None)
    eta=F(r['input_product_error_exact'])
    if eta>=F(1,4):return dict(base,status='EXACT_EULER_DOMAIN_NOT_ESTABLISHED',partial_upper=None)
    rate=F(r['rate_exact']);visits=6*N*rate/(1-N*rate)
    k=6/(1-eta)+8+2*eta
    return dict(base,status='CONDITIONAL_ANALYTIC_PARTS',partial_upper=str((1+k)*visits),
        eta_N=str(eta),product_visit_sum_upper=str(visits),exact_reconstruction_multiplier=str(k),
        rejected_candidates_charged=False,proof_inputs_verified=False)


def categorical_growth_preview(m, *, event_TV=None, event_norm=None, head_norm_product=None):
    """Factored IID p/q bound for literal, possibly nonunitary trajectories.

    2 min(1,16*2^m TV) h^(2^m) beta^(16*2^m).
    Factoring avoids giant rational power expansions; no numeric H4 evaluation.
    It needs independent events AND coherent occurrences under both laws.
    """
    if type(m) is not int or not 0<=m<=12: raise ValueError('E2_THIRTEEN_ROUNDS')
    base=dict(m=m,occurrences=1<<m,events=16*(1<<m),
        model='IID_UNIFORM_53_BIT_WORDS_COUNTERFACTUAL',
        target='ordered_literal_CAR_atom_word_with_same_heads_scalar_and_angle_bits',
        independent_events_and_occurrences_required=True,PCG64_joint_stream_qualified=False,
        numerical_bound=None,actual_H4_bound=None)
    if any(x is None for x in (event_TV,event_norm,head_norm_product)):
        return dict(base,status='UNRESOLVED_INPUTS',factored_upper=None)
    tv,beta,h=map(_q,(event_TV,event_norm,head_norm_product))
    if tv>1 or beta<1 or h<1: raise ValueError('TV_AND_CONSERVATIVE_NORM_DOMAIN')
    return dict(base,status='CONDITIONAL_ANALYTIC_FACTORIZATION',factored_upper=dict(
        prefactor=str(2*min(F(1),base['events']*tv)),
        head_base=str(h),head_power=base['occurrences'],
        event_base=str(beta),event_power=base['events']),
        unitary_distance_two_cap_applied=False,supplied_inputs_verified=False)


def grammar_ceiling(templates):
    """SOURCE grammar ceiling, not an H4 build or measured circuit price.

    At most 40 Gaussian occurrences, six head diagonals, 16 K0 central
    rotations. Ten P gates conservatively include head and outer scalar terms.
    Native gate cost1 and a five-node U witness are proof obligations.
    """
    oneq,cx=40*8+16*2,16*4
    multiplicities=dict(XX=40*28,controlled_RZ=6*8,controlled_RZZ=6*28,phase=10)
    for key,n in multiplicities.items():
        c=templates[key]['lowered_counts']
        if set(c)-{'u','rz','sx','x','cx'} or any(type(v) is not int or v<0 for v in c.values()):
            raise ValueError('OFFICIAL_TEMPLATE_COUNTS')
        oneq+=n*(5*c.get('u',0)+sum(c.get(g,0) for g in ('rz','sx','x')))
        cx+=n*c.get('cx',0)
    return dict(one_qubit_upper=oneq,CX_upper=cx,N_upper=oneq+cx,
        status='CONDITIONAL_SOURCE_GRAMMAR_CEILING',actual_H4_nodes=None,
        proof_obligations=['official definitions for all parameters','unchanged equivalence graph/cost1',
            'U translation <=5 native nodes','fixed native emitter grammar'])


def termination_preview(N):
    count=structural.monotone_budget(N,**uniform.CONDITIONS)
    return dict(structural=count,guard_iterations=1000,
        count_bound_sufficient_for_guard=count['iterations']<=1000,
        all_word_normal_return_proven=False,finite_intermediates_proven=False,
        qualification='strict accepted length decrease is conditional;guard is an exception path',
        actual_H4_bound=None)


def compiler_preview(identity,N,*,target_pair=COMPILER_PAIR,space='full512',
                     coverage='all_trajectory_words',actual_inverse=False,phase_sensitive=True,
                     accounting='run_product_plus_M_to_V',**proof_inputs):
    """Reuse the existing conditional composer only for its UNITARY leg.

    Accepted output is a proof-input preview, never an E2 qualification.
    Actual inverse requires its own error; it is not silently set to adjoint.
    """
    if identity!=binding.target_identity(): raise ValueError('FIXED_E2_P0_SOURCE_IDENTITY')
    if tuple(target_pair)!=COMPILER_PAIR: raise ValueError('COMPILER_GATE_LIST_TARGET_PAIR')
    if space!='full512' or coverage!='all_trajectory_words' or phase_sensitive is not True:
        raise ValueError('FULL512_ALL_WORD_PHASE_SCOPE')
    if actual_inverse is not False: raise ValueError('SEPARATE_ACTUAL_INVERSE_ERROR_REQUIRED')
    if accounting!='run_product_plus_M_to_V': raise ValueError('NO_DIRECT_BOUND_DOUBLE_COUNT')
    result=uniform.uniform_whole_bound(N,conditions=uniform.CONDITIONS,
        exact_unitary_targets=True,**proof_inputs)
    return dict(conditional_composer=result,termination=termination_preview(N),
        target_pair=list(target_pair),actual_inverse_error=None,raw_to_literal_error=None,
        error_evidence=None,all_law_qualified=False,RPE_accuracy='UNQUALIFIED',
        actual_CA_achieved=None,scientific_calls=0,scientific_execution_ready=False,
        mandatory_stop=True)
