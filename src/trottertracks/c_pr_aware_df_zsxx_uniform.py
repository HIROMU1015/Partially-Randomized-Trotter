"""Source-reachable domains and conditional P0 bounds;no molecular execution.

The actual Rust transcendental image is deliberately not invented. Bounds on
ideal gate families are validated;IEEE/matrix-evaluation/MPFR proof inputs
remain conditional. Captured-pair boxes are never accepted as compiler images.
"""
from fractions import Fraction
import math

from . import c_pr_aware_df_zsxx_enclosure as z
from .c_pr_aware_df_c2a import monotone_budget

VERSION='ZSXX_uniform_domain_conditional_composer_v1'
SCOPE='C_PR_AWARE_DF_ZSXX_UNIFORM_READINESS_V1'
FAMILY='artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27'
U=Fraction(1,2**53)
H=Fraction(1,2**1075)  # half the smallest subnormal: an absolute error term
EPS=Fraction.from_float(1e-12)
PI=Fraction.from_float(math.pi)
CONDITIONS=dict(basis_closed=True,target_none=True,zsxx_only=True,
                parameter_free_cancellation=True,no_calibrations=True)


def _q(x):
    v=z.rational(x)
    if v<0:raise ValueError('NONNEGATIVE_EXACT_BOUND')
    return v


def _integer(n,minimum=1):
    if type(n) is not int or n<minimum:raise ValueError('INTEGER_DOMAIN')
    return n


def _unknown(*missing):
    return dict(status='UNRESOLVED',missing=list(missing),whole_law_bound=None,
                actual_H4_bound=None,whole_P0_bound_exact=None)


def ieee_model():
    """An explicit hypothetical model,not a wheel/platform certificate."""
    return dict(kind='conditional_IEEE_binary64_RN',finite_intermediates=True,
        gradual_underflow=True,no_reassociation=True,
        contraction='separate or correctly rounded FMA',platform_verified=False)


def _ieee(model):
    if model is None:return False
    if (model.get('kind')!='conditional_IEEE_binary64_RN'
            or not all(model.get(k) is True for k in
                       ('finite_intermediates','gradual_underflow','no_reassociation'))
            or model.get('contraction')!='separate or correctly rounded FMA'
            or model.get('platform_verified') is not False):
        raise ValueError('UNSUPPORTED_OR_OVERCLAIMED_IEEE_MODEL')
    return True


def product_rate(node_matrix_error):
    """q+b-1 for the actual sequential complex2x2 multiplication order.

    Each real component has four products and three adds (term path<=3).
    gamma3*sum(abs(products)) plus7*h*(1+u)^3 bounds that component.
    sqrt(8)<=4 gives conservative spectral rounding bounds16*gamma3*A*M
    and28*h*(1+u)^3. FMA can only shorten these same expression paths.
    The uniform input-matrix error a is an operator norm,not an entry error.
    """
    a=_q(node_matrix_error);gamma3=3*U/(1-3*U)
    k=16*gamma3;b=28*H*(1+U)**3
    return (1+a)*(1+k)-1+b


def grammar_node_bound(*,official_topology_coverage=False):
    """SOURCE-derived conditional size ceiling,not a built H4 circuit cost.

    U has a five-node RZ/SX witness;the fixed BasisTranslator minimizes sum
    of native gate costs. Counts below come from parameter-independent
    definitions (XX/inverse,closed CRZ,one-control RZZ,P). Finite traces alone
    must not set official_topology_coverage.
    """
    from .c_pr_aware_df_c2a import primitive_grammar
    if official_topology_coverage is not True:return _unknown('all_parameter_official_topology_coverage')
    g=primitive_grammar()
    weights=dict(RZ=1,XX=7*5+6+2+1,CRZ=4,CRZZ=18*5+14+2,P=5)
    counts=dict(RZ=g['gaussian_occurrences_upper']*g['gaussian_RZ_upper'],
        XX=g['gaussian_occurrences_upper']*g['gaussian_XXPlusYY_upper'],
        CRZ=g['head_factors']*g['head_RZ_upper_per_factor'],
        CRZZ=g['head_factors']*g['head_RZZ_upper_per_factor'],
        P=g['scalar_identity_P_upper'])
    tail=g['events']*(g['tail_RZ_upper_per_event']+g['tail_CX_upper_per_event'])
    N=sum(counts[k]*weights[k] for k in counts)+tail
    return dict(status='CONDITIONAL_SOURCE_TOPOLOGY_BOUND',initial_nodes_upper=N,
        counts=counts,translated_weights_upper=weights,tail_native_nodes_upper=tail,
        U_translation_witness_nodes=5,
        restrictions='same fixed raw grammar,official definitions,unchanged equivalence graph;no target/plugins/calibration/opaque gate',
        actual_H4_circuit_built=False,actual_H4_node_count=None,whole_law_bound=None)


def run_bound(length,*,node_matrix_error=None,model=None):
    """All L-gate words under a supplied uniform matrix-evaluation model.

    e_L <= L*p/(1-L*p), where p=(1+a)*(1+16gamma3)-1+28h(1+u)^3.
    This is a priori over words,not a maximum of observed run residuals.
    """
    L=_integer(length)
    ok=_ieee(model)
    if node_matrix_error is None or not ok:return _unknown('uniform_node_matrix_evaluation_and_IEEE_model')
    p=product_rate(node_matrix_error)
    if L*p>=1:return _unknown('small_error_denominator_not_positive')
    eta=L*p/(1-L*p)
    return dict(status='CONDITIONAL_ENCLOSURE',length=L,rate_exact=str(p),
        input_product_error_exact=str(eta),near_unitarity_defect_exact=str(2*eta+eta*eta),
        near_unitarity_defect_norm='operator_2',
        determinant_centers=['1','i','-1','-i'],
        determinant_distance_upper_exact=str(2*eta+eta*eta),
        minimum_singular_value_lower_exact=str(max(Fraction(),1-eta)),
        domain='exists ideal RZ/SX/X product G,det(G) in {1,i,-1,-i},||M-G||2<=eta;Euler poles included',
        all_run_words_under_model=True,model_verified=False,
        compiler_interval_image_enclosed=False,whole_law_bound=None,actual_H4_bound=None)


def theta_branch_cover(lower,upper,*,model=None):
    """Sound branch SUPerset under the explicit basic arithmetic model.

    Subtraction is outward widened,including underflow. General mod/RZ/flip
    branches are ALL retained;this does not enclose any Rust output image.
    """
    lo,hi=z.rational(lower),z.rational(upper)
    if lo>hi:raise ValueError('INTERVAL_ORDER')
    if not _ieee(model):return _unknown('binary64_predicate_model')
    def possibilities(center):
        err=Fraction() if center==0 else U*(max(abs(lo),abs(hi))+abs(center))+H
        a,b=lo-center-err,hi-center+err
        minimum=0 if a<=0<=b else min(abs(a),abs(b))
        maximum=max(abs(a),abs(b))
        return minimum<EPS,maximum>=EPS
    t0,f0=possibilities(0);th,fh=possibilities(PI/2);tp,fp=possibilities(PI)
    routes=[]
    if t0:routes.append('zero_SX')
    if f0 and th:routes.append('one_SX')
    if f0 and fh:routes.append('general_SX_or_X')
    return dict(status='CONDITIONAL_BRANCH_SUPERSET',theta_interval=[str(lo),str(hi)],
        routes=routes,pi_adjust=([x for x,v in [('taken',tp),('not_taken',fp)] if v]
                                if 'general_SX_or_X' in routes else []),
        general_guards=['flip_or_no_flip','RZ_emit_or_omit','SX_pair_or_X','mod_endpoint_or_no_clamp'],
        predicates='strict abs<atol;uncertain comparisons retain both paths',
        compiler_interval_image_enclosed=False,uniform_Rust_error=None,whole_law_bound=None)


def ideal_template_image(gates,*,phase_interval=('0','0')):
    """Validated ideal-gate image on a parameter box,NOT compiler(box).

    Uses the prior Arb gate evaluator. This useful subroutine cannot replace
    the missing f64 Rust branch/transcendental semantics.
    """
    from flint import ctx,arb,acb
    z.runtime_model()
    def ball(pair):
        a,b=map(z.rational,pair)
        if a>b:raise ValueError('INTERVAL_ORDER')
        return z._real((a+b)/2)+arb(0,z._real((b-a)/2))
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        values=[(name,None if interval is None else ball(interval)) for name,interval in gates]
        A=z._product(values)
        A=z._scale(A,(acb(0,1)*ball(phase_interval)).exp())
        entries=[]
        for i in range(2):
            row=[]
            for j in range(2):
                parts=[]
                for part in (A[i,j].real,A[i,j].imag):
                    parts.append([str(z._fraction(part.lower())),str(z._fraction(part.upper()))])
                row.append(parts)
            entries.append(row)
    return dict(status='VALIDATED_IDEAL_TEMPLATE_IMAGE',precision_bits=z.PRECISION,
        entry_real_imag_bounds=entries,compiler_interval_image_enclosed=False,
        Rust_branch_coverage=False,whole_law_bound=None,actual_H4_bound=None)


def translation_rule(rule):
    """Uniform official-definition semantics for finite binary64 parameters.

    CRZ halves can round only at underflow;RZZ copies theta unchanged.
    This does not assert that every full P0 translation uses these rules.
    """
    z.runtime_model()
    if rule=='CRZ_definition':bound=H;nodes=4
    elif rule=='RZZ_definition':bound=Fraction();nodes=3
    elif rule=='RZ_inverse':bound=Fraction();nodes=1
    else:return _unknown('unreviewed_official_translation_rule:'+rule)
    return dict(status='ANALYTIC_RULE_ENCLOSURE_UNDER_IEEE',rule=rule,
        operator_upper_exact=str(bound),definition_nodes=nodes,
        domain='every finite binary64 angle;closed control for CRZ;ideal gate semantics',
        phase_sensitive=True,whole_translation_coverage=False,whole_law_bound=None)


def period_gap():
    """Outward |binary64(2*pi)-mathematical2*pi|,not empirical libm error."""
    from flint import arb,ctx
    z.runtime_model()
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        return z._fraction((z._real(2*PI)-2*arb.pi()).abs_upper())


def phase_bound(additions,sum_abs,*,accumulator_bits=160,remainder_model=None):
    """Conditional MPFR accumulation,final cast,and true-period correction.

    sum_abs includes initial phase and bind/cancel constants as well as the
    rewrite phases. MPFR160 operation coverage is a proof input,not inferred
    from the wrapper's name. Every modulo call needs its own accounting.
    """
    n=_integer(additions,0);bits=_integer(accumulator_bits,2);A=_q(sum_abs)
    if remainder_model is None:return _unknown('MPFR_operation_and_remainder_model')
    if (remainder_model.get('kind')!='conditional_phase_arithmetic'
            or remainder_model.get('accumulator_correctly_rounded') is not True
            or remainder_model.get('no_accumulator_underflow_overflow') is not True
            or remainder_model.get('euclidean_quotient_relation') is not True
            or remainder_model.get('verified_for_fixed_wrapper') is not False):
        raise ValueError('UNSUPPORTED_OR_OVERCLAIMED_PHASE_MODEL')
    rem=remainder_model.get('absolute_remainder_error')
    if rem is None:return _unknown('remainder_absolute_error')
    rem=_q(rem);u=Fraction(1,2**bits)
    if n*u>=1:return _unknown('phase_gamma_denominator_not_positive')
    accumulation=n*u/(1-n*u)*A
    import sys
    if A+accumulation>Fraction.from_float(sys.float_info.max):
        return _unknown('final_binary64_cast_may_overflow')
    cast=U*(A+accumulation)+H
    P=A+accumulation+cast
    quotient_bound=math.ceil(P/(2*PI))+2
    period=quotient_bound*period_gap()
    total=accumulation+cast+rem+period
    return dict(status='CONDITIONAL_ENCLOSURE',phase_operator_upper_exact=str(min(Fraction(2),total)),
        accumulation_exact=str(accumulation),final_cast_exact=str(cast),
        remainder_exact=str(rem),true_period_correction_exact=str(period),
        quotient_abs_upper=quotient_bound,normalizations_accounted=1,phase_fit=False,model_verified=False,
        whole_law_bound=None,actual_H4_bound=None)


def conditional_phase_model():
    """If fmod is exact,the only rem_euclid error is the negative-r+T add.

    Linking fmod is not by itself verification of the entire installed path.
    This explicit conditional model must never be silently adopted for H4.
    """
    return dict(kind='conditional_phase_arithmetic',accumulator_correctly_rounded=True,
        no_accumulator_underflow_overflow=True,euclidean_quotient_relation=True,
        verified_for_fixed_wrapper=False,
        absolute_remainder_error=str(U*(2*PI)+H),
        fmod_semantics='assume exact truncated remainder for finite binary64 operands')


def uniform_whole_bound(N,*,conditions,node_matrix_error=None,model=None,
                       zsxx=None,translation=None,phase=None,exact_unitary_targets=False):
    """All-word proof-input composer,distinct from the captured-word sum.

    sum L<=6N and L<=N imply total run-product error<=6N*p/(1-N*p).
    A uniform M->V bound is charged per ACCEPTED rewrite,at most N times.
    All inputs have to refer to the same reachable D_N and compiler path.
    This schema does not establish supplied proofs. Actual H4 stays null.
    """
    _integer(N);structural=monotone_budget(N,**conditions)
    if structural['accepted_rewrites'] is None:return _unknown('monotone_conditions')
    if exact_unitary_targets is not True:return _unknown('exact_unitary_targets')
    run=run_bound(N,node_matrix_error=node_matrix_error,model=model)
    missing=[]
    if run['status']!='CONDITIONAL_ENCLOSURE':missing.extend(run['missing'])
    for name,row,scope in [('zsxx',zsxx,'M_to_V_all_reachable_D_N'),
                           ('translation',translation,'all_initial_translation_paths'),
                           ('phase',phase,'all_phase_accumulation_and_normalizations')]:
        if row is None:missing.append(name);continue
        if (row.get('kind')!='uniform_proof_input' or row.get('scope')!=scope
                or row.get('norm')!='operator_2' or row.get('phase_sensitive') is not True
                or not row.get('provenance')):
            raise ValueError('UNIFORM_PROOF_SCOPE:'+name)
        if name=='zsxx' and (row.get('compiler_interval_image_enclosed') is not True
                or row.get('covers_D_N')!=N or row.get('all_branches') is not True):
            raise ValueError('CAPTURED_BOX_IS_NOT_COMPILER_IMAGE')
        if row.get('upper_exact') is None:missing.append(name)
        else:_q(row['upper_exact'])
    if missing:return _unknown(*missing)
    p=z.rational(run['rate_exact'])
    product=6*N*p/(1-N*p);synthesis=N*_q(zsxx['upper_exact'])
    total=product+synthesis+_q(translation['upper_exact'])+_q(phase['upper_exact'])
    return dict(status='CONDITIONAL_WHOLE_P0_PROOF_INPUT_BOUND',initial_nodes_upper=N,
        accepted_rewrites_upper=N,accepted_gate_visits_upper=6*N,
        run_product_sum_upper_exact=str(product),ZSXX_sum_upper_exact=str(synthesis),
        whole_P0_bound_exact=str(min(Fraction(2),total)),proof_inputs_mathematically_verified=False,
        rejected_rewrites_charged=False,direct_bound_also_charged=False,
        whole_law_bound=None,actual_H4_bound=None)


def preflight(root,path):
    """Reuse the established byte/package checks,without numerical decode."""
    from pathlib import Path
    import importlib.metadata
    from . import c1_stage1_execution as io
    from .c_pr_aware_df_c2a import P0
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['implementation']!=VERSION or c['executable'] is not False
            or c['status']!='BLOCKED_UNIFORM_RUST_IMAGE' or c['maximum_logical_units'] is not None
            or c['precision_bits']!=z.PRECISION or c['compiler']!=P0
            or c['retry'] or c['resume']):raise ValueError('UNIFORM_TECHNICAL_SCOPE')
    manifest=(root/c['source_manifest']).resolve()
    if not manifest.is_relative_to(root) or io.sha(manifest)!=c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(manifest)['files'],c['saved_inputs']):
        for name,h in group.items():
            target=(root/name).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=h:raise ValueError('SOURCE_OR_INPUT:'+name)
    for name,version in c['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('ENVIRONMENT:'+name)
    from importlib.util import find_spec
    qroot=Path(find_spec('qiskit').origin).parent.resolve()
    for name,h in c['external_python_source_hashes'].items():
        target=(qroot/name).resolve()
        if not target.is_relative_to(qroot) or io.sha(target)!=h:raise ValueError('COMPILER_SOURCE:'+name)
    return dict(status='PASSIVE_IDENTITIES_PASS_SCIENCE_BLOCKED',scientific_calls=0,
        executable=False,maximum_logical_units=None,runtime_model=z.runtime_model())
