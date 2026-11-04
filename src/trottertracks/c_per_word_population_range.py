"""Literal-target per-word and population adapters. Passive imports only.

Reuse fixed-certificate and CAR/Arb arithmetic; never polarize saved U, change
kappa, fit, sample, or use a finite observed maximum as population support.
Every numerical function requires an already authorized caller. Tests invent
all inputs. No H4 array is loaded by this module or its passive entry.
"""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json

from . import c_iid_grid_expected_bias as expected

BASE = 'bd560a45410409573ae849ca5aaf4050343910bf'
FAMILY = 'artifacts/research_c_per_word_population_range/2026-11-04'
CONTRACT = 'docs/research/research_c_per_word_population_range_contract_unapproved_20261104.json'
TARGET = 'literal_saved_atom_with_exact_dyadic_independent_full_CAR_heads_v1'
PRECISION = 192


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _frame(arm, atom):
    name = atom.frame_id
    frames = {arm.model.label+':F'+str(i):i for i in range(len(arm.U))}
    if name not in frames:
        raise ValueError('LITERAL_FRAME_ID')
    return frames[name]


def validate(arm, heads, events, *, h, r):
    """Bind exact input bits, full head matrices, signed atoms and event order.

    Heads are the same binary64 CAR generator outputs used by the independent
    full-reference path, treated as exact dyadics, NOT silently reconstructed
    in exact J/U arithmetic or replaced by stored sector-only generators.
    """
    import numpy as np
    from .c_pr_aware_df_fixed_car import full_partition
    modes = arm.model.modes
    if type(modes) is not int or modes < 2 or modes % 2 or arm.plan.ld != 2 or F(h) <= 0:
        raise ValueError('LD2_POSITIVE_TIME_INTERLEAVED_SPIN')
    n = modes//2; d = 1 << modes
    if (type(r) is not int or r < 1 or len(heads) != 3 or arm.J.shape != arm.U.shape
            or arm.U.shape[1:] != (n,n) or len(arm.U) <= 2
            or arm.U.dtype != np.dtype('float64') or arm.J.dtype != np.dtype('float64')
            or not np.isfinite(arm.U).all() or not np.isfinite(arm.J).all()
            or any(not np.array_equal(J,J.T) for J in arm.J)):
        raise ValueError('FINITE_REAL_SYMMETRIC_J_LITERAL_U')
    groups = full_partition(modes)
    labels = np.empty(d, dtype=int)
    for k, ix in enumerate(groups): labels[ix] = k
    cross = labels[:,None] != labels[None,:]
    for G in heads:
        if (G.shape != (d,d) or G.dtype not in (np.dtype('float64'), np.dtype('complex128'))
                or not np.isfinite(G).all()):
            raise ValueError('FINITE_FULL_FOCK_HEADS')
        if np.any(G[cross] != 0):
            raise ValueError('HEAD_NOT_EXACTLY_SPIN_BLOCK_INVARIANT')
    atoms = {a.atom_id:a for a in arm.plan.atoms}
    if len(atoms) != len(arm.plan.atoms) or not atoms:
        raise ValueError('UNIQUE_COMPLETE_ATOM_CATALOG')
    for atom in atoms.values():
        i = _frame(arm,atom)
        if (i < 2 or len(atom.support) not in (1,2) or len(set(atom.support)) != len(atom.support)
                or any(type(x) is not int or not 0 <= x < modes for x in atom.support)
                or not F(atom.coefficient)):
            raise ValueError('SIGNED_Z_ZZ_TAIL_SUPPORT')
        if not np.array_equal(np.asarray(arm.plan.frames[atom.frame_id]), np.kron(arm.U[i],np.eye(2))):
            raise ValueError('EXACT_SPIN_FRAME_BINDING')
    event_rows = []
    for e in events:
        if (e.rotation_component_id not in atoms or e.taylor_order != 0 or e.phase != 1
                or e.product_component_ids or not np.isfinite(e.rotation_angle)):
            raise ValueError('FINITE_K0_SIGNED_EVENT')
        event_rows.append([e.rotation_component_id,float(e.rotation_angle).hex()])
    def bits(a):
        a = np.ascontiguousarray(a)
        return dict(shape=list(a.shape), dtype=a.dtype.str, SHA256=hashlib.sha256(a.tobytes()).hexdigest())
    target = dict(target=TARGET, candidate=arm.model.source_identity, modes=modes,
        J=bits(arm.J),U=bits(arm.U),heads=[bits(a) for a in heads],
        h=str(F(h)),r=r,tail_identity=float(arm.plan.extracted_identity).hex(),
        atoms=[[a.atom_id,a.frame_id,list(a.support),str(F(a.coefficient))] for a in atoms.values()],
        head_semantics='exact bit values of independent full-CAR head outputs;no Hermitian projection',
        control='ancilla last;D=diag(I,T);all input/output columns;phase fit false')
    if not np.isfinite(arm.plan.extracted_identity):
        raise ValueError('FINITE_EXTRACTED_IDENTITY')
    return dict(target=target, target_id=_digest(target),word=event_rows,
        word_id=_digest(event_rows), groups=groups)


def _literal_block(task):
    """Existing atom_operators/event_unitary/head scheduler, new head routing."""
    import numpy as np
    from flint import ctx, acb
    from . import c_pr_aware_df_fixed_car as car
    from . import c_pr_aware_df_zsxx_enclosure as z
    from .c_pr_aware_df_stage_b import atom_operators
    from .c1_stage3_reference import _ordered_exponentials
    from trotterlib.rte import event_unitary
    arm, heads, events, h, indices, bilinears = task
    with z._BALL_LOCK, ctx.workprec(PRECISION):
        a = car.Arithmetic(); d = len(indices)
        context = NS(d=d,bilinears={key:car.Matrix(z._matrix(E)) for key,E in bilinears.items()})
        ids = {e.rotation_component_id for e in events}
        small_arm = NS(model=arm.model,plan=NS(atoms=tuple(x for x in arm.plan.atoms if x.atom_id in ids),frames=arm.plan.frames))
        atoms = atom_operators(context,small_arm,arithmetic=a)
        G = [car.Matrix(z._matrix(x[np.ix_(indices,indices)])) for x in heads]
        half = tuple((i,F(h)/2) for i in range(3)); cache = {}
        left = _ordered_exponentials(G,half,cache,arithmetic=a)
        right = _ordered_exponentials(G,tuple(reversed(half)),cache,arithmetic=a)
        middle = a.eye(d)
        for event in events: middle = event_unitary(event,atoms,arithmetic=a) @ middle
        phase = (-acb(0,1)*z._real(F(h))*z._real(float(arm.plan.extracted_identity))).exp()
        value = z._scale((right @ middle @ left).ball, phase)
        z._norm(value)
        return dict(indices=indices,entries=car.pack(value))


def build_literal_reference(arm, heads, events, *, h=F(1,10), r=32, parallel=True):
    import numpy as np
    from . import c_pr_aware_df_fixed_car as car
    from . import c_pr_aware_df_fixed_certificate as fixed
    from .c1_stage1 import car_bilinears
    binding = validate(arm,heads,events,h=h,r=r)
    if len(events) != r:
        raise ValueError('REGISTERED_WORD_LENGTH')
    full = car_bilinears(arm.model.modes)
    tasks = [(arm,heads,events,h,ix,{(p,q):full[p,q][np.ix_(ix,ix)]
        for p in range(arm.model.modes) for q in range(arm.model.modes) if p%2 == q%2})
        for ix in binding['groups']]
    plan = fixed.resource_plan(len(tasks))
    if parallel and arm.model.modes >= 6 and plan['workers'] > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            blocks = list(pool.map(_literal_block,tasks))
    else:
        blocks = [_literal_block(t) for t in tasks]
    return dict(status='VALIDATED_LITERAL_REFERENCE_BLOCKS',target=TARGET,
        binding=binding,system_modes=arm.model.modes,system_dimension=1<<arm.model.modes,
        blocks=blocks,resource_plan=plan,precision_bits=PRECISION,
        input_U_replaced=False,kappa_projected=False,polar_bridge_added=False,
        head_target='same exact dyadic full-reference heads',whole_law_bound=None)


def compare_literal(raw_record, reference, *, parallel=True):
    """Reuse full controlled column action; no polar bridge or norm<=2 cap."""
    from flint import arb,ctx
    from . import c_pr_aware_df_fixed_car as car
    from . import c_pr_aware_df_fixed_certificate as fixed
    from . import c_pr_aware_df_zsxx_enclosure as z
    fixed._validate(raw_record)
    d = 2*reference['system_dimension']
    if (reference['target'] != TARGET or reference['status'] != 'VALIDATED_LITERAL_REFERENCE_BLOCKS'
            or (1<<raw_record['qubits']) != d):
        raise ValueError('EXACT_LITERAL_CONTROLLED_TARGET')
    plan = fixed.resource_plan(d)
    if parallel and d >= 128 and plan['workers'] > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),
            initializer=car._init_compare,initargs=(raw_record,reference)) as pool:
            rows = list(pool.map(car._compare_column,range(d)))
    else:
        car._init_compare(raw_record,reference)
        rows = [car._compare_column(i) for i in range(d)]
    with z._BALL_LOCK,ctx.workprec(PRECISION):
        upper = z._fraction(sum((z._real(F(x['upper_exact']))**2 for x in rows),arb(0)).sqrt().upper())
    return dict(status='VALIDATED_RAW_LITERAL_OPERATOR',target=TARGET,
        upper_exact=str(upper),lower_exact=str(max(F(x['lower_exact']) for x in rows)),
        kind='rigorous_enclosure',scope='per_word_full512' if d==512 else 'per_word_full_controlled_artificial',
        leg='raw_to_literal_atom',word_id=reference['binding']['word_id'],
        target_id=reference['binding']['target_id'],raw_id=_digest(raw_record),
        dimension=d,columns_evaluated=d,phase_fit=False,rounding_enclosed=True,
        reference_uncertainty='all CAR/gate/norm arithmetic included;not added twice',
        precision_bits=PRECISION,columns=rows,resource_plan=plan,
        provenance='literal CAR blocks + fixed full-column Arb action',
        upper_method='outward Frobenius of all certified column distances;no cap2',
        all_word_uniform=False,whole_law_bound=None,actual_CA_achieved=None)


def evaluate_word(raw_record, compiled_record, reference, *, parallel=True):
    from . import c_pr_aware_df_fixed_certificate as fixed
    if raw_record['qubits'] != compiled_record['qubits']:
        raise ValueError('RAW_COMPILED_DIMENSION')
    compiled_raw = fixed.compare(compiled_record,raw_record,parallel=parallel)
    raw_literal = compare_literal(raw_record,reference,parallel=parallel)
    if raw_literal['raw_id'] != _digest(raw_record):
        raise ValueError('TRIANGLE_RAW_IDENTITY')
    u = F(compiled_raw['upper_exact'])+F(raw_literal['upper_exact'])
    lower = max(F(),F(raw_literal['lower_exact'])-F(compiled_raw['upper_exact']))
    return dict(status='VALIDATED_PER_WORD_COMPILED_LITERAL_OPERATOR',
        target=TARGET,binding=reference['binding'],kind='rigorous_enclosure',
        upper_exact=str(u),lower_exact=str(lower),compiled_raw=compiled_raw,raw_literal=raw_literal,
        raw_id=_digest(raw_record),compiled_id=_digest(compiled_record),
        phase_fit=False,rounding_enclosed=True,all_word_uniform=False,
        provenance='same raw record and literal target;two disjoint enclosure legs',
        actual_CA_achieved=None,Stage_C_ready=False)


def grid_support(catalog, table, *, expected_ids):
    """Exact saved q counts; zero counts are excluded only with exact evidence."""
    if set(catalog) != set(expected_ids) or len(catalog) != len(expected_ids):
        raise ValueError('COMPLETE_216_CATALOG_BINDING')
    counts = table['counts']; D = table['denominator']
    if (table.get('mapping') != 'searchsorted_right' or table.get('bits') != 53
            or D != 1<<53 or len(counts) != len(expected_ids)
            or any(type(x) is not int or x < 0 for x in counts) or sum(counts) != D):
        raise ValueError('EXACT_53_BIT_Q_SUPPORT')
    return dict(positive=[key for key,n in zip(expected_ids,counts) if n>0],
        excluded_exact_zero=[key for key,n in zip(expected_ids,counts) if n==0],
        counts_by_id=dict(zip(expected_ids,counts)),denominator=D)


def head_norm(G, groups, *, h):
    """Exact antisymmetric coefficients, no Hermitian projection of input.

    ||exp(-itG)||<=exp(|t|*||(G-G*)/2||). Spin blocks form a full-Fock direct
    sum. Their anti-part Frobenius upper is much sharper than ||G||_F.
    """
    from flint import arb,ctx
    from . import c_pr_aware_df_zsxx_enclosure as z
    def q(x):
        x=complex(x);return F(x.real),F(x.imag)
    squares=[]
    for ix in groups:
        s=F()
        for i in ix:
            for j in ix:
                a,b=q(G[i,j]);c,d=q(G[j,i]);s+=((a-c)**2+(b+d)**2)/4
        squares.append(s)
    anti=expected.sqrt_upper(max(squares))
    with z._BALL_LOCK,ctx.workprec(PRECISION):
        norm=(z._real(abs(F(h))/2)*z._real(anti)).exp()
        endpoint=upper_record(norm)
        if endpoint['upper_exact'] is None:
            raise ValueError('HEAD_NORM_REQUIRES_STRUCTURED_ARITHMETIC_NO_CLIP')
        upper=F(endpoint['upper_exact'])
    return dict(half_time=str(F(h)/2),anti_norm_upper_exact=str(anti),
        half_head_norm_upper_exact=str(upper),exact_Hermitian=anti==0,
        method='full-Fock direct-sum logarithmic norm;exact dyadic anti-part;outward exp')


def frame_gram(U):
    """Exact real dyadic Gram matrix, no floating subtraction or polar target."""
    q=[[F(float(x)) for x in row] for row in U];n=len(q)
    gram=[[sum((q[k][i]*q[k][j] for k in range(n)),F()) for j in range(n)] for i in range(n)]
    return dict(gram=[[str(x) for x in row] for row in gram],
        q=[str(gram[i][i]) for i in range(n)],
        Z_norm_upper=[str(max(F(1),abs(1-2*gram[i][i]))) for i in range(n)],
        exact_orthogonal=all(gram[i][j]==int(i==j) for i in range(n) for j in range(n)))


def event_norm(atom, event, frame):
    """CAR column norms and commutator bound, valid for nonorthogonal U.

    ||Z_u||=max(1,|1-2||u||^2|). For ZZ, ||A*-A||<=
    8|<u,v>| ||u|| ||v|| (zero for opposite spins). Then
    ||cI-isA||^2 <= 1+s^2(b^2-1)+|cs|*||A*-A||.
    This keeps c^2+s^2=1 exactly; no loose |c|+|s| triangle.
    """
    from flint import arb,ctx
    from . import c_pr_aware_df_zsxx_enclosure as z
    support=atom.support
    b=F(1)
    for p in support:b*=F(frame['Z_norm_upper'][p//2])
    anti2=F()
    if len(support)==2:
        p,q=support
        cross=F(frame['gram'][p//2][q//2]) if p%2==q%2 else F()
        anti2=64*cross*cross*F(frame['q'][p//2])*F(frame['q'][q//2])
    with z._BALL_LOCK,ctx.workprec(PRECISION):
        if b==1 and anti2==0:
            upper=F(1)
        else:
            t=z._real(float(event.rotation_angle));s=t.sin().abs_upper();c=t.cos().abs_upper()
            value=1+s*s*z._real(b*b-1)+s*c*z._real(anti2).sqrt()
            upper=z._fraction(value.sqrt().upper())
    return dict(atom_id=atom.atom_id,frame_id=atom.frame_id,support=list(support),
        atom_norm_upper_exact=str(b),anti_part_squared_upper_exact=str(anti2),
        event_norm_upper_exact=str(max(F(1),upper)),
        method='literal CAR column-density spectrum and CAR commutator;outward event norm',
        sign_preserved=True,input_U_replaced=False)


def population_norm(arm, heads, catalog, table, *, expected_ids, h=F(1,10), r=32, call=lambda k,l,f:f()):
    """All positive-q length-r words, no sample extrema or observed residuals.

    C norm1 is a theorem for every accepted exact P0 gate output. Totality of
    the compiler on *all* q words remains a separate, unsupplied premise.
    """
    binding=validate(arm,heads,tuple(catalog.values()),h=h,r=r)
    if any(k!=v.rotation_component_id for k,v in catalog.items()):raise ValueError('EVENT_CATALOG_KEY')
    support=call('support_gate','exact saved q support',lambda:grid_support(catalog,table,expected_ids=expected_ids))
    if not support['positive']:raise ValueError('NONEMPTY_Q_SUPPORT')
    hs=[call('head_norm',str(i),lambda G=G:head_norm(G,binding['groups'],h=h)) for i,G in enumerate(heads)]
    frames={i:call('frame_norm',str(i),lambda U=U:frame_gram(U)) for i,U in enumerate(arm.U)}
    atoms={a.atom_id:a for a in arm.plan.atoms}
    rows={key:call('event_norm',key,lambda key=key:event_norm(atoms[key],catalog[key],frames[_frame(arm,atoms[key])]))
        for key in expected_ids}
    def assemble():
        from flint import ctx
        from . import c_pr_aware_df_zsxx_enclosure as z
        head=F(1)
        for row in hs:head*=F(row['half_head_norm_upper_exact'])**2
        event=max(F(rows[key]['event_norm_upper_exact']) for key in support['positive'])
        event_power=power_upper(event,r)
        if event_power['upper_exact'] is None:
            return dict(status='VALIDATED_LITERAL_NORM_STRUCTURED_NOT_CALIBRATION_READY',
                target=TARGET,target_id=binding['target_id'],
                D_norm_upper_exact=None,d_range_upper_exact=None,
                head_norm_product_exact=str(head),event_power=event_power,
                calibrated=False,calibration_ready=False,clipped=False,
                reason='large finite norm growth;structured upper endpoint retained')
        # Compact outward endpoint avoids an exponentially long rational
        # denominator in later m12 profiles. The mathematical target is intact.
        with z._BALL_LOCK,ctx.workprec(PRECISION):
            endpoint=upper_record(z._real(head)*z._real(F(event_power['upper_exact'])))
        if endpoint['upper_exact'] is None:
            return dict(status='VALIDATED_LITERAL_NORM_STRUCTURED_NOT_CALIBRATION_READY',
                target=TARGET,target_id=binding['target_id'],D_norm_upper_exact=None,d_range_upper_exact=None,
                D_norm_upper=endpoint,calibration_ready=False,clipped=False)
        a=max(F(1),F(endpoint['upper_exact']))
        return dict(status='VALIDATED_LITERAL_POPULATION_NORM',target=TARGET,target_id=binding['target_id'],
            D_norm_upper_exact=str(a),accepted_P0_C_norm_exact='1',d_range_upper_exact=str(1+a),
            word_length=r,support=support,heads=hs,frames=frames,events=rows,
            event_angle_bits={k:float(v.rotation_angle).hex() for k,v in catalog.items()},
            range_qualification='conditional_on_total_compiler_domain',
            all_q_words_compile_success_qualified=False,calibration_ready=False,
            no_sample_maximum_used=True,all_word_norm_bound=True,
            all_word_synthesis_error_bound=None,actual_CA_achieved=None)
    value=call('population_norm','all legal length-r words',assemble)
    if value['D_norm_upper_exact'] is None:
        value['round_growth']=None
        return value
    value['round_growth']=[call('growth_profile',str(m),lambda m=m:dict(m=m,occurrences=1<<m,
        D_product_norm=power_upper(value['D_norm_upper_exact'],1<<m),
        axis_support_formula='1+D_product_norm_upper',
        interpreted_as_all_word_synthesis_bound=False)) for m in range(13)]
    return value


def upper_record(ball):
    """Compact exact outward endpoint; never clip a large value to a small one.

    A dyadic mantissa/exponent remains usable when a decimal integer string
    would exceed Python's serialization limit. This is an output-format choice,
    not a scientific/resource bound. Such profiles cannot enter old rational
    slots until the caller supports the structured endpoint.
    """
    from . import c_pr_aware_df_zsxx_enclosure as z
    point=ball.upper()
    if not point.is_finite():raise ValueError('NONFINITE_ENCLOSURE')
    mantissa,exponent=map(int,point.man_exp())
    if mantissa < 0:raise ValueError('NONNEGATIVE_ENCLOSURE')
    return dict(upper_dyadic=dict(mantissa=str(mantissa),exponent2=exponent),
        upper_exact=str(z._fraction(point)) if abs(exponent)+mantissa.bit_length() <= 8192 else None,
        clipped=False)


def power_upper(value, n):
    from flint import ctx
    from . import c_pr_aware_df_zsxx_enclosure as z
    if type(n) is not int or n < 1:raise ValueError('POSITIVE_OCCURRENCES')
    with z._BALL_LOCK,ctx.workprec(PRECISION):
        # Positive exact rational input; repeated squaring encloses every op.
        x=z._real(expected.nonnegative(value));out=z._real(1);k=n
        while k:
            if k&1:out*=x
            k//=2
            if k:x*=x
        return upper_record(out)


def calibration_inputs(samples, population, *, compiler_domain=None):
    """Qualify shared target/range before invoking old confidence functions."""
    if population.get('status') != 'VALIDATED_LITERAL_POPULATION_NORM':
        return dict(status='UNRESOLVED',support=None,samples=None)
    if (F(population['d_range_upper_exact']) != 1+F(population['D_norm_upper_exact'])
            or population.get('accepted_P0_C_norm_exact') != '1'
            or population.get('no_sample_maximum_used') is not True):
        raise ValueError('POPULATION_RANGE_DERIVATION')
    if (not compiler_domain or compiler_domain.get('kind') != 'structural_totality_argument'
            or compiler_domain.get('scope') != 'all_legal_q_words'
            or compiler_domain.get('gates') != 'exact_ideal_P0_unitaries'
            or not compiler_domain.get('provenance')):
        return dict(status='NOT_READY_COMPILER_TOTALITY_PREMISE',support=None,samples=None)
    rows=[]
    for s in samples:
        if s.get('kind') != 'rigorous_enclosure' or s.get('upper_exact') is None:
            return dict(status='UNRESOLVED_PER_SAMPLE',support=None,samples=None)
        binding=s['binding']
        if (binding['target_id'] != population['target_id']
                or any(population['event_angle_bits'].get(key) != angle for key,angle in binding['word'])):
            raise ValueError('SAMPLE_POPULATION_TARGET_OR_ANGLE')
        if s.get('phase_fit') is not False or s.get('rounding_enclosed') is not True:
            raise ValueError('TRUE_NUMERICAL_ENCLOSURE_REQUIRED')
        lo=expected.nonnegative(s['lower_exact']);hi=expected.nonnegative(s['upper_exact'])
        M=F(population['d_range_upper_exact'])
        if lo>hi or lo>M:raise ValueError('INTERVAL_POPULATION_CONTRADICTION')
        # A certified loose upper may exceed a separately proved range. The
        # intersection is valid and explicit, never sample-maximum clipping.
        rows.append(dict(kind='rigorous_enclosure',lower_exact=str(lo),upper_exact=str(min(hi,M)),
            provenance=s['provenance'],range_intersection_applied=hi>M))
    return dict(status='CONDITIONAL_CALIBRATION_INPUTS',samples=rows,
        support=dict(qualification='deterministic_population_bound',
            upper_exact=population['d_range_upper_exact'],
            provenance='CAR/heads population norm plus supplied all-q total compiler premise'),
        statistical_confidence_computed=False,PCG64_joint_stream_qualified=False,
        all_word_synthesis_qualified=False,Stage_C_ready=False)


def threshold_tail(samples, *, support, threshold, delta, policy):
    """Use old exact zero-bad binomial bound; straddles never count as good."""
    from .c_pr_aware_df_guarantee_strategy import zero_failure_tail_limit
    rows=tuple(samples);expected.sampling_gate(policy,len(rows))
    if support is None or any(s.get('upper_exact') is None for s in rows):
        return dict(status='UNRESOLVED',bound=None)
    # Reuse validation/range rules, not a newly implemented statistics engine.
    checked=expected.mean_upper(rows,support=support,delta=delta,policy=policy)
    if checked['bound'] is None:return dict(status='UNRESOLVED',bound=None)
    M=expected.nonnegative(support['upper_exact']);t=expected.nonnegative(threshold)
    if t>=M:raise ValueError('THRESHOLD_STRICTLY_WITHIN_RANGE')
    good=bad=straddle=0;indicators=[]
    for row in rows:
        lo,hi=map(F,(row['lower_exact'],row['upper_exact']))
        if hi<=t:good+=1;bounds=('0','0')
        elif lo>t:bad+=1;bounds=('1','1')
        else:straddle+=1;bounds=('0','1')
        indicators.append(dict(kind='rigorous_enclosure',lower_exact=bounds[0],upper_exact=bounds[1],provenance='certified threshold interval'))
    if bad+straddle==0:
        p=F(zero_failure_tail_limit(len(rows),delta,iid=True,fresh_holdout=True,certified_decisions=True)['tail_probability_upper_exact'])
        method='existing zero_failure_tail_limit'
    else:
        p=F(expected.mean_upper(indicators,support=dict(qualification='deterministic_population_bound',upper_exact='1',
            provenance='Bernoulli exceedance indicator'),delta=delta,policy=policy)['bound'])
        method='existing one-sided bounded mean of conservative bad indicators'
    return dict(status='CONDITIONAL_TAIL_EXPECTATION_BOUND',bound=str(t+(M-t)*p),
        threshold=str(t),good=good,bad=bad,straddle=straddle,tail_probability_upper=str(p),
        delta_cal=str(expected.probability(delta)),method=method,
        all_word_uniform=False,actual_CA_achieved=None)


def variance_sensitive(samples, *, support, delta, policy):
    """Maurer-Pontil2009 Thm4 plus deterministic interval uncertainty.

    New minimal scalar adapter because old code only has Hoeffding/zero-bad.
    Projection contraction bounds the sample standard deviation by midpoint
    std + ||per-sample radii||2/sqrt(N-1). No low-variance assumption is made.
    """
    rows=tuple(samples)
    if len(rows)<2:raise ValueError('VARIANCE_NEEDS_N_AT_LEAST_TWO')
    checked=expected.mean_upper(rows,support=support,delta=delta,policy=policy)
    if checked['bound'] is None:return dict(status='UNRESOLVED',bound=None)
    M=F(support['upper_exact']);n=len(rows)
    mid=[(F(x['lower_exact'])+F(x['upper_exact']))/2 for x in rows]
    radii=[(F(x['upper_exact'])-F(x['lower_exact']))/2 for x in rows]
    avg=sum(mid,F())/n
    v=sum(((x-avg)**2 for x in mid),F())/(n-1)
    sigma=expected.sqrt_upper(v)+expected.sqrt_upper(sum((x*x for x in radii),F())/(n-1))
    log=expected.log_inverse_upper(expected.probability(delta)/2)
    extra=expected.sqrt_upper(2*sigma*sigma*log/n)+7*M*log/(3*(n-1))
    mean_upper=sum((F(x['upper_exact']) for x in rows),F())/n
    return dict(status='CONDITIONAL_EMPIRICAL_BERNSTEIN_BOUND',bound=str(min(M,mean_upper+extra)),
        sample_variance_upper_exact=str(sigma*sigma),midpoint_sample_variance_exact=str(v),
        numerical_interval_uncertainty_included=True,delta_cal=str(F(delta)),
        theorem='Maurer-Pontil2009 Theorem4;scaled [0,M]',all_word_uniform=False,
        actual_CA_achieved=None)


def propagate_calibrated_mean(mu, population, *, shells):
    """Outward Arb companion of the old both-norm expectation formula.

    Only the scalar evaluation changes to keep m12 rational outputs compact.
    No per-sample observation, B2 mean bound, or actual-stream premise enters.
    Unknown numerical inputs return thirteen null rows, never zeros.
    """
    if set(shells)!={'head','X','Y'}:raise ValueError('ALL_THREE_SHELLS')
    a=population.get('D_norm_upper_exact')
    if mu is None or a is None or any(v is None for v in shells.values()):
        return [dict(m=m,bound=None,status='UNRESOLVED') for m in range(13)]
    a=expected.nonnegative(a);mu=expected.nonnegative(mu)
    if a<1:raise ValueError('BOTH_MAP_NORM_GROWTH_REQUIRED')
    errors={k:expected.nonnegative(v) for k,v in shells.items()}
    from flint import ctx
    from . import c_pr_aware_df_zsxx_enclosure as z
    rows=[]
    with z._BALL_LOCK,ctx.workprec(PRECISION):
        for m in range(13):
            n=1<<m;A=z._real(a);growth=A**n
            repeated=growth*z._real(min(F(2),n*mu/a))
            axis={k:A*growth*z._real(errors['head'])+A*A*repeated+A*growth*z._real(errors[k]) for k in ('X','Y')}
            signal=2*A*A*growth*(axis['X']**2+axis['Y']**2).sqrt()
            enclosure=upper_record(signal)
            rows.append(dict(m=m,occurrences=n,bound=enclosure['upper_exact'],enclosure=enclosure,
                status='CONDITIONAL_EXPECTATION_PROPAGATION',
                numerical_formula='existing c_iid_grid_expected_bias.propagate_expected,Arb scalar enclosure',
                all_word_synthesis_qualified=False,PCG64_joint_stream_qualified=False,actual_CA_achieved=None))
    return rows
