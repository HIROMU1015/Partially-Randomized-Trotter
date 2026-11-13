"""Full-H pure RTE: supplied paper kappa -> explicit q x R layouts. Metadata only.

The shared B count rule r_m = ceil(kappa (lambda_R T_m)^2) gives the total
number of paired finite-K events (one abstract rotation each). This module only
splits that total into an explicit grouping q and events per group R, so the
existing D layout mapper, B.generate and C baseline signal/price paths can be
used unchanged.

Nothing here loads a Hamiltonian/state, evaluates a finite normalizer, signal,
bias, N, price or G, samples events, builds or compiles a circuit, or selects /
adopts kappa, K or a grouping. The supplied kappa is echoed, never back-solved.
q is the Full-H outer grouping (not a PF qPF); signals of the ideal finite law
depend on s=qR only, but compiled-price identity keeps q and R separately.
"""
from copy import deepcopy
from fractions import Fraction

MODULE = 'src/trottertracks/full_h_kappa_layout.py'
BRANCH = 'codex/d-compiled-residual-c-minus-s-20261113'
B_COUNT_SOURCE = '437ba4acdb82708b6fc98e3cd119a2a734f7cf64'   # C-pinned B controller (same blob at b9b9ffa6)
A_DEFINITION_SOURCE = '510fbacd01a2d855202de4c045d8ba1972e5ff3d'
KAPPA_KIND = 'PAPER_TOTAL_ROTATIONS'
DEFINITION = 'PAPER_V2_E21_KAPPA_ROTATIONS_OVER_LAMBDA_R_SQUARED_TIME_SQUARED'
PAPER = dict(url='https://arxiv.org/pdf/2503.05647v2', sections=['A.2', 'E.3'], equations=['E21', 'E22'])
GROUPINGS = ('DYADIC_RPE', 'EXPLICIT')


def kappa_descriptor(value, *, proposal_source):
    """Named, source-bound kappa for D.full_h_rte_spec(kappa=...). Not adopted."""
    value = _positive(value, 'kappa')
    if not proposal_source:
        raise ValueError('Kappa proposal SOURCE required')
    return dict(kind=KAPPA_KIND, value=dict(numerator=value.numerator, denominator=value.denominator),
                definition=DEFINITION, unit='TOTAL_PAIRED_FINITE_K_EVENTS_EQUAL_ABSTRACT_ROTATIONS',
                lambda_scope='FULL_H_SAMPLER_LAMBDA_R_INCLUDING_ONE_BODY',
                definition_source=dict(paper=deepcopy(PAPER),
                    total_count=dict(commit=B_COUNT_SOURCE, path='src/trottertracks/pr_rpe_rmse_controller.py',
                                     function='paper_event_mapper'),
                    A_definition=dict(commit=A_DEFINITION_SOURCE, path='src/trottertracks/pr_rpe_rmse.py',
                                      constant='KAPPA_DEFINITION'),
                    grouping=dict(branch=BRANCH, path=MODULE, function='layouts')),
                proposal_source=deepcopy(proposal_source),
                analytic_optimum_claim=False, compiled_optimum_claim=False, scientifically_adopted=False)


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + ' must be positive')
    if isinstance(value, dict) and set(value) == {'numerator', 'denominator'}:
        if type(value['numerator']) is not int or type(value['denominator']) is not int or value['denominator'] <= 0:
            raise ValueError(name + ' must be an exact rational')
        value = Fraction(value['numerator'], value['denominator'])
    elif type(value) is float:
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError(name + ' must be finite')
        value = Fraction.from_float(value)
    elif type(value) in (int, Fraction):
        value = Fraction(value)
    else:
        raise ValueError(name + ' must be int, binary64 or exact rational')
    if value <= 0:
        raise ValueError(name + ' must be positive')
    return value


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _exact(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def layouts(spec, *, B, grouping='DYADIC_RPE', q_by_round=None):
    """Return ({m: {'q','R_outer'}}, rows) for D.layout_mapper / C.configure.

    The total target is B.paper_event_mapper's exact FULL_H count. R is the
    smallest integer with q R >= target; the overshoot is below q and is
    reported. q follows the named rule (DYADIC_RPE: q=2^m) or explicit integers.
    """
    if spec.get('method') != 'FULL_H_RTE' or spec.get('L_D') != 0 or spec.get('delta_pf') is not None:
        raise ValueError('Full-H pure RTE spec required; PR tails and PF steps have their own mappers')
    kappa = spec.get('kappa') or {}
    if kappa.get('kind') != KAPPA_KIND or kappa.get('definition') != DEFINITION or not kappa.get('definition_source'):
        raise ValueError('Named paper total-rotation kappa required')
    if spec['split'].get('one_body_policy') != 'RANDOMIZED' or spec['split'].get('randomized_scope') != 'FULL_H_INCLUDING_ONE_BODY':
        raise ValueError('Full-H lambda must include randomized one-body terms')
    if type(spec['finite_K']) is not int or spec['finite_K'] < 0 or spec['finite_K'] % 2:
        raise ValueError('Even finite K required')
    M = spec['M']
    if type(M) is not int or M < 0:
        raise ValueError('Nonnegative integer M required')
    if grouping not in GROUPINGS:
        raise ValueError('Named grouping rule required')
    if (grouping == 'EXPLICIT') != (q_by_round is not None):
        raise ValueError('Explicit q_by_round only with EXPLICIT grouping')
    k = _positive(kappa['value'], 'kappa'); lam = _positive(spec['split']['lambda_R'], 'lambda_R')
    h = _positive(spec['h'], 'h')
    out = {}; rows = []
    for m in range(M + 1):
        T = h * (1 << m)
        mapped = B.paper_event_mapper(deepcopy(spec), dict(m=m, T=_exact(T), q_PF=None, piece_count=1))
        target = mapped['paper_total_target']
        if type(target) is not int or target < 1 or mapped['R_outer'] is not None or mapped['event_count'] != target:
            raise ValueError('Shared B Full-H total count changed')
        if target != _ceil(k * (lam * T) ** 2):
            raise ValueError('Total count differs from kappa (lambda_R T)^2')
        q = (1 << m) if grouping == 'DYADIC_RPE' else (q_by_round or {}).get(m)
        if type(q) is not int or q < 1:
            raise ValueError('Positive integer grouping q required for every round')
        R = -(-target // q); s = q * R
        if not target <= s < target + q:
            raise ValueError('Integer layout does not cover the target minimally')
        x = (lam * T) ** 2
        out[m] = dict(q=q, R_outer=R)
        rows.append(dict(m=m, T=_exact(T), q_RPE=1 << m, q_grouping=q, R_per_group=R, event_count=s,
            total_target=target, overshoot=s - target, kappa_requested=_exact(k),
            kappa_effective=_exact(Fraction(s) / x), kappa_effective_float=float(Fraction(s) / x),
            tau=_exact(lam * T / s), tau_float=float(lam * T / s),
            paper_logB_envelope=_exact(x / s), paper_logB_envelope_float=float(x / s),
            envelope_definition='sum over events of tau^2 = (lambda_R T)^2/s <= 1/kappa; finite-K logB not evaluated',
            event_tau='lambda_R*(T/q)/R_outer', grouping=grouping, signal_depends_on='s=qR',
            price_identity_keeps='q_and_R', finite_normalizer=None, signal=None, N=None, price=None))
    return out, rows


def check_configuration(config, *, B, rows):
    """Post-generation checks the kappa-supplied D/B route does not run itself."""
    s = config['spec']
    B._explicit_preparation(s)
    if s['kappa'] is None or s['kappa'].get('kind') != KAPPA_KIND:
        raise ValueError('Kappa-supplied configuration required')
    if len(config['rounds']) != len(rows):
        raise ValueError('Round coverage differs')
    for v2, row in zip(config['rounds'], rows):
        mapping = v2['mapping']
        if v2['resolution'] != 'GENERATED' or mapping is None:
            raise ValueError('Unresolved Full-H layout')
        B._explicit_layout(mapping, s)
        layout = mapping['physical_layout']
        if (layout['q'] != row['q_grouping'] or layout['R_outer'] != row['R_per_group']
                or mapping['event_count'] != row['event_count'] or v2['structure']['event_count'] != row['event_count']
                or not B.p.same(mapping['kappa'], s['kappa']) or not B.p.same(v2['law_binding']['kappa'], s['kappa'])
                or v2['q_PF'] is not None or v2['R_outer'] is not None):
            raise ValueError('Generated Full-H kappa layout differs at m=' + str(v2['m']))
    return True
