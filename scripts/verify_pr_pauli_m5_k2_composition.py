"""Artificial check: does the provider evaluate PR-Pauli 4th(m5_best), K=2 as the specified composition?

Two separate questions are kept apart:
  (A) evaluation  - the provider's raw / corrected / exact-tail signals and normalizer equal an
      independent reference that spells out the specified composition;
  (B) approximation - how far the specified 4th-order composition is from exp(-iTH) (reported
      separately; it says nothing about (A)).

Artificial system: 3 qubits, one-body + 10 two-body "fragments" (head = one-body + fragments 1..L_D)
and a signed Pauli tail dictionary A = sum_l c_l P_l, lambda = sum_l |c_l|; head and tail do not
commute. Jobs come from B's pinned generate/validate and P1 ``job`` with a synthetic Pauli
descriptor, so the provider takes its Pauli branch exactly as in the H4 runs.

The reference does not call the provider's composition (``_macros``/``evaluate``) nor the RTE
primitives (``finite_taylor_operator``/``finite_rte_distribution``). It states explicitly:
  * the m5_best signed weights (h46_m5_plan TAIL, FORMULA_ID PF_M5_BEST_20D64C2_BINARY64_V1),
    palindrome tail+middle+tail, middle weight negative;
  * per outer step and piece j: signed time t_j = T w_j / q, events c_j, event time t_j / c_j;
  * head order: exp(-i t_j/2 H_0), exp(-i t_j/2 H_1) ... exp(-i t_j/2 H_LD), tail, then the mirror;
  * the integer event allocation per outer step (exact rational largest remainder, minimum one);
  * the K=2 paired-Taylor event mean (degree K+1 polynomial, via eigenvalues) and the normalizer
    b(tau) = sum_{k even <= K} |tau|^k/k! sqrt(1 + (|tau|/(k+1))^2), tau = lambda * event time;
  * for negative pieces, the event law itself (explicit rotation-form events) against P/b.
No H4 data, signal, compile or Q3.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np  # noqa: E402
from trottertracks import df_head_pauli_tail_p1 as p1  # noqa: E402
from trottertracks import pf_pr_rpe_pipeline as pp  # noqa: E402
from trottertracks import pr_rpe_signal_provider as P  # noqa: E402

# Specification of 4th(m5_best) (src/trotterlib/h46_m5_plan.py TAIL / weights), restated here.
SPEC_TAIL = (.11225428783206336, .14181961643121604, .1211920365713068, .13781643533753404, .12547397927357543)
SPEC_FORMULA_ID = 'PF_M5_BEST_20D64C2_BINARY64_V1'
K = 2
SOURCE = dict(fixture='SYNTHETIC_ONLY')
DICTIONARY_SOURCE = dict(fixture='SYNTHETIC_PAULI_DICTIONARY')
TOL_AGREE = 1e-11
DETECT_FACTOR = 1e3          # a deliberate error must move a signal/log_B by >= 1000 x this round's agreement level


def spec_weights():
    return tuple(reversed(SPEC_TAIL)) + (1 - 2 * sum(SPEC_TAIL),) + SPEC_TAIL


def allocation_ref(weights, total):
    """Per outer step: one event per piece, the rest by |w| with exact largest remainder (ties: lower index)."""
    n = len(weights); mags = [Fraction(abs(w)) for w in weights]; total_mag = sum(mags)
    quota = [(total - n) * m / total_mag for m in mags]
    out = [1 + math.floor(x) for x in quota]
    order = sorted(range(n), key=lambda i: (-(quota[i] - math.floor(quota[i])), i))
    for i in order[:total - sum(out)]:
        out[i] += 1
    return out


def normalizer_ref(tau, k_max=K):
    a = abs(tau)
    return math.fsum(a ** k / math.factorial(k) * math.sqrt(1 + (a / (k + 1)) ** 2) for k in range(0, k_max + 1, 2))


# ---------------------------------------------------------------- artificial system
PAULI = dict(I=np.eye(2, dtype=complex), X=np.array([[0, 1], [1, 0]], complex),
             Y=np.array([[0, -1j], [1j, 0]], complex), Z=np.diag([1., -1.]).astype(complex))


def pauli(label):
    out = np.eye(1, dtype=complex)
    for ch in label:
        out = np.kron(out, PAULI[ch])
    return out


def system(seed=20261116, lam=Fraction(3, 2)):
    rng = np.random.default_rng(seed); n = 3
    one = sum(rng.uniform(.2, .6) * pauli(''.join('Z' if j == q else 'I' for j in range(n)))
              + rng.uniform(.1, .4) * pauli(''.join('X' if j == q else 'I' for j in range(n))) for q in range(n))
    labels = [a + b for a in 'XYZ' for b in 'XYZ']
    fragments = []
    for i in range(10):
        q = i % n; r = (i + 1) % n; mat = np.zeros((8, 8), complex)
        for _ in range(3):
            pair = labels[rng.integers(len(labels))]; s = ['I'] * n; s[q] = pair[0]; s[r] = pair[1]
            mat += rng.uniform(-.7, .7) * pauli(''.join(s))
        fragments.append(mat)
    # Signed dyadic coefficients with sum |c| = 96/64 = 3/2 exactly (lambda of the spec).
    tail = [(20, 'XYZ'), (-14, 'ZZI'), (9, 'YIX'), (-17, 'IXY'), (11, 'ZXZ'), (-13, 'XIY'), (12, 'IYY')]
    if Fraction(sum(abs(c) for c, _ in tail), 64) != lam:
        raise ValueError('ARTIFICIAL_LAMBDA')
    coeffs = [c / 64 for c, _ in tail]; tail_labels = [l for _, l in tail]
    state = rng.normal(size=8) + 1j * rng.normal(size=8); state /= np.linalg.norm(state)
    return dict(matrices=[one] + fragments, tail=list(zip(coeffs, tail_labels)), lam=math.fsum(abs(c) for c in coeffs),
                state=state, constant=.17, scalar=.013)


def tail_matrix(sysm):
    return sum(c * pauli(lbl) for c, lbl in sysm['tail'])


def configuration(b, *, prefix, h, delta, kappa, M, lam, formula='4th(m5_best)', scalar=.013):
    rational = lambda s: dict(numerator=Fraction(s).numerator, denominator=Fraction(s).denominator)
    desc = dict(kind=p1.KIND, dictionary_sha256='0' * 64, source=deepcopy(DICTIONARY_SOURCE), prefix=prefix,
                mode_mapping='SYNTHETIC', coefficient_policy='SYNTHETIC', scalar_policy='KEEP_SAVED_sR_AND_c_ONCE',
                event_space='FULL_2_TO_N_NOT_PROJECTED_PAULI', P0_operator_residual_source=dict(fixture='NONE'))
    profile = deepcopy(pp.PROFILE); profile['compiler'] = 'MOCK_COMPILER'
    spec = dict(method='PR', M=M, L_D=prefix, delta_pf=float(Fraction(delta)), formula=formula, h=float(Fraction(h)),
                binding=dict(system='MOCK', rank=10, hamiltonian='MOCK_H', state='MOCK_S', snapshot='MOCK_SNAPSHOT',
                             decomposition='MOCK_DF', source=SOURCE),
                split=dict(L_D=prefix, source=SOURCE, randomized_scope='TAIL_AFTER_PREFIX', lambda_R=rational(lam),
                           lambda_source=dict(DICTIONARY_SOURCE, selector=['lambda_P']), one_body_policy='DETERMINISTIC',
                           deterministic_scope='PREFIX', tail_dictionary=deepcopy(desc), scalar=scalar),
                law=dict(model='ideal_finite_RTE_law', tail_dictionary=deepcopy(desc), source=SOURCE,
                         event_model='EXISTING_PAIRED_TAYLOR_SIGNED_PAULI_INVOLUTIONS',
                         scalar='SAVED_sR_AND_c_ONCE_AT_TOTAL_TIME', actual_sampler_qualified=False),
                kappa=dict(kind='PAPER_TOTAL_ROTATIONS', value=rational(kappa), definition_source=SOURCE),
                event_mapping=dict(kind='PAPER_TOTAL_UNIFORM_OUTER_DRAFT', source=SOURCE), finite_K=K,
                rmse_policy=dict(family='PR_PAPER_RMSE', id='MOCK_POLICY', source=SOURCE, target_rmse_ha=rational('1/1000'),
                                 error_contract=dict(fixture='MOCK_ONLY'), scientific_status='PROPOSED'), profile=profile)
    config = b.generate(spec); b.validate(config)
    return config


def context(sysm, config, *, budget=math.inf):
    s = config['spec']; prefix = s['L_D']; lam = float(Fraction(s['split']['lambda_R']['numerator'], s['split']['lambda_R']['denominator']))
    if abs(lam - sysm['lam']) > 1e-15:
        raise ValueError('ARTIFICIAL_LAMBDA_NOT_DICTIONARY_L1')
    tails = {prefix: dict(scalar=sysm['scalar'], lambda_R=lam, source=s['split']['lambda_source'])}
    ctx = P.make_context(sysm['matrices'], sysm['state'], sysm['constant'], tails, binding=s['binding'], source=SOURCE)
    matrix = tail_matrix(sysm); matrix.setflags(write=False)
    ctx.tails[prefix].update(matrix=matrix, dictionary_identity=deepcopy(s['split']['tail_dictionary']))
    ctx.normalizers = {}; ctx.normalizer_budget = budget
    return ctx


def exact_reference(sysm, prefix, T, binding):
    H = sum(sysm['matrices'][:prefix + 1]) + tail_matrix(sysm) + (sysm['constant'] + sysm['scalar']) * np.eye(8)
    w, v = np.linalg.eigh(H); value = complex(np.vdot(sysm['state'], v @ (np.exp(-1j * T * w) * (v.conj().T @ sysm['state']))))
    z = dict(real=value.real, imag=value.imag)
    return dict(binding=deepcopy(binding), T=T, source=SOURCE, evidence=dict(reference=z, independent_reference=z, signal_difference=0.))


# ---------------------------------------------------------------- independent reference
class Spectral:
    def __init__(self, matrix):
        self.w, self.v = np.linalg.eigh(matrix); self.vh = self.v.conj().T

    def exp(self, t):                       # exp(-i t H)
        return self.v @ (np.exp(-1j * t * self.w)[:, None] * self.vh)

    def taylor(self, dt, k_max=K):          # sum_{n <= K+1} (-i dt H)^n / n!
        x = -1j * dt * self.w
        return self.v @ (sum(x ** n / math.factorial(n) for n in range(k_max + 2))[:, None] * self.vh)


def reference(sysm, prefix, T, q, R, *, weights=None, mutate=None):
    """Sequential state propagation of the specified composition (no provider/RTE primitive)."""
    weights = list(spec_weights() if weights is None else weights); counts = allocation_ref(weights, R)
    lam = sysm['lam']; heads = [Spectral(m) for m in sysm['matrices'][:prefix + 1]]; A = Spectral(tail_matrix(sysm))
    if mutate == 'head_order_reversed':
        heads = heads[::-1]
    if mutate == 'allocation_shifted':
        i = counts.index(max(counts)); counts[i] -= 1; counts[(i + 1) % len(counts)] += 1
    v = np.repeat(sysm['state'][:, None], 3, axis=1); log_b = 0.; pieces = []
    for _ in range(q):
        for w, count in zip(weights, counts):
            t = T * w / q
            if mutate == 'negative_piece_sign_dropped':
                t = abs(t)
            dt = t / count; tau = lam * dt
            b = normalizer_ref(dt if mutate == 'normalizer_without_lambda' else tau)
            half = [h.exp(t / 2) for h in heads]
            if mutate == 'tail_before_head':
                order = [('tail', None)] + [('head', u) for u in half] + [('head', u) for u in half[::-1]]
            else:
                order = [('head', u) for u in half] + [('tail', None)] + [('head', u) for u in reversed(half)]
            poly = A.taylor(dt)
            for kind, u in order:
                if kind == 'head':
                    v = u @ v
                else:
                    v[:, 0] = A.exp(t) @ v[:, 0]
                    v[:, 1] = np.linalg.matrix_power(poly, count) @ v[:, 1]
                    v[:, 2] = np.linalg.matrix_power(poly / b, count) @ v[:, 2]
            log_b += count * math.log(b)
            if len(pieces) < len(weights):
                pieces.append(dict(signed_time=t, events=count, event_time=dt, tau=tau, b=b))
    v = v * np.exp(-1j * T * (sysm['constant'] + sysm['scalar']))
    means = [complex(np.vdot(sysm['state'], v[:, i])) for i in range(3)]
    return dict(exact_tail=means[0], corrected=means[1], raw=means[2], log_B=log_b, pieces=pieces, allocation=counts)


def event_law_mean(sysm, dt, k_max=K):
    """Explicit rotation-form events of the paired-Taylor law (signed tau): sum_events prob * U_event."""
    lam = sysm['lam']; tau = lam * dt; terms = [(abs(c) / lam, 1 if c > 0 else -1, pauli(l)) for c, l in sysm['tail']]
    b = normalizer_ref(tau, k_max); mean = np.zeros((8, 8), complex); total = 0.
    for k in range(0, k_max + 1, 2):
        weight_k = abs(tau) ** k / math.factorial(k) * math.sqrt(1 + (abs(tau) / (k + 1)) ** 2)
        theta = math.atan(tau / (k + 1))
        for prefix in itertools.product(terms, repeat=k):
            left = np.eye(8, dtype=complex); p_left = 1.
            for p, s, mat in prefix:
                left = left @ (s * mat); p_left *= p
            for p, s, mat in terms:
                prob = weight_k / b * p_left * p
                rotation = math.cos(theta) * np.eye(8) - 1j * math.sin(theta) * s * mat
                mean += prob * ((-1) ** (k // 2)) * (left @ rotation); total += prob
    return mean, total, b


# ---------------------------------------------------------------- comparisons
def provider_signals(out):
    s = out['signal']; z = lambda key: complex(s[key]['real'], s[key]['imag'])
    return dict(exact_tail=z('exact_tail'), corrected=z('corrected'), raw=z('raw'), log_B=s['log_B'])


def compare(sysm, config, b, m):
    job = p1.job(config, m, dict(fixture='SYNTHETIC_SOURCE'), b)
    c = job['physical_context']['physical_candidate']; st = job['physical_context']['structure']
    ctx = context(sysm, config)
    out = P.evaluate(ctx, job, exact_reference(sysm, config['spec']['L_D'], c['T'], config['spec']['binding']), root=ROOT)
    ref = reference(sysm, config['spec']['L_D'], c['T'], c['q'], c['R_outer'])
    got = provider_signals(out)
    diffs = {k: abs(got[k] - ref[k]) for k in ('exact_tail', 'corrected', 'raw', 'log_B')}
    piece = max(abs(a[k] - r[k]) / max(1., abs(r[k])) for a, r in zip(out['piece_diagnostics'], ref['pieces'])
                for k in ('signed_time', 'event_time', 'tau', 'b'))
    events_equal = [a['events'] for a in out['piece_diagnostics']] == [r['events'] for r in ref['pieces']]
    mutations = {}
    for name in ('head_order_reversed', 'negative_piece_sign_dropped', 'allocation_shifted', 'tail_before_head',
                 'normalizer_without_lambda'):
        if name == 'allocation_shifted' and max(ref['allocation']) == 1:
            continue                                   # R = 11: the one-per-piece allocation is the only one
        mut = reference(sysm, config['spec']['L_D'], c['T'], c['q'], c['R_outer'], mutate=name)
        mutations[name] = max(abs(got[k] - mut[k]) for k in ('exact_tail', 'corrected', 'raw', 'log_B'))
    negative = [dict(index=i, **{k: r[k] for k in ('signed_time', 'events', 'event_time', 'tau', 'b')})
                for i, r in enumerate(ref['pieces']) if r['signed_time'] < 0]
    return dict(m=m, prefix=config['spec']['L_D'], T=c['T'], q=c['q'], delta=c['delta'], R_outer=c['R_outer'],
                status=out['status'], gate_pass=out['numerical'].get('gate_pass'), allocation=st['allocation'],
                allocation_equals_reference=st['allocation'] == ref['allocation'],
                weights_equal_spec=st['signed_weights'] == list(spec_weights()), formula_id=st['formula_id'],
                signals=dict(raw=str(got['raw']), corrected=str(got['corrected']), exact_tail=str(got['exact_tail']), log_B=got['log_B']),
                differences=diffs, max_piece_relative_difference=piece, piece_events_equal=events_equal,
                negative_pieces=negative, mutation_differences=mutations,
                agree=max(diffs.values()) <= TOL_AGREE and piece <= 1e-14 and events_equal,
                mutations_detected=all(v >= DETECT_FACTOR * max(max(diffs.values()), 1e-14) for v in mutations.values()),
                min_mutation_over_agreement=min(mutations.values()) / max(max(diffs.values()), 1e-14))


CASES = (  # (prefix, h, delta, kappa, M); actual steps T/q are binary64-exact (see non_dyadic_step_refusal)
    (1, '1/10', '1/10', '2', 3),      # q = 1, 2, 4, 8; one event per piece
    (2, '1/2', '1/2', '6', 4),        # q = 2^m; multi-event pieces at m3, m4
    (3, '1/4', '1', '12', 4),         # q = 1 then 2, 4; multi-event pieces
    (2, '1', '4', '40', 3),           # hundreds of events per piece
    (2, '1', '4', '1/4', 2),          # one event per piece, |tau| up to 1.7 (negative middle piece)
)


def non_dyadic_step_refusal(sysm, b):
    """A B round whose actual step T/q is not binary64-exact (q = 7): P1 job rounds delta to float."""
    config = configuration(b, prefix=2, h='2/5', delta='1/2', kappa='3', M=3, lam=Fraction(3, 2))
    row = config['rounds'][3]; job = p1.job(config, 3, dict(fixture='SYNTHETIC_SOURCE'), b)
    try:
        P.evaluate(context(sysm, config), job, exact_reference(sysm, 2, job['physical_context']['physical_candidate']['T'],
                                                               config['spec']['binding']), root=ROOT)
        outcome = 'EVALUATED'
    except ValueError as exc:
        outcome = 'REFUSED: ' + str(exc)
    return dict(q_PF=row['q_PF'], actual_delta_exact=str(b.exact_scalar(row['actual_delta_pf'])),
                actual_delta_float=job['physical_context']['physical_candidate']['delta'], outcome=outcome)


def run(out_path=None):
    b = p1.b_api(ROOT); sysm = system(); report = dict(spec=dict(formula_id=SPEC_FORMULA_ID, weights=list(spec_weights())),
                                                       system=dict(qubits=3, head_fragments=10, tail=[dict(coefficient=c, pauli=l) for c, l in sysm['tail']],
                                                                   lam=sysm['lam'], constant=sysm['constant'], scalar=sysm['scalar']))
    w = spec_weights()
    report['spec_checks'] = dict(palindrome=list(w) == list(reversed(w)), sum_minus_one=math.fsum(w) - 1,
                                 cubic_sum=math.fsum(x ** 3 for x in w), negative_pieces=[i for i, x in enumerate(w) if x < 0])
    head = sum(sysm['matrices'][:3]); A = tail_matrix(sysm)
    report['noncommuting'] = dict(head_tail=float(np.linalg.norm(head @ A - A @ head)),
                                  one_body_fragment1=float(np.linalg.norm(sysm['matrices'][0] @ sysm['matrices'][1] - sysm['matrices'][1] @ sysm['matrices'][0])),
                                  fragment1_fragment2=float(np.linalg.norm(sysm['matrices'][1] @ sysm['matrices'][2] - sysm['matrices'][2] @ sysm['matrices'][1])))
    rows = []
    for prefix, h, delta, kappa, M in CASES:
        config = configuration(b, prefix=prefix, h=h, delta=delta, kappa=kappa, M=M, lam=Fraction(3, 2))
        for m in range(M + 1):
            rows.append(dict(case=dict(prefix=prefix, h=h, delta=delta, kappa=kappa), **compare(sysm, config, b, m)))
    report['rounds'] = rows
    # Event law of the negative pieces (and one positive) against the K+1 polynomial / normalizer.
    law = []
    for r in rows:
        for piece in r['negative_pieces'][:1]:
            mean, total, bb = event_law_mean(sysm, piece['event_time'])
            poly = Spectral(A).taylor(piece['event_time'])
            law.append(dict(m=r['m'], prefix=r['prefix'], event_time=piece['event_time'], tau=piece['tau'],
                            probability_sum_minus_one=total - 1, b_minus_provider_b=bb - piece['b'],
                            law_mean_times_b_minus_polynomial=float(np.abs(mean * bb - poly).max())))
    mean, total, bb = event_law_mean(sysm, .21)
    law.append(dict(positive_control=True, event_time=.21, probability_sum_minus_one=total - 1,
                    law_mean_times_b_minus_polynomial=float(np.abs(mean * bb - Spectral(A).taylor(.21)).max())))
    report['event_law'] = law
    # Allocation rule over many R (B's frozen float rule vs exact rational restatement).
    mism = [R for R in range(11, 3001) if list(pp._schedule().allocation(list(w), R)) != allocation_ref(w, R)]
    report['allocation_R_11_3000_mismatches'] = mism
    # (B) approximation, separate: deterministic composition with exact tail vs exp(-iTH), fixed T, q doubling.
    approx = {}
    H = head + A
    for label, weights in (('4th(m5_best)', spec_weights()), ('2nd', (1.,))):
        errs = []
        for q in (1, 2, 4, 8):
            ref = reference(sysm, 2, 1.6, q, len(weights), weights=weights)
            wv, vv = np.linalg.eigh(H + (sysm['constant'] + sysm['scalar']) * np.eye(8))
            exact = complex(np.vdot(sysm['state'], vv @ (np.exp(-1j * 1.6 * wv) * (vv.conj().T @ sysm['state']))))
            errs.append(abs(ref['exact_tail'] - exact))
        approx[label] = dict(errors=errs, observed_orders=[math.log2(errs[i] / errs[i + 1]) for i in range(3)])
    report['approximation_separate_question'] = approx
    report['normalizer_budget'] = normalizer_budget_demo(sysm, b)
    report['non_dyadic_step_refusal'] = non_dyadic_step_refusal(sysm, b)
    report['summary'] = dict(rounds=len(rows), all_agree=all(r['agree'] for r in rows), all_gate_pass=all(r['gate_pass'] for r in rows),
                             all_mutations_detected=all(r['mutations_detected'] for r in rows),
                             max_signal_difference=max(max(r['differences'][k] for k in ('exact_tail', 'corrected', 'raw')) for r in rows),
                             max_log_B_difference=max(r['differences']['log_B'] for r in rows),
                             min_mutation_difference=min(min(r['mutation_differences'].values()) for r in rows),
                             min_mutation_over_agreement=min(r['min_mutation_over_agreement'] for r in rows),
                             negative_pieces_checked=sum(len(r['negative_pieces']) for r in rows),
                             weights_and_allocation=all(r['weights_equal_spec'] and r['allocation_equals_reference'] for r in rows),
                             event_law_max=max(x['law_mean_times_b_minus_polynomial'] for x in law),
                             new_H4_signal=0, compile=0, Q3=0)
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=1, default=str) + '\n')
    return report


def normalizer_budget_demo(sysm, b):
    """Shared normalizer cache over all rounds (as the runners use it): budget = rounds vs one per piece."""
    config = configuration(b, prefix=2, h='1/2', delta='1/2', kappa='6', M=4, lam=Fraction(3, 2))
    out = {}
    for label, budget in (('one_per_round', len(config['rounds'])),
                          ('one_per_piece', sum(len(r['structure']['signed_weights']) for r in config['rounds']))):
        ctx = context(sysm, config, budget=budget); statuses = []
        for m in range(len(config['rounds'])):
            job = p1.job(config, m, dict(fixture='SYNTHETIC_SOURCE'), b)
            r = P.evaluate(ctx, job, exact_reference(sysm, 2, job['physical_context']['physical_candidate']['T'], config['spec']['binding']), root=ROOT)
            statuses.append(r['status'] if r['status'] != 'NUMERICAL_FAILURE' else 'NUMERICAL_FAILURE: ' + r.get('error', {}).get('message', ''))
        out[label] = dict(budget=budget, statuses=statuses, distinct_normalizers=len(ctx.normalizers))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out')
    a = parser.parse_args()
    value = run(a.out)
    print(json.dumps(dict(summary=value['summary'], spec_checks=value['spec_checks'], noncommuting=value['noncommuting'],
                          allocation_mismatches=value['allocation_R_11_3000_mismatches'][:10],
                          approximation={k: v['observed_orders'] for k, v in value['approximation_separate_question'].items()},
                          normalizer_budget=value['normalizer_budget'], non_dyadic=value['non_dyadic_step_refusal']), indent=1, default=str))
