"""Opt-in local-support certificate, not a molecular loader or P0 replacement.

Qiskit definitions only propose a local expansion: an independent Arb comparison
to the existing raw primitive encloses its error. Exact disjoint commutations and
inverse cancellations expose a shared CX skeleton. One-qubit SU(2) distances and
the explicitly retained scalar phase are then composed by a unitary telescope.
No full-register column enumeration, sector projection, or literal target occurs.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from fractions import Fraction as F
from functools import lru_cache
import heapq
import multiprocessing

from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z
from .c_per_word_population_range import _digest

VERSION = 'local_definition_SU2_CX_telescope_Arb192_v1'


def validate(record):
    fixed._validate(record)
    if 'target' in record or record.get('unitary') is False:
        raise ValueError('NATIVE_UNITARY_RECORD_ONLY_NO_LITERAL_TARGET')


def gate_key(g):
    return g['name'], tuple(g['qubits']), tuple(float.fromhex(x).hex() for x in g['params_hex'])


def _primitive(g):
    return dict(name=g['name'], qubits=list(range(len(g['qubits']))), params_hex=list(g['params_hex']))


def _definition(g):
    """Candidate only. Its correctness is NOT assumed from Qiskit definitions."""
    from qiskit.circuit.library import XXPlusYYGate, CRZGate, RZZGate
    values = [float.fromhex(x) for x in g['params_hex']]
    if g['name'] == 'xx_plus_yy': op = XXPlusYYGate(*values)
    elif g['name'] == 'crz': op = CRZGate(*values)
    elif g['name'] == 'rzz': op = RZZGate(*values)
    elif g['name'] == 'crzz': op = RZZGate(*values).control(1, annotated=False)
    else: raise ValueError('UNSUPPORTED_TRANSLATION_PRIMITIVE')

    def walk(op, qs, active):
        params = [float(x) for x in op.params]
        if op.name == 'cx' or (op.name in fixed.ARITY and fixed.ARITY[op.name] == 1):
            return [dict(name=op.name, qubits=list(qs), params_hex=[x.hex() for x in params])], F()
        if op.name in ('u', 'u3') and len(params) == 3:
            theta, phi, lam = params
            return [dict(name=n, qubits=list(qs), params_hex=[v.hex()])
                    for n, v in [('p', lam), ('ry', theta), ('p', phi)]], F()
        key = (type(op).__module__, type(op).__name__, op.name, tuple(x.hex() for x in params))
        if key in active: raise ValueError('CYCLIC_GATE_DEFINITION')
        circuit = op.definition
        if circuit is None or circuit.parameters or circuit.num_clbits:
            raise ValueError('UNRESOLVED_LOCAL_DEFINITION')
        phase = F(float(circuit.global_phase)); result = []
        for item in circuit.data:
            inner = tuple(qs[circuit.find_bit(q).index] for q in item.qubits)
            gates, p = walk(item.operation, inner, active | {key})
            result.extend(gates); phase += p
        return result, phase

    gates, phase = walk(op, tuple(range(len(g['qubits']))), set())
    candidate = dict(fixed.identity(len(g['qubits'])), gates=gates)
    validate(candidate)
    if any(len(x['qubits']) != 1 and x['name'] != 'cx' for x in gates):
        raise ValueError('NOT_ONEQ_CX_EXPANSION')
    return dict(primitive=_primitive(g), candidate=candidate, phase_exact_rad=str(phase))


def expand(record):
    validate(record); definitions = {}; gates = []; phase = F(float.fromhex(record['phase_hex']))
    for g in record['gates']:
        if len(g['qubits']) == 1 or g['name'] == 'cx':
            gates.append(deepcopy(g)); continue
        key = _digest(_primitive(g))
        if key not in definitions: definitions[key] = dict(_definition(g), multiplicity=0)
        d = definitions[key]; d['multiplicity'] += 1; phase += F(d['phase_exact_rad'])
        for item in d['candidate']['gates']:
            gates.append(dict(item, qubits=[g['qubits'][q] for q in item['qubits']]))
    return dict(record=dict(fixed.identity(record['qubits']), gates=gates),
                phase_exact_rad=str(phase), definitions=definitions, original_id=_digest(record))


def _inverse(a, b):
    if a['qubits'] != b['qubits']: return False
    na, nb = a['name'], b['name']
    if na == nb and na in ('x', 'z', 'h', 'cx'): return True
    if (na, nb) in (('sx', 'sxdg'), ('sxdg', 'sx'), ('s', 'sdg'), ('sdg', 's')): return True
    return (na == nb and na in ('rz', 'ry', 'p')
            and F(float.fromhex(a['params_hex'][0])) == -F(float.fromhex(b['params_hex'][0])))


def cancel_disjoint_inverses(record):
    """Cancel only an exact inverse with no intervening gate on its support."""
    last = [None] * record['qubits']; nodes = []; cancelled = []
    for index, g in enumerate(record['gates']):
        if g['name'] == 'id': continue
        previous = [last[q] for q in g['qubits']]
        older = previous[0]
        if older is not None and all(x == older for x in previous) and _inverse(nodes[older]['gate'], g):
            node = nodes[older]; node['active'] = False
            for q, p in zip(g['qubits'], node['previous']): last[q] = p
            cancelled.append([node['index'], index]); continue
        node = dict(gate=deepcopy(g), previous=previous, index=index, active=True)
        nodes.append(node)
        for q in g['qubits']: last[q] = len(nodes)-1
    return [x['gate'] for x in nodes if x['active']], cancelled


def _events(r, commute_CX=False):
    """Push only known commuting 1Q suffixes to the right of closed CX."""
    gates = r['gates']
    pending = [[] for _ in range(r['qubits'])]; last = [None] * r['qubits']; nodes = []
    for g in gates:
        if len(g['qubits']) == 1:
            pending[g['qubits'][0]].append(dict(g, qubits=[0])); continue
        if g['name'] != 'cx': raise ValueError('NON_CX_BARRIER')
        qs = g['qubits']; deps = sorted({last[q] for q in qs if last[q] is not None})
        before = []; after = []
        allowed = ({'rz','p','z','s','sdg','id'}, {'x','sx','sxdg','id'}) if commute_CX else (set(), set())
        for q, commute in zip(qs, allowed):
            cut = len(pending[q])
            while cut and pending[q][cut-1]['name'] in commute: cut -= 1
            before.append(pending[q][:cut]); after.append(pending[q][cut:])
        nodes.append(dict(gate=g, blocks=before, deps=deps, successors=[]))
        for d in deps: nodes[d]['successors'].append(len(nodes)-1)
        for q, tail in zip(qs, after): pending[q] = tail; last[q] = len(nodes)-1
    degree = [len(x['deps']) for x in nodes]; heap = []
    for i, n in enumerate(nodes):
        if degree[i] == 0: heapq.heappush(heap, (gate_key(n['gate']), i))
    order = []
    while heap:
        _, i = heapq.heappop(heap); order.append(i)
        for j in nodes[i]['successors']:
            degree[j] -= 1
            if degree[j] == 0: heapq.heappush(heap, (gate_key(nodes[j]['gate']), j))
    if len(order) != len(nodes): raise ValueError('INTERNAL_DAG_CYCLE')
    return dict(events=[dict(gate=nodes[i]['gate'], blocks=nodes[i]['blocks']) for i in order], terminal=pending)


def skeleton(expansion, *, commute_CX=False):
    r = deepcopy(expansion['record']); removed = 0
    while True:
        gates, cancelled = cancel_disjoint_inverses(r); removed += len(cancelled)
        n = _events(dict(r, gates=gates), commute_CX); flattened = []
        for event in n['events']:
            for q, block in zip(event['gate']['qubits'], event['blocks']):
                flattened.extend(dict(g, qubits=[q]) for g in block)
            flattened.append(event['gate'])
        for q, block in enumerate(n['terminal']):
            flattened.extend(dict(g, qubits=[q]) for g in block)
        again, pairs = cancel_disjoint_inverses(dict(r, gates=flattened))
        removed += len(pairs)
        if not pairs:
            return dict(n, cancelled_inverse_pair_count=removed,
                exact_commutations='disjoint supports'+(';control diagonal or target X-axis through closed CX' if commute_CX else ''))
        # Every repeated pass removes at least two gates; no arbitrary iteration cap.
        r['gates'] = again


def plan(left, right, *, commute_CX=False):
    validate(left); validate(right)
    if left['qubits'] != right['qubits']: raise ValueError('SAME_REGISTER_REQUIRED')
    if type(commute_CX) is not bool: raise ValueError('STRUCTURAL_STRATEGY')
    expansions = [expand(x) for x in (left, right)]; sk = [skeleton(x, commute_CX=commute_CX) for x in expansions]
    common = ([gate_key(x['gate']) for x in sk[0]['events']] ==
              [gate_key(x['gate']) for x in sk[1]['events']])
    pairs = []
    if common:
        for a, b in zip(sk[0]['events'], sk[1]['events']):
            for q, ga, gb in zip(a['gate']['qubits'], a['blocks'], b['blocks']):
                pairs.append(dict(qubit=q, left=ga, right=gb))
        for q, (a, b) in enumerate(zip(sk[0]['terminal'], sk[1]['terminal'])):
            pairs.append(dict(qubit=q, left=a, right=b))
    value = dict(version=VERSION, left_id=_digest(left), right_id=_digest(right),
        qubits=left['qubits'], expansions=expansions, skeletons=sk, block_pairs=pairs,
        correspondence=common, commute_CX=commute_CX,
        theorem='exact registered commutations and inverse cancellations;shared CX skeleton')
    return dict(value, plan_id=_digest(value))


def _validate_plan(p):
    if (type(p['qubits']) is not int or p['qubits'] < 1 or type(p['commute_CX']) is not bool
            or len(p['expansions']) != 2 or len(p['skeletons']) != 2):
        raise ValueError('PLAN_REGISTER_SCOPE')
    for e, expected_id in zip(p['expansions'], (p['left_id'],p['right_id'])):
        validate(e['record'])
        if e['record']['qubits'] != p['qubits'] or e['original_id'] != expected_id:
            raise ValueError('PLAN_TARGET_REGISTER_IDENTITY')
        for key, value in e['definitions'].items():
            if (key != _digest(value['primitive']) or type(value['multiplicity']) is not int
                    or value['multiplicity'] < 1 or len(value['primitive']['qubits']) != value['candidate']['qubits']):
                raise ValueError('LOCAL_DEFINITION_SCOPE')
            validate(value['candidate'])
    sk = [skeleton(e,commute_CX=p['commute_CX']) for e in p['expansions']]
    common = [gate_key(x['gate']) for x in sk[0]['events']] == [gate_key(x['gate']) for x in sk[1]['events']]
    pairs = []
    if common:
        for a,b in zip(sk[0]['events'],sk[1]['events']):
            pairs.extend(dict(qubit=q,left=ga,right=gb) for q,ga,gb in zip(a['gate']['qubits'],a['blocks'],b['blocks']))
        pairs.extend(dict(qubit=q,left=a,right=b) for q,(a,b) in enumerate(zip(sk[0]['terminal'],sk[1]['terminal'])))
    if sk != p['skeletons'] or common is not p['correspondence'] or pairs != p['block_pairs']:
        raise ValueError('COMPLETE_STRUCTURAL_PARTITION')


def _phase(pi_coeff=F(), rad=F()):
    from flint import acb, arb
    return (acb(0, 1) * (z._real(pi_coeff)*arb.pi() + z._real(rad))).exp()


def _midpoint_matrix(ball):
    from flint import acb, acb_mat
    mid = acb_mat([[acb(ball[i,j].real.mid(), ball[i,j].imag.mid())
                    for j in range(ball.ncols())] for i in range(ball.nrows())])
    return mid, F(z._norm(ball-mid)['upper_exact'])


def _oneq(gates, cache):
    from flint import acb_mat
    v = acb_mat([[1,0],[0,1]]); error = F(); pi = F(); rad = F()
    phases = dict(id=F(), x=F(1,2), z=F(1,2), h=F(1,2), sx=F(1,4),
                  sxdg=F(-1,4), s=F(1,4), sdg=F(-1,4), rz=F(), ry=F(), p=F())
    for g in gates:
        key = (g['name'], tuple(g['params_hex']))
        if g['name'] not in phases or g['qubits'] != [0]: raise ValueError('ONEQ_UNITARY_ONLY')
        if key not in cache: cache[key] = fixed._local(key[0], [float.fromhex(x) for x in key[1]])
        v, delta = _midpoint_matrix(cache[key]*v); error += delta; pi += phases[g['name']]
        if g['name'] == 'p': rad += F(float.fromhex(g['params_hex'][0]))/2
    v, delta = _midpoint_matrix(z._scale(v, _phase(-pi, -rad)))
    return v, error+delta, pi, rad


def _block_job(pair):
    from flint import ctx, arb
    with z._BALL_LOCK, ctx.workprec(192):
        cache = {}; a, ea, pa, ra = _oneq(pair['left'], cache); b, eb, pb, rb = _oneq(pair['right'], cache)
        options = []
        for sign in (1,-1):
            norm = F(z._norm(z._scale(a, sign)-b)['upper_exact'])+ea+eb
            upper = z._fraction((z._real(norm)/arb(2).sqrt()).upper())
            options.append((upper, sign))
        upper, sign = min(options, key=lambda x: (x[0], -x[1]))
        return dict(upper_exact=str(min(F(2), upper)), sign=sign,
            delta_pi_exact=str(pa-pb+(sign == -1)), delta_rad_exact=str(ra-rb),
            method='SU2 difference operator norm = Frobenius/sqrt(2);outward recentered errors',
            phase_compensation_retained=True)


@lru_cache(None)
def _embedding_layout(n, qs):
    d = 1 << n; mask = sum(1 << q for q in qs)
    local = tuple(sum(((i >> q)&1) << j for j,q in enumerate(qs)) for i in range(d))
    return tuple((i,j,local[i],local[j]) for i in range(d) for j in range(d) if (i & ~mask) == (j & ~mask))


def _definition_job(definition):
    from flint import ctx, acb_mat
    d = definition; n = d['candidate']['qubits']; size = 1 << n
    with z._BALL_LOCK, ctx.workprec(192):
        v = acb_mat([[int(i==j) for j in range(size)] for i in range(size)]); error = F(); cache = {}
        for g in d['candidate']['gates']:
            key = (g['name'], tuple(g['params_hex']))
            if key not in cache: cache[key] = fixed._local(key[0], [float.fromhex(x) for x in key[1]])
            B = acb_mat(size,size)
            for i,j,a,b in _embedding_layout(n, tuple(g['qubits'])): B[i,j] = cache[key][a,b]
            v, delta = _midpoint_matrix(B*v); error += delta
        v, delta = _midpoint_matrix(z._scale(v, _phase(rad=F(d['phase_exact_rad'])))); error += delta
        g = d['primitive']; original = fixed._local(g['name'], [float.fromhex(x) for x in g['params_hex']])
        upper = min(F(2), F(z._norm(original-v)['upper_exact'])+error)
        return dict(upper_exact=str(upper), support_dimension=size, precision_bits=192,
            rounding_enclosed=True, phase_fit=False,
            method='local full matrix Frobenius + outward recenter radius;unitary embedding preserves operator norm')


def _map(function, jobs, parallel):
    jobs = list(jobs); resources = fixed.resource_plan(max(1, len(jobs)))
    # Process startup dominates tiny exact-symbolic/2x2 cases; no resource cap.
    if parallel and len(jobs) > 64 and resources['workers'] > 1:
        with ProcessPoolExecutor(max_workers=resources['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            return list(pool.map(function, jobs)), resources
    resources.update(workers_used=1, reason='small local task set;process startup exceeds local arithmetic')
    return [function(x) for x in jobs], resources


def evaluate(p, *, parallel=True):
    from flint import ctx
    z.runtime_model()
    identity = {k:v for k,v in p.items() if k != 'plan_id'}
    if _digest(identity) != p['plan_id'] or p['version'] != VERSION: raise ValueError('PLAN_IDENTITY')
    _validate_plan(p)
    d = 1 << p['qubits']
    if not p['correspondence']:
        return dict(status='NOT_APPLICABLE_NO_SHARED_CX_SKELETON', upper_exact=None,
                    full_operator_coverage=False, dimension=d, dense_fallback_used=False)
    definitions = {}; counts = Counter()
    for e in p['expansions']:
        for key, value in e['definitions'].items():
            definitions[key] = value; counts[key] += value['multiplicity']
    local, dp = _map(_definition_job, definitions.values(), parallel)
    witnesses = dict(zip(definitions, local))
    # Identical block bodies are evaluated once, keeping multiplicities and phase.
    blocks = {}; multiplicities = Counter()
    for pair in p['block_pairs']:
        key = _digest(dict(left=pair['left'], right=pair['right']))
        blocks[key] = pair; multiplicities[key] += 1
    results, bp = _map(_block_job, blocks.values(), parallel); bw = dict(zip(blocks, results))
    with z._BALL_LOCK, ctx.workprec(192):
        translation = sum((F(witnesses[k]['upper_exact'])*counts[k] for k in definitions), F())
        local_sum = sum((F(bw[k]['upper_exact'])*multiplicities[k] for k in blocks), F())
        pi = sum((F(bw[k]['delta_pi_exact'])*multiplicities[k] for k in blocks), F())
        rad = F(p['expansions'][0]['phase_exact_rad'])-F(p['expansions'][1]['phase_exact_rad'])
        rad += sum((F(bw[k]['delta_rad_exact'])*multiplicities[k] for k in blocks), F())
        phase = z._fraction((_phase(pi, rad)-1).abs_upper())
        upper = min(F(2), translation+local_sum+phase)
    return dict(status='VALIDATED_LOCAL_SUPPORT_NATIVE_PAIR', version=VERSION, kind='rigorous_enclosure',
        norm='full_operator_2', dimension=d, full_operator_coverage=True,
        coverage_proof='local operator embedding tensor identity + exact structural normalization + unitary telescope',
        columns_evaluated=0, sector_projection=False, precision_bits=192, rounding_enclosed=True,
        phase_fit=False, discrete_SU2_sign_gauge_with_compensated_phase=True,
        left_id=p['left_id'], right_id=p['right_id'], plan_id=p['plan_id'],
        commute_CX=p['commute_CX'],
        upper_exact=str(upper), lower_exact='0', translation_upper_exact=str(translation),
        blocks_upper_exact=str(local_sum), global_phase_upper_exact=str(phase),
        global_delta_pi_exact=str(pi), global_delta_rad_exact=str(rad),
        unique_definitions=len(definitions), unique_blocks=len(blocks), block_occurrences=len(p['block_pairs']),
        definition_witnesses=witnesses, block_witnesses=bw,
        definition_multiplicities=dict(counts), block_multiplicities=dict(multiplicities),
        definition_resources=dp, block_resources=bp, nonunitary_literal_target_supported=False,
        all_word_uniform=False, all_q_words_compile_success_qualified=False,
        calibration_ready=False, actual_CA_achieved=None, Stage_C_ready=False)


def compare(left, right, *, parallel=True, commute_CX=False):
    return evaluate(plan(left, right, commute_CX=commute_CX), parallel=parallel)


def check_binding(cert, left, right):
    """Persisted target/scope checks, not an authentication or numerical replay."""
    validate(left); validate(right)
    if (cert.get('status') != 'VALIDATED_LOCAL_SUPPORT_NATIVE_PAIR' or cert.get('version') != VERSION
            or left['qubits'] != right['qubits'] or cert.get('dimension') != 1 << left['qubits']
            or cert.get('left_id') != _digest(left) or cert.get('right_id') != _digest(right)
            or cert.get('precision_bits') != 192 or cert.get('rounding_enclosed') is not True
            or cert.get('phase_fit') is not False or cert.get('full_operator_coverage') is not True
            or cert.get('columns_evaluated') != 0 or cert.get('sector_projection') is not False):
        raise ValueError('LOCAL_CERTIFICATE_TARGET_OR_QUALIFICATION')
    terms = sum((F(cert[k]) for k in ('translation_upper_exact','blocks_upper_exact','global_phase_upper_exact')), F())
    if (any(F(cert[k]) < 0 for k in ('translation_upper_exact','blocks_upper_exact','global_phase_upper_exact'))
            or F(cert['upper_exact']) != min(F(2), terms) or F(cert['lower_exact']) != 0):
        raise ValueError('TELESCOPE_COMPOSITION')
    return True
