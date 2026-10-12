"""Independent stdlib saved-witness checker; never imports the generator.

Recomputes coefficient trials, lower certificates and every accepted proof.
No optimizer/eigensolver/truth/circuit calls. Numerical proposal quality is not
certified; finite failures are not upgraded to mathematical infeasibility.
"""
from fractions import Fraction as F
from itertools import combinations
import hashlib
import json
import math


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def hash_value(x): return hashlib.sha256(canonical(x)).hexdigest()
def encode(x):
    if isinstance(x, F): return [x.numerator, x.denominator]
    if isinstance(x, dict): return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [encode(v) for v in x]
    return x


class Checker:
    def __init__(self, bit_cap=4096):
        self.bit_cap = bit_cap
        self.checks = 0
        self.max_bits = 0
        self.pool = {}
        self.cache = {}

    def req(self, test, reason):
        self.checks += 1
        if not test: raise ValueError(reason)

    def q(self, x):
        x = F(x)
        bits = max(abs(x.numerator).bit_length(), x.denominator.bit_length())
        self.max_bits = max(self.max_bits, bits)
        self.req(bits <= self.bit_cap, 'checker rational bit cap')
        return x

    def f(self, x):
        self.req(isinstance(x, list) and len(x) == 2 and all(type(v) is int for v in x) and x[1] > 0, 'rational schema')
        return self.q(F(*x))

    def mul(self, x, y): return self.q(self.q(x) * self.q(y))
    def add(self, x, y): return self.q(self.q(x) + self.q(y))
    def sub(self, x, y): return self.q(self.q(x) - self.q(y))
    def div(self, x, y): return self.q(self.q(x) / self.q(y))
    def total(self, xs):
        s = F(0)
        for x in xs: s = self.add(s, x)
        return s

    def matrix(self, raw):
        self.req(isinstance(raw, list) and bool(raw) and all(isinstance(r, list) and len(r) == len(raw) for r in raw), 'matrix shape')
        M = [[self.f(x) for x in row] for row in raw]
        self.req(all(M[i][j] == M[j][i] for i in range(len(M)) for j in range(len(M))), 'matrix symmetry')
        return M

    def shift(self, M, z): return [[self.sub(x, z) if i == j else self.q(x) for j, x in enumerate(r)] for i, r in enumerate(M)]
    def scale(self, M, z): return [[self.mul(x, z) for x in r] for r in M]
    def plus(self, A, B): return [[self.add(x, y) for x, y in zip(r, s)] for r, s in zip(A, B)]
    def outer(self, v): return [[self.mul(x, y) for y in v] for x in v]

    def proof(self, M, key, strict=False):
        self.req(key in self.pool, 'missing PSD proof')
        p = self.pool[key]
        self.req(type(p['accepted']) is bool and type(p['strict']) is bool and p['strict'] == strict, 'PSD flags')
        expected_id = 'psd:' + hash_value({'matrix': encode(M), 'strict': strict})
        self.req(key == p['id'] == expected_id and p['matrix_sha256'] == hash_value(encode(M)), 'PSD matrix identity')
        if key in self.cache:
            self.req(p == self.cache[key][0], 'changed cached PSD proof')
            return self.cache[key][1:]
        n = len(M)
        self.req(bool(n) and all(len(r) == n for r in M) and all(M[i][j] == M[j][i] for i in range(n) for j in range(n)), 'PSD shape/symmetry')
        L = [[F(i == j) for j in range(n)] for i in range(n)]
        ds = []
        status = 'PSD'
        for i in range(n):
            d = self.sub(M[i][i], self.total(self.mul(self.mul(L[i][k], L[i][k]), ds[k]) for k in range(i)))
            ds.append(d)
            if d < 0: status = 'NEGATIVE_PIVOT'; break
            residual = [self.sub(M[j][i], self.total(self.mul(self.mul(L[j][k], L[i][k]), ds[k]) for k in range(i))) for j in range(i + 1, n)]
            if d == 0:
                if strict: status = 'ZERO_PIVOT_NOT_SPD'; break
                if any(residual): status = 'ZERO_PIVOT_NONZERO_ROW'; break
            else:
                for j, r in zip(range(i + 1, n), residual): L[j][i] = self.div(r, d)
        valid = status == 'PSD'
        self.req(p['status'] == status and p['accepted'] == valid and [self.f(x) for x in p['pivots']] == ds, 'independent PSD status/pivots')
        if valid:
            self.req([[self.f(x) for x in r] for r in p['L']] == L, 'independent LDL factors')
            for i in range(n):
                for j in range(n): self.req(self.total(self.mul(self.mul(L[i][k], ds[k]), L[j][k]) for k in range(n)) == M[i][j], 'LDL exact reconstruction')
        self.cache[key] = (p, valid, L, ds)
        return valid, L, ds

    def spectrum(self, G, k, s):
        self.req(type(s['k']) is int and s['k'] == k and 0 <= k <= len(G), 'spectrum particle count')
        if s.get('empty'):
            self.req(k == 0 and self.f(s['lower']) == 0, 'empty spectrum')
            return F(0), True
        if s.get('full_trace'):
            low = self.total(G[i][i] for i in range(len(G)))
            self.req(k == len(G) and self.f(s['lower']) == low, 'full spectrum trace')
            return low, True
        Z = self.matrix(s['Z'])
        self.req(len(Z) == len(G), 'spectral Z dimension')
        tau = self.f(s['tau'])
        zp = self.proof(Z, s['Z_proof'])[0]
        rp = self.proof(self.plus(self.shift(G, tau), Z), s['residual_proof'])[0]
        lower = self.sub(self.mul(k, tau), self.total(Z[i][i] for i in range(len(Z))))
        self.req(self.f(s['lower']) == lower and type(s['valid_spectral']) is bool and s['valid_spectral'] == (zp and rp), 'spectral lower/validity')
        return lower, zp and rp

    def decode(self, case):
        self.req(set(case) == {'id', 'base', 'modes', 'particles', 'active', 'external', 'scale', 'amplitude', 'direction', 'h', 'density'}, 'unknown/oracle input field')
        n, nu, active, d = case['modes'], case['particles'], case['active'], case['external']
        self.req(type(n) is int and 2 <= n <= 10 and type(nu) is int and 1 <= nu <= n - 1 and type(d) is int and 0 <= d < n, 'input scope')
        self.req(active == [i for i in range(n) if i != d] and all(type(i) is int for i in active), 'ordered partition')
        h = self.matrix(case['h'])
        self.req(len(h) == n, 'one-body dimension')
        den = {}
        for r in case['density']:
            self.req(set(r) == {'i', 'j', 'coefficient'}, 'density schema')
            i, j = r['i'], r['j']
            self.req(type(i) is int and type(j) is int and 0 <= i < n and 0 <= j < n and i != j, 'density indices')
            key = tuple(sorted((i, j))); value = self.f(r['coefficient'])
            self.req(key not in den and value >= 0, 'unordered nonnegative density')
            den[key] = value
        scale = self.f(case['scale'])
        self.req(scale > 0, 'scale')
        return h, den, active, d, nu, self.div(scale, 250)

    def diagonal(self, h, den, occ):
        return self.total([h[i][i] for i in occ] + [x for (i, j), x in den.items() if i in occ and j in occ])

    def upper(self, h, den, active, nu, record):
        dets = list(combinations(active, nu))
        values = [self.diagonal(h, den, occ) for occ in dets]
        expected = [{'kind': 'determinant', 'occupations': [list(occ)], 'value': encode(x), 'relative_sign': 1} for occ, x in zip(dets, values)]
        count = 0
        for i, j in combinations(range(len(dets)), 2):
            if count == 128: break
            pset, qset = set(dets[i]) - set(dets[j]), set(dets[j]) - set(dets[i])
            if len(pset) == len(qset) == 1:
                p, q = next(iter(pset)), next(iter(qset))
                hop = self.mul(h[p][q], (-1) ** (dets[i].index(p) + dets[j].index(q)))
                if hop:
                    count += 1
                    value = self.sub(self.div(self.add(values[i], values[j]), 2), abs(hop))
                    expected.append({'kind': 'pair', 'occupations': [list(dets[i]), list(dets[j])], 'relative_sign': -1 if hop > 0 else 1,
                                     'matrix_element': encode(hop), 'value': encode(value)})
        self.req(record['trials'] == expected and record['determinants'] == len(dets) and record['pairs'] == count, 'coefficient trial enumeration')
        chosen = min(range(len(expected)), key=lambda i: (self.f(expected[i]['value']), i))
        self.req(record['selected'] == chosen and record['U'] == expected[chosen]['value'], 'coefficient upper minimum')
        return self.f(record['U'])

    def bounds(self, h, den, active, d, nu, bounds):
        A = [[h[i][j] for j in active] for i in active]
        m = len(A); b0 = bounds['c0']
        r = min(self.sub(A[i][i], self.total(abs(A[i][j]) for j in range(m) if i != j)) for i in range(m))
        span = self.sub(max(A[i][i] for i in range(m)), min(A[i][i] for i in range(m)))
        self.req(self.f(b0['gershgorin']) == r and len(b0['old_lambda']) == 5, 'Gershgorin/old grid')
        lowers = [r]
        for row, rho in zip(b0['old_lambda'], (F(0), F(1, 20), F(1, 10), F(3, 20), F(1, 5))):
            lam = self.add(r, self.mul(rho, span))
            valid = self.proof(self.shift(A, lam), row['proof'], strict=True)[0]
            self.req(self.f(row['rho']) == rho and self.f(row['lambda']) == lam and row['SPD'] == valid, 'old lambda SPD')
            if valid: lowers.append(lam)
        offsets = [1, 2, 4, 8, 16, 32]
        self.req(len(b0['refined']) <= 6, 'refined cap')
        floor = None
        step = self.div(max([F(1)] + [abs(x) for row in A for x in row]), 1 << 24)
        for row, offset in zip(b0['refined'], offsets):
            lam = self.f(row['lambda'])
            fp = self.add(lam, self.mul(offset, step))
            floor = fp if floor is None else floor
            self.req(row['offset'] == offset and fp == floor and self.div(floor, step).denominator == 1, 'refined candidate grid')
            valid = self.proof(self.shift(A, lam), row['proof'])[0]
            self.req(row['valid'] == valid, 'refined PSD status')
            if valid: lowers.append(lam)
        ell = max(lowers)
        c0 = self.add(h[d][d], self.mul(nu - 1, ell))
        self.req(self.f(b0['ell']) == ell and self.f(b0['c']) == c0, 'c0 lower')
        ext = [den.get(tuple(sorted((i, d))), F(0)) for i in active]
        G = [[self.add(A[i][j], ext[i]) if i == j else A[i][j] for j in range(m)] for i in range(m)]
        bo = bounds['c_occ']
        c_occ = self.bound_spectral(G, nu - 1, h[d][d], bo)
        bd = bounds['c_dh']
        c_dh = None
        if bd['c'] is not None:
            occs = list(combinations(active, nu - 1))
            expected = [{'occupation': list(occ), 'energy': encode(self.diagonal(h, den, (*occ, d)))} for occ in occs]
            self.req(bd['diagonal_rows'] == expected and len(expected) <= 256, 'complete external diagonal minimum')
            minimum = min(self.f(x['energy']) for x in expected)
            self.req(self.f(bd['diagonal_minimum']) == minimum, 'diagonal minimum')
            O = [[A[i][j] if i != j else F(0) for j in range(m)] for i in range(m)]
            c_dh = self.bound_spectral(O, nu - 1, minimum, bd)
        else:
            self.req(bd['witness'] is None and 'UNRESOLVED' in bd['status'], 'missing density lower reason')
        self.req(self.f(bounds['c_star']) == max(x for x in (c0, c_occ, c_dh) if x is not None), 'c-star maximum')
        expected_components = [name for name, value in zip(('c0', 'c_occ', 'c_dh'), (c0, c_occ, c_dh)) if value is not None]
        self.req(bounds['available_star_components'] == expected_components, 'available lower components')
        return A, (c0, c_occ, self.f(bounds['c_star']))

    def bound_spectral(self, G, k, constant, record):
        self.req(len(record['attempts']) <= 6, 'bound spectral pair cap')
        valid_witnesses = []
        for s in record['attempts']:
            lower, valid = self.spectrum(G, k, s)
            if valid: valid_witnesses.append(s)
        if record['c'] is None:
            self.req(record['witness'] is None and not valid_witnesses and 'UNRESOLVED' in record['status'], 'bound unresolved reason')
            return None
        s = record['witness']
        low, valid = self.spectrum(G, k, s)
        best = max(valid_witnesses, key=lambda w: self.f(w['lower'])) if valid_witnesses else None
        self.req(valid and (s.get('empty') or s.get('full_trace') or s == best), 'bound maximum verified witness')
        c = self.add(constant, low)
        self.req(self.f(record['c']) == c, 'spectral external lower')
        return c

    def witness(self, w, A, v, nu, U, c, delta, binding):
        self.req(w['binding'] == binding, 'witness input/source/U/c/delta binding')
        D = self.add(self.sub(c, U), delta)
        k = F(0) if w['kind'] == 'norm' else self.f(w['kappa'])
        if w['kind'] == 'norm':
            low = self.total(self.mul(x, x) for x in v)
            margin = self.sub(self.mul(delta, D), low); psd = True
        elif w['kind'] == 'old':
            mu = self.f(w['mu'])
            M = self.plus(self.shift(self.scale(A, k), mu), self.scale(self.outer(v), -1))
            psd = self.proof(M, w['proof'])[0]
            margin = self.sub(self.mul(delta, D), self.sub(self.mul(k, U), self.mul(nu, mu)))
        elif w['kind'] == 'occ':
            K = self.plus(self.scale(A, k), self.scale(self.outer(v), -1))
            for s in w.get('spectral_attempts', []): self.spectrum(K, nu, s)
            if w['spectral'] is None:
                self.req(not w['accepted'], 'uncertified spectrum accepted')
                return False, False
            lower, psd = self.spectrum(K, nu, w['spectral'])
            margin = self.add(self.sub(lower, self.mul(k, U)), self.mul(delta, D))
        else: raise ValueError('witness kind')
        accepted = c > U and 0 <= k <= D and margin >= 0 and psd
        self.req(self.f(w['D']) == D and self.f(w['affine_margin']) == margin and w['PSD_valid'] == psd and type(w['accepted']) is bool and w['accepted'] == accepted, 'witness exact margins/acceptance')
        return accepted, psd

    def audit_case(self, case, record, expected_source):
        h, den, active, d, nu, delta = self.decode(case)
        case_hash = hash_value(case)
        self.req(record['schema_version'] == 1 and record['input_id'] == case['id'] and record['base'] == case['base'] and record['case_sha256'] == case_hash and record['source_fingerprint'] == expected_source, 'record identity')
        self.req(record['truth_calls'] == 0 and record['next_scientific_stage_authorized'] is False, 'scope')
        self.pool, self.cache = record['proof_pool'], {}
        self.resource_audit(record)
        if not record['scientific_complete']:
            self.req(record['upper'] is None and record['bounds'] is None and 'UNRESOLVED' in record['case_unresolved_reason'], 'whole-case unresolved')
            self.req(all(not m['accepted'] and not m['attempts'] for row in record['raw_rows'] for m in row['methods']), 'unresolved case certificate claim')
            self.req(all(not r['occ_specific_gain'] for r in record['operational_rows']), 'unresolved case gain')
            return False
        U = self.upper(h, den, active, nu, record['upper'])
        A, cvals = self.bounds(h, den, active, d, nu, record['bounds'])
        v = [h[i][d] for i in active]
        self.req(len(record['raw_rows']) == 3 and len(record['operational_rows']) == 3, 'row count')
        for row, c, name in zip(record['raw_rows'], cvals, ('c0', 'c_occ', 'c_star')):
            self.req(row['row'] == name and row['c'] == (encode(c) if c is not None else None) and row['U'] == encode(U) and row['delta'] == encode(delta), 'common row task')
            self.req([m['method'] for m in row['methods']] == ['NORM', 'FIVE', 'OLD_DIRECT', 'OCC'], 'method order')
            for m in row['methods']:
                self.req(type(m['accepted']) is bool, 'method boolean')
                if c is None:
                    self.req(not m['accepted'] and not m['attempts'] and m['status'] == 'UNRESOLVED_LOWER_DEPENDENCY', 'missing common lower')
                    continue
                bind = {'input_sha256': case_hash, 'source_fingerprint': expected_source, 'U': encode(U), 'c': encode(c), 'delta': encode(delta), 'row': name}
                selected = []
                cap = 1 if m['method'] == 'NORM' else 5 if m['method'] == 'FIVE' else 10
                self.req(len(m['attempts']) <= cap, 'candidate cap')
                for w in m['attempts']:
                    ok, _ = self.witness(w, A, v, nu, U, c, delta, bind)
                    if ok: selected.append(w)
                self.req(m['accepted'] == bool(selected) and m['witness'] == (selected[0] if selected else None), 'raw selected certificate')
                if m['method'] == 'NORM': self.req(len(m['attempts']) == 1, 'exact norm verdict missing')
                if m['method'] == 'FIVE':
                    expected = []
                    for lower in record['bounds']['c0']['old_lambda']:
                        if not lower['SPD']: continue
                        lam = self.f(lower['lambda'])
                        _, L, ds = self.proof(self.shift(A, lam), lower['proof'], strict=True)
                        y = []
                        for i in range(len(v)): y.append(self.sub(v[i], self.total(self.mul(L[i][j], y[j]) for j in range(i))))
                        k = self.total(self.div(self.mul(t, t), dd) for t, dd in zip(y, ds))
                        expected.append((encode(k), encode(self.mul(k, lam))))
                    self.req([(w['kappa'], w['mu']) for w in m['attempts']] == expected[:len(m['attempts'])], 'fixed FIVE inverse-quadratic candidates')
                    self.req(len(m['attempts']) == len(expected) or bool(m.get('partial_failure')), 'unexplained truncated FIVE candidates')
                if m['method'] in ('OLD_DIRECT', 'OCC') and m['proposal'] is not None:
                    p = m['proposal']
                    self.req(p['role'] == 'proposal_only' and 1 <= p['calls'] <= 128 and math.isfinite(p['x']) and 0 <= p['x'] <= 1, 'finite proposal budget')
                    candidates = {F(0), F(1)}
                    x = self.q(F(p['x']))
                    for bits in (16, 24, 32, 40):
                        y = self.mul(x, 1 << bits)
                        candidates.update((self.div(math.floor(y), 1 << bits), self.div(math.ceil(y), 1 << bits)))
                    D = self.add(self.sub(c, U), delta)
                    expected_k = [self.mul(D, xr) for xr in sorted(candidates)]
                    self.req([self.f(w['kappa']) for w in m['attempts']] == expected_k[:len(m['attempts'])], 'fixed dyadic kappa candidates')
            self.req(row['raw_base_or'] == any(m['accepted'] for m in row['methods'][:3]), 'raw OR')
        self.operational_audit(record, A, v, nu, U, delta, expected_source)
        return record['operational_rows'][-1]['occ_specific_gain']

    def operational_audit(self, record, A, v, nu, U, delta, source):
        rows = record['raw_rows']
        for index, (row, op) in enumerate(zip(rows, record['operational_rows'])):
            self.req(op['row'] == row['row'], 'operational row identity')
            if row['c'] is None:
                self.req(not op['baseline_star'] and not op['occ_operational'], 'missing lower operational success')
                continue
            c = self.f(row['c']); expected_refs = []
            for origin in rows[:index + 1]:
                if origin['c'] is None or self.f(origin['c']) > c: continue
                for m in origin['methods']:
                    for j, w in enumerate(m['attempts']):
                        if not w.get('PSD_valid') or w.get('spectral', True) is None: continue
                        # Re-evaluate at common target without changing the original proof.
                        k = F(0) if w['kind'] == 'norm' else self.f(w['kappa'])
                        D = self.add(self.sub(c, U), delta)
                        if w['kind'] == 'norm': margin = self.sub(self.mul(delta, D), self.total(self.mul(x, x) for x in v))
                        elif w['kind'] == 'old': margin = self.sub(self.mul(delta, D), self.sub(self.mul(k, U), self.mul(nu, self.f(w['mu']))))
                        else: margin = self.add(self.sub(self.f(w['spectral']['lower']), self.mul(k, U)), self.mul(delta, D))
                        if c > U and 0 <= k <= D and margin >= 0:
                            expected_refs.append({'row': origin['row'], 'method': m['method'], 'attempt': j})
            baseline_refs = [r for r in expected_refs if r['method'] != 'OCC']
            occ_refs = [r for r in expected_refs if r['method'] == 'OCC']
            self.req([w['origin'] for w in op['baseline_certificates']] == baseline_refs and [w['origin'] for w in op['same_method_inherited']] == occ_refs, 'complete retained candidate inheritance')
            bind = {'input_sha256': record['case_sha256'], 'source_fingerprint': source, 'U': encode(U), 'c': encode(c), 'delta': encode(delta), 'row': row['row']}
            for moved in op['baseline_certificates'] + op['same_method_inherited']:
                ref = moved['origin']; original = next(m for r in rows if r['row'] == ref['row'] for m in r['methods'] if m['method'] == ref['method'])['attempts'][ref['attempt']]
                for key in ('kind', 'kappa', 'mu', 'proof', 'spectral'):
                    self.req(moved.get(key) == original.get(key), 'transport altered certificate')
                self.req(self.witness(moved, A, v, nu, U, c, delta, bind)[0], 'invalid transport')
            self.req([f['origin'] for f in op['fallback']] == baseline_refs, 'fallback source list')
            for fallback, baseline in zip(op['fallback'], op['baseline_certificates']):
                self.req(fallback['certificate'] == baseline and self.witness(fallback['embedded_occ'], A, v, nu, U, c, delta, bind)[0], 'explicit OCC embedding')
                embedded = fallback['embedded_occ']
                if baseline['kind'] == 'norm':
                    self.req(self.f(embedded['kappa']) == 0 and self.f(embedded['spectral']['tau']) == 0 and self.matrix(embedded['spectral']['Z']) == self.outer(v), 'norm embedding parameters')
                else:
                    self.req(embedded['kappa'] == baseline['kappa'] and embedded['spectral']['tau'] == baseline['mu'] and all(x == 0 for r in self.matrix(embedded['spectral']['Z']) for x in r), 'old embedding parameters')
            raw_occ = row['methods'][-1]['accepted']
            self.req(op['baseline_star'] == bool(baseline_refs) and op['raw_occ_generated'] == raw_occ and op['occ_specific_gain'] == (raw_occ and not baseline_refs) and op['occ_operational'] == bool(raw_occ or occ_refs or baseline_refs), 'gain attribution/operational verdict')

    def resource_audit(self, record):
        profiles = {'U': (0, 0), 'C0': (1, 11), 'COCC': (1, 12), 'CDH': (1, 12), 'NORM': (0, 0), 'FIVE': (0, 5), 'OLD_DIRECT': (128, 10), 'OCC': (138, 120), 'TRANSPORT': (0, 0), 'FALLBACK': (0, 3), 'SECONDARY': (16, 6)}
        total_eig = total_psd = eig_requests = psd_requests = hits = 0
        ids = set()
        for event in record['events']:
            self.req(event['id'] not in ids and event['kind'] in profiles, 'event identity')
            ids.add(event['id'])
            eig_cap, psd_cap = profiles[event['kind']]
            self.req(type(event['eig']) is int and 0 <= event['eig'] <= eig_cap and type(event['psd']) is int and 0 <= event['psd'] <= psd_cap, 'reserved stage budget')
            total_eig += event['eig']; total_psd += event['psd']
            counted = {'eig': 0, 'psd': 0}
            for r in event['primitive_requests']:
                self.req(r['category'] in counted and type(r['cache_hit']) is bool, 'primitive request')
                eig_requests += r['category'] == 'eig'; psd_requests += r['category'] == 'psd'; hits += r['cache_hit']
                if not r['cache_hit'] and not r.get('denied'): counted[r['category']] += 1
                for k in ('actual_wall_seconds', 'actual_cpu_seconds', 'reference_wall_seconds', 'reference_cpu_seconds'):
                    self.req(isinstance(r[k], (int, float)) and math.isfinite(r[k]) and r[k] >= 0, 'primitive timing')
            self.req(counted == {'eig': event['eig'], 'psd': event['psd']}, 'event primitive counter consistency')
        for event in record['events']: self.req(set(event['dependencies']) <= ids, 'event dependency closure')
        counts = record['resources']['counts']
        self.req(counts['eig'] == total_eig <= 850 and counts['psd'] == total_psd <= 512 and counts['eig_requests'] == eig_requests and counts['psd_requests'] == psd_requests and counts['cache_hits'] == hits, 'case counters')
        self.req(0 <= counts['occ_spectral_pairs'] <= 180 and 0 <= counts['bound_spectral_pairs'] <= 12, 'separate spectral pair caps')
        for result in record.get('standalone_reconstructed_cost', {}).values():
            self.req(set(result['event_ids']) <= ids and 'not a measured cold' in result['qualification'], 'standalone cost qualification')
            selected = set(result['event_ids']); primitive_keys = set(); wall = cpu = 0.
            for event in record['events']:
                if event['id'] not in selected: continue
                self.req(set(event['dependencies']) <= selected, 'charged producer dependency')
                wall += event['overhead_wall_seconds']; cpu += event['overhead_cpu_seconds']
                for request in event['primitive_requests']:
                    if request['key'] not in primitive_keys:
                        primitive_keys.add(request['key']); wall += request['reference_wall_seconds']; cpu += request['reference_cpu_seconds']
            self.req(result['unique_primitives'] == len(primitive_keys) and abs(wall - result['reconstructed_wall_seconds']) < 1e-12 and abs(cpu - result['reconstructed_cpu_seconds']) < 1e-12, 'reconstructed cost sum')

    def secondary(self, case, primary, record, seal):
        self.req(record['input_id'] == case['id'] and record['primary_seal'] == seal, 'secondary seal binding')
        self.req(record['case_sha256'] == primary['case_sha256'] and record['source_fingerprint'] == primary['source_fingerprint'], 'secondary input/source identity')
        eligible = primary['operational_rows'][-1]['occ_specific_gain']
        self.req(record['eligible'] == eligible, 'secondary eligibility')
        s = record['separator']
        if s is None:
            self.req(record['status'] != 'EXACT_SEPARATOR', 'separator missing')
            return False
        self.req(eligible and record['status'] == 'EXACT_SEPARATOR', 'unexpected secondary certificate')
        h, den, active, d, nu, delta = self.decode(case)
        A = [[h[i][j] for j in active] for i in active]; v = [h[i][d] for i in active]
        U, c = self.f(primary['upper']['U']), self.f(primary['bounds']['c_star'])
        self.req(self.f(s['U']) == U and self.f(s['c']) == c and self.f(s['delta']) == delta, 'separator common task')
        D = self.add(self.sub(c, U), delta)
        R = self.shift(A, self.div(U, nu))
        M0 = self.plus(self.scale([[F(i == j) for j in range(len(A))] for i in range(len(A))], self.div(self.mul(delta, D), nu)), self.scale(self.outer(v), -1))
        W = self.matrix(s['W'])
        self.req(len(W) == len(A), 'separator dimension')
        self.pool = record['proof_pool']; self.cache = {}
        valid = self.proof(W, s['proof'])[0]
        tr = self.total(W[i][i] for i in range(len(W)))
        intercept = self.total(self.mul(W[i][j], M0[j][i]) for i in range(len(W)) for j in range(len(W)))
        slope = self.total(self.mul(W[i][j], R[j][i]) for i in range(len(W)) for j in range(len(W)))
        endpoint = self.add(intercept, self.mul(D, slope))
        self.req(valid and tr > 0 and self.f(s['trace']) == tr and [self.f(x) for x in s['endpoint_traces']] == [intercept, endpoint] and max(intercept, endpoint) < 0, 'both strict separator endpoints')
        norm = self.sub(self.mul(delta, D), self.total(self.mul(x, x) for x in v))
        self.req(norm < 0, 'NORM must also be strictly false for family separation')
        self.resource_audit(record)
        pc = primary['resources']['counts']; sc = record['resources']['counts']
        self.req(pc['eig'] + sc['eig'] <= 850 and pc['psd'] + sc['psd'] <= 512, 'primary+secondary case budget')
        return True


def check_saved(pack, records, secondary, expected_source, seal):
    checker = Checker()
    cases = pack['cases']
    checker.req(len(cases) == len(records) == len(secondary), 'complete case coverage')
    gains, separators = [], []
    for case, record in zip(cases, records):
        if checker.audit_case(case, record, expected_source): gains.append(case['base'])
    for case, primary, extra in zip(cases, records, secondary):
        if checker.secondary(case, primary, extra, seal): separators.append(case['base'])
    return {'status': 'INDEPENDENT_SAVED_WITNESSES_PASS', 'checks': checker.checks,
            'max_checker_rational_bits': checker.max_bits, 'inputs': len(cases),
            'gain_bases': sorted(set(gains)), 'separator_bases': sorted(set(separators)),
            'numerical_eigensolver_calls': 0, 'ground_solves': 0, 'optimizer_calls': 0,
            'mandatory_stop': True, 'next_scientific_stage_authorized': False}
