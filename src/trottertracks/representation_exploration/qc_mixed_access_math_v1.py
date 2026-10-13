"""Classical ideal PF primitives. No molecular engine or quantum compiler.

Modes are little endian; spatial ERI are (pq|rs). Only real inputs, <=8 modes.
Function evaluations here are reference arithmetic, never free quantum oracles.
"""
from collections import Counter
from functools import lru_cache
import itertools
import math
import time

import numpy as np


class ContractError(ValueError):
    pass


class Budget:
    def __init__(self, limits=None):
        self.limits = limits or {}
        self.counts = Counter()

    def tick(self, key, amount=1):
        self.counts[key] += amount
        if key in self.limits and self.counts[key] > self.limits[key]:
            raise ContractError('RESOURCE_STOP:' + key)


def validate_input(h, eri, constant):
    h, eri = np.asarray(h), np.asarray(eri)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= len(h) <= 4:
        raise ContractError('SPATIAL_DIMENSION')
    if eri.shape != (len(h),) * 4:
        raise ContractError('ERI_SHAPE')
    if not np.isrealobj(h) or not np.isrealobj(eri):
        raise ContractError('REAL_INPUT_REQUIRED')
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(eri)) or not math.isfinite(constant):
        raise ContractError('FINITE_INPUT_REQUIRED')
    if np.max(np.abs(h-h.T)) > 1e-10:
        raise ContractError('ONE_BODY_HERMITICITY')
    return np.asarray(h, float), np.asarray(eri, float), float(constant)


@lru_cache(maxsize=65536)
def normal_form(ops):
    """Integer CAR rewrite; creators ascending, annihilators descending."""
    for i in range(len(ops)-1):
        p, a = ops[i]; q, b = ops[i+1]
        if a == b and p == q:
            return ()
        violation = a < b or (a == b == 1 and p > q) or (a == b == 0 and p < q)
        if violation:
            swapped = ops[:i] + (ops[i+1], ops[i]) + ops[i+2:]
            result = Counter({k: -v for k, v in normal_form(swapped)})
            if a == 0 and b == 1 and p == q:
                for k, v in normal_form(ops[:i] + ops[i+2:]):
                    result[k] += v
            return tuple(sorted((k, v) for k, v in result.items() if v))
    return ((ops, 1),)


def raw_terms(h, eri, constant):
    ns = len(h)
    yield (), constant
    for p, q in itertools.product(range(ns), repeat=2):
        for spin in (0, 1):
            if h[p, q] != 0:
                yield ((2*p+spin, 1), (2*q+spin, 0)), float(h[p, q])
    for p, q, r, s in itertools.product(range(ns), repeat=4):
        if eri[p, q, r, s] == 0:
            continue
        for sigma, tau in itertools.product((0, 1), repeat=2):
            yield ((2*p+sigma, 1), (2*r+tau, 1),
                   (2*s+tau, 0), (2*q+sigma, 0)), float(eri[p, q, r, s])/2


def canonical_terms(h, eri, constant):
    result = {}
    for ops, coefficient in raw_terms(h, eri, constant):
        for key, factor in normal_form(ops):
            result[key] = result.get(key, 0.) + factor*coefficient
    return {k: v for k, v in result.items() if v != 0}


def ladder_action(n, ops):
    states = np.arange(1 << n, dtype=np.int64)
    dest = states.copy(); sign = np.ones(len(states)); valid = np.ones(len(states), bool)
    for mode, create in reversed(ops):
        occupied = (dest >> mode) & 1
        valid &= occupied == (0 if create else 1)
        parity = np.array([(int(x) & ((1 << mode)-1)).bit_count() & 1 for x in dest])
        sign *= 1-2*parity
        dest ^= 1 << mode
    return states[valid], dest[valid], sign[valid]


def term_matrix(n, terms):
    matrix = np.zeros((1 << n, 1 << n), complex)
    for ops, coefficient in terms.items():
        src, dst, sign = ladder_action(n, ops)
        matrix[dst, src] += coefficient*sign
    return matrix


def spin_matrix(spatial):
    return np.kron(spatial, np.eye(2))


def one_body_matrix(spin):
    return term_matrix(len(spin), {((p,1),(q,0)): float(spin[p,q])
                                  for p, q in itertools.product(range(len(spin)), repeat=2)
                                  if spin[p,q] != 0})


def excitation_groups(n, terms):
    eps = np.zeros(n); density = np.zeros((n,n)); scalar = terms.get((), 0.)
    groups = {}; max_hermiticity = 0.
    for ops, value in terms.items():
        if not ops:
            continue
        p = tuple(i for i, a in ops if a); q = tuple(i for i, a in ops if not a)
        ps, qs = set(p), set(q)
        if ps == qs:
            if len(p) == 1:
                eps[p[0]] += value
            elif len(p) == 2:
                density[p[0],p[1]] += value; density[p[1],p[0]] += value
            else:
                raise ContractError('DIAGONAL_DEGREE')
            continue
        common = ps & qs
        create, annih = tuple(sorted(ps-common)), tuple(sorted(qs-common))
        # Orient each Hermitian transition once; reverse orientation is checked.
        if create > annih:
            continue
        adj = tuple((i, 1-a) for i,a in reversed(ops))
        adj_nf = normal_form(adj)
        if len(adj_nf) != 1:
            raise ContractError('ADJOINT_NORMAL_FORM')
        adj_key, adj_sign = adj_nf[0]
        other = terms.get(adj_key, 0.)*adj_sign
        max_hermiticity = max(max_hermiticity, abs(value-other))
        coefficient = (value+other)/2
        key = (create, annih)
        group = groups.setdefault(key, {'create': create, 'annihilate': tuple(reversed(annih)),
                                       'g0': 0., 'g_weights': {}})
        if common:
            if len(common) != 1 or len(create) != 1:
                raise ContractError('UNSUPPORTED_GROUP')
            spectator = next(iter(common))
            form = normal_form(((spectator,1),(spectator,0),(create[0],1),(annih[0],0)))
            assert len(form) == 1 and form[0][0] == ops
            coefficient /= form[0][1]
            group['g_weights'][spectator] = group['g_weights'].get(spectator, 0.)+coefficient
        else:
            group['g0'] += coefficient
    if max_hermiticity > 1e-10:
        raise ContractError('CANONICAL_HERMITICITY')
    states = np.arange(1 << n); occupations = ((states[:,None] >> np.arange(n)) & 1).astype(float)
    diagonal = occupations @ eps
    for i in range(n):
        for j in range(i+1,n):
            diagonal += density[i,j]*occupations[:,i]*occupations[:,j]
    enriched = []
    for key in sorted(groups):
        group = groups[key]; p = group['create']; q = group['annihilate']
        src, dst, signs = ladder_action(n, tuple((i,1) for i in p)+tuple((i,0) for i in q))
        amplitude = np.full(len(src), group['g0'])
        for spectator, weight in group['g_weights'].items():
            amplitude += weight*((src >> spectator)&1)
        support = sorted(set(p+q)); u = {i:int(i in p) for i in support}; v = {i:int(i in q) for i in support}
        delta0 = sum(eps[i]*(u[i]-v[i]) for i in support)
        delta0 += sum(density[i,j]*(u[i]*u[j]-v[i]*v[j]) for i in support for j in support if i<j)
        weights = {l:sum(density[l,i]*(u[i]-v[i]) for i in support) for l in range(n) if l not in support}
        parity_mask = [l for l in range(n) if l not in support and sum(i>l for i in p+q)%2]
        enriched.append({**group, 'src':src, 'dst':dst, 'amplitude':amplitude*signs,
                         'delta':diagonal[dst]-diagonal[src], 'delta0':float(delta0),
                         'delta_weights':weights, 'parity_mask':parity_mask})
    return eps, density, scalar, diagonal, enriched


def pauli_terms(n, terms):
    """JW algebra in binary symplectic words; phase multiplication exact."""
    def mul(left, right):
        out = {}
        for (x,z), a in left.items():
            for (y,w), b in right.items():
                key = (x^y,z^w)
                out[key] = out.get(key,0j) + a*b*((-1)**((z&y).bit_count()))
        return out
    accum = {}
    for ops, coefficient in terms.items():
        polynomial = {(0,0):complex(coefficient)}
        for mode, create in ops:
            mask = (1 << mode)-1
            # a=(X+iY)/2; XZ=-iY, so a=(X-XZ)/2.
            ladder = {(1<<mode,mask):.5, (1<<mode,mask|(1<<mode)):(.5 if create else -.5)}
            polynomial = mul(polynomial,ladder)
        for key, value in polynomial.items():
            accum[key] = accum.get(key,0j)+value
    result = []
    for (x,z), value in sorted(accum.items()):
        coefficient = value / (1j**((x&z).bit_count()))
        if abs(coefficient.imag) > 1e-10:
            raise ContractError('PAULI_HERMITICITY')
        if coefficient.real != 0:
            result.append((x,z,float(coefficient.real)))
    return result


def qr_plan(u):
    work = np.array(u, float, copy=True); rotations = []
    for column in range(len(u)-1):
        for row in range(len(u)-1,column,-1):
            a,b = work[row-1,column],work[row,column]
            radius = math.hypot(a,b)
            if radius == 0:
                continue
            c,s = a/radius,b/radius
            top,bottom = work[row-1].copy(),work[row].copy()
            work[row-1] = c*top+s*bottom; work[row] = -s*top+c*bottom
            rotations.append((row-1,row,c,s))
    if np.linalg.norm(work-np.diag(np.diag(work)),ord='fro') > 1e-9:
        raise ContractError('GAUSSIAN_QR_RESIDUAL')
    signs = np.where(np.diag(work)<0,-1.,1.)
    operations = [('sign',2*i+spin) for i,d in enumerate(signs) if d<0 for spin in (0,1)]
    for i,j,c,s in reversed(rotations):
        operations += [('givens',2*i+spin,2*j+spin,c,-s) for spin in (0,1)]
    return operations


def gaussian_apply(rows, operations, inverse=False, budget=None):
    out = rows.copy(); n = (len(rows)-1).bit_length(); states = np.arange(len(rows))
    for op in (reversed(operations) if inverse else operations):
        if budget: budget.tick('primitive_applications')
        if op[0] == 'sign':
            out *= (1-2*((states>>op[1])&1))[:,None]
        else:
            _,p,q,c,s = op; s = -s if inverse else s
            ip = states[((states>>p)&1)==1]; ip = ip[((ip>>q)&1)==0]
            iq = ip ^ ((1<<p)|(1<<q))
            mask = ((1<<q)-1)^((1<<(p+1))-1)
            sign = np.array([(-1)**((int(i)&mask).bit_count()) for i in ip])
            a,b = out[ip].copy(),out[iq].copy()
            out[ip] = c*a+s*sign[:,None]*b; out[iq] = -s*sign[:,None]*a+c*b
    return out


def fock_frame(u):
    """Independent determinant reference, batched minors by particle number."""
    n = len(u); out = np.zeros((1<<n,1<<n),complex); out[0,0] = 1
    for k in range(1,n+1):
        subsets = list(itertools.combinations(range(n),k)); labels = [sum(1<<i for i in s) for s in subsets]
        indices = np.array(subsets)
        minors = u[indices[:,None,:,None],indices[None,:,None,:]]
        out[np.ix_(labels,labels)] = np.linalg.det(minors)
    return out


def factorize(h, eri, budget):
    ns = len(h); n = 2*ns; supermatrix = eri.reshape(ns*ns,ns*ns)
    budget.tick('DF_small_matrix_eigh'); eigenvalues,vectors = np.linalg.eigh((supermatrix+supermatrix.T)/2)
    correction = h-.5*np.einsum('pqqs->ps',eri)
    budget.tick('DF_small_matrix_eigh'); eta,u = np.linalg.eigh((correction+correction.T)/2)
    occupations = ((np.arange(1<<n)[:,None] >> np.arange(n)) & 1).astype(float)
    actors = [{'kind':'gaussian','operations':qr_plan(u), 'energy':occupations @ np.repeat(eta,2),
               'eta':np.repeat(eta,2), 'weight':None, 'orbital_matrix':u}]
    factors = []
    order = sorted(range(len(eigenvalues)),key=lambda i:(-abs(eigenvalues[i]),i))
    for index in order:
        lam = float(eigenvalues[index]); f = vectors[:,index].reshape(ns,ns); f = (f+f.T)/2
        budget.tick('DF_small_matrix_eigh'); spectrum,frame = np.linalg.eigh(f)
        spun = np.repeat(spectrum,2)
        actors.append({'kind':'gaussian','operations':qr_plan(frame),
                       'energy':.5*lam*(occupations @ spun)**2, 'eta':spun, 'weight':lam,
                       'orbital_matrix':frame})
        factors.append((lam,f))
    reconstructed = sum((lam*np.einsum('pq,rs->pqrs',f,f) for lam,f in factors), np.zeros_like(eri))
    return actors, factors, float(np.max(np.abs(reconstructed-eri)))


def build_model(h, eri, constant, budget=None):
    budget = budget or Budget(); h,eri,constant = validate_input(h,eri,constant); n = 2*len(h)
    started=time.perf_counter(); timings={}
    terms = canonical_terms(h,eri,constant); timings['normal_ordering_seconds']=time.perf_counter()-started
    started=time.perf_counter(); eps,j,scalar,diagonal,groups = excitation_groups(n,terms)
    timings['excitation_descriptor_seconds']=time.perf_counter()-started
    started=time.perf_counter(); paulis = pauli_terms(n,terms)
    timings['Pauli_descriptor_seconds']=time.perf_counter()-started
    started=time.perf_counter(); gaussian,factors,df_residual = factorize(h,eri,budget)
    timings['DF_basis_seconds']=time.perf_counter()-started
    started=time.perf_counter()
    original = term_matrix(n,terms); rebuilt = np.diag(diagonal+scalar).astype(complex)
    for group in groups:
        a,b,g = group['src'],group['dst'],group['amplitude']; rebuilt[b,a] += g; rebuilt[a,b] += g
    if np.linalg.norm(original-rebuilt,ord='fro') > 1e-8:
        raise ContractError('EXCITATION_RECONSTRUCTION')
    if np.linalg.norm(original-original.conj().T,'fro')>1e-10:
        raise ContractError('ORIGINAL_HERMITICITY')
    particle_counts=np.array([i.bit_count() for i in range(1<<n)])
    if np.linalg.norm(original*(particle_counts[:,None]-particle_counts[None,:]),'fro')>1e-10:
        raise ContractError('ORIGINAL_PARTICLE_CONSERVATION')
    if df_residual > 1e-10:
        raise ContractError('DF_COEFFICIENT_RECONSTRUCTION')
    timings['matrix_identity_seconds']=time.perf_counter()-started
    return {'n':n,'h':h,'eri':eri,'scalar':scalar,'terms':terms,'eps':eps,'density':j,'preprocessing_timings':timings,
            'diagonal':diagonal,'groups':groups,'paulis':paulis,'gaussian':gaussian,
            'factors':factors,'H':original,'df_coefficient_residual':df_residual}


def stages(method, time, model):
    """Chronological stage IR;actor exponential fusion only where valid."""
    if method.endswith('4'):
        s = 1/(4-4**(1/3)); out = []
        for weight in (s,s,1-4*s,s,s):
            out.extend(stages(method[0]+'2',time*weight,model))
        if method=='D4':
            # Move commuting nuclear scalar phases out of the palindrome so
            # the same DF-fragment fusion available to E/R is also applied.
            scalar_time=sum(node[2] for node in out if node[0]=='SCALAR')
            out=[('SCALAR',0,scalar_time,False)]+[node for node in out if node[0]!='SCALAR']
        return fuse(out)
    if method == 'R2':
        t = time/2; m = len(model['groups'])
        # P(-t)^dagger first, then P(t).
        out = [('A',0,t,False)] + [('R',j,-t,True) for j in range(m)]
        out += [('R',j,t,False) for j in reversed(range(m))] + [('A',0,t,False)]
        return fuse(out)
    kinds = [('A',0)]+[('E',j) for j in range(len(model['groups']))] if method=='E2' else (
        [('D',j) for j in range(len(model['gaussian']))] if method=='D2' else (
            [('P',j) for j in range(len(model['paulis']))] if method=='P2' else None))
    if kinds is None:
        raise ContractError('METHOD')
    out = [(kind,index,time/2,False) for kind,index in kinds+list(reversed(kinds))]
    if method=='D2': out = [('SCALAR',0,time,False)]+out
    return fuse(out)


def fuse(nodes):
    out = []
    for node in nodes:
        kind,index,t,adj = node
        if t == 0:
            continue
        if out and kind!='R' and out[-1][:2]==node[:2] and out[-1][3]==adj:
            previous = out.pop(); t += previous[2]
            if t == 0: continue
            node = (kind,index,t,adj)
        out.append(node)
    return out


def apply_stage(rows, node, model, budget=None):
    if budget: budget.tick('primitive_applications')
    kind,index,t,adj = node; out = np.array(rows,dtype=complex,copy=True)
    if kind=='SCALAR': return np.exp(-1j*t*model['scalar'])*out
    if kind=='A': return np.exp(-1j*t*(model['diagonal']+model['scalar']))[:,None]*out
    if kind=='P':
        x,z,g = model['paulis'][index]; states = np.arange(len(out)); perm = states^x
        phase = np.array([(1j**((x&z).bit_count()))*((-1)**((int(i)&z).bit_count())) for i in perm])
        return math.cos(t*g)*out-1j*math.sin(t*g)*phase[:,None]*out[perm]
    if kind=='D':
        actor = model['gaussian'][index]
        out = gaussian_apply(out,actor['operations'],True,budget)
        out *= np.exp(-1j*t*actor['energy'])[:,None]
        return gaussian_apply(out,actor['operations'],False,budget)
    group = model['groups'][index]; a,b,g = group['dst'],group['src'],group['amplitude']
    if kind=='E':
        c = np.cos(t*g); s = -1j*np.sin(t*g)
        aa,bb = out[a].copy(),out[b].copy()
        out[a]=c[:,None]*aa+s[:,None]*bb; out[b]=s[:,None]*aa+c[:,None]*bb
        return out
    if kind!='R': raise ContractError('ACTOR')
    d = group['delta']/2; rho = np.hypot(d,g)
    f = np.divide(np.sin(t*rho),rho,out=np.full_like(rho,t),where=rho!=0)
    c = np.cos(t*rho); phase = np.exp(1j*t*d)
    c00=phase*(c-1j*f*d); c01=-1j*phase*f*g
    c10=-1j*phase.conjugate()*f*g; c11=phase.conjugate()*(c+1j*f*d)
    if adj: c00,c01,c10,c11=c00.conjugate(),c10.conjugate(),c01.conjugate(),c11.conjugate()
    aa,bb=out[a].copy(),out[b].copy()
    out[a]=c00[:,None]*aa+c01[:,None]*bb; out[b]=c10[:,None]*aa+c11[:,None]*bb
    return out


def seed_matrix(method,time,model,budget=None):
    rows = np.eye(1<<model['n'],dtype=complex)
    for node in stages(method,time,model): rows=apply_stage(rows,node,model,budget)
    return rows


def power_doubling(seed, exponent, budget=None):
    if type(exponent) is not int or exponent<1 or exponent&(exponent-1):
        raise ContractError('DOUBLING_POWER')
    out=seed.copy()
    for _ in range(exponent.bit_length()-1):
        if budget: budget.tick('dense_power_products')
        out=out@out
    return out


def taylor_reference(H,time,budget=None):
    if budget: budget.tick('reference_matrices')
    a=-1j*time*H; norm=float(np.linalg.norm(a,ord=1))
    squarings=max(0,math.ceil(math.log2(2*norm))) if norm else 0
    if squarings>32: raise ContractError('TAYLOR_SCALING_CAP')
    a=a/(2**squarings); result=np.eye(len(a),dtype=complex); term=result.copy()
    degree=0
    for k in range(1,129):
        term=term@a/k; result+=term; degree=k
        if np.linalg.norm(term,ord=1)<1e-18: break
    scaled=float(np.linalg.norm(a,ord=1))
    tail=math.exp(scaled)*scaled**(degree+1)/math.factorial(degree+1)
    for _ in range(squarings): result=result@result
    return result,{'degree':degree,'squarings':squarings,'scaled_analytic_tail':tail,
                   'qualification':'binary64 crosscheck,not outward-roundoff certificate'}


def norm_certificate(error,budget=None):
    if budget: budget.tick('norm_SVD_including_checks')
    _,sigma,vh=np.linalg.svd(error,full_matrices=False)
    return float(sigma[0]),sigma,vh
