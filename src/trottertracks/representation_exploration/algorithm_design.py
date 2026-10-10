"""Finite N1/N2 construction feasibility; coefficient-only N3 certification.

Small synthetic development only. No production factor fitting, molecular data,
RTE-law changes, fault-tolerant synthesis or scalable/global optimum claims.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
import resource
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from .construction_comparison import (
    DiagonalBlock, apply_pauli, gaussian_circuit as legacy_gaussian_circuit, orbital_paulis, padd,
    pmul, pscale, real_terms, wrapper,
)
from .mechanisms import matrix

SEED = 20261011
METRICS = ('rz', 'cx', 'sx', 'x', 'size', 'depth')
TOL = 3e-9
N1_TIME, N1_BUDGET, EPSILON = .6, .02, .04
Q_GRID = (1, 2, 4, 8)


def opnorm(a):
    return float(np.linalg.norm(a, 2))


def second_quantize_one_body(g):
    """Bounded JW action for this 4/6-mode batch; older 3-mode path stays frozen."""
    g = np.asarray(g, complex); n = len(g)
    if g.shape != (n, n) or not 1 <= n <= 6 or not np.all(np.isfinite(g)):
        raise ValueError('finite square one-body matrix, at most six modes, required')
    out = np.zeros((1 << n, 1 << n), complex)
    for state in range(1 << n):
        for q in range(n):
            if not (state >> q & 1):
                continue
            removed = state ^ (1 << q)
            sign_q = (-1) ** ((state & ((1 << q)-1)).bit_count())
            for p in range(n):
                if removed >> p & 1:
                    continue
                sign_p = (-1) ** ((removed & ((1 << p)-1)).bit_count())
                out[removed | (1 << p), state] += sign_q*sign_p*g[p, q]
    return out


def sector_indices(n, particles):
    if not 0 <= particles <= n:
        raise ValueError('invalid particle count')
    return [s for s in range(1 << n) if s.bit_count() == particles]


def sector_norm(values, particles):
    """Exact mathematical occupation norm; binary64 evaluation, no rounding certificate."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError('finite real spectrum required')
    sector_indices(len(values), particles)
    ordered = np.sort(values)
    if particles == 0:
        return 0.
    return max(abs(float(np.sum(ordered[:particles]))),
               abs(float(np.sum(ordered[-particles:]))))


def square_bound(eta, approximate, particles, weight=1.):
    return abs(weight) * sector_norm(np.asarray(eta)-approximate, particles) * (
        sector_norm(eta, particles)+sector_norm(approximate, particles))


def partitions(n):
    for cuts in itertools.product((False, True), repeat=n-1):
        groups, start = [], 0
        for i, cut in enumerate(cuts, 1):
            if cut:
                groups.append(list(range(start, i))); start = i
        yield groups + [list(range(start, n))]


def subspace_frame(u, groups):
    """Coordinate-projector Gram-Schmidt within each degenerate subspace.

    Column assignment and phases are applied to ALL candidate families. No
    many-body truth information is used. This is a finite gauge heuristic.
    """
    n = len(u); frame = np.zeros_like(u, dtype=complex)
    for group in groups:
        projector = u[:, group] @ u[:, group].conj().T
        chosen = []
        for i in range(n):
            v = projector[:, i].copy()
            for _ in range(2):
                for a in chosen:
                    v -= a * np.vdot(a, v)
            length = np.linalg.norm(v)
            if length > 1e-11:
                chosen.append(v/length)
            if len(chosen) == len(group):
                break
        if len(chosen) != len(group):
            raise ValueError('ill-conditioned projector completion')
        frame[:, group] = np.column_stack(chosen)
    if opnorm(frame.conj().T@frame-np.eye(n)) > 1e-10:
        raise ValueError('non-unitary frame')
    return frame


def ordered_frame(frame, eta):
    rows, cols = linear_sum_assignment(-abs(frame)**2)
    assert np.array_equal(rows, np.arange(len(frame)))
    frame, eta = np.asarray(frame[:, cols],complex).copy(), np.asarray(eta)[cols].copy()
    for j in range(len(frame)):
        i = int(np.argmax(abs(frame[:, j])))
        frame[:, j] *= np.exp(-1j*np.angle(frame[i, j]))
    return frame, eta


def gaussian_circuit(u):
    """Local absolute Gamma(U) convention: number-preserving vacuum phase is one.

    The reused helper's RZ decomposition can leave a global phase. Repair from
    vacuum action only; no change to old helper/source or dense-unitary fallback.
    """
    qc=legacy_gaussian_circuit(u)
    vacuum=Statevector.from_int(0,1<<len(u)).evolve(qc).data
    if np.linalg.norm(vacuum[1:])>1e-12 or abs(abs(vacuum[0])-1)>1e-12:
        raise ValueError('Gaussian helper failed number-preserving vacuum check')
    qc.global_phase-=float(np.angle(vacuum[0]))
    return qc


def spectral_candidates(g, particles, weight, budget):
    if not np.allclose(g, g.conj().T, atol=1e-12, rtol=0):
        raise ValueError('Hermitian factor required')
    eta, u = np.linalg.eigh(g); n = len(eta)
    specifications = [('exact', eta.copy(), [[i] for i in range(n)])]
    for groups in partitions(n):
        et = eta.copy()
        for group in groups:
            et[group] = float(np.mean(eta[group]))
        specifications.append(('cluster', et, groups))
    # Strong finite shifted/truncated comparator: every subset, not a chosen rank.
    for bits in itertools.product((0, 1), repeat=n):
        chosen = [i for i, b in enumerate(bits) if b]
        if chosen:
            et = eta.copy(); et[chosen] = float(np.mean(eta[chosen]))
            specifications.append(('shifted_cutoff', et,
                                   [chosen]+[[i] for i in range(n) if i not in chosen]))
        et = eta.copy(); et[chosen] = 0.
        specifications.append(('zero_cutoff', et,
                               ([chosen] if chosen else [])+[[i] for i in range(n) if i not in chosen]))
    out, seen = [], set()
    for family, et, groups in specifications:
        bound = square_bound(eta, et, particles, weight)
        if bound > budget:
            continue
        key = (family, tuple(et))
        if key in seen:
            continue
        seen.add(key)
        gauge = subspace_frame(u, groups)
        frame, reordered = ordered_frame(gauge, et)
        original, original_eta = ordered_frame(u, et)
        approx = frame@np.diag(reordered)@frame.conj().T
        same = u@np.diag(et)@u.conj().T
        if opnorm(approx-same) > 1e-10:
            raise AssertionError('gauge changed approximated factor')
        out.append({'family':family, 'eta':eta.tolist(), 'approx_eta':reordered.tolist(),
                    'frame':frame, 'ungauged_frame':original, 'ungauged_eta':original_eta,
                    'g':approx, 'groups':groups, 'bound':bound,
                    'eigen_reconstruction_error':opnorm(u@np.diag(eta)@u.conj().T-g),
                    'gauge_residual':opnorm(approx-same)})
    return out


def fock_frame(u):
    n = len(u); occ = [[i for i in range(n) if s >> i & 1] for s in range(1 << n)]
    out = np.zeros((1 << n, 1 << n), complex)
    for i, a in enumerate(occ):
        for j, b in enumerate(occ):
            if len(a) == len(b):
                out[i, j] = np.linalg.det(u[np.ix_(a, b)]) if a else 1.
    return out


def embed(logical, system, workspace=0, controlled=True):
    rows = (1 << (system+workspace+(1 if controlled else 0)))
    cols = 1 << (system+(1 if controlled else 0))
    ans = np.zeros((rows, cols), complex)
    for s in range(1 << system):
        for a in range(2 if controlled else 1):
            ans[s+(a << (system+workspace)), :] = logical[s+(a << system), :]
    return ans


def native_action(ir, columns):
    out = columns.copy(); n = ir['qubits']
    indices = np.arange(1 << n)
    for item in ir['operations']:
        name, qs, params = item['name'], item['qubits'], item['parameters']
        target = qs[-1]; lo = indices[(indices & (1 << target)) == 0]; hi = lo | (1 << target)
        if name == 'rz':
            out[lo] *= np.exp(-.5j*params[0]); out[hi] *= np.exp(.5j*params[0])
        elif name == 'sx':
            a, b = out[lo].copy(), out[hi].copy()
            out[lo] = ((1+1j)*a+(1-1j)*b)/2
            out[hi] = ((1-1j)*a+(1+1j)*b)/2
        elif name in ('x', 'cx'):
            if name == 'cx':
                lo = lo[(lo & (1 << qs[0])) != 0]; hi = lo | (1 << target)
            a = out[lo].copy(); out[lo], out[hi] = out[hi], a
        else:
            raise ValueError('non-native gate')
    return np.exp(1j*ir['global_phase']) * out


class Audit:
    def __init__(self):
        self.records = []

    def compile(self, qc, logical, system, label, workspace=0, controlled=True):
        if len(self.records) >= 384 or qc.num_qubits > 13:
            raise ValueError('local circuit/qubit cap')
        initial = embed(np.eye(len(logical)), system, workspace, controlled)
        target = embed(logical, system, workspace, controlled)
        if qc.num_qubits <= 6:
            built = Operator(qc).data @ initial
        else:
            built = np.column_stack([Statevector(initial[:, j]).evolve(qc).data
                                     for j in range(initial.shape[1])])
        built_error = opnorm(built-target)
        if built_error > TOL:
            raise AssertionError((label, 'built', built_error))
        compiled = transpile(qc, basis_gates=['rz','sx','x','cx'], optimization_level=1,
                             seed_transpiler=SEED, num_processes=1, qubits_initially_zero=False)
        if compiled.size() > 20000:
            raise ValueError('native gate cap')
        ir = {'label':label, 'qubits':qc.num_qubits,
              'qubits_initially_zero':False,
              'global_phase':float(compiled.global_phase), 'operations':[
                  {'name':x.operation.name, 'qubits':[compiled.find_bit(q).index for q in x.qubits],
                   'parameters':[float(v) for v in x.operation.params]} for x in compiled.data],
              'system':system, 'workspace':workspace, 'controlled':controlled,
              'logical_reference':matrix(logical)}
        raw_action = native_action(ir, initial)
        raw_error = opnorm(raw_action-target)
        # Qiskit 1.3 can lose a scalar phase in optimization. Only a scalar
        # mismatch against the independently checked built action is repairable;
        # relative controlled phase or workspace leakage must still fail.
        overlap = np.vdot(built, raw_action)/initial.shape[1]
        scalar_residual = opnorm(raw_action-overlap*built)
        phase_repair = 0.
        if raw_error > TOL:
            if abs(abs(overlap)-1) > TOL or scalar_residual > TOL:
                raise AssertionError((label, 'non-scalar compiler discrepancy', raw_error))
            phase_repair = -float(np.angle(overlap))
            ir['global_phase'] += phase_repair
        ir['compiler_phase_audit'] = {'raw_absolute_error':raw_error,
                                     'scalar_residual':scalar_residual,
                                     'global_phase_correction':phase_repair}
        error = opnorm(native_action(ir, initial)-target)
        if error > TOL:
            raise AssertionError((label, 'absolute native/clean-workspace action', error))
        ir['sha256'] = hashlib.sha256(json.dumps(ir, sort_keys=True, separators=(',',':')).encode()).hexdigest()
        self.records.append(ir)
        counts = compiled.count_ops()
        return {'label':label, 'ir_sha256':ir['sha256'], **{m:int(counts.get(m,0)) for m in METRICS[:4]},
                'size':compiled.size(), 'depth':compiled.depth(), 'qubits':qc.num_qubits,
                'workspace':workspace, 'built_residual':built_error, 'native_residual':error,
                'raw_native_residual':raw_error,'compiler_global_phase_correction':phase_repair}


def rotation(n, i, j, angle):
    v = np.eye(n); c, s = math.cos(angle), math.sin(angle)
    v[i,i] = v[j,j] = c; v[i,j] = -s; v[j,i] = s
    return v


def n1_input(kind):
    n = 4; w = rotation(n,0,1,.6)@rotation(n,2,3,-.4)
    v = rotation(n,0,2,.35)@rotation(n,1,3,-.22)
    if kind != 'planted_local_gauge':
        v = v@rotation(n,0,3,.29)@rotation(n,1,2,-.37)
    v2 = rotation(n,0,3,.31)@rotation(n,1,2,.27)@rotation(n,0,2,-.17)
    spectra = ([1.,1.002,-.4,-.397], [.7,.703,-.2,-.196])
    if kind == 'separated_spectrum':
        spectra = ([1.,1.25,-.4,-.1], [.7,.95,-.2,.1])
    frames = [v@w, v2@rotation(n,0,1,-.43)@rotation(n,2,3,.33)]
    return [u@np.diag(e)@u.T for u,e in zip(frames,spectra)], [.7,-.4], np.diag([.08,-.04,.03,-.02]), .11


def square_terms(eta, weight):
    p = orbital_paulis(np.diag(eta)); return real_terms(pscale(pmul(p,p), weight))


def df_operator(gs, weights, h1, scalar):
    n = len(h1); out = second_quantize_one_body(h1)+scalar*np.eye(1 << n)
    for g,w in zip(gs,weights,strict=True):
        f = second_quantize_one_body(g); out += w*(f@f)
    return out


def product_circuit(candidates, weights, h1, scalar, time_total, q, paulis=None):
    n = len(h1); anc = n; qc = QuantumCircuit(n+1)
    if paulis is None:
        blocks = [DiagonalBlock('h1',real_terms(orbital_paulis(h1)),QuantumCircuit(n))]
        blocks += [DiagonalBlock(str(i),square_terms(c['approx_eta'],w),gaussian_circuit(c['frame']))
                   for i,(c,w) in enumerate(zip(candidates,weights))]
    else:
        blocks = [DiagonalBlock(p,{p:c},QuantumCircuit(n)) for p,c in paulis.items()]
    op = np.eye(1 << n, dtype=complex); dt = time_total/q
    for _ in range(q):
        for block in blocks + list(reversed(blocks)):
            block.apply(qc,anc,dt/2); op = expm(-.5j*dt*block.dense())@op
    if paulis is None:
        qc.p(-time_total*scalar,anc); op *= np.exp(-1j*time_total*scalar)
    return qc, op


def precision(bias, xy_cost, epsilon=EPSILON):
    margin = epsilon/math.sqrt(2)-bias
    shots = math.ceil(2*math.log(80)/margin**2) if margin > 0 else None
    return {'epsilon_complex':epsilon, 'axis_epsilon':epsilon/math.sqrt(2),
            'normalization':1., 'bias_budget_used':bias, 'shots_per_axis':shots,
            'xy_cost':xy_cost, 'work':None if shots is None else {m:shots*xy_cost[m] for m in METRICS}}


def run_n1(kind):
    audit = Audit(); gs, weights, h1, scalar = n1_input(kind); n, nu = 4, 2
    lists = [spectral_candidates(g,nu,w,N1_BUDGET) for g,w in zip(gs,weights)]
    for factor, candidates in enumerate(lists):
        for i,c in enumerate(candidates):
            qc = gaussian_circuit(c['frame'])
            c['basis_cost'] = audit.compile(qc,fock_frame(c['frame']),n,f'N1/{kind}/basis/{factor}/{i}',controlled=False)
    def choose(families, metric):
        combos = [p for p in itertools.product(*lists) if all(c['family'] in families for c in p)
                  and sum(c['bound'] for c in p) <= N1_BUDGET]
        return min(combos,key=lambda p:(sum(c['basis_cost'][metric] for c in p),
                                       sum(c['basis_cost']['cx' if metric=='rz' else 'rz'] for c in p),
                                       sum(c['bound'] for c in p)))
    chosen = [('native_exact', choose({'exact'},'rz')),
              ('N1_min_basis_rz', choose({'exact','cluster'},'rz')),
              ('N1_min_basis_cx', choose({'exact','cluster'},'cx')),
              ('shifted_cutoff_min_basis_rz',choose({'exact','shifted_cutoff'},'rz')),
              ('shifted_cutoff_min_basis_cx',choose({'exact','shifted_cutoff'},'cx')),
              ('zero_cutoff_min_basis_rz',choose({'exact','zero_cutoff'},'rz'))]
    paired = [dict(c,frame=c['ungauged_frame'],approx_eta=c['ungauged_eta'].tolist()) for c in chosen[1][1]]
    chosen.append(('N1_same_approx_ungauged',paired))
    full_terms = real_terms(padd(orbital_paulis(h1),{'I'*n:scalar},
                                *[pscale(pmul(orbital_paulis(g),orbital_paulis(g)),w) for g,w in zip(gs,weights)]))
    approx_terms = real_terms(padd(orbital_paulis(h1),{'I'*n:scalar},
                                  *[pscale(pmul(orbital_paulis(c['g']),orbital_paulis(c['g'])),w) for c,w in zip(chosen[1][1],weights)]))
    chosen += [('whole_Pauli_original',chosen[0][1]),('whole_Pauli_N1_approx',chosen[1][1])]
    h = df_operator(gs,weights,h1,scalar); idx = sector_indices(n,nu); target = expm(-1j*N1_TIME*h)
    records, rows = [], []
    for name, cs in chosen:
        hs = df_operator([c['g'] for c in cs],weights,h1,scalar)
        bound = sum(c['bound'] for c in cs)
        if name == 'whole_Pauli_original':
            hs, bound = h, 0.
        actual_model = opnorm((h-hs)[np.ix_(idx,idx)])
        if actual_model > bound+1e-10:
            raise AssertionError('model bound failed')
        records.append({'name':name, 'weights':weights, 'factors':[matrix(c['g']) for c in cs],
                        'frames':[matrix(c['frame']) for c in cs], 'spectra':[c['approx_eta'] for c in cs],
                        'model_bound':bound,'actual_model_sector_error':actual_model,
                        'paulis':full_terms if name=='whole_Pauli_original' else approx_terms if name=='whole_Pauli_N1_approx' else None})
        for q in Q_GRID:
            qc, op = product_circuit(cs,weights,h1,scalar,N1_TIME,q,records[-1]['paulis'])
            pf = opnorm((op-expm(-1j*N1_TIME*hs))[np.ix_(idx,idx)])
            actual = opnorm((op-target)[np.ix_(idx,idx)])
            accounted = N1_TIME*bound+pf
            if actual > accounted+1e-10:
                raise AssertionError('total target-error triangle failed')
            costs = []
            for axis in ('X','Y'):
                circ, ref = wrapper(qc,op,axis,[(0,'x'),(1,'x')])
                costs.append(audit.compile(circ,ref,n,f'N1/{kind}/{name}/q{q}/{axis}'))
            xy = {m:sum(c[m] for c in costs) for m in METRICS}
            rows.append({'candidate':name,'q':q,'delta':N1_TIME/q,'operator':matrix(op),
                         'model_bound':bound,'pf_sector_bias':pf,'actual_original_sector_bias':actual,
                         'accounted_original_bias':accounted,'costs':costs,'resources':precision(accounted,xy)})
    serial_lists = [[{k:matrix(v) if isinstance(v,np.ndarray) else v for k,v in c.items()} for c in cs] for cs in lists]
    return {'track':'N1','context':kind,'modes':n,'particles':nu,'sector_dimension':len(idx),
            'df_rank':2,'weights':weights,'one_body':matrix(h1),'scalar':scalar,'input_factors':[matrix(g) for g in gs],
            'hamiltonian':matrix(h),'time':N1_TIME,'model_budget':N1_BUDGET,'q_grid':list(Q_GRID),
            'generator_candidates':serial_lists,'selected':records,'rows':rows,
            'fragment_commutator_norm':opnorm(second_quantize_one_body(gs[0])@second_quantize_one_body(gs[0])@second_quantize_one_body(gs[1])@second_quantize_one_body(gs[1])-second_quantize_one_body(gs[1])@second_quantize_one_body(gs[1])@second_quantize_one_body(gs[0])@second_quantize_one_body(gs[0])),
            'selector_information':'n-by-n factor eigensystems and native basis costs only; dense sector truth evaluated afterward',
            'native_ir':audit.records}


def density_bound(e):
    return float(np.sum(abs(np.diag(e)))+2*np.sum(abs(np.triu(e,1))))


def charges(j, budget=.0021):
    """J-only finite dictionary search, k<=2; exponential small-input generator."""
    j = np.asarray(j,float); n = len(j)
    if n > 4 or j.shape != (n,n) or not np.allclose(j,j.T,atol=1e-12,rtol=0):
        raise ValueError('symmetric J, n<=4 required')
    atoms = [np.array(x,float) for x in itertools.product((-1,0,1),repeat=n)
             if any(x) and next(v for v in x if v) == 1]
    eligible = []; examined = 0
    for k in (1,2):
        for inds in itertools.combinations(range(len(atoms)),k):
            s = np.column_stack([atoms[i] for i in inds]); pairs = list(itertools.combinations_with_replacement(range(k),2))
            basis = [np.outer(s[:,a],s[:,b])+(np.outer(s[:,b],s[:,a]) if a!=b else 0) for a,b in pairs]
            coeff = np.linalg.lstsq(np.column_stack([m.ravel() for m in basis]),j.ravel(),rcond=None)[0]
            km = np.zeros((k,k))
            for (a,b),c in zip(pairs,coeff):km[a,b]=km[b,a]=c
            error = j-s@km@s.T; bound = density_bound(error); examined += 1
            if bound <= budget:
                widths = [max(2,math.ceil(math.log2(2*max(np.sum(v>0),np.sum(v<0))+1))) for v in s.T]
                score = (sum(int(np.count_nonzero(v))*b for v,b in zip(s.T,widths)),sum(widths),k,bound,inds)
                eligible.append((score,s,km,error,bound,widths))
    if not eligible:
        return {'status':'NO_ELIGIBLE_COMPRESSED_CANDIDATE','examined':examined,'eligible':0}
    _,s,km,error,bound,widths = min(eligible,key=lambda row:row[0])
    return {'status':'COMPRESSED_CANDIDATE_FOUND','examined':examined,'eligible':len(eligible),
            'S':s.astype(int).tolist(),'K':km.tolist(),'E':error.tolist(),'bound':bound,'widths':widths,
            'selector':'min controlled-add work proxy, workspace, k, coefficient bound; not native/global optimum'}


def increment(qc, control, register, sign):
    operations = [(register[:i],register[i]) for i in reversed(range(1,len(register)))]+[([],register[0])]
    if sign < 0:operations = list(reversed(operations))
    for lower,target in operations:
        if lower:qc.mcx([control,*lower],target)
        else:qc.cx(control,target)


def charge_circuit(s, k, widths, duration, *, signed=True):
    s,k = np.asarray(s,int),np.asarray(k,float); n, count = s.shape
    workspace = sum(widths); anc=n+workspace; qc=QuantumCircuit(anc+1); compute=QuantumCircuit(anc+1)
    registers=[]; offset=n
    for a,b in enumerate(widths):
        reg=list(range(offset,offset+b));offset+=b;registers.append(reg)
        for i in range(n):
            if s[i,a]:increment(compute,i,reg,int(s[i,a]))
    qc.compose(compute,inplace=True)
    bits=[(a,q,(-2**i if signed and i==len(reg)-1 else 2**i)) for a,reg in enumerate(registers) for i,q in enumerate(reg)]
    for i,(a,p,x) in enumerate(bits):
        coefficient=k[a,a]*x*x
        if coefficient:qc.cp(-duration*coefficient,anc,p)
        for b,q,y in bits[i+1:]:
            coefficient=2*k[a,b]*x*y
            if coefficient:qc.mcp(-duration*coefficient,[anc,p],q)
    qc.compose(compute.inverse(),inplace=True)
    return qc,workspace,{'compute_size':compute.size(),'compute_ops':dict(compute.count_ops()),'uncompute_size':compute.size()}


def density_terms(j):
    n=len(j); out={}
    for i,l in itertools.product(range(n),repeat=2):
        if j[i,l]:
            gi=np.zeros((n,n));gl=gi.copy();gi[i,i]=1;gl[l,l]=1
            out=padd(out,pscale(pmul(orbital_paulis(gi),orbital_paulis(gl)),j[i,l]))
    return real_terms(out)


def n2_input(kind):
    s=np.array([[1,0],[1,1],[0,1],[-1,0]])
    k=np.array([[.3,-.1],[-.1,.2]]);j=s@k@s.T
    if kind=='perturbed_signed':j[0,3]+=.001;j[3,0]+=.001
    if kind=='uniform_hamming':j=.3*np.ones((4,4))
    if kind=='dense_real_rank2':
        v=np.array([[-.8,.3],[.31,-.21],[.53,.71],[-.27,.47]]);j=v@k@v.T
    if kind=='sparse_no_collective':j=np.array([[0,.3,0,0],[.3,0,0,0],[0,0,0,-.2],[0,0,-.2,0]])
    return j


def run_n2(kind):
    audit=Audit();j=n2_input(kind);construction=charges(j);n=4;duration=.7
    bits=np.array([[s>>i&1 for i in range(n)] for s in range(1 << n)])
    target_values=np.einsum('bi,ij,bj->b',bits,j,bits)
    candidates=[('direct_original',j,None)]
    if 'S' in construction:
        approximate=np.asarray(construction['S'])@np.asarray(construction['K'])@np.asarray(construction['S']).T
        candidates += [('charge_generated',approximate,construction),('direct_same_approx',approximate,None)]
    if kind=='uniform_hamming':
        candidates.append(('known_unsigned_HWP',j,{'S':[[1]]*n,'K':[[.3]],
                                                  'widths':[math.ceil(math.log2(n+1))],'unsigned':True}))
    rows=[]
    for name,model,con in candidates:
        values=np.einsum('bi,ij,bj->b',bits,model,bits);op=np.diag(np.exp(-1j*duration*values))
        bound=density_bound(j-model);actual=float(np.max(abs(target_values-values)))
        if actual>bound+1e-12:raise AssertionError('N2 model bound')
        if con:
            circ,workspace,struct=charge_circuit(con['S'],con['K'],con['widths'],duration,
                                                signed=not con.get('unsigned',False))
        else:
            circ=QuantumCircuit(n+1);workspace=0;struct={'compute_size':0,'uncompute_size':0}
            for label,c in density_terms(model).items():apply_pauli(circ,label,angle=duration*c,control=n)
        costs=[]
        for axis in ('X','Y'):
            if workspace:
                wrapped=QuantumCircuit(n+workspace+1);anc=n+workspace
                wrapped.x(0);wrapped.x(1);wrapped.h(anc);wrapped.compose(circ,inplace=True)
                if axis=='Y':wrapped.sdg(anc)
                wrapped.h(anc)
                # Logical ordinary control has ancilla directly above system.
                dummy=QuantumCircuit(n+1);wrapped_small,ref=wrapper(dummy,op,axis,[(0,'x'),(1,'x')])
            else:wrapped,ref=wrapper(circ,op,axis,[(0,'x'),(1,'x')])
            costs.append(audit.compile(wrapped,ref,n,f'N2/{kind}/{name}/{axis}',workspace))
        rows.append({'candidate':name,'model_J':model.tolist(),'model_bound':bound,'actual_model_full_error':actual,
                     'operator':matrix(op),'actual_original_bias':float(np.max(abs(np.exp(-1j*duration*target_values)-np.exp(-1j*duration*values)))),
                     'arithmetic':struct,'costs':costs,'resources':precision(duration*bound,{m:sum(c[m] for c in costs) for m in METRICS})})
    return {'track':'N2','context':kind,'modes':n,'J':j.tolist(),'time':duration,'model_budget':.0021,
            'construction':construction,'rows':rows,'native_ir':audit.records,
            'known_HWP_comparison':'uniform J is a known Hamming-weight mechanism; no new mechanism credit, generated signed representation uses one extra sign bit'}


def active_certificate(energies, hoppings, density, active, particles=2):
    """Coefficient lower bounds; no many-body matrix, true ground or true gap input."""
    n=len(energies);external=[i for i in range(n) if i not in active]
    trial=set(sorted(active)[:particles]);u=sum(energies[i] for i in trial)+sum(v for i,j,v in density if i in trial and j in trial)
    vnorm=sum(abs(v) for _,_,v in hoppings)+sum(abs(v) for _,_,v in density)
    if not external:return {'status':'FULL_SPACE','U':u,'delta':0.,'classes':[]}
    off=sum(abs(v) for i,j,v in hoppings if i in external or j in external)
    classes=[]
    for pos,orbital in enumerate(external):
        free=[i for i in range(n) if i not in external[:pos+1]]
        if len(free)<particles-1:continue
        ref=energies[orbital]+sum(sorted(energies[i] for i in free)[:particles-1])
        beta=sum(abs(v) for i,j,v in hoppings if orbital in (i,j) and (j if i==orbital else i) in active)
        c=ref-vnorm;lower=c-(len(external)-1)*off;gap=lower-u
        classes.append({'first_external':orbital,'reference_lower':ref,'C_block_lower':c,
                        'block_offdiag_upper':off,'d':lower,'beta':beta,'gap_lower':gap})
    if any(row['gap_lower']<=0 for row in classes):
        return {'status':'UNRESOLVED_LOWER_BOUND','U':u,'delta':None,'classes':classes}
    lo=0.;hi=sum(row['beta']**2/row['gap_lower'] for row in classes)
    for _ in range(100):
        mid=(lo+hi)/2;rhs=sum(row['beta']**2/(row['gap_lower']+mid) for row in classes)
        if mid>=rhs:hi=mid
        else:lo=mid
    return {'status':'COEFFICIENT_BOUND_AVAILABLE','U':u,'delta':hi,'classes':classes,
            'bound_kind':'mathematical sufficient coefficient bound evaluated in binary64; no interval certificate'}


def run_n3():
    rows=[];n=6;nu=2
    hops=[(1,2,.2),(0,3,.1),(2,4,.08),(3,5,.04),(4,5,.05)];density=[(0,1,.3)]
    for name,external_energies in [('gapped',[6.,8.]),('small_gap',[.4,.6])]:
        energies=[-.2,-.1,.1,.25,*external_energies];active=list(range(4));history=[]
        while True:
            cert=active_certificate(energies,hops,density,active,nu)
            history.append({'active':active.copy(),'certificate':cert})
            if cert['delta'] is not None and cert['delta']<=.004:break
            external=[i for i in range(n) if i not in active]
            if not external:break
            scores={c['first_external']:c['beta'] for c in cert['classes']}
            active.append(max(external,key=lambda i:(scores.get(i,0.),-i)));active.sort()
        # Exact truth is strictly post-selector, used only to evaluate certificate.
        g=np.diag(energies)
        for i,j,v in hops:g[i,j]=g[j,i]=v
        full=second_quantize_one_body(g)
        for i,j,v in density:
            ni=np.diag([float(s>>i&1) for s in range(1 << n)])
            nj=np.diag([float(s>>j&1) for s in range(1 << n)])
            full+=v*ni@nj
        idx=sector_indices(n,nu);p=[s for s in idx if all(not(s>>i&1) for i in range(n) if i not in active)]
        truth=float(np.linalg.eigvalsh(full[np.ix_(idx,idx)])[0]);a=float(np.linalg.eigvalsh(full[np.ix_(p,p)])[0])
        if a-truth>cert['delta']+1e-12:raise AssertionError('N3 bound failed')
        initial_p=[s for s in idx if s<1 << 4]
        rows.append({'context':name,'modes':n,'particles':nu,'energies':energies,'hoppings':hops,'density':density,
                     'history':history,'selected_active':active,'certificate':cert,'ground_truth_post_selection':truth,
                     'active_ground_post_selection':a,'actual_energy_shift':a-truth,
                     'initial_active_dimension':len(initial_p),'initial_active_offdiagonal_norm':opnorm(full[np.ix_(initial_p,initial_p)]-np.diag(np.diag(full[np.ix_(initial_p,initial_p)]))),
                     'time_evolution_guarantee':None,'quantum_resource_comparison':None})
    return {'track':'N3','rows':rows,'native_ir':[],
            'task':'fixed two-particle ground energy feasibility only; no true-ground/gap constructor input'}


def run_job(task):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    started=time.monotonic()
    kind,context=task
    result=run_n1(context) if kind=='N1' else run_n2(context) if kind=='N2' else run_n3()
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result['worker_resources']={'wall_seconds':time.monotonic()-started,'pid':__import__('os').getpid(),
                                'cpu_seconds_cumulative':usage.ru_utime+usage.ru_stime,'peak_rss_bytes':usage.ru_maxrss*1024}
    return result


TASKS = [('N1',x) for x in ('planted_local_gauge','dense_group_frame','separated_spectrum')] + [
    ('N2',x) for x in ('signed_overlap','perturbed_signed','uniform_hamming','dense_real_rank2','sparse_no_collective')] + [('N3','coefficient_feasibility')]
