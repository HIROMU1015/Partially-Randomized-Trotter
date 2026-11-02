"""Independent, bounded synthetic audit. Never invokes a production runner.

The supplied GPT ZIP is read as provenance only; no code is imported from it.
Exact formal logarithms and JW action on bit strings are independent checks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["QISKIT_PARALLEL"] = "FALSE"

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/track_b_df_sparse_correction_audit/2026-10-11"
sys.path.insert(0, str(ROOT / "src"))
DEGREE = 5
CHECKS = []


def check(name, ok, details=None):
    CHECKS.append({"name": name, "pass": bool(ok), "details": details})
    if not ok:
        raise AssertionError(name)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1536 * 1024:
        raise MemoryError("Audit RSS guard: 1536 MiB")


def plus(*xs):
    out = defaultdict(F)
    for x in xs:
        for word, value in x.items():
            out[word] += value
    return {k: v for k, v in out.items() if v}


def times(x, y, degree=DEGREE):
    out = defaultdict(F)
    for a, ca in x.items():
        for b, cb in y.items():
            if len(a) + len(b) <= degree:
                out[a + b] += ca * cb
    return {k: v for k, v in out.items() if v}


def scaled(x, scalar):
    return {k: v * scalar for k, v in x.items() if v * scalar}


def bracket(x, y):
    return plus(times(x, y), scaled(times(y, x), -1))


def exp_series(x):
    out = {(): F(1)}
    power = out
    for degree in range(1, DEGREE + 1):
        power = times(power, x)
        out = plus(out, scaled(power, F(1, math.factorial(degree))))
    return out


def log_series(x):
    increment = plus(x, {(): F(-1)})
    out = {}
    power = {(): F(1)}
    for k in range(1, DEGREE + 1):
        power = times(power, increment)
        out = plus(out, scaled(power, F((-1) ** (k + 1), k)))
    return out


def grade(x, n):
    return {k: v for k, v in x.items() if len(k) == n}


def phase_a():
    records = []
    for count in (1, 2, 3, 4, 5):
        symbols = [{(j,): F(1)} for j in range(count)]
        factors = [exp_series(scaled(s, F(1, 2))) for s in symbols[:-1]]
        sequence = factors + [exp_series(symbols[-1])] + list(reversed(factors))
        step = {(): F(1)}
        for factor in sequence:
            step = times(step, factor)
        log = log_series(step)
        prediction = {}
        for j in range(count - 1):
            tail = plus(*symbols[j + 1:])
            prediction = plus(prediction,
                scaled(bracket(symbols[j], bracket(symbols[j], tail)), F(-1, 24)),
                scaled(bracket(tail, bracket(symbols[j], tail)), F(-1, 12)))
        check(f"A.{count}.exact_log_cubic", grade(log, 3) == prediction)
        check(f"A.{count}.symmetric_log_even_zero", not grade(log, 2) and not grade(log, 4))
        correction = exp_series(scaled(prediction, F(-1, 2)))
        ideal = exp_series(plus(*symbols))
        corrected = times(times(correction, step), correction)
        residual = plus(corrected, scaled(ideal, -1))
        check(f"A.{count}.corrected_degree4", all(not grade(residual, k) for k in range(5)))
        central_p3 = {w: c for w, c in exp_series(symbols[-1]).items() if len(w) <= 3}
        bad = {(): F(1)}
        for factor in factors + [central_p3] + list(reversed(factors)):
            bad = times(bad, factor)
        bad = plus(times(times(correction, bad), correction), scaled(ideal, -1))
        check(f"A.{count}.P3_degree4_residual", grade(bad, 4) == {(count - 1,) * 4: F(-1, 24)})
        if count > 1:
            wrong = exp_series(scaled(prediction, F(1, 2)))
            wrong = plus(times(times(wrong, step), wrong), scaled(ideal, -1))
            check(f"A.{count}.wrong_sign_detected", grade(wrong, 3) == scaled(prediction, 2))
        records.append({"blocks": count, "cubic_words": len(prediction),
                        "cubic_log": {','.join(map(str,w)):str(c) for w,c in prediction.items()},
                        "P3_quartic": "-R^4/24"})
    return {"status": "INDEPENDENT_VALIDATION_PASS", "method": "Exact log(I+X) of palindromic series, not GPT script execution", "fixtures": records}


def comm(a, b):
    return a @ b - b @ a


def jw_annihilate(n):
    import numpy as np
    out = []
    for mode in range(n):
        a = np.zeros((1 << n, 1 << n), complex)
        states = np.arange(1 << n)
        for state in states:
            if state & (1 << mode):
                a[state ^ (1 << mode), state] = (-1) ** (int(state & ((1 << mode) - 1)).bit_count())
        out.append(a)
    return out


def primitive(columns, sign=1):
    import numpy as np
    ops = jw_annihilate(columns.shape[0])
    eye = np.eye(1 << columns.shape[0], dtype=complex)
    result = sign * eye
    for col in columns.T:
        a = sum(col[k].conjugate() * ops[k] for k in range(len(ops)))
        result = result @ (eye - 2 * a.conj().T @ a)
    return result


def gaussian_by_generator(w):
    """Second quantize a one-particle log, distinct from GPT's minors code."""
    import numpy as np
    from scipy.linalg import expm, logm
    ops = jw_annihilate(len(w))
    log = logm(w)
    log = (log - log.conj().T) / 2
    generator = sum(log[i, j] * ops[i].conj().T @ ops[j]
                    for i in range(len(w)) for j in range(len(w)))
    return expm(generator)


def random_unitary(n, rng):
    import numpy as np
    q, r = np.linalg.qr(rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n)))
    return q * (np.diag(r)/abs(np.diag(r))).conjugate()


def phase_b():
    import numpy as np
    from scipy.linalg import expm, norm
    rng = np.random.default_rng(5101107)
    records = []
    # One genuine rank-six and one exact-degenerate common-span fixture.
    for n, active, ranks in [(7,6,(2,2,2)),(5,3,(1,2,2)),(4,2,(0,1,2))]:
        w = random_unitary(n, rng)
        frame = w[:,:active]
        cols = [random_unitary(active,rng)[:,:r] for r in ranks]
        full = [primitive(frame @ v,s) for v,s in zip(cols,(-1,1,-1))]
        local = [primitive(v,s) for v,s in zip(cols,(-1,1,-1))]
        t = comm(full[0],comm(full[1],full[2]))
        small = comm(local[0],comm(local[1],local[2]))
        g = gaussian_by_generator(w)
        lifted = g @ np.kron(np.eye(1 << (n-active)),small) @ g.conj().T
        error = float(norm(t-lifted,2))
        unitary = expm(1j*.03125*t/4)
        lifted_u = g @ np.kron(np.eye(1 << (n-active)),expm(1j*.03125*small/4)) @ g.conj().T
        u_error = float(norm(unitary-lifted_u,2))
        check(f"B.{n}.{active}.lift", error < 4e-11, error)
        check(f"B.{n}.{active}.rotation", u_error < 4e-11, u_error)
        check(f"B.{n}.{active}.Hermitian_bound", norm(t-t.conj().T,2)<4e-11 and norm(t,2)<=4+4e-11)
        controlled = np.block([[np.eye(len(t)),np.zeros_like(t)],[np.zeros_like(t),unitary]])
        check(f"B.{n}.{active}.control_zero_identity", norm(controlled[:len(t),:len(t)]-np.eye(len(t)),2)==0)
        records.append({"system_modes":n,"active_modes":active,"primitive_ranks":ranks,
                        "lift_residual":error,"rotation_residual":u_error,
                        "max_cross_primitive_overlap":float(max(abs((cols[i].conj().T@cols[j])).max(initial=0) for i in range(3) for j in range(i))),
                        "actual_norm_diagnostic_only":float(norm(small,2)),"kappa_used":4})
    return {"status":"CONDITIONAL_PASS", "general_support_proof":"Finite-dimensional even CAR algebra on span of <=6 orbital columns; no numerical rank dropping permitted", "fixtures":records,"interval_certified_rank_selection":"NOT_IMPLEMENTED"}


def enumerate_terms(blocks):
    out = []
    for j in range(len(blocks)-1):
        tail = [q for block in blocks[j+1:] for q in block]
        for a in blocks[j]:
            for b in blocks[j]:
                for c in tail:
                    out.append((a[0]*b[0]*c[0]/24,comm(a[1],comm(b[1],c[1]))))
        for a in tail:
            for b in blocks[j]:
                for c in tail:
                    out.append((a[0]*b[0]*c[0]/12,comm(a[1],comm(b[1],c[1]))))
    return out


def correction_mean(terms, delta, gamma):
    import numpy as np
    from scipy.linalg import expm, norm
    eye = np.eye(len(terms[0][1]),dtype=complex)
    a = sum(c for c,t in terms)
    if a == 0:
        return eye, {"A":0,"B":0,"eta":0,"probability_sum":1,"error":0,"bound":0}
    b = 4*a
    eta = min(1,gamma*delta)
    # Safe universal kappa, without reading a synthetic operator's exact norm.
    mean = (1-eta)*eye
    k = np.zeros_like(eye)
    for c,t in terms:
        mean += eta*c/a*expm(1j*b*delta**3*t/(8*eta))
        k += c*t
    error = float(norm(mean-expm(1j*delta**3*k/2),2))
    # Lambda and ||K|| upper bounded by B: fully accessible, not norm oracle.
    bound = delta**6*(b*b/eta+b*b)/8
    return mean, {"A":a,"B":b,"eta":eta,"probability_sum":1,
                  "error":error,"bound":bound,"added_weight_multiplier":1}


def phase_c_d():
    import numpy as np
    from scipy.linalg import expm, norm
    from trotterlib.rte import finite_taylor_operator, finite_rte_distribution
    rng = np.random.default_rng(6121107)
    qs = [primitive(random_unitary(3,rng)[:,:r],s) for r,s in [(1,1),(2,-1),(1,-1)]]
    weights = [.45,.35,.25]
    blocks = [[(h,q)] for h,q in zip(weights,qs)]
    hs = [h*q for h,q in zip(weights,qs)]
    hsum = sum(hs)
    terms = enumerate_terms(blocks)
    a_formula = sum(weights[j]**2*sum(weights[j+1:])/24 + weights[j]*sum(weights[j+1:])**2/12 for j in range(2))
    check("C.hierarchical_mass_exact", abs(sum(c for c,t in terms)-a_formula)<1e-15)
    rows = []
    for delta in [.125,.0625,.03125]:
        corr, meta = correction_mean(terms,delta,1.25)
        check(f"C.bound.{delta}", meta['error']<=meta['bound']+3e-14,meta)
        check(f"C.contraction.{delta}", norm(corr,2)<=1+3e-13)
        forward = expm(-1j*delta*hs[0]/2) @ expm(-1j*delta*hs[1]/2)
        reverse = expm(-1j*delta*hs[1]/2) @ expm(-1j*delta*hs[0]/2)
        mid = finite_taylor_operator(hs[2],delta,4)
        own = sum(np.linalg.matrix_power(-1j*delta*hs[2],j)/math.factorial(j) for j in range(6))
        check(f"A.source_cutoff4_degree5.{delta}", norm(mid-own,2)<3e-14)
        mean = corr @ forward @ mid @ reverse @ corr
        exact = expm(-1j*delta*hsum)
        x = sum(weights)*delta
        y = meta['B']*delta**3
        tail = lambda z,k:sum(z**j/math.factorial(j) for j in range(k,40))
        sym = 2*tail(x,5)+y*tail(x,2)+math.exp(x)*tail(y,2)
        rte = math.exp(weights[2]*delta)*(weights[2]*delta)**6/720
        local_bound = sym+2*meta['bound']+rte
        n = round(1/delta)
        global_bound = math.expm1(n*math.log1p(local_bound))
        local_error = float(norm(mean-exact,2))
        global_error = float(norm(np.linalg.matrix_power(mean,n)-expm(-1j*hsum),2))
        check(f"D.local_bound.{delta}", local_error<=local_bound+3e-14)
        check(f"D.global_bound.{delta}", global_error<=global_bound+3e-13)
        rows.append({"delta":delta,"local_error":local_error,"local_bound":local_bound,
                     "global_error":global_error,"global_bound":global_bound,
                     "expected_requested_corrections":2*n*meta['eta'],**meta})
    _, saturation = correction_mean(terms,.125,12)
    check("C.saturation", saturation['eta']==1 and saturation['error']<=saturation['bound']+3e-14)
    zero, z = correction_mean([(0,np.zeros_like(qs[0]))],.125,1)
    check("C.A_zero", z['A']==0 and norm(zero-np.eye(len(zero)))==0)
    commuting=[(.25,comm(qs[0],comm(qs[0],qs[0])))]
    zero, z = correction_mean(commuting,.125,1)
    check("C.Lambda_zero_identity", norm(zero-np.eye(len(zero)))<3e-14)
    check("A.pure_D_nonzero", norm(comm(hs[0],comm(hs[0],hs[1]))/24+comm(hs[1],comm(hs[0],hs[1]))/12,2)>1e-5)
    for cutoff in [0,2,4]:
        d = finite_rte_distribution(.0625,cutoff)
        check(f"A.cutoff.{cutoff}.probability", abs(sum(d.order_probabilities)-1)<1e-14)
    return {"C":{"status":"CONDITIONAL_PASS","rows":rows,"saturation":saturation,
                 "tiny_eta":"Algebraically eliminate theta/eta factors; log-domain probability needed for a future digital sampler; not implemented",
                 "finite_bit_bias":"Not zero: sum |p_tilde-p| min(2,|theta|)+sum p_tilde epsilon; residual budget must be preregistered",
                 "negative_time":"OUT_OF_SCOPE_UNPROVED","production_sampler":"NOT_IMPLEMENTED"},
            "D":{"status":"CONDITIONAL_PASS","assumptions":["fixed positive gamma, fixed t/input, independent slots and steps", "central paired cutoff>=4/P5", "finite-bit/native/DF errors separately budgeted"],"rows":rows}}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['A','B','CD'],required=True)
    args=parser.parse_args(); start=time.monotonic(); cpu=time.process_time()
    report={'phase':args.phase,'scope':'Independent mathematical and small synthetic technical audit only',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'production_runner_calls':0,'molecule_NPZ_GPU_calls':0,'trajectory_or_quantum_shots':0}
    try:
        report['result']={'A':phase_a,'B':phase_b,'CD':phase_c_d}[args.phase]()
        report['status']='FOCUSED_CHECKS_PASS'
    except Exception as exc:
        report['status']='AUDIT_STOP'; report['exception_type']=type(exc).__name__; report['reason']=str(exc)
        report['mandatory_STOP']=True
    report['checks']=CHECKS; report['checks_passed']=sum(c['pass'] for c in CHECKS)
    report['wall_seconds']=time.monotonic()-start; report['CPU_seconds']=time.process_time()-cpu
    report['peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    report['python']=platform.python_version(); report['threads']=1
    path=ART/f'phase_{args.phase}_checks.json'
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['result','checks']}))
    return 0 if report['status']=='FOCUSED_CHECKS_PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
