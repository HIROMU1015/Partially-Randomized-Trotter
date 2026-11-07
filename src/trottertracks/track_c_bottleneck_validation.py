"""Fixed-policy price holdout and short joint-circuit checks.

The optimized source points, exact alpha allocation and integer shots are
frozen before these independent MC streams. No selection uses holdout prices.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
import os
import subprocess
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .track_c_bottleneck import OUT, ROOT, saved, account
from .track_c_bottleneck_runwise import build_singletons
from .track_c_bottleneck_select import replacement_prices, new_point
from .track_c_d_resources import r_part, r_key, metrics, assemble, modular_parts
from .track_c_compiler_a import compile_reference
from .track_c_h4 import common, save_json
from .track_c_h4_price import wrapper
from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase


def holdout_price(task):
    t, n, rare, j = task
    seed = int.from_bytes(hashlib.sha256(f'TrackC_bottleneck_HOLDOUT:{t}:{n}:{rare}:{j}'.encode()).digest()[:8], 'big')
    old, probability, choices = r_part(t, n, rare, seed)
    old_native = compile_reference(old)
    proposal, _ = build_singletons(t, n, choices)
    proposal_native = compile_reference(proposal)
    before, after = metrics(old_native), metrics(proposal_native)
    choice = 'singletons' if after[0] < before[0] else 'full'
    record = dict(t=t, n=n, rare=rare, j=j, seed=seed, probability=probability if rare else 1-probability,
                  choices=choices, old=before, proposal=after, selected=after if choice == 'singletons' else before, choice=choice)
    save_json(OUT/'holdout_prices'/f'{r_key(t,n)}_{int(rare)}_{j}.json', record)
    return record


def short_check(task):
    case, h, rate, steps = task
    old_tails, new_tails, decisions = [], [], []
    for cycle in range(steps):
        n = max(1, math.ceil(h*rate/.02))
        old, _, choices = r_part(h, n, cycle == 0, 881903+cycle)
        new, _ = build_singletons(h, n, choices)
        choose = metrics(compile_reference(new))[0] < metrics(compile_reference(old))[0]
        old_tails.append(old); new_tails.append(new if choose else old); decisions.append(choose)
    old_source = assemble(case, h, steps, old_tails)
    new_source = assemble(case, h, steps, new_tails)
    parts = modular_parts(case, h, steps, new_tails)
    scalar = QuantumCircuit(9); p = common()['preps'][3]
    scalar.p(-steps*h*(p.constant_coefficient+p.extracted_identity_coefficient), 8)
    pre = QuantumCircuit(9); pre.h(8); pre = compile_reference(pre)
    c = common(); rev = [int(f'{int(i):08b}'[::-1], 2) for i in c['sector'].basis_indices]
    initial = np.zeros(512, complex); initial[rev] = c['state']
    rows = []
    for axis in ('X', 'Y'):
        read = QuantumCircuit(9, 1); read.compose(scalar, inplace=True)
        if axis == 'Y': read.sdg(8)
        read.h(8); read.measure(8, 0); read = compile_reference(read)
        modular = QuantumCircuit(9, 1); modular.compose(pre, inplace=True)
        for part in parts: modular.compose(part, inplace=True)
        modular.compose(read, inplace=True)
        whole = compile_reference(wrapper([new_source], axis))
        check = None
        if steps == 1:
            vectors = [Statevector(initial).evolve(q.remove_final_measurements(inplace=False)).data
                       for q in (wrapper([old_source], axis), wrapper([new_source], axis), whole, modular)]
            readout = lambda v: float(np.vdot(v[:256], v[:256]).real-np.vdot(v[256:], v[256:]).real)
            means = [readout(v) for v in vectors]
            errors = []
            for v in vectors[1:]:
                phase = np.vdot(vectors[0], v); phase /= abs(phase)
                errors.append(float(np.linalg.norm(vectors[0]-v/phase)))
            check = dict(means=means, readout_errors=[abs(v-means[0]) for v in means[1:]], projective_errors=errors)
            assert max(check['readout_errors']+errors) < 1e-8, check
        rows.append(dict(axis=axis, whole_DAG=metrics(whole), modular_DAG=metrics(modular),
                         modular_serial_depth_upper=sum(q.depth() for q in [pre, *parts, read]), state=check))
    result = dict(case=case, h=h, rate=rate, steps=steps, singleton_decisions=decisions, rows=rows)
    save_json(OUT/'short_checks'/f'{case["name"]}_{h}_{rate}_{steps}.json', result)
    return result


def fixed_policy_result(policy, records):
    rs = {}
    for r in records: rs.setdefault(r_key(r['t'], r['n']), {}).setdefault(str(r['rare']), []).append(r)
    for strata in rs.values():
        for records in strata.values(): records.sort(key=lambda r:r['j'])
    cert = json.loads((OUT/'REPLACEMENT_CERTIFICATE.json').read_text())
    # Qualifications already include the replacement supplement; supply the
    # saved unsupplemented row to avoid applying that error twice.
    qs = dict(saved()[1])
    for q in json.loads((OUT/'JOINT_NATIVE_SOURCE.json').read_text())['rows']:
        qs[(q['name'],q['m'],q['h'],q['rate'])] = q
    from .track_c_bottleneck import identity
    points = [new_point(s['signal'], qs[identity(s['signal'])], rs, cert) for s in policy['selected']]
    ns = np.array([r['shots'] for r in policy['rounds']])
    result = account(points, ns)
    weights = {}
    for p, nn in zip(points, ns):
        for recipe in p['signal']['recipe']:
            key = r_key(recipe['time'], recipe['r'])
            weights[key] = weights.get(key, 0)+int(nn.sum())*p['signal']['steps']
    covariance = np.zeros((4,4)); paired = np.zeros((4,4)); delta = np.zeros(4)
    old_rte = np.zeros(4)
    for key, weight in weights.items():
        for records in rs[key].values():
            probability = records[0]['probability']
            values = weight*probability*np.array([r['selected'] for r in records])
            old = weight*probability*np.array([r['old'] for r in records])
            covariance += np.cov(values, rowvar=False, ddof=1)/len(records)
            paired += np.cov(values-old, rowvar=False, ddof=1)/len(records)
            delta += (values-old).mean(axis=0); old_rte += old.mean(axis=0)
    result.update(covariance_MC=covariance.tolist(), MC_SE=np.sqrt(np.diag(covariance)).tolist(),
                  baseline_same_points_shots_G=(np.array(result['totals']['total'])-delta).tolist(),
                  paired_basis_saving=(-delta).tolist(), paired_difference_MC_SE=np.sqrt(np.diag(paired)).tolist())
    phases = [math.atan2(p['signal']['raw']['imag'], p['signal']['raw']['real']) for p in points]
    decode = reconstruct_rpe_phase(phases, [1<<m for m in range(18)])
    error = abs(-decode['final_phase_estimate']/.02-common()['energy'])
    branch = all(abs(r['base_phase_estimate']+.02*common()['energy']) <= .4/(1<<m)+1e-10 for m,r in enumerate(decode['rounds']))
    assert branch and error <= 1.5936e-4
    alpha = sum(Fraction.from_float(a) for row in policy['alpha'] for a in row)
    assert alpha <= Fraction(1,20)
    result.update(primary=policy['primary'], alpha=policy['alpha'], alpha_sum_exact=str(alpha),
                  decoder=decode, mean_branch_pass=branch, mean_energy_error=error,
                  energy_statistical_bound=.4/(.02*(1<<17)), all_source_native_closed=all(p['qualification']['closed'] for p in points),
                  note='Mean decoder uses original canonical mean. New implementation mean is enclosed by the uniform replacement disk. Fresh-bit/Hoeffding guarantee uses supplemented rho,beta; no E0 supplied to decoder.')
    return result


def run():
    if (OUT/'VALIDATION.json').exists(): raise RuntimeError('Do not overwrite validation')
    (OUT/'holdout_prices').mkdir(exist_ok=False); (OUT/'short_checks').mkdir(exist_ok=False)
    optimized = json.loads((OUT/'OPTIMIZATION.json').read_text())
    policies = [p for p in optimized['optimized'] if p['family']=='PR']
    cells = sorted({(r['time'],r['r']) for p in policies for s in p['selected'] for r in s['signal']['recipe']})
    cases = {r['signal']['name']:r['signal']['case'] for p in saved()[0].values() for r in p['selected']}
    checks = [(cases[name],h,rate,n) for name,h,rate in (('original',.02,32),('tuned_0123',.16,32),('tuned_0321',.32,64)) for n in (1,2)]
    prices, short = [], []
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        tasks = [(t,n,rare,j) for t,n in cells for rare in (False,True) for j in range(16)]
        for f in as_completed(pool.submit(holdout_price,t) for t in tasks): prices.append(f.result())
        for f in as_completed(pool.submit(short_check,t) for t in checks): short.append(f.result())
    result = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                  frozen_selection_source=optimized['source_commit'], cells=cells, new_samples=len(prices),
                  policies=[fixed_policy_result(p,prices) for p in policies], short_checks=short,
                  statistical='Independent seeds per cell/stratum/index. Sum cell covariance after shared round/cache weights. Paired covariance for baseline versus singleton implementation. No reoptimization on holdout.',
                  scope='Same canonical K2 sampling law and return aggregation. Conditional saved-H4 model, exact ground, ideal native, fresh bits. Depth totals are serial upper bounds; short whole/modular DAG depths are explicit.')
    save_json(OUT/'VALIDATION.json', result); print('HOLDOUT_AND_SHORT_CHECKS_COMPLETE',len(prices),flush=True)


if __name__ == '__main__': run()
