"""Limited full-H DF RTE study; historical SOURCE/results are read-only.

Only preparation of the one-body Z components and the independent oracle
extension are new. Sampling, finite means, native lowering, compilation and
shot/cost accounting reuse the completed B3 study's functions.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import traceback
from types import SimpleNamespace

import h46_long_time_signal as signal

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09'
PRIVATE = PUBLIC/'PRIVATE'
B3 = ROOT/'artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26'
BASE = ROOT/'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25'
BASE_PRIVATE = Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025')
COHORT = 'FULL_H_LIGHT_EXPLORATION_20261109'
read, save = signal.read, signal.save
CONTEXT = {}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def setup():
    signal.install()
    signal.PUBLIC, signal.PRIVATE = PUBLIC, PRIVATE
    signal.PREFIX = {'H4': (0,), 'H6': (0,)}
    import h46_long_time_native as native
    native.PUBLIC, native.PRIVATE = PUBLIC, PRIVATE
    native.trial_seed = trial_seed
    # The previously validated accelerated writer emits the same canonical bytes.
    from h46_b3_fast_fingerprint import use_fast_fingerprint
    from trottertracks.resource_applicability import h6_hash_reuse_v1
    h6_hash_reuse_v1.use_fingerprint_reuse = use_fast_fingerprint
    return native


def trial_seed(row, cohort, index, outer, piece):
    return int(digest(['DF_FULL_H_LIGHT_20261109', cohort, row['id'], index, outer, piece])[:16], 16)


def prepare_full_h(ham, *, base=None):
    """Append one-body eigenbasis Zs to the existing, ordered DF tail.

    sum eps_j n_j = (sum eps_j)/2 I - sum eps_j/2 Z_j.
    The extracted scalar is applied once; no deterministic block survives.
    No threshold, DF fragment reorder or component merging is introduced.
    """
    from h46_new2_native import prepare
    from trotterlib.df_rte_tail import DFDiagonalPauliComponent, prepare_df_rte_event_inputs
    # H8 supplies its restored, saved Gaussian preparation.
    if base is None:
        base = prepare(ham, 0)
    require(len(base.deterministic_blocks) == 1 and
            base.deterministic_blocks[0].block_kind == 'one_body', 'ONE_BODY_SOURCE_BLOCK')
    one = base.deterministic_blocks[0]
    tail = base.tail_extraction
    definition = tail.basis_registry.register(one.runtime_basis_operations,
        num_system_qubits=ham.n_qubits, basis_id=one.basis_id)
    extra = tuple(DFDiagonalPauliComponent(
        component_id=f'full-h-one-body:Z{j}', coefficient=-float(e)/2,
        coefficient_abs=abs(float(e))/2, coefficient_sign=1 if -e >= 0 else -1,
        df_fragment_id='full-h-one-body', basis_id=definition.basis_id,
        basis_hash=definition.basis_hash, diagonal_pauli_support=(j,),
        basis_change_operations=definition.metadata.operations)
        for j, e in enumerate(one.diagonal_eigenvalues) if e != 0.)
    require(extra, 'ONE_BODY_NONSCALAR_COMPONENTS_REQUIRED')
    components = tail.components + extra
    one_scalar = math.fsum(one.diagonal_eigenvalues)/2
    scalar = math.fsum((tail.deterministic_identity_coefficient, one_scalar))
    lam = math.fsum(c.coefficient_abs for c in components)
    em = tail.extraction_metadata
    identity_count = int(one_scalar != 0.)
    added = len(extra) + identity_count
    em = replace(em, threshold_input_component_count=em.threshold_input_component_count+added,
        threshold_retained_component_count=em.threshold_retained_component_count+added,
        retained_identity_component_count=em.retained_identity_component_count+identity_count,
        extracted_identity_component_count=em.extracted_identity_component_count+identity_count,
        randomized_component_count=len(components), extracted_identity_coefficient=scalar,
        randomized_coefficient_l1=lam, normalization_policy='full_h_df_plus_one_body_exact_Z')
    nm = replace(tail.normalization_metadata, input_component_count=len(components),
        retained_component_count=len(components), normalization_policy='full_h_prethresholded_components')
    tail_hash = digest(dict(policy='full_h_df_plus_one_body_v1', two_body_tail=tail.tail_hash,
        one_body_components=[asdict(c) for c in extra], one_body_scalar=one_scalar))
    tail = replace(tail, tail_id='full-h-'+tail.tail_id, tail_hash=tail_hash,
        components=components, identity_coefficient=scalar,
        deterministic_identity_coefficient=scalar, rte_lambda_r=lam,
        extraction_metadata=em, normalization_metadata=nm,
        referenced_basis_ids=tuple(sorted(set(tail.referenced_basis_ids + (definition.basis_id,)))))
    return replace(base, preparation_hash=digest(dict(base=base.preparation_hash, full_tail=tail_hash)),
        deterministic_blocks=(), deterministic_block_order=(), tail_extraction=tail,
        rte_preparation=prepare_df_rte_event_inputs(tail), exact_rte_lambda_r=lam,
        extracted_identity_coefficient=scalar)


def full_oracle(ham, full):
    from h46_new2_native import OccupationOracle
    from trotterlib.df_trotter.decompose import diag_hermitian

    class FullOracle(OccupationOracle):
        def component(self, comp, vector):
            if comp.df_fragment_id != 'full-h-one-body':
                return super().component(comp, vector)
            value = vector
            for mode in comp.diagonal_pauli_support:
                key = 'one_body', mode
                if key not in self.numbers:
                    column = self.one_rotation[:, mode]
                    self.numbers[key] = self.one_body(__import__('numpy').outer(column, column.conj()))
                value = value - 2*(self.numbers[key] @ value)
            return comp.coefficient_sign*value

    oracle = FullOracle(ham, full, prefix=0)
    oracle.one_rotation = diag_hermitian(ham.one_body, sort='descending_abs', assume_hermitian=True)[0]
    oracle.specs = ()
    return oracle


def context(system):
    if system in CONTEXT:
        return CONTEXT[system]
    import numpy as np
    import qiskit
    from functools import partial
    from h46_m5_tol_native import restore
    from h46_new2_core import occupation_one_body, spectrum
    import trotterlib.rte_compiled_cost as costs
    require(qiskit.__version__ == '1.3.0' and np.__version__ == '1.26.4', 'SAVED_RUNTIME')
    costs.transpile = partial(qiskit.transpile, num_processes=1)
    info = signal.origin_config()['systems'][system]
    for item in info['inputs'].values():
        require(signal.sha(item['path']) == item['sha256'], 'SAVED_INPUT_SHA')
    ham, sector, full, state, md = restore(dict(system=system, **info))
    require(md['hamiltonian_hash'] == info['binding']['hamiltonian'] and
        md['state_hash'] == info['binding']['state'] and ham.n_blocks == signal.RANK[system], 'SAVED_INPUT_BINDING')
    old = read(BASE_PRIVATE/(system+'_context.json'))
    require(old['binding'] == info['binding'], 'SAVED_REFERENCE_BINDING')
    reference = old['reference']['0.8']
    # Read and verify the actual saved reference bytes as well as its saved signal.
    refpath = Path(info['reference_root'])/'reference.npz'
    receipt = read(refpath.parent/'receipt.json')
    require(signal.sha(refpath) == receipt['reference_sha256'], 'REFERENCE_BYTES')
    particles = sector.nelec_alpha + sector.nelec_beta
    whole = [i for i in range(1 << ham.n_qubits) if i.bit_count() == particles]
    pos = {b:i for i,b in enumerate(whole)}
    chosen = [pos[int(b)] for b in sector.basis_indices]
    with np.load(refpath, allow_pickle=False) as archive:
        z = complex(np.vdot(state, archive['reference'][chosen]))
    require(abs(z-complex(reference['real'],reference['imag'])) < 1e-12, 'REFERENCE_SIGNAL')
    prep = prepare_full_h(ham)
    matrix = occupation_one_body(ham.one_body, sector)
    for w,g in zip(ham.lambdas, ham.g_matrices):
        m = occupation_one_body(g, sector)
        matrix += w*(m@m)
    from trotterlib.df_hamiltonian import df_linear_operator
    operator, _ = df_linear_operator(ham, sector, backend='python')
    probe = np.arange(1,len(state)+1) + 1j*np.arange(len(state),0,-1)
    probe /= np.linalg.norm(probe)
    matrix_error = float(np.linalg.norm(operator@probe - (matrix@probe + ham.constant*probe)))
    require(matrix_error < 1e-9, 'SAVED_HAMILTONIAN_RECONSTRUCTION')
    scalar = prep.extracted_identity_coefficient
    oracle = full_oracle(ham, full)
    qstate = np.zeros(1 << ham.n_qubits, complex)
    qstate[oracle.qindices] = oracle.initial
    ctx = SimpleNamespace(system=system,ham=ham,sector=sector,full=full,state=state,
        specs=(), tails={0:dict(scalar=scalar,lam=prep.exact_rte_lambda_r,
        spec=spectrum(matrix-scalar*np.eye(len(state))))}, refs={.8:z},
        oracle=oracle,qstate=qstate,preparations={0:prep})
    one = [c for c in prep.rte_preparation.symbolic_tail.components if c.df_fragment_id=='full-h-one-body']
    save(PRIVATE/(system+'_context.json'),dict(binding=info['binding'],reference={'0.8':reference},
        reference_file_sha256=receipt['reference_sha256'],matrix_reconstruction_error=matrix_error,
        lambda_R=prep.exact_rte_lambda_r,scalar=ham.constant+scalar,constant=ham.constant,
        extracted_scalar=scalar,deterministic_blocks=0,one_body_components=len(one),
        one_body_probability_mass=math.fsum(c.probability for c in one),
        one_body_lambda_R=math.fsum(c.coefficient_abs for c in one),
        two_body_fragments=list(prep.randomized_block_indices),
        components=len(prep.rte_preparation.symbolic_tail.components),
        sector_dimension=len(state),preparation_hash=prep.preparation_hash,
        input_regeneration=0,new_reference_calls=0))
    CONTEXT[system] = ctx
    return ctx


def validate(system, row):
    import numpy as np
    from trotterlib.rte import finite_rte_distribution, _make_event
    from trotterlib.df_rte_circuit import DFRTEEventCircuitRequest
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    from trotterlib.df_trotter.circuit import simulate_statevector
    ctx = context(system)
    prep = ctx.preparations[0]
    comps = prep.rte_preparation.symbolic_tail.components
    require(not prep.deterministic_blocks and not ctx.specs and not ctx.oracle.specs, 'NO_DETERMINISTIC_EVOLUTION')
    require(tuple(prep.randomized_block_indices)==tuple(range(ctx.ham.n_blocks)), 'ALL_DF_ORDER_PRESERVED')
    require(abs(math.fsum(c.probability for c in comps)-1) < 1e-13, 'COMPONENT_MASS')
    require(max(abs(c.probability-c.coefficient_abs/prep.exact_rte_lambda_r) for c in comps)<1e-15, 'COMPONENT_NORMALIZATION')
    oracle = ctx.oracle
    rng = np.random.default_rng(20261109)
    vectors = [oracle.initial] + [rng.normal(size=len(oracle.initial))+1j*rng.normal(size=len(oracle.initial)) for _ in range(2)]
    errors = []; one_errors = []
    one_scalar = float(np.trace(ctx.ham.one_body).real)/2
    for vector in vectors:
        v = vector/np.linalg.norm(vector)
        exact = ctx.ham.constant*v + oracle.one_body(ctx.ham.one_body)@v
        for w,g in zip(ctx.ham.lambdas,ctx.ham.g_matrices):
            m = oracle.one_body(g); exact += w*(m@(m@v))
        actual = (ctx.ham.constant+prep.extracted_identity_coefficient)*v
        one_actual = one_scalar*v
        for c in comps:
            part = c.coefficient_abs*oracle.component(c,v)
            actual = actual + part
            if c.df_fragment_id=='full-h-one-body': one_actual = one_actual + part
        errors.append(float(np.linalg.norm(actual-exact)))
        one_errors.append(float(np.linalg.norm(one_actual-oracle.one_body(ctx.ham.one_body)@v)))
    require(max(errors+one_errors)<1e-9, 'FULL_H_COMPONENT_RECONSTRUCTION')
    builder = QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
    lookup = {c.component_id:c for c in comps}
    one_idx = next(i for i,c in enumerate(comps) if c.df_fragment_id=='full-h-one-body')
    two_idx = max((i for i,c in enumerate(comps) if c.df_fragment_id!='full-h-one-body'), key=lambda i: comps[i].probability)
    checks = []
    for sign in (1,-1):
        dist = finite_rte_distribution(prep.exact_rte_lambda_r*(sign*row['T']/row['s']),row['K'])
        for order in dist.orders:
            indices = [one_idx] + [two_idx if j%2==0 else one_idx for j in range(order)]
            event = _make_event(indices,comps,dist,dist.orders.index(order))
            circuit = builder.build_event(DFRTEEventCircuitRequest(event,prep.rte_preparation.component_specs,
                controlled=True,ancilla_qubit=ctx.ham.n_qubits)).circuit
            low = np.r_[ctx.qstate,np.zeros_like(ctx.qstate)]
            high = np.r_[np.zeros_like(ctx.qstate),ctx.qstate]
            expected = np.zeros_like(ctx.qstate)
            expected[oracle.qindices] = oracle.event(event,lookup,oracle.initial)
            actual = simulate_statevector(circuit,high)
            err = [float(np.linalg.norm(actual[len(expected):]-expected)),
                float(np.linalg.norm(actual[:len(expected)])),
                float(np.linalg.norm(simulate_statevector(circuit,low)-low)),
                float(np.linalg.norm(simulate_statevector(circuit.inverse(),actual)-high))]
            require(max(err)<1e-9, 'ONE_BODY_MIXED_EVENT_AND_ACTUAL_ADJOINT')
            checks.append(dict(sign=sign,order=order,errors=err,conditional=True,in_cost_samples=False))
    result = dict(system=system,status='PASS',full_h_action_errors=errors,
        one_body_action_errors=one_errors,deterministic_blocks=0,
        component_probability_sum=math.fsum(c.probability for c in comps),
        conditional_event_checks=checks,phase_alignment=False)
    save(PUBLIC/(system+'_validation.json'),result)
    return result


def run_signal(system, row):
    ctx = context(system)
    record = signal.evolve_batch(ctx,[row])[0]
    record['origin'] = 'FULL_H_FIXED_SAVED_INPUT_FINITE_MEAN'
    save(PUBLIC/(system+'_signal.json'),record)
    endpoints = signal.qualify(record,row)
    save(PUBLIC/(system+'_endpoints.json'),endpoints)
    return endpoints


def cost_job(job):
    native = setup()
    row, _, _ = job
    native.CONTEXT[row['system']] = context(row['system'])
    return native.cost_trial(job)


def run_cost(system, row):
    native = setup()
    native.CONTEXT[system] = context(system)
    endpoints = read(PUBLIC/(system+'_endpoints.json'))
    if not any(ep['status']=='ELIGIBLE' for ep in endpoints):
        save(PUBLIC/(system+'_cost_skipped.json'),dict(reason='No eligible precision for fixed candidate',n=0))
        return
    # Trial zero verifies the actual full wrapper and compiled vectors before
    # releasing the other eight trajectories. No failed trial is retried.
    first = cost_job((row,COHORT,0))
    from pf_common_run_contract import resources
    obs = resources()
    workers = max(1,min(8, int(obs['idle_cpu']), obs['available_ram']//max(2**30,2*first['peak_RSS_bytes'])))
    save(PRIVATE/(system+'_resources.json'),dict(observation=obs,workers=workers,
        allocation='up to the 8 remaining fixed trials; fresh idle CPU/RAM',BLAS_threads_per_worker=1))
    trials = [first]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        pending = {pool.submit(cost_job,(row,COHORT,i)):i for i in range(1,9)}
        for future in as_completed(pending):
            trials.append(future.result())
    trials.sort(key=lambda t:t['index'])
    signal_log = read(PUBLIC/(system+'_signal.json'))['signal']['log_B']
    require(max(abs(t['log_B']-signal_log) for t in trials)<1e-9,'NATIVE_SIGNAL_NORMALIZATION')
    require(len(trials)==9 and {t['index'] for t in trials}==set(range(9)), 'FIXED_NINE_TRIALS')
    result = native.summarize_trials(row,COHORT,trials)
    one_ids = {c.component_id for c in native.CONTEXT[system].preparations[0].rte_preparation.symbolic_tail.components if c.df_fragment_id=='full-h-one-body'}
    observed = 0
    for i in range(9):
        data = read(PRIVATE/'cost'/COHORT/row['id']/str(i)/'events_PRIVATE.json')
        observed += sum(app['component_id'] in one_ids for events in data['events'] for event in events for app in event['application_sequence'])
    result['observed_one_body_applications'] = observed
    require(observed>0, 'ONE_BODY_ACTUALLY_SAMPLED')
    save(PUBLIC/(system+'_costs.json'),result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['validate','signal','cost'])
    p.add_argument('system',choices=['H4','H6'])
    args = p.parse_args()
    setup()
    row = next(r for r in read(PUBLIC/'selection.json')['candidates'] if r['system']==args.system)
    try:
        if args.command=='validate': result = validate(args.system,row)
        elif args.command=='signal':
            require(read(PUBLIC/(args.system+'_validation.json'))['status']=='PASS','VALIDATION_REQUIRED')
            result = run_signal(args.system,row)
        else:
            require(read(PUBLIC/(args.system+'_validation.json'))['status']=='PASS','VALIDATION_REQUIRED')
            result = run_cost(args.system,row)
        print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        save(PUBLIC/(args.system+'_'+args.command+'_STOP.json'),dict(reason=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
