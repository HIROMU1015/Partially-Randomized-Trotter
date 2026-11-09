"""Fixed B4-2 saved-input scalar evaluation; explicit --execute only.

No sampler, state vector, eigensolver, circuit builder, compiler or price calls.
Existing B4-1 adapter is unchanged. Output is fresh/no retry and always STOPs.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
BASE = '31cd67c91abda3c9c5d4b643c434c040b8ceaeca'
B1_RESULT = 'd3913866e6ade4ab97ac02645fed91b2d5fcfa75'
B1_SOURCE = '8c6eeb02e89f6cf14870789719493e75482f62f8'
CARRIER = '3649b75af191916daf44abb91e14d34c43b83e8f'
CDF_SOURCE = '4df8eca29c59ffd55e2dd898f05da16bb246ddbe'
OLD = 'artifacts/research_c_near_practical_r32/2026-11-03/run01/'
B1 = 'artifacts/track_d_b1_c1_validation/2026-11-06/run01/'
FAMILY = 'artifacts/track_d_b4_2_rpe_shots/2026-11-06'
OUTPUT = ROOT / FAMILY / 'run01'
ORIGINAL_H = '1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'
ENDPOINT = 'a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
PHYSICAL_H = '92a8d27bdebb53937f6361c106a13ef90b9110930d4bf4ddb577796cfb208739'
SOURCE_FILES = ('src/trottertracks/track_d_original_h_accuracy.py',
                'src/trottertracks/c1_stage3_accuracy.py',
                'src/trottertracks/c1_stage1.py',
                'src/trotterlib/rpe_resource_accounting.py',
                'src/trotterlib/rpe_target_round_horizon_validation.py',
                'src/trotterlib/rpe_four_round_phase_validation.py')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, data):
    (OUTPUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                        allow_nan=False) + '\n')


def read_fixed(commit, path, records, expected=None, *, decode=True):
    data = git('show', commit + ':' + path)
    digest = sha(data)
    if expected is not None and digest != expected:
        raise ValueError('INPUT_BINDING:' + path)
    records.append(dict(commit=commit, path=path, bytes=len(data), sha256=digest))
    return json.loads(data) if decode else data


def validate_policy(p):
    expected = dict(atoms=4, distance_A=1., basis='STO-3G', rank=8, Nalpha=2,
                    Nbeta=2, sector_dimension=36, system_modes=8, ld=2,
                    frame_order=list(range(8)), delta='1/10', h='1/10', s=1, r=32,
                    K=0, rounds=list(range(13)), original_g_kappa=True,
                    scalar='nuclear once', state_preparation='EXCLUDED')
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('CDF_POLICY_BINDING')


def prepare():
    records = []
    source = {p: sha((ROOT / p).read_bytes()) for p in SOURCE_FILES}
    for p, digest in source.items():
        if digest != sha(git('show', BASE + ':' + p)):
            raise ValueError('B4_1_SOURCE_CHANGED:' + p)
    glue = 'scripts/run_track_d_b4_2.py'
    if (ROOT / glue).read_bytes() != git('show', 'HEAD:' + glue):
        raise ValueError('UNCOMMITTED_EXECUTION_SOURCE')
    source[glue] = sha((ROOT / glue).read_bytes())
    manifest = read_fixed(B1_RESULT, B1 + 'SOURCE.json', records)
    run = read_fixed(B1_RESULT, B1 + 'result.json', records)
    b1 = read_fixed(B1_RESULT, B1 + 'B1/result.json', records)
    stop = read_fixed(B1_RESULT, B1 + 'STOP.json', records)
    if (manifest['execution_source'] != B1_SOURCE or run['source_commit'] != B1_SOURCE
            or not stop['mandatory_stop'] or stop['retry'] or stop['resume']):
        raise ValueError('B1_SOURCE_OR_STOP')
    read_fixed(B1_SOURCE, 'src/trottertracks/track_d_b1_c1.py', records,
               manifest['new_source']['src/trottertracks/track_d_b1_c1.py'], decode=False)
    historical = {}
    for p, binding in manifest['B1_inputs'].items():
        if binding['commit'] != CARRIER:
            raise ValueError('B1_CARRIER')
        # The old own-DF reference signals/N are never used by this evaluation.
        historical[p] = read_fixed(CARRIER, p, records, binding['sha256'],
                                   decode=not p.endswith('B/references.json'))
    contract_path = 'docs/research/research_c_near_practical_r32_downstream_contract_20261103.json'
    c = historical[contract_path]
    validate_policy(c['policy'])
    identity, law = historical[OLD+'PR/A/identity.json'], historical[OLD+'PR/A/law.json']
    if (identity['candidate'] != ENDPOINT or identity['saved_DF']['candidate_sha256'] != ENDPOINT
            or identity['saved_DF']['actual_rank'] != 8 or law['status'] != 'PASS'
            or law['ideal_law'] != 'finite_RTE_IID_reference'
            or law['implemented_finite_bit_law_error'] is not None):
        raise ValueError('ENDPOINT_OR_LAW')
    for p, expected in ((c['input'], ORIGINAL_H), (c['physical_reference'], PHYSICAL_H),
                        ('artifacts/research_c_near_practical/2026-11-03/run01/PR/endpoint.npz', ENDPOINT)):
        if c['saved_inputs'][p] != expected or manifest['original_saved_inputs'][p] != expected:
            raise ValueError('ORIGINAL_H_CDF_PHYSICAL_BINDING')
        read_fixed(CARRIER, p, records, expected, decode=False)  # hash only, no array evaluation
    read_fixed(CARRIER, OLD+'SOURCE.json', records, c['source_manifest_sha256'])
    physical = read_fixed(CARRIER, OLD+'shared/physical_ground.json', records)
    norm = read_fixed(CARRIER, OLD+'PR/A/normalization.json', records)
    read_fixed(CDF_SOURCE, 'src/trottertracks/c_near_practical_r32.py', records, decode=False)
    read_fixed(CDF_SOURCE, 'src/trottertracks/c_pr_aware_df_stage_b_runner.py', records, decode=False)
    if (b1['status'] != 'CONDITIONAL_OPERATOR_SUFFICIENT'
            or b1['model'] != 'exact_original_H_ground_in_invariant_Nalpha2_Nbeta2_sector36'
            or b1['law'] != 'ideal_finite_RTE_IID_exact_native'
            or b1['df_counting'] != 'DF enters evolution/signal error once; no extra Weyl energy addition'
            or b1['rigorous_enclosure'] or b1['own_DF_signal_or_N_reused']
            or b1['state_vector_used'] or b1['N_evaluated']
            or [r['m'] for r in b1['rows']] != list(range(13))):
        raise ValueError('B1_STATE_ERROR_OR_ROUNDS')
    if norm['r'] != 32 or [r['attenuation'] for r in norm['rows']] != [r['attenuation'] for r in b1['rows']]:
        raise ValueError('ATTENUATION_BINDING')
    if (physical['status'] != 'PASS' or not -10 <= physical['energy_Ha'] <= 10
            or not 0 <= physical['residual'] <= 1e-10
            or not 0 <= physical['normalization_error'] <= 1e-12 or physical['alias_margin'] <= 0):
        raise ValueError('SAVED_PHYSICAL_PRIOR_UNQUALIFIED')
    return source, records, b1, physical, law


def evaluate(b1, physical, law, rows=None):
    from trottertracks import track_d_original_h_accuracy as adapter
    target = adapter.OriginalHTarget(energy_ha=.00159, base_time=.1, beta_total=math.pi/6,
        beta_stat=math.pi/12, alpha_axis=.01/26, alpha_total=.01,
        maximum_round_index=12, energy_window_ha=(-10., 10.))
    binding = adapter.SignalBinding(original_h=ORIGINAL_H, cdf=ENDPOINT,
        sector='Nalpha2_Nbeta2_sector36; Qiskit indices=bit_reverse_indices(8)[saved sector basis]',
        law='CDF8 ideal finite-RTE IID; original frames LD2; ordered S2/s1/r32/K0; h=0.1',
        source='B1:'+B1_SOURCE+';saved:'+B1_RESULT+';adapter:'+BASE,
        implementation='ideal exact-native original-H signal; no compiled implementation qualification')
    alias = dict(status='CONDITIONAL_NUMERICAL_PRIOR_PASS', energy_window_ha=[-10.,10.],
        margin_rad=math.pi-(10*target.base_time+target.beta_total), saved_physical_ground=physical,
        physical_H_sha256=PHYSICAL_H, original_H_packet_sha256=ORIGINAL_H,
        source_binding='CDF science execute loads original physical_reference.H and calls ground(H,shared,physical)',
        label_is_not_binding=True, rigorous_enclosure=False,
        qualification='saved binary64 physical lowest eigenpair/residual; exact original-H ground assumed; no new solve')
    if alias['margin_rad'] <= 0 or alias['margin_rad'] != physical['alias_margin']:
        raise ValueError('PRIOR_PHASE_MARGIN')
    horizon = target.horizon()
    plans = []
    if rows is None:
        rows = []
    for saved in b1['rows']:
        evidence = adapter.OriginalHRoundEvidence.from_b1_row(saved, target=target, binding=binding,
            implementation_signal_error=0., finite_law_signal_error=0., law_model=adapter.IDEAL,
            alias_branch_verified=True, iid_per_shot=True,
            operator_provenance=B1_RESULT+':'+B1+'B1/result.json; saved binary64 norm plus guard',
            alias_provenance=CARRIER+':'+OLD+'shared/physical_ground.json; bound to original physical H',
            iid_provenance='ideal independent finite-RTE trajectory per shot and occurrence: explicit model premise',
            implementation_provenance='exact native: explicit model definition, not an actual zero-error bound',
            law_provenance='ideal IID law: explicit model definition, not a finite-bit law certificate')
        plan = adapter.assess_round(target, binding, evidence)
        plans.append(plan)
        row = dict(m=saved['m'], q=saved['q'], time=saved['time'], attenuation=saved['attenuation'],
            original_H_corrected_operator_error=saved['original_H_corrected_operator_error'],
            status=plan['status'], N_X=plan['shots'], N_Y=plan['shots'],
            statistical_phase_budget=target.beta_stat, alpha_X=target.alpha_axis, alpha_Y=target.alpha_axis)
        if plan['status'] != 'CONDITIONAL_ROUND_PASS':
            rows.append(row)
            return dict(status='ROUND_CONDITION_NOT_MET', rows=rows, total_shots=None,
                        mandatory_stop=True, failing_round=saved['m'])
        row.update(raw_signal_error=plan['raw_signal_error'], radius_lower=plan['radius_lower'],
            deterministic_phase_upper=plan['beta_deterministic'],
            statistical_phase_allowance=target.beta_total-plan['beta_deterministic'],
            phase_slack=target.beta_total-plan['beta_total'],
            epsilon_coordinate_X=plan['epsilon_coordinate'], epsilon_coordinate_Y=plan['epsilon_coordinate'],
            radius_positive=plan['radius_lower']>0, phase_budget_met=plan['beta_total']<=target.beta_total)
        # Same saved B1 geometry, now connected to N; no re-transport or state action.
        for key, oldkey in [('raw_signal_error','raw_signal_error'),('radius_lower','observed_radius_lower'),
                            ('deterministic_phase_upper','deterministic_phase_upper')]:
            if not math.isclose(row[key], saved[oldkey], rel_tol=0., abs_tol=2e-15):
                raise ValueError('B1_GEOMETRY_DISCREPANCY:'+str(saved['m']))
        rows.append(row)
    energy = adapter.assess_energy(target, binding, plans)
    total = sum(r['N_X']+r['N_Y'] for r in rows) if energy['same_accuracy'] else None
    return dict(status='CONDITIONAL_IDEAL_13_ROUND_PASS' if energy['same_accuracy'] else 'GLOBAL_CONDITION_NOT_MET',
        target=asdict(target), binding=asdict(binding), rows=rows, alias=alias, horizon=horizon,
        energy=energy, total_shots=total, shot_definition='sum_m(N_m_X+N_m_Y); per-axis sufficient Hoeffding shots',
        ideal_lane=dict(law='ideal finite-RTE IID', native='exact', implementation_raw_error=0.,law_raw_error=0.,
                        premise=b1['norm_qualification'], rigorous_enclosure=False, state='exact original-H ground assumed'),
        actual_lane=dict(status='UNQUALIFIED', N=None, total_shots=None, implemented_finite_bit_law_error=None,
            shot_occurrence_IID_bridge=None, synthesis_compiled_error=None, Reference_chi_adjoint_leakage_error=None,
            all_law_all_round_correctness_coverage=None),
        decoder=dict(id=adapter.DECODER, phase_energy_sign='phi=-hE; E=-phi/h',
            conditions='consecutive dyadic q, per-round total phase below pi/3, alias prior, coordinate error events',
            measurement_data_used=False, success_probability_evaluated=False),
        Stage_C_handoff=dict(round_axis_N='rows.N_X/N_Y and axis_shots.csv', state_preparation='EXCLUDED',
            measurement='X=Re(z),Y=Im(z); H/core/(Y:Sdg)/H; bit0:+1,bit1:-1',
            normalization=law['ideal_law'], circuit_price=None, total_circuit_resources=None,
            same_N_for_Reference_symmetric='not certified; requires full-path/law/implementation error qualification'),
        mandatory_stop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        p.error('explicit --execute required; no evaluation performed')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    start_memory = next((line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')), None)
    result, source, rows = None, None, []
    try:
        hashes, inputs, b1, physical, law = prepare()
        source = dict(execution_source_commit=git('rev-parse','HEAD').decode().strip(), adapter_base=BASE,
            B1_scientific_source=B1_SOURCE, B1_result=B1_RESULT, CDF_scientific_source=CDF_SOURCE,
            saved_carrier=CARRIER, relevant_source_sha256=hashes, fixed_inputs=inputs,
            python=platform.python_version(), packages={k:importlib.metadata.version(k)
                for k in ('numpy','scipy','openfermion','openfermionpyscf','qiskit')})
        save('SOURCE.json', source)
        save('preflight.json', dict(status='SOURCE_INPUT_BINDINGS_PASS',
             physical_alias_qualification='saved numerical record bound by fixed generating SOURCE; not state label',
             B1_DF_counting='already in original-H operator error once', new_state_vectors_used=False))
        result = evaluate(b1, physical, law, rows)
        for name, digest in hashes.items():
            if sha((ROOT/name).read_bytes()) != digest:
                raise ValueError('EXECUTION_SOURCE_CHANGED:'+name)
        save('result.json', result)
        if result.get('rows'):
            with (OUTPUT/'axis_shots.csv').open('w', newline='') as f:
                writer=csv.DictWriter(f, fieldnames=('m','q','time','axis','N','alpha','qualification'))
                writer.writeheader()
                for row in result['rows']:
                    for axis in ('X','Y'):
                        writer.writerow(dict(m=row['m'],q=row['q'],time=row['time'],axis=axis,N=row['N_'+axis],
                                             alpha=row['alpha_'+axis],qualification=row['status']))
        print(json.dumps(dict(status=result['status'],total_shots=result.get('total_shots'),
                              energy=result.get('energy'),mandatory_stop=True)))
    except BaseException as exc:
        save('failure.json',dict(error=repr(exc),mandatory_stop=True))
        save('partial_rows.json',rows)
        raise
    finally:
        save('STOP.json',dict(mandatory_stop=True,status=result['status'] if result else 'FAILED_OR_UNQUALIFIED',
            retry=0,resume=0,scientific_run_attempts=1,new_sampling=0,quantum_shots=0,Monte_Carlo_trials=0,
            ground_solves=0,DF_refits=0,B2_started=False,circuit_builds=0,compile_calls=0,
            total_circuit_cost_evaluated=False,C4_modified=False,detailed_scientific_review=False))
        memory = next((line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')),None)
        save('resources.json',dict(wall_seconds=time.monotonic()-start,
            process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            affinity_CPUs=len(os.sched_getaffinity(0)),MemAvailable_at_start_KiB=start_memory,
            MemAvailable_at_end_KiB=memory,
            policy='one short scalar process; no artificial CPU/RAM cap; existing venv read-only; no shared changes'))


if __name__ == '__main__':
    main()
