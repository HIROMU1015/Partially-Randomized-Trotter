"""Five fixed arms / fifteen points, diagnostic only; frozen kernels unchanged.

HP powers use the exact values of the already constructed binary64 matrices.
There is no unitary projection, new PF, eligibility decision, shot or circuit.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import time

import mpmath
import numpy as np

from . import b2_p1_strong_pf_validation as old

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_COMMIT = '2583ce4957ed44e914ce6571e3e676830439dfa7'
ORIGINAL_SOURCE = 'a7940b58c6cb939d77e9f8e44d214b3787b7e101'
ORIGINAL_PATH = 'artifacts/track_a_b2_p1_strong_pf/2026-11-07/run01/result.json'
SCOPE = (('Full_S2', 4, (0, 16, 17)), ('Full_S2', 8, (0, 15, 16)),
         ('Yoshida4', 2, (0, 16, 17)), ('Yoshida4', 4, (0, 15, 16)),
         ('Yoshida4', 8, (0, 14, 15)))
PRECISIONS = (80, 120)


@lru_cache(None)
def hp_helpers():
    """Reuse the preceding artificial diagnostic's exact binary64 converter."""
    path = ROOT/'scripts/diagnose_b2_p1_strong_pf_numerics.py'
    spec = importlib.util.spec_from_file_location('b2_existing_hp_tools', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def maxabs(matrix):
    value = float(np.max(np.abs(matrix), initial=0))
    old.p.require(math.isfinite(value), 'NONFINITE_MATRIX_DIAGNOSTIC')
    return value


def defect(matrix):
    return maxabs(matrix.conj().T@matrix-np.eye(len(matrix)))


def hp_text(ctx, value):
    return ctx.nstr(value, n=ctx.dps)


def hp_signal_record(ctx, value):
    return dict(real=hp_text(ctx, value.real), imag=hp_text(ctx, value.imag),
                radius=hp_text(ctx, abs(value)), phase_rad=hp_text(ctx, ctx.arg(value)))


def hp_difference(ctx, left, right):
    old.p.require(abs(left)>0 and abs(right)>0, 'UNDEFINED_HP_PHASE')
    return dict(delta=hp_signal_record(ctx, left-right),
        complex_absolute=hp_text(ctx, abs(left-right)),
        radius_absolute=hp_text(ctx, abs(abs(left)-abs(right))),
        phase_rad=hp_text(ctx, abs(ctx.arg(left*ctx.conj(right)))))


def mp_copy(ctx, matrix):
    # Cross-context conversion without binary64 or decimal intermediary.
    return ctx.matrix([[ctx.mpc(x.real, x.imag) for x in row] for row in matrix.tolist()])


def hp_expectation(matrix, psi):
    return (psi.H*matrix*psi)[0]


def power_diagnosis(native, car, psi64, rounds, records64):
    """One arm: shared square ladders, only requested points get signals.

    Each square is mpmath matrix ** 2 (existing integer-power implementation).
    No signals/qualifications are computed at intermediate, unrequested rounds.
    """
    helpers = hp_helpers(); contexts = {}; powers = {}; states = {}
    for digits in PRECISIONS:
        ctx = mpmath.mp.clone(); ctx.dps = digits; contexts[digits] = ctx
        states[digits] = helpers.mp_matrix(ctx, np.asarray(psi64)[:, None])
        powers[digits] = {name:helpers.mp_matrix(ctx, a) for name,a in (('N',native),('C',car))}
    ctx = contexts[120]; rows = []; match = ctx.mpf(old.p.MATCH)
    for m in range(max(rounds)+1):
        if m:
            for digits in PRECISIONS:
                for name in ('N','C'):
                    powers[digits][name] = powers[digits][name]**2
        if m not in rounds:
            continue
        signals = {}; stability = {}
        for name in ('N','C'):
            high = powers[120][name]; low = mp_copy(ctx, powers[80][name])
            z_high = hp_expectation(high, states[120])
            low_signal = hp_expectation(powers[80][name], states[80])
            z_low = ctx.mpc(low_signal.real, low_signal.imag)
            differences = hp_difference(ctx, z_high, z_low)
            matrix_difference = helpers.error(ctx, high, low)
            old.p.require(matrix_difference<=match and all(ctx.mpf(differences[k])<=match
                for k in ('complex_absolute','radius_absolute','phase_rad')), 'HP_PRECISION_NOT_STABLE')
            z64 = records64[m][name]['value']
            z64_hp = ctx.mpc(z64.real, z64.imag)
            signals[name] = (z64_hp, z_high)
            power64 = records64[m][name]['matrix']
            contract64 = hp_expectation(helpers.mp_matrix(ctx, power64), states[120])
            stability[name] = dict(dps=list(PRECISIONS),
                operator_max_abs_80_vs_120=hp_text(ctx, matrix_difference),
                signal_80=hp_signal_record(contexts[80], low_signal),
                signal_120=hp_signal_record(ctx, z_high), differences=differences,
                float64_power_HP_contraction=hp_signal_record(ctx, contract64),
                matrix_power_only_difference=hp_signal_record(ctx, contract64-z_high),
                contraction_rounding_difference=hp_signal_record(ctx, z64_hp-contract64),
                precision_stable=True)
        n64, nh = signals['N']; c64, ch = signals['C']
        dn = n64-nh; dc = c64-ch; ds = nh-ch
        net = dn-dc; total = n64-c64; residual = total-(dn+ds-dc)
        old.p.require(abs(residual)<=match, 'COMPLEX_DECOMPOSITION_IDENTITY')
        # Threshold-scale descriptions, not fitted percentages or claims about
        # which construction is closer to an exact physical PF operator.
        if abs(ds)>match and abs(net)<=match:
            classification = 'STEP_DIFFERENCE_REMAINS_ABOVE_FIXED_THRESHOLD'
        elif abs(ds)<=match and abs(net)>match:
            classification = 'POWER_PROCESSING_DIFFERENCE_ABOVE_FIXED_THRESHOLD'
        elif abs(ds)>match and abs(net)>match:
            classification = 'BOTH_TERMS_ABOVE_FIXED_THRESHOLD'
        else:
            classification = 'INCONCLUSIVE'
        cross = 2*ctx.re(ds*ctx.conj(net))
        cosine = ctx.re(ds*ctx.conj(net))/(abs(ds)*abs(net)) if ds and net else None
        rows.append(dict(m=m, Q=1<<m, T_Ha_inverse=(1<<m)*old.p.DELTA,
            signals=dict(N64=old.p.signal_record(records64[m]['N']['value']),
                C64=old.p.signal_record(records64[m]['C']['value']),
                N_HP=hp_signal_record(ctx,nh), C_HP=hp_signal_record(ctx,ch)),
            comparisons=dict(N64_vs_NHP=hp_difference(ctx,n64,nh),
                C64_vs_CHP=hp_difference(ctx,c64,ch), NHP_vs_CHP=hp_difference(ctx,nh,ch),
                N64_vs_C64=hp_difference(ctx,n64,c64)),
            precision_check=stability,
            decomposition=dict(D_N_power=hp_signal_record(ctx,dn),D_C_power=hp_signal_record(ctx,dc),
                D_step=hp_signal_record(ctx,ds),D_power_net=hp_signal_record(ctx,net),
                D_total=hp_signal_record(ctx,total),identity_residual=hp_text(ctx,abs(residual)),
                cross_term_2Re_step_conj_power=hp_text(ctx,cross),
                direction_cosine=None if cosine is None else hp_text(ctx,cosine),
                destructive_interference=bool(cross<0),
                classification=classification,qualification_changed=False),
            original_binary64_reproduction=records64[m]['reproduction'],
            status='DIAGNOSTIC_POINT_COMPLETE_NO_REQUALIFICATION'))
        print('HP diagnostic m='+str(m)+' complete', flush=True)
    return rows


def gaussian_diagnostics(inputs, model, reports, generators):
    occupation = np.asarray([[(j>>bit)&1 for bit in range(model.modes)] for j in range(1<<model.modes)])
    bases = []; energies = []; rows = []; ix = np.ix_(inputs['indices'], inputs['indices'])
    for i, block in enumerate(model.blocks):
        orbital = np.asarray(block.rotation)
        basis, leak = old.sector_lift(orbital, inputs['ref'], inputs['indices'])
        eigenvalues = np.asarray(reports[i]['eigenvalues'])
        linear = occupation@eigenvalues
        energy = linear if i==0 else inputs['ham'].lambdas[i-1]*linear**2
        generator = (basis*energy[inputs['indices']])@basis.conj().T
        original = inputs['ham'].one_body if i==0 else inputs['ham'].g_matrices[i-1]
        rows.append(dict(block_id=block.block_id, orbital_unitarity_max_abs=defect(orbital),
            orbital_generator_reconstruction_max_abs=maxabs((orbital*eigenvalues)@orbital.conj().T-original),
            sector_Fock_unitarity_max_abs=defect(basis), sector_leakage_max_abs=leak,
            sector_generator_vs_CAR_max_abs=maxabs(generator-generators[i]),
            signed_factor_preserved=True))
        old.p.require(all(math.isfinite(v) and v<=old.p.MATCH for k,v in rows[-1].items()
            if k.endswith('max_abs')), 'GAUSSIAN_REFERENCE_INELIGIBLE')
        bases.append(basis); energies.append(energy)
    return occupation, bases, energies, rows


def schedule_diagnostics(inputs, model, schedule, bases, energies, cache):
    groups = old.strong.native.runs(model, schedule, optimized=True)
    transitions = {}; diagonals = {}
    for k, group in enumerate(groups):
        key = tuple((i,str(t)) for i,t in group)
        if key not in diagonals:
            actual = old.diagonal_values(old.strong.native.combined_diagonal(model, group), model.modes)
            direct_phase = sum((float(t)*energies[i] for i,t in group), np.zeros(1<<model.modes))
            direct = np.exp(-1j*direct_phase)
            difference = maxabs(actual-direct)
            old.p.require(difference<=old.p.MATCH, 'DIAGONAL_REFERENCE_INELIGIBLE')
            diagonals[key] = dict(group=[dict(block=i,time_fraction=str(t)) for i,t in group],
                primitive_vs_occupation_max_abs=difference,
                diagonal_unitarity_max_abs=maxabs(abs(actual)**2-1))
        if k+1<len(groups):
            i,j = group[0][0],groups[k+1][0][0]
            if (i,j) not in cache:
                vi=np.asarray(model.blocks[i].rotation);vj=np.asarray(model.blocks[j].rotation)
                orbital=vj.conj().T@vi
                relative, leak=old.sector_lift(orbital,inputs['ref'],inputs['indices'])
                difference=maxabs(relative-bases[j].conj().T@bases[i])
                old.p.require(difference<=old.p.MATCH, 'TRANSITION_REFERENCE_INELIGIBLE')
                cache[i,j]=dict(previous=i,following=j,orbital_unitarity_max_abs=defect(orbital),
                    sector_unitarity_max_abs=defect(relative),homomorphism_max_abs=difference,
                    sector_leakage_max_abs=leak)
            transitions[i,j]=cache[i,j]
    return dict(group_count=len(groups), transitions=list(transitions.values()),
                unique_diagonal_groups=list(diagonals.values()),
                order_and_signed_stage_source='unchanged strong_baseline_schedule',
                scalar_once=True, factor_identity_in_existing_primitives=True)


def reproduce64(native, car, psi, rounds, saved_arm):
    records = {}
    for m in rounds:
        prior=next(r for r in saved_arm['rounds'] if r['m']==m)
        item={}
        for name,step,field in (('N',native,prior['signals']['PF']),('C',car,prior['independent_PF'])):
            power=np.linalg.matrix_power(step,1<<m)
            value=complex(np.vdot(psi,power@psi))
            old.p.require(old.p.signal_record(value)==field,'ORIGINAL_BINARY64_SIGNAL_NOT_REPRODUCED')
            item[name]=dict(matrix=power,value=value)
        difference=old.p.signal_difference(item['N']['value'],item['C']['value'])
        old.p.require(difference==prior['qualification']['independent_difference'],
                      'ORIGINAL_STOP_DIFFERENCE_NOT_REPRODUCED')
        item['reproduction']=dict(exact_saved_scalar_equality=True,
            original_status=prior['status'],original_difference=difference,
            original_result_commit=ORIGINAL_COMMIT,original_execution_SOURCE=ORIGINAL_SOURCE)
        records[m]=item
    return records


def write_result(output, result):
    (output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def run(private_dir, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source=old.p.git('rev-parse','HEAD').decode().strip()
    paths=['src/trotterlib/b2_p1_strong_pf_h4_diagnosis.py',
        'scripts/run_b2_p1_strong_pf_h4_diagnosis.py','tests/test_b2_p1_strong_pf_h4_diagnosis.py']
    for path in paths:
        old.p.require((ROOT/path).read_bytes()==old.p.git('show',source+':'+path),'COMMITTED_DIAGNOSTIC_SOURCE')
    result=dict(schema_version='b2_p1_strong_pf_fixed_15_h4_diagnosis_v1',execution_SOURCE=source,
        SOURCE_SHA256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        started_UTC=datetime.now(timezone.utc).isoformat(),arms=[],mandatory_stop=True,
        historical_qualification_changed=False,algorithm_repaired=False,
        HP_semantics='Exact stored binary64 step and state values; powers/contractions only at 80/120 decimal digits',
        interpretation='D_power includes binary64 contraction rounding; auxiliary contraction split supplied',
        precision_dps=list(PRECISIONS),independent_tolerance=old.p.MATCH,
        classification_rule='Compare |D_step| and |D_N_power-D_C_power| to unchanged 1e-9; no absolute-value percentages',
        no_projection_normalization_phase_fit=True, guarantee_status='not_certified')
    try:
        original,meta=old.saved.fixed_json(ORIGINAL_COMMIT,ORIGINAL_PATH)
        old.p.require(original['execution_SOURCE']==ORIGINAL_SOURCE and
            original['status']=='FIXED_EIGHT_ARM_SCREEN_COMPLETE','ORIGINAL_RESULT_SOURCE')
        for path,sha in original['SOURCE_SHA256'].items():
            old.p.require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha and
                (ROOT/path).read_bytes()==old.p.git('show',ORIGINAL_SOURCE+':'+path),'FROZEN_KERNEL_SOURCE')
        prior_helper='scripts/diagnose_b2_p1_strong_pf_numerics.py'
        old.p.require((ROOT/prior_helper).read_bytes()==old.p.git('show','e217d145bd99c2b34302017390ecb57987c06b90:'+prior_helper),'REUSED_HP_CONVERTER_SOURCE')
        _,_,identity,conditions,refs=old.checked_references()
        old.p.require(original['scientific_context']==identity and original['P1_conditions']==conditions,'SHARED_P1_CONTEXT')
        inputs=old.p.load_private_inputs(private_dir)
        old.p.require(all(inputs['public'][k]==v for k,v in identity.items()),'PRIVATE_INPUT_CONTEXT')
        result.update(original_result=meta,frozen_kernel_SOURCE_SHA256=original['SOURCE_SHA256'],
            reused_HP_converter_SHA256=hashlib.sha256((ROOT/prior_helper).read_bytes()).hexdigest(),
            input_identity=inputs['public'],scientific_context=identity,P1_conditions=conditions,
            saved_reference_inputs=refs,historical_qualified_arms=original['qualified_arms'])
        ham=inputs['ham'];indices=inputs['indices'];ix=np.ix_(indices,indices)
        full_generators=[inputs['one_body'],*inputs['fragments']]
        outside=np.setdiff1d(np.arange(len(inputs['state'])),indices)
        leaks=[max(maxabs(g[np.ix_(outside,indices)]),maxabs(g[np.ix_(indices,outside)])) for g in full_generators]
        old.p.require(max(leaks)<=old.p.MATCH,'SECTOR_INVARIANCE')
        generators=tuple(g[ix] for g in full_generators)
        model,reports=old.strong.prepare_xdf_basis(ham.one_body,ham.lambdas,ham.g_matrices,
            ham.constant,source_identity=ORIGINAL_SOURCE,input_identity=identity['df_hamiltonian'],basis_policy=old.strong.SPIN)
        bases_cache={};relative_cache={};exponential_cache={};jobs=[]
        for formula,s,rounds in SCOPE:
            arm_id=formula+'_s'+str(s);prior=next(a for a in original['arms'] if a['arm_id']==arm_id)
            old.p.require(prior['status']=='STOP_ARM_INELIGIBLE' and prior['first_ineligible_round']==rounds[-1], 'FIXED_STOP_ARM')
            schedule=old.strong_baseline_schedule(len(model.blocks),substeps=s,formula=formula,delta_time=old.p.DELTA)
            old.p.require([dict(block_index=i,time_fraction=str(t),time=float(t)) for i,t in schedule]==prior['base_delta_schedule'], 'FIXED_SCHEDULE')
            native,leak=old.native_mean_step(model,schedule,inputs['ref'],indices,basis_cache=bases_cache,relative_cache=relative_cache)
            car=np.exp(-1j*ham.constant*old.p.DELTA)*old._ordered_exponentials(generators,schedule,exponential_cache)
            comparison=maxabs(native-car);nu=defect(native)
            old.p.require(comparison==prior['base_step_independent_max_abs'] and
                nu==prior['base_step_unitarity_max_abs'] and leak==prior['native_sector_leakage_max_abs'],'ORIGINAL_BASE_STEP_NOT_REPRODUCED')
            e=car.conj().T@(native-car)
            arm=dict(arm_id=arm_id,formula=formula,substeps=s,requested_rounds=list(rounds),rounds=[],
                original_status=prior['status'],original_first_stop=prior['first_ineligible_round'],
                base_step=dict(independent_entrywise_max_abs=comparison,native_unitarity_entrywise_max_abs=nu,
                    CAR_unitarity_entrywise_max_abs=defect(car),native_matrix_hash=old.p._array_hash(native),
                    CAR_matrix_hash=old.p._array_hash(car),native_sector_leakage=leak,
                    relative_perturbation_Hermitian_max_abs=maxabs((e+e.conj().T)/2),
                    relative_perturbation_antiHermitian_max_abs=maxabs((e-e.conj().T)/2),
                    relative_perturbation_definition='C.conj().T @ (N-C); linearized amplitude/phase diagnostic only',
                    exact_old_base_diagnostic_reproduction=True),status='BASE_STEP_REPRODUCED')
            result['arms'].append(arm);jobs.append((native,car,inputs['sector_state'],rounds,prior,schedule))
            print(arm_id+' original base diagnostics reproduced',flush=True)
        _,bases,energies,basis_rows=gaussian_diagnostics(inputs,model,reports,generators)
        result.update(gaussian_basis_diagnostics=basis_rows,generator_sector_leakage_max_abs=leaks)
        transition_cache={};prepared=[]
        for arm,job in zip(result['arms'],jobs,strict=True):
            native,car,psi,rounds,prior,schedule=job
            arm['components']=schedule_diagnostics(inputs,model,schedule,bases,energies,transition_cache)
            records=reproduce64(native,car,psi,rounds,prior)
            arm['all_original_binary64_points_reproduced']=True
            prepared.append((native,car,psi,rounds,records))
        result['shared_input_and_binary64_reproduction']='PASS'
        result['environment_before_HP']=dict(host=platform.node(),python=platform.python_version(),numpy=np.__version__,
            scipy=importlib.metadata.version('scipy'),mpmath=mpmath.__version__,CPU_affinity=len(os.sched_getaffinity(0)),
            load_average=os.getloadavg(),meminfo_available_kB=next(line.strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')),
            worker_processes=len(SCOPE),worker_policy='one per approved independent arm; no BLAS/CPU/RAM caps')
        write_result(output,result)
        with ProcessPoolExecutor(max_workers=len(SCOPE),mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(power_diagnosis,*job):i for i,job in enumerate(prepared)}
            for future in as_completed(futures):
                i=futures[future];rows=future.result();arm=result['arms'][i]
                old.p.require([r['m'] for r in rows]==arm['requested_rounds'],'EXACTLY_THREE_APPROVED_POINTS')
                arm.update(rounds=rows,status='ARM_DIAGNOSIS_COMPLETE_NO_REQUALIFICATION')
                write_result(output,result)
                print(arm['arm_id']+' HP diagnostic complete',flush=True)
        old.p.require(sum(len(a['rounds']) for a in result['arms'])==15,'EXACTLY_FIFTEEN_POINTS')
        result['status']='FIXED_FIFTEEN_POINT_DIAGNOSIS_COMPLETE_NO_REQUALIFICATION'
        for name,path in inputs['files'].items():
            old.p.require(hashlib.sha256(path.read_bytes()).hexdigest()==inputs['public']['file_SHA256'][name],'PRIVATE_INPUT_UNCHANGED')
    except Exception as error:
        result.update(status='STOP_DIAGNOSTIC_INELIGIBLE',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;write_result(output,result)
    fields=['arm_id','m','Q','T_Ha_inverse','N64_real','N64_imag','C64_real','C64_imag',
        'N_HP_real','N_HP_imag','C_HP_real','C_HP_imag','D_N_power_abs','D_C_power_abs',
        'D_step_abs','D_power_net_abs','D_total_abs','direction_cosine','identity_residual','classification','status']
    with (output/'rounds.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for arm in result['arms']:
            for row in arm['rounds']:
                d=row['decomposition'];record={k:row[k] for k in ('m','Q','T_Ha_inverse','status')};record['arm_id']=arm['arm_id']
                for name,sig in row['signals'].items():
                    record[name+'_real']=sig['real'];record[name+'_imag']=sig['imag']
                for name in ('D_N_power','D_C_power','D_step','D_power_net','D_total'):
                    record[name+'_abs']=d[name]['radius']
                record.update({k:d[k] for k in ('direction_cosine','identity_residual','classification')});writer.writerow(record)
    return result
