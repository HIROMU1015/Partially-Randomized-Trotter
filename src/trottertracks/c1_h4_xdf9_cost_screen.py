"""Dormant XDF9 native/optimistic-prescription cost screen; no science on import.

Reuse the precision native builder, fixed P0, RPE solver, serial composition and
existing rational optimistic aggregator. Real molecules are only read by run().
Structural checks are deliberately not a full-controlled-operator certificate.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import csv
import importlib.metadata
import math
import os
import resource
import subprocess
import time
import traceback

from . import c1_h4_native_fair as fair
from . import c1_h4_xdf9_pf as pf
from . import c1_h4_ca_resources as ca
from . import c1_stage3_cost_only as cost
from . import c1_stage3_accuracy as accuracy

io, native, integrated = fair.io, fair.native, fair.integrated
SCOPE = 'C1_H4_XDF9_NATIVE_OPTIMISTIC_COST_SCREEN_V1'
FAMILY = 'artifacts/research_c_c1_h4_xdf9_cost_screen'
CONDITIONS = dict(pf.CONDITIONS, route=fair.CONDITIONS['route'],
    serial_policy=fair.CONDITIONS['serial_policy'], full_operator_coverage=False)
TARGET = dict(ca.TARGET)
MONOTONICITY = 'nonnegative_extra_signal_error_fixed_Hoeffding_ceil_and_serial_v1'
LIMITS = dict(saved_input_load=1, saved_identity=1, packet_gate=1, native_model=1,
    gaussian_build=10, model_structure=1, kernel_build=2, raw_structure=2,
    compile=2, compiled_structure=2, metrics=2, optimistic_round=26,
    round_serial_cost=26, optimistic_aggregate=2, screen_compare=4)
MAX_CALLS = sum(LIMITS.values())
FORBIDDEN = ('new_fit', 'optimizer', 'rank_search', 'DF_recomputation', 'PF_recomputation',
    'full_operator', 'state_action', 'synthesis_evaluation', 'CA_evaluation',
    'quantum_sampling', 'baseline_build_compile', 'baseline_reaggregation', 'retry', 'resume')
RECORD_KEYS = {'PF', 'PF_SOURCE', 'PF_STOP', 'PF_contract', 'native', 'native_SOURCE',
    'native_STOP', 'native_contract', 'fairness', 'resources', 'resources_SOURCE',
    'resources_STOP', 'resources_contract'}
BASELINES = {'1': {'XDF': 128445454592, 'CDF': 113219727909},
             '4': {'XDF': 470217718215, 'CDF': 413834761607}}
LINEAGE = dict(PF_SOURCE='9a4cda44a45675c2518b2252a98e78eeeb320942',
    native_SOURCE=ca.NATIVE_SOURCE, resources_SOURCE='18045333a1438740160d7b9e44e78139ad59f2b5')


def preflight(root, path, input_path):
    """Bytes/policy/versions only; NPZ/QPY and saved numerical results untouched."""
    root = Path(root).resolve(); c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions'] != CONDITIONS or c['accuracy_target'] != TARGET
            or c['compiler'] != fair.profile_dict() or c['gaussian'] != fair.GAUSSIAN
            or c['compiler_phase'] != fair.PHASE or c['monotonicity'] != MONOTONICITY
            or c['lineage'] != LINEAGE or c['baseline_total_RZ'] != BASELINES
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or c['fresh_output'] != FAMILY+'/<execution-date>/run01'
            or any(c[k] for k in FORBIDDEN)):
        raise ValueError('XDF9_COST_SCREEN_SCOPE')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, expected in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != expected:
                raise ValueError('SOURCE_OR_INPUT: '+name)
    if (io.sha(input_path) != io.P_H4_SHA256 or c['original_input']['sha256'] != io.P_H4_SHA256
            or set(c['records']) != RECORD_KEYS
            or not set(c['records'].values()) | {c['baseline_csv']} <= c['saved_inputs'].keys()):
        raise ValueError('ORIGINAL_INPUT_OR_UNPINNED_RECORD')
    for name, version in c['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('ENVIRONMENT: '+name)
    return c


def fresh_output(root, output):
    from datetime import date
    root, output = Path(root).resolve(), Path(output).resolve()
    parts = output.relative_to(root/FAMILY).parts
    if (len(parts) != 2 or date.fromisoformat(parts[0]).isoformat() != parts[0]
            or parts[1] != 'run01' or output.exists()):
        raise ValueError('FRESH_RUN01_NO_RETRY_RESUME')
    return output


def load_bundle(root, c, input_path):
    packet = io.load_input(input_path, evaluation=True)
    for value in packet.values(): value.setflags(write=False)
    data = {k: io.read(Path(root)/p) for k, p in c['records'].items()}
    with (Path(root)/c['baseline_csv']).open() as stream:
        data['baseline_csv'] = list(csv.DictReader(stream))
    return dict(packet=packet, evidence=data)


def saved_identity(bundle, c):
    """Only attach saved results. Never regenerate old PF, shots or totals."""
    from decimal import Decimal
    b = bundle['evidence']
    for name, units in [('PF', 95), ('native', 305), ('resources', 119)]:
        stop, source = b[name+'_STOP'], b[name+'_SOURCE']
        if (stop['failed'] or not stop['mandatory_stop'] or stop['scientific_units'] != units
                or stop['retry'] or stop['resume'] or source['execution_commit'] != LINEAGE[name+'_SOURCE']):
            raise ValueError('SAVED_RUN_IDENTITY: '+name)
    if (b['PF_contract']['scope'] != pf.SCOPE or b['PF_contract']['conditions'] != pf.CONDITIONS
            or b['PF_contract']['original_input']['sha256'] != c['original_input']['sha256']
            or b['native_contract']['conditions'] != fair.CONDITIONS
            or b['native_contract']['original_input']['sha256'] != c['original_input']['sha256']
            or b['resources_contract']['conditions'] != fair.CONDITIONS
            or b['resources_contract']['accuracy_target'] != TARGET):
        raise ValueError('SAVED_CONDITIONS_OR_TARGET')
    for record in (b['native_contract'], b['resources_contract'], b['fairness']):
        if (record['compiler'] != fair.profile_dict() or record['gaussian'] != fair.GAUSSIAN
                or record['compiler_phase'] != fair.PHASE): raise ValueError('SAVED_PRECISION_P0')
    if (b['fairness']['status'] != 'PASS' or b['fairness']['route'] != CONDITIONS['route']
            or b['fairness']['serial_policy'] != CONDITIONS['serial_policy']
            or b['fairness']['additional_relative_Gaussian_fusion']
            or b['native']['status'] != 'H4_FAIR_NATIVE_FULL512_PILOT_COMPLETE'
            or b['resources']['status'] != 'H4_SAVED_CA_RPE_SERIAL_RESOURCES_COMPLETE'):
        raise ValueError('SAVED_NATIVE_OR_COST_POLICY')
    f = b['PF']
    accuracy._finite(f['saved_DF']['saved_norm_plus_allowance_Ha'], 'saved DF norm plus guard')
    if (f['status'] != 'XDF9_STAGE1_COMPLETE' or f['representation'] != 'PASS'
            or f['ground']['status'] != 'PASS' or f['saved_DF']['rank'] != 9
            or f['saved_DF']['status'] != 'PASS'
            or f['saved_DF']['saved_norm_plus_allowance_Ha'] > .00019875):
        raise ValueError('SAVED_XDF9_DF_GROUND')
    for s in (1, 4):
        v = f['candidates'][str(s)]
        if v['summary']['status'] != 'POTENTIALLY_FEASIBLE' or [r['m'] for r in v['rounds']] != list(range(13)):
            raise ValueError('SAVED_PF_COVERAGE')
        for r in v['rounds']:
            if (r['target'] != 'XDF9' or r['s'] != s or r['time'] != .1*(1<<r['m'])
                    or r['kernel_time'] != .1/s or r['occurrences'] != s*(1<<r['m'])
                    or r['attenuation'] != 1 or r['rte_corrected_operator_error'] != 0
                    or r['sufficient_condition'] != 'PASS'):
                raise ValueError('SAVED_PF_TARGET_OR_LAW')
            accuracy._finite(r['pf_operator_error'], 'saved PF')
    lookup = {x['candidate']: x for x in b['baseline_csv']}
    if len(lookup) != 4: raise ValueError('BASELINE_CSV_COVERAGE')
    for s in (1, 4):
        for arm in ('CDF', 'XDF'):
            key = f'{arm}:s{s}'; record = b['resources']['candidates'][key]
            expected = BASELINES[str(s)][arm]
            if (record['status'] != 'NUMERICAL_PILOT_CA_PASS'
                    or Decimal(str(record['cost']['totals']['rz_count'])) != expected
                    or Decimal(lookup[key]['rz_count']) != expected
                    or lookup[key]['status'] != record['status']):
                raise ValueError('SAVED_BASELINE_TOTAL_IDENTITY')
    shells = {k: ca.reused.metric_record(v, measurement=int(k != 'head'))
        for k, v in b['native']['shell_metrics'].items()}
    if set(shells) != {'head', 'X', 'Y'}: raise ValueError('SAVED_SHELL_COVERAGE')
    return dict(status='PASS', shell_metrics=shells, baselines=BASELINES,
        full_operator_correctness_of_XDF9=False, actual_synthesis_error=None)


def packet_gate(packet):
    fair.stage.finite_arrays(*packet.values())
    kappa = pf.spatial_spin_lift(packet['df_one_body'])
    for G in packet['df_g_matrices'][:9]: pf.spatial_spin_lift(G)
    check = fair.component_check(kappa, fair.stage.corrected_kappa(packet['one_mo'], packet['eri_chemist']),
        1e-10, label='original-g kappa')
    if len(packet['df_lambdas']) < 9 or len(packet['df_g_matrices']) < 9:
        raise ValueError('PREFIX9_REQUIRED')
    return dict(status='PASS', kappa=check, rank=9, frame_order=list(range(9)),
        scalar=float(packet['nuclear_repulsion']), signed_lambdas_preserved=True)


def prepare_model(packet, input_identity):
    return native.prepare_xdf(packet['df_one_body'], packet['df_lambdas'][:9],
        packet['df_g_matrices'][:9], float(packet['nuclear_repulsion']),
        source_identity=input_identity+':prefix9', input_identity=input_identity)


def model_structure(model, builder, packet):
    if (model.label != 'XDF' or model.modes != len(packet['df_one_body'])
            or [b.block_id for b in model.blocks] != ['one_body']+[f'F{i}' for i in range(9)]
            or model.constant != float(packet['nuclear_repulsion'])
            or not isinstance(builder, fair.PrecisionBuilder)):
        raise ValueError('MODEL_ORDER_OR_SCALAR')
    for block in model.blocks:
        r = builder.basis_reports['XDF:'+block.block_id]
        if (r['version'] != fair.GAUSSIAN or not math.isfinite(r['single_particle_max_entry'])
                or r['single_particle_max_entry'] > 1e-10): raise ValueError('GAUSSIAN_TRANSFER')
    return dict(status='PASS', gaussian=fair.GAUSSIAN, compiler_phase=fair.PHASE,
        full_operator_correctness=False, coverage='single-particle transfers and native model structure')


def _signature(op, qubits):
    from qiskit.circuit import Gate
    if not isinstance(op, Gate) or op.num_clbits or getattr(op, 'condition', None) is not None:
        raise ValueError('UNITARY_GATE_STRUCTURE_REQUIRED')
    params = tuple(float(x) for x in op.params)
    if not all(math.isfinite(x) for x in params): raise ValueError('NONFINITE_GATE_PARAMETER')
    return (op.base_class.__module__+'.'+op.base_class.__name__, op.name,
        tuple(qubits), params, getattr(op, 'ctrl_state', None))


def circuit_structure(qc, modes, *, compiled):
    from qiskit.circuit.library import RZGate, SXGate, XGate, CXGate
    if (qc.num_qubits != modes+1 or qc.num_clbits or qc.parameters
            or not math.isfinite(float(qc.global_phase))): raise ValueError('COHERENT_BOUND_CIRCUIT_REQUIRED')
    allowed = {'rz','sx','x','cx'} if compiled else {'rz','xx_plus_yy','crz','crzz','p'}
    classes = {'rz':RZGate, 'sx':SXGate, 'x':XGate, 'cx':CXGate}
    signatures = []
    for item in qc.data:
        if item.operation.name not in allowed: raise ValueError('UNEXPECTED_GATE_OR_NONUNITARY')
        if compiled and item.operation.base_class is not classes[item.operation.name]:
            raise ValueError('KNOWN_P0_GATE_CLASS_REQUIRED')
        signatures.append(_signature(item.operation, [qc.find_bit(q).index for q in item.qubits]))
    return signatures


def raw_structure(qc, model, builder, h):
    """Audit instruction itinerary using existing emitters, not full matrices.

    The recording sink allocates no additional kernel QuantumCircuit. Basis
    inverse, central relative phases and scalar-on-control are compared exactly.
    The emitter itself is independently CAR-tested on artificial circuits.
    """
    from qiskit.circuit.library import PhaseGate
    from trotterlib.df_trotter.ops import append_diagonal_primitives
    expected = []
    class Sink:
        def append(self, op, qubits): expected.append(_signature(op, qubits))
    for run in native.runs(model, native.ordered_schedule(len(model.blocks), h), optimized=True):
        b = builder._basis(model, model.blocks[run[0][0]])
        for item in b.inverse().data:
            expected.append(_signature(item.operation, [b.find_bit(q).index for q in item.qubits]))
        append_diagonal_primitives(Sink(), native.combined_diagonal(model, run),
            controlled=True, ancilla_qubit=model.modes)
        for item in b.data:
            expected.append(_signature(item.operation, [b.find_bit(q).index for q in item.qubits]))
    expected.append(_signature(PhaseGate(-model.constant*float(h)), [model.modes]))
    if float(qc.global_phase) != 0 or circuit_structure(qc, model.modes, compiled=False) != expected:
        raise ValueError('RAW_FRAME_TIME_GAUSSIAN_OR_CONTROL_PHASE')
    return dict(status='PASS', control_relative_phase='exact native itinerary including I terms and scalar',
        scope='structural itinerary only; full controlled operator and compile equivalence unqualified')


def optimistic_round(target, row, target_id):
    base = dict(m=row['m'], target_id=target_id, actual_synthesis_signal_error=None,
        optimistic_synthesis_signal_error=0., CA_achieved=None, guaranteed_actual_shots=None,
        universal_shots_lower_bound=False, monotonicity=MONOTONICITY,
        floating_point_rounding_enclosed=False)
    if row.get('pf_operator_error') is None:
        return dict(base, status='UNRESOLVED_PF', axes=None)
    if row['attenuation'] != 1 or row['rte_corrected_operator_error'] != 0:
        raise ValueError('DETERMINISTIC_LAW_REQUIRED')
    try:
        plan = accuracy.assess_round(target, accuracy.RoundEvidence(m=row['m'], attenuation=1.,
            pf_operator_error=row['pf_operator_error'], rte_corrected_operator_error=0.,
            synthesis_signal_error=0., finite_law_signal_error=0., target_id=target_id,
            state_model='own_DF_ground_state_oracle', alias_branch_verified=True,
            iid_per_shot=True, law_model='deterministic'))
    except (OverflowError, ZeroDivisionError):
        return dict(base, status='UNRESOLVED_SHOT_RANGE', axes=None)
    if plan['status'] != 'CONDITIONAL_ROUND_PASS':
        return dict(base, status='INFEASIBLE_OPTIMISTIC_PRESCRIPTION', axes=None, solver=plan)
    n = plan['shots']
    if type(n) is not int or n < 1: raise ValueError('POSITIVE_INTEGER_SHOTS_REQUIRED')
    return dict(base, status='OPTIMISTIC_FORMULA_ESTIMATE', solver=plan,
        axes={a:dict(optimistic_shots_decimal=str(n)) for a in ('X','Y')})


def serial_cost_pair(kernel_metrics, shells, s, m, target_id):
    packet = dict(id=target_id, s=s, kernel_metrics=kernel_metrics, shell_metrics=shells)
    return ca.reused.serial_round_cost(packet, m)


def aggregate_screen(s, rounds, costs):
    states = [x['status'] for x in rounds]
    if len(rounds) != 13 or any(x.startswith('UNRESOLVED') for x in states):
        return dict(status='UNRESOLVED', optimistic_total_RZ_decimal=None)
    if any(x != 'OPTIMISTIC_FORMULA_ESTIMATE' for x in states):
        return dict(status='INFEASIBLE_OPTIMISTIC_PRESCRIPTION', optimistic_total_RZ_decimal=None)
    candidate = integrated.Candidate('XDF9', 'D', s)
    try:
        value = cost.aggregate_optimistic(candidate, dict(rounds=rounds), costs)
    except OverflowError:
        return dict(status='UNRESOLVED_COST_RANGE', optimistic_total_RZ_decimal=None)
    exact = Fraction(value['point_totals_rational']['rz_count'])
    if exact.denominator != 1 or exact < 0: raise ValueError('INTEGER_SERIAL_RZ_REQUIRED')
    return dict(status='OPTIMISTIC_MODEL_COST_AVAILABLE', optimistic_total_RZ_decimal=str(exact.numerator),
        aggregation=value, actual_total_cost=None, CA_achieved=None,
        support_scope='singleton counts of this fixed deterministic compiled circuit; no accuracy claim',
        analytic_prescription_lower_bound=True, rigorous_numerical_lower_bound=None,
        qualification='Fixed sufficient-shots prescription/serial/native-structure numerical screen only')


def compare_screen(candidate, saved_total, baseline):
    value = candidate.get('optimistic_total_RZ_decimal')
    if value is None:
        return dict(status='UNRESOLVED', baseline=baseline, strictly_beating_excluded=None)
    if type(saved_total) is not int or saved_total < 0: raise ValueError('SAVED_INTEGER_BASELINE')
    excluded = int(value) >= saved_total
    return dict(status='EXCLUDED_STRICT_IMPROVEMENT_FIXED_MODEL_NUMERICAL' if excluded
        else 'COMPETITIVENESS_NOT_EXCLUDED', baseline=baseline,
        optimistic_total_RZ_decimal=value, saved_total_RZ_decimal=str(saved_total),
        strictly_beating_excluded=excluded, unconditional_resource_superiority=False,
        full_operator_correctness=False, floating_point_rounding_enclosed=False,
        interpretation='Conditional on fixed implementation, nonnegative extra signal error, monotone prescription and serial policy; not a universal lower bound or final CA')


def run(root, path, input_path, output):
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    root = Path(root).resolve(); c = preflight(root, path, input_path)
    output = fresh_output(root, output); output.mkdir(parents=True, exist_ok=False)
    (output/'circuits').mkdir(); (output/'candidates').mkdir()
    for key, leaf in [('MPLCONFIGDIR','matplotlib'), ('TMPDIR','tmp'), ('XDG_CACHE_HOME','cache')]:
        cache = root/'.cache/xdf9-cost-screen'/leaf; cache.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache)
    calls = integrated.Calls(output, LIMITS, MAX_CALLS)
    serial = fair.CompileSerialCalls(calls); plan = integrated.resource_plan(task_count=2, full_dimension=256)
    tick, cpu = time.monotonic(), time.process_time(); partial, worker_errors = {}, []
    try:
        io.save(output/'contract.json', c)
        io.save(output/'SOURCE.json', dict(io.read(root/c['source_manifest']), execution_commit=
            subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()))
        io.save(output/'resource_plan.json', plan)
        bundle = calls('saved_input_load', 'original whitelist and saved scalar evidence', lambda:load_bundle(root,c,input_path))
        verified = calls('saved_identity', 'PF9, precision shells, saved totals', lambda:saved_identity(bundle,c))
        io.save(output/'saved_identity.json', verified)
        io.save(output/'packet_gate.json', calls('packet_gate','original kappa/scalar/prefix9',lambda:packet_gate(bundle['packet'])))
        with threadpool_limits(limits=1 if plan['workers'] > 1 else None):
            model = calls('native_model','signed prefix9',lambda:prepare_model(bundle['packet'],c['original_input']['sha256']))
            builder = fair.PrecisionBuilder()
            for block in model.blocks:
                calls('gaussian_build',block.block_id,lambda block=block:fair.basis_request(builder,model,block))
            io.save(output/'model_structure.json',calls('model_structure','all10 blocks',lambda:model_structure(model,builder,bundle['packet'])))
            io.save(output/'gaussian.json',builder.basis_reports)
            raw = {}
            for s in (1,4):
                h = Fraction(1,10*s)
                raw[s] = calls('kernel_build',f's{s}',lambda h=h:builder.ordered(model,h,optimized=True))
                io.save(output/f'raw_structure_s{s}.json',calls('raw_structure',f's{s}',lambda s=s,h=h:raw_structure(raw[s],model,builder,h)))
                with (output/'circuits'/f'XDF9_s{s}_raw.qpy').open('xb') as f:qpy.dump(raw[s],f)
            # All inverse/itinerary creation is finished before the P0 workers.
            def candidate(s):
                key = integrated.Candidate('XDF9','D',s).id
                compiled = serial('compile',key,lambda:fair.compile_checked(raw[s],fair.CompilerProfile()))
                gate = calls('compiled_structure',key,lambda:dict(status='PASS',
                    instruction_count=len(circuit_structure(compiled,model.modes,compiled=True)),
                    full_operator_correctness=False,compile_operator_equivalence=None,
                    qualification='fixed P0 and finite known-unitary gate structure only'))
                io.save(output/f'compiled_structure_s{s}.json',gate)
                with (output/'circuits'/f'XDF9_s{s}_compiled.qpy').open('xb') as f:qpy.dump(compiled,f)
                def measure():
                    value = circuit_metrics(compiled)
                    ca.reused.metric_record(value,measurement=0)
                    return value
                metrics = calls('metrics',key,measure)
                rounds, costs = [], []
                for row in bundle['evidence']['PF']['candidates'][str(s)]['rounds']:
                    r = calls('optimistic_round',f'{key}:m{row["m"]}:XY',lambda row=row:optimistic_round(accuracy.AccuracyTarget(**TARGET),row,key))
                    rounds.append(r)
                    pair = calls('round_serial_cost',f'{key}:m{row["m"]}:XY',lambda row=row:serial_cost_pair(metrics,verified['shell_metrics'],s,row['m'],key))
                    costs.extend(pair)
                    io.save(output/'candidates'/f's{s}_m{row["m"]}.json',dict(shots=r,costs=pair))
                value = calls('optimistic_aggregate',key,lambda:aggregate_screen(s,rounds,costs))
                comparisons = {a:calls('screen_compare',key+':'+a,lambda a=a:compare_screen(value,verified['baselines'][str(s)][a],a+'8')) for a in ('XDF','CDF')}
                status = ('INFEASIBLE_OPTIMISTIC_PRESCRIPTION' if value['status'] == 'INFEASIBLE_OPTIMISTIC_PRESCRIPTION' else
                    'UNRESOLVED' if value['optimistic_total_RZ_decimal'] is None else
                    'NO_STRICT_IMPROVEMENT_VS_XDF8_FIXED_MODEL_NUMERICAL' if comparisons['XDF']['strictly_beating_excluded']
                    else 'COMPETITIVENESS_NOT_EXCLUDED')
                result = dict(s=s,status=status,metrics=metrics,rounds=rounds,cost=value,comparisons=comparisons,
                    full_operator_correctness=False,actual_synthesis_error=None,CA_achieved=None)
                io.save(output/'candidates'/f's{s}.json',result)
                return s,result
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                futures = [pool.submit(candidate,s) for s in (1,4)]
                for f in as_completed(futures):
                    try:
                        s,value = f.result(); partial[str(s)] = value
                    except BaseException as error:
                        worker_errors.append(dict(error=str(error),traceback=traceback.format_exc()))
                        calls.stop.set()
            if calls.stop.is_set(): raise ValueError('XDF9_COST_SCREEN_CORRECTNESS_STOP')
        result = dict(status='XDF9_NATIVE_OPTIMISTIC_SCREEN_COMPLETE',candidates=partial,
            actual_synthesis_error=None,CA_achieved=None,guaranteed_total_resource_superiority=False,
            full_operator_correctness=False,mandatory_stop=True)
        io.save(output/'result.json',result); return result
    except BaseException as e:
        calls.stop.set(); io.save(output/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),
            first_failure=calls.first_failure,workers=worker_errors)); raise
    finally:
        io.save(output/'partial_candidates.json',partial)
        io.save(output/'STOP.json',dict(failed=calls.stop.is_set(),call_counts=dict(calls.counts),
            scientific_units=sum(calls.counts.values()),retry=0,resume=0,mandatory_stop=True))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-tick,
            process_CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
