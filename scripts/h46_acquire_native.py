"""Native acquisition glue for the fixed 12 groups; frozen science kernels reused.

Costs only: no shot recomputation, weighted mean, q ranking or winner selection.
Each process caches saved preparations/oracles and executes paired wrappers.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

CONTEXT = {}
CONFIG = None
STOP = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def verify(c):
    for item in c['sealed_files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('SEALED_SOURCE_INPUT_RUNTIME_MISMATCH:'+item['path'])
    if sha(c['plan_path']) != c['plan_sha256'] or sha(c['seed_registry_path']) != c['seed_registry_sha256']:
        raise ValueError('PLAN_OR_SEED_SEAL')


def initialize(config_path, stop):
    global CONFIG, STOP
    CONFIG = json.loads(Path(config_path).read_text()); STOP = stop
    verify(CONFIG)
    sys.path.insert(0, str(Path(CONFIG['source_root'])/'scripts'))
    sys.path.insert(1, str(Path(CONFIG['provider_root'])/'src'))
    class NoGPU:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('cupy','cupyx','cuquantum'):
                raise ImportError('CPU-only sealed native series')
    sys.meta_path.insert(0, NoGPU())
    import qiskit
    from functools import partial
    import trotterlib.rte_compiled_cost as costs
    costs.transpile = partial(qiskit.transpile, num_processes=1)


def context(c, system):
    if system in CONTEXT:
        return CONTEXT[system]
    import numpy as np
    from h46_new2_native import prepare, OccupationOracle
    if system == 'H4':
        from trotterlib.pr2_new_series_validation import _load_snapshot_once
        ham,sector,full,state,metadata,_ = _load_snapshot_once(Path(c['inputs']['H4_snapshot.npz']['path']))
    else:
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        ham,sector,full,state,metadata,_ = load_h6_snapshot(Path(c['inputs']['H6_snapshot.npz']['path']),
            json.loads(Path(c['inputs']['H6_snapshot_receipt.json']['path']).read_text()),
            json.loads(Path(c['inputs']['H6_df_receipt.json']['path']).read_text()))
    prefix = 3 if system == 'H4' else 5
    prep = prepare(ham, prefix)
    oracle = OccupationOracle(ham, full, prefix=prefix)
    qstate = np.zeros(1 << ham.n_qubits, complex); qstate[oracle.qindices] = oracle.initial
    low = np.r_[qstate, np.zeros_like(qstate)]; high = np.r_[np.zeros_like(qstate), qstate]
    CONTEXT[system] = (ham, sector, prep, oracle, qstate, low, high)
    return CONTEXT[system]


def task_id(group, sample):
    if type(sample) is not int or not 0 <= sample < 8:
        raise ValueError('SAMPLE_SCOPE')
    return group['id']+'/sample_%02d'%sample


def reserve_axis(ledger, group, sample, axis, evidence):
    if axis not in ('cosine','sine'):
        raise ValueError('AXIS_SCOPE')
    path = Path(ledger)/(group['id']+'__%02d__'%sample+axis+'.json')
    write(path, {'state':'ENTERED_NO_REFUND','group':group['id'],'sample':sample,'axis':axis,**evidence})
    return path


def run_task(group, sample):
    c, stop = CONFIG, STOP
    out = Path(c['output'])/task_id(group, sample); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); checks = []; records = []
    def checkpoint(phase):
        if stop.is_set(): raise RuntimeError('CAMPAIGN_STOP_NO_FURTHER_CALLS')
        with (out/'progress.jsonl').open('a') as f:
            f.write(json.dumps({'phase':phase,'elapsed':time.monotonic()-started})+'\n')
    def check(error, name):
        error = float(error); checks.append({'check':name,'error':error})
        if not math.isfinite(error) or error > c['plan']['numeric_tolerance']:
            raise ValueError('NATIVE_CORRECTNESS:'+name+':'+str(error))
    try:
        import numpy as np
        import qiskit
        from trotterlib.rte import CompilerSettings
        from trotterlib.rte_compiled_cost import transpile_and_measure_cost
        from trotterlib.df_trotter.circuit import simulate_statevector
        from trotterlib.df_rte_circuit import DFRTEEventCircuitRequest
        from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
        from trottertracks.resource_applicability.ax2b_native_df_v5 import build_native_hadamard_wrapper
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint
        from h46_new2_native import make_requests, native_evolution, forced_event_request
        from h46_new2_core import matrix_only_tail_coefficients
        if group not in c['plan']['groups'] or task_id(group,sample) not in c['plan']['tasks']:
            raise ValueError('UNREGISTERED_TASK')
        checkpoint('preparation')
        ham,sector,prep,oracle,qstate,low,high = context(c,group['system'])
        scalar,lam = matrix_only_tail_coefficients(ham,group['prefix'])
        old = c['group_binding'][group['id']]
        if prep.preparation_hash != old['preparation_hash']:
            raise ValueError('PREPARATION_IDENTITY')
        if tuple(prep.deterministic_fragment_indices) != tuple(range(group['prefix'])):
            raise ValueError('GENERATION_ORDER')
        check(abs(prep.extracted_identity_coefficient-scalar),'full_Fock_scalar')
        check(abs(prep.exact_rte_lambda_r-lam),'full_Fock_lambda')
        check(abs(old['lambda_R']-lam),'previous_lambda')
        check(abs(old['scalar']-ham.constant-scalar),'previous_scalar')
        if old['shots']['eligibility_under_declared_allowance'] != 'ELIGIBLE':
            raise ValueError('SAVED_ELIGIBILITY_BINDING')
        registry = json.loads(Path(c['seed_registry_path']).read_text())
        seeds = registry['tasks'][task_id(group,sample)]
        checkpoint('sample_event')
        requests = make_requests(prep,group['label'],group['q'],T=.8,total_steps=32,
            seed_for=lambda outer,piece:seeds[f'{outer}:{piece}'])
        log_B = math.fsum(len(r.rte_occurrence.events)*math.log(r.rte_distribution.exact_finite_distribution) for r in requests)
        check(abs(log_B-old['log_B']),'sample_normalization')
        events = [[event.to_dict() for event in request.rte_occurrence.events] for request in requests]
        event_digest = hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest()
        write(out/'events_PRIVATE.json',{'events':events,'seed_namespace':c['plan']['seed_namespace'],'conditional':False})
        checkpoint('native_build')
        evolution = native_evolution(requests,group['label'],group['q'],.8)
        z = None
        if sample == 0:
            expected = oracle.trajectory(requests)
            check(np.linalg.norm(simulate_statevector(evolution.circuit,low)-low),'native_control_low')
            actual = simulate_statevector(evolution.circuit,high)
            check(np.linalg.norm(actual[:len(qstate)]),'native_control_leakage')
            check(np.linalg.norm(actual[len(qstate):]-expected),'native_occupation_trajectory')
            z = complex(np.vdot(qstate,expected))
        compiler = CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
        for axis in ('cosine','sine'):
            checkpoint('wrapper_'+axis)
            wrapper = build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=10**7)
            fingerprint = canonical_qiskit_circuit_fingerprint(wrapper)
            checkpoint('compile_'+axis)
            reserve_axis(c['call_ledger'],group,sample,axis,{'event_digest':event_digest,'fingerprint':fingerprint,'SOURCE':c['SOURCE']})
            before_compile = time.monotonic()
            cost = transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
            if cost.compiler_settings_hash != c['plan']['compiler_settings_hash']:
                raise ValueError('COMPILER_PROFILE')
            if sample == 0:
                before = simulate_statevector(wrapper.remove_final_measurements(inplace=False),low)
                observed = float(np.vdot(before[:len(qstate)],before[:len(qstate)]).real-np.vdot(before[len(qstate):],before[len(qstate):]).real)
                check(abs(observed-(z.real if axis=='cosine' else z.imag)),'native_hadamard_'+axis)
                after = simulate_statevector(cost.transpiled_circuit.remove_final_measurements(inplace=False),low)
                check(np.linalg.norm(after-before),'compiled_statevector_'+axis)
            metrics = {key:int(getattr(cost,key)) for key in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')}
            row = {'sample':sample,'axis':axis,'event_digest':event_digest,'fingerprint':fingerprint,
                   'compiler_settings_hash':cost.compiler_settings_hash,'metrics':metrics,
                   'wall_seconds':time.monotonic()-before_compile,'untranspiled_instructions':len(wrapper.data)}
            records.append(row);write(out/('cost_'+axis+'.json'),row)
            del cost,wrapper;gc.collect()
        if sample == 0 and group['label']=='4th(new_2)':
            checkpoint('forced_correctness_no_compile')
            builder = QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
            components = {v.component_id:v for v in prep.rte_preparation.symbolic_tail.components}
            for sign in (-1,1):
                r = next(r for r in requests if r.step_time*sign>0)
                for order in (0,2,4):
                    event = forced_event_request(r,order)
                    request = DFRTEEventCircuitRequest(event,prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=ham.n_qubits)
                    circuit = builder.build_event(request).circuit
                    expected = np.zeros_like(qstate);expected[oracle.qindices] = oracle.event(event,components,oracle.initial)
                    actual = simulate_statevector(circuit,high)
                    check(np.linalg.norm(actual[len(qstate):]-expected),f'forced_order{order}_sign{sign}')
                    check(np.linalg.norm(simulate_statevector(circuit,low)-low),f'forced_low{order}_sign{sign}')
        checkpoint('complete')
        result = {'status':'ACQUIRED_VALIDATED_NO_ACCOUNTING','SOURCE':c['SOURCE'],'provider_SOURCE':c['provider_SOURCE'],
                  'plan_sha256':c['plan_sha256'],'seed_registry_sha256':c['seed_registry_sha256'],
                  'seed_namespace':c['plan']['seed_namespace'],'group':group,'sample':sample,
                  'preparation_hash':prep.preparation_hash,'input_sha256':c['inputs'][group['system']+'_snapshot.npz']['sha256'],
                  'records':records,'validation_checks':checks,'forced_events_enter_cost_sample':False,
                  'shot_accounting_recomputed':False,'weighted_cost_computed':False,'ranking_computed':False,
                  'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'imported_provider_modules':{name:str(Path(m.__file__).resolve()) for name,m in sys.modules.items()
                      if name.startswith(('trotterlib.','trottertracks.')) and getattr(m,'__file__',None)}}
        write(out/'result.json',result)
        return {'group':group['id'],'sample':sample,'status':'PASS','wall_seconds':result['wall_seconds']}
    except BaseException as exc:
        stop.set()
        write(out/'STOP.json',{'type':type(exc).__name__,'reason':str(exc),'checks':checks,'traceback':traceback.format_exc(),'retry':False})
        raise
