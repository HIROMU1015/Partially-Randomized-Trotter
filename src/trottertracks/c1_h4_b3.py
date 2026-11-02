"""B3-only connection to the frozen spin basis and relative Gaussian builder.

No changes to B0/B1/B2, their ROUTES, contracts, or defaults. Imports and
preflight are passive; real molecule work requires separate execution permission.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from fractions import Fraction
import math
import os
from pathlib import Path
import resource
import subprocess
import time

from . import c1_h4_strong_baseline as strong

fair, io, native, integrated = strong.fair, strong.io, strong.native, strong.integrated
ROUTE = 'B3_XDF_spin_relative'
SCOPE = 'C1_H4_B3_SPIN_RELATIVE_S1_V1'
FAMILY = 'artifacts/research_c_c1_h4_b3'
COMPARATORS = ('B1_XDF_spin', 'B2_XDF_relative', 'B2_CDF_relative')
PREVIOUS_SOURCE = '880006396326a4d46f4a4030564279e92fadb9cf'
PREVIOUS_PUBLICATION = 'e0c89f611eac92c939edde67890f5cc4391d098e'
CONDITIONS = dict(fair.CONDITIONS, substeps=[1], route=ROUTE)
RECORDS = {'result', 'SOURCE', 'STOP', 'contract', 'identity', 'spin',
           'B0_result', 'B0_SOURCE', 'B0_STOP', 'fairness'}
LIMITS = dict(saved_input_load=1, saved_evidence_load=1, saved_identity=1,
    native_model=1, model_gate=1, gaussian_build=9, relative_gaussian=16,
    reference_load=1, shell_operator_load=6, ground_load=1, kernel_build=1,
    fusion_gate=1, raw_gate=1, compile=1, operator_decode=2, segment_assessment=1,
    adjoint_build=2, adjoint_operator=2, adjoint_gate=2, signal_reference=1,
    signal_prepare=2, signal_axis=4, signal_gate=3, metrics=1, cost_comparison=3)
MAX_CALLS = sum(LIMITS.values())
FORBIDDEN = ('new_fit', 'optimizer', 'rank_search', 'DF_PF_recalculation',
    'ground_eigensolve', 'reference_regeneration', 'old_route_build_compile',
    'CDF_numeric_decode', 's4_build', 'round_synthesis', 'CA_evaluation',
    'shots', 'total_cost', 'retry', 'resume')


def preflight(root, path, input_path):
    root = Path(root).resolve(); c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT_DRAFT'
            or c['route'] != ROUTE or c['comparators'] != list(COMPARATORS)
            or c['conditions'] != CONDITIONS or c['allowances'] != strong.ALLOWANCES
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or c['previous_scientific_SOURCE'] != PREVIOUS_SOURCE
            or c['previous_publication'] != PREVIOUS_PUBLICATION
            or c['compiler'] != fair.profile_dict() or c['gaussian'] != fair.GAUSSIAN
            or c['compiler_phase'] != fair.PHASE or any(c[k] for k in FORBIDDEN)
            or c['fresh_output'] != FAMILY+'/<execution-date>/run01'):
        raise ValueError('B3_SCOPE')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, want in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != want:
                raise ValueError('SOURCE_OR_INPUT: '+name)
    # Reuse the old byte/conditions/environment checks, not its science runner.
    previous, legacy = strong.preflight(root, root/c['previous_contract'], input_path)
    required = [c['previous_contract'], c['reference'], c['ground'], *c['records'].values(),
                *[p for x in c['shells'].values() for p in x.values()]]
    if (set(c['records']) != RECORDS or not set(required) <= c['saved_inputs'].keys()
            or c['original_input'] != previous['original_input']
            or c['reference'] != previous['references']['XDF']
            or c['shells'] != previous['shells'] or c['ground'] != legacy['ground_states']['XDF']):
        raise ValueError('B3_REFERENCE_OR_INPUT_BINDING')
    return c, previous


def saved_identity(records, previous):
    """Saved scalar/metadata only; old PASS never becomes B3 correctness."""
    r, stop, source, contract = (records[k] for k in ('result', 'STOP', 'SOURCE', 'contract'))
    if (r['status'] != 'H4_STRONG_BASELINE_S1_NUMERICAL_COMPLETE' or r['B3'] != 'NOT_RUN'
            or source['execution_commit'] != PREVIOUS_SOURCE or stop['failed']
            or stop['scientific_units'] != 227 or stop['retry'] or stop['resume']
            or not stop['mandatory_stop'] or contract != previous
            or set(r['candidates']) != set(strong.ROUTES)
            or records['identity']['status'] != 'PASS' or len(records['spin']) != 9):
        raise ValueError('SAVED_STRONG_BASELINE_IDENTITY')
    for arm, bound in records['identity']['DF_bounds_Ha'].items():
        if not math.isfinite(bound) or not 0 <= bound <= .00019875:
            raise ValueError('SAVED_DF_RESERVE: '+arm)
    b0 = strong.baseline_gate(records['B0_result'], records['B0_SOURCE'], records['B0_STOP'], previous)
    f = records['fairness']
    if (f['status'] != 'PASS' or f['compiler'] != fair.profile_dict()
            or f['gaussian'] != fair.GAUSSIAN or f['compiler_phase'] != fair.PHASE
            or f['additional_relative_Gaussian_fusion']):
        raise ValueError('SAVED_B0_PRECISION')
    for route in COMPARATORS:
        x = r['candidates'][route]; s = x['segment']
        if (x['s'] != 1 or s['input_dimension'] != 512 or s['output_dimension'] != 512
                or s['global_phase_fit'] or not math.isfinite(s['compiled_reference']['max_entry'])
                or s['compiled_reference']['max_entry'] > 1e-10
                or any(not math.isfinite(g['error']) or g['error'] > 2e-10 for g in x['XY'].values())):
            raise ValueError('SAVED_ROUTE_CORRECTNESS: '+route)
        if any(type(x['metrics'][k]) is not int or x['metrics'][k] < 0 for k in
               ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')):
            raise ValueError('SAVED_NATIVE_METRICS: '+route)
    return dict(status='PASS', baseline_metrics={r: records['result']['candidates'][r]['metrics'] for r in COMPARATORS},
                B0_metrics=b0['metrics'], B3_correctness='NOT_YET_EVALUATED')


def prepare(packet, input_identity):
    return strong.prepare_xdf_basis(packet['df_one_body'], packet['df_lambdas'][:8],
        packet['df_g_matrices'][:8], float(packet['nuclear_repulsion']),
        source_identity=input_identity+':prefix8', input_identity=input_identity, basis_policy=strong.SPIN)


def model_gate(model, reports, saved_spin):
    import numpy as np
    if ([b.block_id for b in model.blocks] != ['one_body']+[f'F{i}' for i in range(8)]
            or model.label != 'XDF' or len(reports) != len(saved_spin) or len(reports) != 9):
        raise ValueError('B3_FRAME_ORDER')
    differences = []
    for now, old in zip(reports, saved_spin, strict=True):
        if now['status'] != 'PASS' or now['policy'] != strong.SPIN or not now['exact_spin_structure']:
            raise ValueError('B3_SPIN_RECIPE')
        for key in ('status', 'policy', 'spin_order', 'exact_spin_structure', 'original_index', 'signed_lambda'):
            if now.get(key) != old.get(key): raise ValueError('B1_B3_BASIS_RECIPE: '+key)
        a, b = np.asarray(now['eigenvalues']), np.asarray(old['eigenvalues'])
        if a.shape != b.shape or not np.isfinite(b).all(): raise ValueError('B1_BASIS_SCHEMA')
        difference = float(np.max(abs(a-b)))
        if difference > 1e-10: raise ValueError('B1_EIGENVALUE_ORDER')
        differences.append(difference)
    return dict(status='PASS', B1_recipe=strong.SPIN, frame_order=list(range(8)),
                max_saved_eigenvalue_difference=max(differences), saved_U_bitwise_claim=False)


def execute(model, ground, reference, shells, builder, baseline, output, call, plan):
    """One new kernel; all synthesis, phase, inverse, signal and metric math reused."""
    import numpy as np
    from qiskit import qpy
    from .c1_h6_native import compare_metrics
    from .c1_h6_state_action import ancilla_z
    from .pr_dictionary_validation.gates import signal_gate
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    output = Path(output); d = len(reference); serial = fair.CompileSerialCalls(call)
    def save_q(kind, q):
        with (output/'circuits'/(ROUTE+'_'+kind+'.qpy')).open('xb') as f: qpy.dump(q, f)
    def decode(kind, qc, call_kind):
        a = call(call_kind, kind, lambda: fair.decode(qc))
        with (output/'operators'/(kind+'.npz')).open('xb') as f: np.savez_compressed(f, matrix=a)
        return kind, a
    raw = call('kernel_build', ROUTE, lambda: builder.ordered(model, Fraction(1, 10), optimized=True))
    raw.metadata = dict(raw.metadata or {}, route=ROUTE)
    save_q('raw', raw)
    def fusion():
        groups = native.runs(model, native.ordered_schedule(9, '1/10'), optimized=True)
        pairs = [(model.blocks[a[0][0]].block_id, model.blocks[b[0][0]].block_id) for a, b in zip(groups, groups[1:])]
        if ([(x['previous'], x['following']) for x in builder.fusion_records] != pairs
                or any(x['multiplication'] != 'V_following.conj().T @ V_previous' for x in builder.fusion_records)):
            raise ValueError('B3_FUSION_DIRECTION_OR_ORDER')
        return dict(status='PASS', route=ROUTE, records=builder.fusion_records)
    io.save(output/'fusion.json', call('fusion_gate', ROUTE, fusion))
    matrices = dict([decode('raw', raw, 'operator_decode')]); ideal = fair.controlled(reference)
    io.save(output/'raw_gate.json', call('raw_gate', ROUTE, lambda: fair.component_check(matrices['raw'], ideal, 1e-10, label='B3 raw/reference')))
    # Existing Qiskit translator mutation stays serialized; three later matrix
    # decodes are independent. No new raw/adjoint inverse creation races P0.
    adj_raw = call('adjoint_build', 'raw', lambda: native.actual_adjoint(raw)); save_q('raw_adjoint', adj_raw)
    compiled = serial('compile', ROUTE, lambda: fair.compile_checked(raw, fair.CompilerProfile())); save_q('compiled', compiled)
    adj_compiled = call('adjoint_build', 'compiled', lambda: native.actual_adjoint(compiled)); save_q('compiled_adjoint', adj_compiled)
    jobs = [('compiled', compiled, 'operator_decode'), ('raw_adjoint', adj_raw, 'adjoint_operator'), ('compiled_adjoint', adj_compiled, 'adjoint_operator')]
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures = [pool.submit(decode, *j) for j in jobs]
        for f in futures:
            key, value = f.result(); matrices[key] = value
    segment = call('segment_assessment', ROUTE, lambda: fair.synthesis.segment(matrices['raw'], matrices['compiled'], ideal, ROUTE))
    io.save(output/'segment.json', segment)
    for kind in ('raw', 'compiled'):
        def check(kind=kind):
            a = matrices[kind+'_adjoint']
            return dict(dagger=fair.component_check(a, matrices[kind].conj().T, 1e-10, label='actual dagger'),
                reference=fair.component_check(a, ideal.conj().T, 1e-10, label='adjoint/reference'),
                recovery=fair.component_check(a@matrices[kind], np.eye(2*d), 1e-10, label='actual recovery'))
        io.save(output/(kind+'_adjoint.json'), call('adjoint_gate', kind, check))
    indices = fair.stage.sector_indices(model.modes//2, model.modes//4, model.modes//4)
    psi = np.zeros(d, complex); psi[indices] = ground
    expected = call('signal_reference', ROUTE, lambda: complex(np.vdot(psi, reference@psi))); signals = {}
    for kind in ('raw', 'compiled'):
        state = call('signal_prepare', kind, lambda: matrices[kind]@shells['head'][kind]@np.r_[psi, np.zeros(d)])
        values = {a: call('signal_axis', kind+':'+a, lambda a=a: float(ancilla_z((shells[a][kind]@state)[None, :])[0])) for a in ('X', 'Y')}
        signals[kind] = complex(values['X'], values['Y'])
    xy = {}
    for name, x, y in [('raw_reference', signals['raw'], expected), ('compiled_raw', signals['compiled'], signals['raw']), ('compiled_reference', signals['compiled'], expected)]:
        xy[name] = call('signal_gate', name, lambda x=x, y=y: signal_gate(x.real, x.imag, y, 2e-10))
    io.save(output/'XY.json', xy)
    metrics = call('metrics', ROUTE, lambda: circuit_metrics(compiled))
    comparisons = [call('cost_comparison', ROUTE+'/'+r, lambda r=r: compare_metrics(metrics, baseline['baseline_metrics'][r], ROUTE+'/'+r)) for r in COMPARATORS]
    return dict(status='H4_B3_S1_NUMERICAL_COMPLETE', route=ROUTE, s=1, segment=segment, XY=xy,
        metrics=metrics, comparisons=comparisons, saved_B0_metrics=baseline['B0_metrics'],
        CA_achieved=None, shots=None, total_cost=None, rigorous_enclosure=None, mandatory_stop=True)


def run(root, path, input_path, output):
    import numpy as np
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve(); c, previous = preflight(root, path, input_path)
    output = Path(output).resolve(); parts = output.relative_to(root/FAMILY).parts
    if len(parts) != 2 or parts[1] != 'run01' or date.fromisoformat(parts[0]).isoformat() != parts[0] or output.exists():
        raise ValueError('FRESH_B3_RUN01_NO_RETRY_RESUME')
    output.mkdir(parents=True, exist_ok=False)
    for n in ('circuits', 'operators'): (output/n).mkdir()
    for key, leaf in [('XDG_CACHE_HOME', ''), ('MPLCONFIGDIR', 'matplotlib'), ('TMPDIR', 'tmp')]:
        p = root/'.cache/h4-b3'/leaf; p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
    call = integrated.Calls(output, LIMITS, MAX_CALLS); plan = integrated.resource_plan(task_count=3, full_dimension=256)
    plan['independent_tasks'] = ['compiled_operator', 'raw_adjoint_operator', 'compiled_adjoint_operator']
    tick, cpu = time.monotonic(), time.process_time()
    try:
        io.save(output/'contract.json', c); io.save(output/'SOURCE.json', dict(io.read(root/c['source_manifest']), execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()))
        io.save(output/'resource_plan.json', plan)
        packet = call('saved_input_load', 'original XDF8', lambda: io.load_input(input_path, evaluation=True))
        records = call('saved_evidence_load', 'B0/B1/B2', lambda: {k: io.read(root/p) for k, p in c['records'].items()})
        baseline = call('saved_identity', 'saved provenance', lambda: saved_identity(records, previous)); io.save(output/'saved_identity.json', baseline)
        with threadpool_limits(limits=1 if plan['workers'] > 1 else None):
            model, reports = call('native_model', ROUTE, lambda: prepare(packet, c['original_input']['sha256']))
            io.save(output/'model_gate.json', call('model_gate', ROUTE, lambda: model_gate(model, reports, records['spin'])))
            io.save(output/'spin_basis.json', reports)
            builder = strong.StrongBuilder(relative_fusion=True, basis_policy=strong.SPIN, call=call)
            for block in model.blocks: call('gaussian_build', block.block_id, lambda block=block: fair.basis_request(builder, model, block))
            reference = call('reference_load', 'original independent XDF CAR/PF', lambda: fair.matrix_load(root/c['reference'], 'matrix', (256, 256)))
            shells = {a: {k: call('shell_operator_load', a+':'+k, lambda p=p: fair.matrix_load(root/p, 'matrix', (512, 512))) for k, p in v.items()} for a, v in c['shells'].items()}
            def load_ground():
                state = fair.matrix_load(root/c['ground'], 'sector_state', (36,))
                if abs(np.vdot(state, state)-1) > 1e-12: raise ValueError('SAVED_GROUND_NORMALIZATION')
                return state
            ground = call('ground_load', 'saved own XDF oracle', load_ground)
            result = execute(model, ground, reference, shells, builder, baseline, output, call, plan)
        io.save(output/'result.json', result); return result
    except BaseException as e:
        call.stop.set(); io.save(output/'failure.json', dict(error=str(e), first_failure=call.first_failure)); raise
    finally:
        io.save(output/'STOP.json', dict(mandatory_stop=True, failed=call.stop.is_set(), scientific_units=sum(call.counts.values()), call_counts=dict(call.counts), retry=0, resume=0))
        io.save(output/'resources.json', dict(wall_seconds=time.monotonic()-tick, process_CPU_seconds=time.process_time()-cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
