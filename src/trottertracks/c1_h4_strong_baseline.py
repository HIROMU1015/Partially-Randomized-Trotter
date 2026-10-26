"""Opt-in spin-adapted XDF bases and phase-preserving Gaussian boundaries.

Imports and byte preflight are passive. Molecular preparation/build/evaluation
is exclusively in run(), after separate scientific execution permission.
The old NativeBuilder, PrecisionBuilder and preparation defaults are unchanged.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from fractions import Fraction
import math
import os
from pathlib import Path
import resource
import subprocess
import time

from . import c1_h4_native_fair as fair
from . import c1_h6_native_precision as precision

io, native, integrated = fair.io, fair.native, fair.integrated
VERSION = 'spin_lift_relative_gaussian_v1'
LEGACY = 'legacy_spin_orbital_eigh'
SPIN = 'exact_spin_lift_spatial_eigh_v1'
SCOPE = 'C1_H4_STRONG_BASELINE_S1_V1'
FAMILY = 'artifacts/research_c_c1_h4_strong_baseline'
ROUTES = ('B1_XDF_spin', 'B2_CDF_relative', 'B2_XDF_relative')
ALLOWANCES = dict(matrix_component_abs=1e-10, operator_component_abs=1e-10,
    signal_complex=2e-10, rtol=0, global_phase_fit=False,
    qualification='binary64 full512 numerical pilot; no rounding enclosure')
LOAD_LIMITS = {k: fair.LIMITS[k] for k in ('input_load', 'export_load', 'evidence_load',
    'cdf_operator_load', 'sector_generator_load', 'cdf_sector_load', 'ground_load', 'saved_kernel_load')}
LIMITS = dict(**LOAD_LIMITS, identity_gate=1, baseline_gate=1, native_model=3, spin_basis_gate=9,
    reference_load=2, shell_operator_load=6, car_context=1, xdf_generator=9,
    gaussian_build=27, relative_gaussian=32, local_reference=9, local_build=9,
    local_operator=9, local_gate=9, kernel_build=3, fusion_gate=3, compile=3,
    operator_decode=6, segment_assessment=3, adjoint_build=6, adjoint_operator=6,
    adjoint_assessment=6, signal_reference=2, signal_prepare=6, signal_axis=12,
    signal_gate=9, metrics=3, cost_comparison=5)
MAX_CALLS = sum(LIMITS.values())
FORBIDDEN = ('new_fit', 'optimizer', 'rank_search', 'DF_PF_recalculation', 'B0_rebuild',
    'B3', 's4_build', 'round_synthesis', 'CA_evaluation', 'shots', 'total_cost', 'retry', 'resume')


def basis_certificate(matrix, rotation, eigenvalues):
    """Certify G itself, not only a squared fragment or nearly degenerate space.

    This does not offer a near-degeneracy rotation policy: a proposed basis must
    reconstruct the original G. Residuals are numerical checks, not enclosures.
    """
    import numpy as np
    a, u, eta = np.asarray(matrix, complex), np.asarray(rotation, complex), np.asarray(eigenvalues)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or u.shape != a.shape
            or eta.shape != (len(a),) or not all(np.isfinite(x).all() for x in (a, u, eta))
            or np.iscomplexobj(eta) and np.any(eta.imag != 0)):
        raise ValueError('BASIS_SCHEMA')
    return dict(status='PASS', hermitian=fair.component_check(a, a.conj().T, 1e-10, label='original G Hermitian'),
        orthogonality=fair.component_check(u.conj().T @ u, np.eye(len(a)), 1e-10, label='basis unitarity'),
        reconstruction=fair.component_check((u * eta.real) @ u.conj().T, a, 1e-10, label='original G reconstruction'),
        near_degeneracy_rotation=False)


def spin_adapted_eigh(matrix):
    """Exact interleaved alpha/beta spin lift only; never drop off-spin entries."""
    import numpy as np
    from trotterlib.df_trotter.decompose import diag_hermitian
    a = np.asarray(matrix, complex)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a) or len(a) % 2
            or not np.isfinite(a).all()
            or not np.array_equal(a, np.kron(a[::2, ::2], np.eye(2)))):
        raise ValueError('EXACT_INTERLEAVED_SPIN_LIFT_REQUIRED')
    fair.component_check(a, a.conj().T, 1e-10, label='spin source Hermitian')
    spatial, eta = diag_hermitian(a[::2, ::2], assume_hermitian=True)
    u, values = np.kron(spatial, np.eye(2)), np.repeat(eta, 2)
    report = basis_certificate(a, u, values)
    report.update(policy=SPIN, spin_order='mode 2p=alpha,2p+1=beta',
                  exact_spin_structure=True, eigenvalues=values.tolist())
    return u, values, report


def prepare_xdf_basis(one, lambdas, matrices, constant, *, source_identity, input_identity,
                      basis_policy=LEGACY):
    if basis_policy == LEGACY:
        return native.prepare_xdf(one, lambdas, matrices, constant,
            source_identity=source_identity, input_identity=input_identity), ()
    if basis_policy != SPIN:
        raise ValueError('UNKNOWN_BASIS_POLICY')
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
    lambdas, matrices = tuple(lambdas), tuple(matrices)
    if len(lambdas) != len(matrices):
        raise ValueError('XDF_RANK')
    u, eps, r = spin_adapted_eigh(one)
    blocks = [native.NativeBlock('one_body', native._rotation(u), one_body_diagonal_primitives(eps, 1.))]
    reports = [r]
    for i, (lam, g) in enumerate(zip(lambdas, matrices, strict=True)):
        if not math.isfinite(float(lam)):
            raise ValueError('NONFINITE_SIGNED_LAMBDA')
        u, eta, r = spin_adapted_eigh(g)
        blocks.append(native.NativeBlock(f'F{i}', native._rotation(u), df_squared_diagonal_primitives(eta, lam, 1.)))
        reports.append(dict(r, original_index=i, signed_lambda=float(lam)))
    return native.NativeModel('XDF', tuple(blocks), float(constant), source_identity, input_identity), tuple(reports)


def vacuum_phase(circuit):
    """Analytic vacuum phase of this fixed Gaussian emitter; no phase fitting.

    RZ(phi)|0>=exp(-i phi/2)|0>; XXPlusYY leaves |00> unchanged.
    Refuse other gates rather than assuming their phase or using a state probe.
    """
    from qiskit.circuit.library import RZGate, XXPlusYYGate
    if circuit.num_clbits or circuit.parameters:
        raise ValueError('GAUSSIAN_PHASE_SCHEMA')
    terms = [float(circuit.global_phase)]
    for item in circuit.data:
        g = item.operation
        if isinstance(g, RZGate):
            terms.append(-float(g.params[0])/2)
        elif not isinstance(g, XXPlusYYGate):
            raise ValueError('UNKNOWN_GAUSSIAN_VACUUM_PHASE')
    value = math.fsum(terms)
    if not math.isfinite(value):
        raise ValueError('NONFINITE_GAUSSIAN_PHASE')
    return value


class StrongBuilder(precision.PrecisionBuilder):
    """Existing central emitter and same-basis fsum; optional boundary adapter."""
    def __init__(self, *, relative_fusion=False, basis_policy=LEGACY, call=None):
        super().__init__()
        if type(relative_fusion) is not bool or basis_policy not in (LEGACY, SPIN):
            raise ValueError('EXPLICIT_ROUTE_REQUIRED')
        self.relative_fusion, self.basis_policy = relative_fusion, basis_policy
        self.science_call = call or (lambda kind, label, fn: fn())
        self.relative_cache, self.fusion_records = {}, []

    def transition(self, model, previous, following):
        """Chronological B_i then B_j^-1 => matrix B_j^dagger B_i."""
        import numpy as np
        key = (VERSION, precision.VERSION, self.basis_policy, model.source_identity,
               model.input_identity, previous.rotation, following.rotation)
        if key not in self.relative_cache:
            bi, bj = self._basis(model, previous), self._basis(model, following)
            relative = np.asarray(following.rotation).conj().T @ np.asarray(previous.rotation)
            def make():
                circuit, report = precision.gaussian(relative)
                phi = math.fsum([vacuum_phase(bi), -vacuum_phase(bj), -vacuum_phase(circuit)])
                circuit.global_phase = float(circuit.global_phase) + phi
                return circuit, dict(version=VERSION,
                    multiplication='V_following.conj().T @ V_previous', phase_correction=phi,
                    vacuum_phase_target=math.fsum([vacuum_phase(bi), -vacuum_phase(bj)]),
                    phase_method='analytic RZ vacuum phase; no fitting', gaussian=report)
            self.relative_cache[key] = self.science_call('relative_gaussian',
                model.label+':'+previous.block_id+'->'+following.block_id, make)
        circuit, record = self.relative_cache[key]
        # Cache mathematical rotations, but record the requested fragment IDs.
        value = dict(record, previous=previous.block_id, following=following.block_id)
        self.fusion_records.append(value)
        return circuit

    def sequence(self, model, schedule, *, optimized, scalar_time):
        if not self.relative_fusion:
            return super().sequence(model, schedule, optimized=optimized, scalar_time=scalar_time)
        from qiskit import QuantumCircuit
        from trotterlib.df_trotter.ops import append_diagonal_primitives
        groups = native.runs(model, schedule, optimized=optimized)
        if not groups:
            raise ValueError('EMPTY_EVOLUTION')
        qc = QuantumCircuit(model.modes+1)
        first = model.blocks[groups[0][0][0]]
        qc.compose(self._basis(model, first).inverse(), range(model.modes), inplace=True)
        for i, group in enumerate(groups):
            block = model.blocks[group[0][0]]
            append_diagonal_primitives(qc, native.combined_diagonal(model, group),
                                      controlled=True, ancilla_qubit=model.modes)
            if i+1 < len(groups):
                following = model.blocks[groups[i+1][0][0]]
                qc.compose(self.transition(model, block, following), range(model.modes), inplace=True)
            else:
                qc.compose(self._basis(model, block), range(model.modes), inplace=True)
        qc.p(-model.constant*float(Fraction(scalar_time)), model.modes)
        qc.metadata = dict(strong_baseline=VERSION, relative_fusion=True, basis_policy=self.basis_policy,
            scalar_once=True, phase_policy='analytic boundary vacuum phase; controlled I terms retained')
        return qc


def preflight(root, path, input_path):
    """SOURCE/input bytes and frozen settings only; no NPZ/QPY numeric decode."""
    root = Path(root).resolve(); c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT_DRAFT'
            or c['routes'] != list(ROUTES) or c['allowances'] != ALLOWANCES
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or c['adapter_version'] != VERSION or c['basis_policy'] != SPIN
            or any(c[k] for k in FORBIDDEN)):
        raise ValueError('STRONG_BASELINE_SCOPE')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, want in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != want:
                raise ValueError('SOURCE_OR_INPUT: '+name)
    legacy = fair.preflight(root, root/c['legacy_contract'], input_path)
    if (c['fresh_output'] != FAMILY+'/<execution-date>/run01'
            or c['conditions'] != dict(fair.CONDITIONS, substeps=[1], route='B0 saved; B1 XDF spin; B2 both legacy bases relative fusion')
            or c['compiler'] != fair.profile_dict() or c['gaussian'] != fair.GAUSSIAN
            or c['compiler_phase'] != fair.PHASE):
        raise ValueError('SCIENTIFIC_CONDITIONS')
    required = [c['legacy_contract'], c['baseline_result'], c['baseline_SOURCE'], c['baseline_STOP'],
                *c['references'].values(), *[p for d in c['shells'].values() for p in d.values()]]
    if set(c['references']) != set(fair.ARMS) or set(c['shells']) != set(fair.SHELLS) or not set(required) <= c['saved_inputs'].keys():
        raise ValueError('UNPINNED_REFERENCE')
    return c, legacy


def baseline_gate(saved, source, stop, c):
    if (saved['status'] != 'H4_FAIR_NATIVE_FULL512_PILOT_COMPLETE'
            or source['execution_commit'] != c['baseline_scientific_SOURCE']
            or stop['failed'] or stop['scientific_units'] != 305 or not stop['mandatory_stop']):
        raise ValueError('BASELINE_PROVENANCE')
    for arm in fair.ARMS:
        a = saved['candidates'][arm+'_s1']
        if (a['arm'] != arm or a['s'] != 1 or a['segment']['input_dimension'] != 512
                or a['segment']['output_dimension'] != 512
                or a['segment']['global_phase_fit']
                or a['segment']['compiled_reference']['max_entry'] > 1e-10):
            raise ValueError('BASELINE_SCOPE')
    return dict(status='PASS', rebuild=False, metrics={a: saved['candidates'][a+'_s1']['metrics'] for a in fair.ARMS})


def prepare_models(data, legacy, call):
    old = fair.prepare_models(data, legacy, call)
    p = data['packet']
    spin, reports = call('native_model', 'XDF exact spin basis', lambda: prepare_xdf_basis(
        p['df_one_body'], p['df_lambdas'][:8], p['df_g_matrices'][:8], float(p['nuclear_repulsion']),
        source_identity=old['XDF'].source_identity, input_identity=old['XDF'].input_identity, basis_policy=SPIN))
    for i, r in enumerate(reports):
        call('spin_basis_gate', str(i), lambda r=r: r)
    return dict(B1_XDF_spin=spin, B2_CDF_relative=old['CDF'], B2_XDF_relative=old['XDF']), reports


def execute(data, models, c, root, output, call, plan, baseline):
    """Future s1-only science; old independent CAR/PF targets remain independent."""
    import numpy as np
    from scipy.linalg import expm
    from qiskit import qpy
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    from .pr_dictionary_validation.gates import signal_gate
    from .c1_h6_native import compare_metrics
    from .c1_h6_state_action import ancilla_z
    d = 1 << data['modes']; profile = fair.CompilerProfile()
    serial = fair.CompileSerialCalls(call)
    references = {a: call('reference_load', a, lambda p=p: fair.matrix_load(root/p, 'matrix', (d, d))) for a, p in c['references'].items()}
    shells = {a: {kind: call('shell_operator_load', a+':'+kind, lambda p=p: fair.matrix_load(root/p, 'matrix', (2*d, 2*d)))
                  for kind, p in v.items()} for a, v in c['shells'].items()}
    car = call('car_context', 'original G CAR', lambda: fair.stage.car_bilinears(data['modes']))
    p = data['packet']
    generators = [call('xdf_generator', 'one_body', lambda: native.xdf_car_generator(p['df_one_body'], None, car)+float(p['nuclear_repulsion'])*np.eye(d))]
    generators += [call('xdf_generator', 'F'+str(i), lambda i=i: native.xdf_car_generator(p['df_g_matrices'][i], p['df_lambdas'][i], car)) for i in range(8)]
    builders = {r: StrongBuilder(relative_fusion=r.startswith('B2'), basis_policy=SPIN if r.startswith('B1') else LEGACY, call=call) for r in ROUTES}
    for r, model in models.items():
        for b in model.blocks:
            call('gaussian_build', r+':'+b.block_id, lambda r=r, b=b: fair.basis_request(builders[r], models[r], b))
    def qsave(label, kind, qc):
        with (output/'circuits'/f'{label}_{kind}.qpy').open('xb') as f: qpy.dump(qc, f)
    def msave(label, matrix):
        with (output/'operators'/f'{label}.npz').open('xb') as f: np.savez_compressed(f, matrix=matrix)
    # B1 changes all nine diagonal bases; B2 reuses the old complete local factors.
    for i in range(9):
        target = call('local_reference', str(i), lambda i=i: expm(-.05j*generators[i]))
        qc = call('local_build', str(i), lambda i=i: builders[ROUTES[0]].local(models[ROUTES[0]], i, Fraction(1, 20)))
        qsave('B1_local'+str(i), 'raw', qc)
        actual = call('local_operator', str(i), lambda qc=qc: fair.decode(qc))
        io.save(output/'local'/f'{i}.json', call('local_gate', str(i), lambda i=i: fair.local_gate(actual, target, 'B1:'+str(i))))
    expected = {}
    for arm in fair.ARMS:
        psi = np.zeros(d, complex); psi[data['indices']] = data['grounds'][arm]
        expected[arm] = call('signal_reference', arm, lambda arm=arm, psi=psi: complex(np.vdot(psi, references[arm] @ psi)))
    # Build all raw/inverse components before parallel P0, whose translator is shared.
    raws = {}
    for r in ROUTES:
        raws[r] = call('kernel_build', r, lambda r=r: builders[r].ordered(models[r], Fraction(1, 10), optimized=True))
        qsave(r, 'raw', raws[r])
        def fusion(r=r):
            model = models[r]; groups = native.runs(model, native.ordered_schedule(9, '1/10'), optimized=True)
            pairs = [(model.blocks[a[0][0]].block_id, model.blocks[b[0][0]].block_id) for a, b in zip(groups, groups[1:])]
            records = builders[r].fusion_records
            if ([(x['previous'], x['following']) for x in records] != (pairs if r.startswith('B2') else [])
                    or any(x['multiplication'] != 'V_following.conj().T @ V_previous' for x in records)):
                raise ValueError('FUSION_ORDER_OR_DIRECTION')
            return dict(status='PASS', records=records, schedule=[(i, str(t)) for i, t in native.ordered_schedule(9, '1/10')])
        io.save(output/(r+'_fusion.json'), call('fusion_gate', r, fusion))
    def candidate(r):
        arm = models[r].label; raw = raws[r]
        compiled = serial('compile', r, lambda: fair.compile_checked(raw, profile)); qsave(r, 'compiled', compiled)
        mats = {k: call('operator_decode', r+':'+k, lambda qc=qc: fair.decode(qc)) for k, qc in [('raw', raw), ('compiled', compiled)]}
        for k, a in mats.items(): msave(r+'_'+k, a)
        ideal = fair.controlled(references[arm])
        report = call('segment_assessment', r, lambda: fair.synthesis.segment(mats['raw'], mats['compiled'], ideal, r))
        io.save(output/'candidates'/(r+'_segment.json'), report)
        for kind, qc in [('raw', raw), ('compiled', compiled)]:
            def inverse(qc=qc):
                with serial.lock:
                    if call.stop.is_set(): raise integrated.Stopped('adjoint STOP')
                    return native.actual_adjoint(qc)
            adj = call('adjoint_build', r+':'+kind, inverse); qsave(r, kind+'_adjoint', adj)
            value = call('adjoint_operator', r+':'+kind, lambda: fair.decode(adj))
            def check():
                return dict(dagger=fair.component_check(value, mats[kind].conj().T, 1e-10, label='actual dagger'),
                    reference=fair.component_check(value, ideal.conj().T, 1e-10, label='adjoint/reference'),
                    recovery=fair.component_check(value @ mats[kind], np.eye(2*d), 1e-10, label='actual recovery'))
            io.save(output/'candidates'/(r+'_'+kind+'_adjoint.json'), call('adjoint_assessment', r+':'+kind, check))
        psi = np.zeros(d, complex); psi[data['indices']] = data['grounds'][arm]
        signals = {}
        for kind in ('raw', 'compiled'):
            state = call('signal_prepare', r+':'+kind, lambda: mats[kind] @ shells['head'][kind] @ np.r_[psi, np.zeros(d)])
            axes = {a: call('signal_axis', r+':'+kind+':'+a, lambda a=a: float(ancilla_z((shells[a][kind] @ state)[None, :])[0])) for a in ('X', 'Y')}
            signals[kind] = complex(axes['X'], axes['Y'])
        gates = {}
        for name, x, y in [('raw_reference', signals['raw'], expected[arm]), ('compiled_raw', signals['compiled'], signals['raw']), ('compiled_reference', signals['compiled'], expected[arm])]:
            gates[name] = call('signal_gate', r+':'+name, lambda x=x, y=y: signal_gate(x.real, x.imag, y, 2e-10))
        # Metrics are obtained only after operator, adjoint and all signal gates PASS.
        metrics = call('metrics', r, lambda: circuit_metrics(compiled))
        value = dict(route=r, arm=arm, s=1, segment=report, XY=gates, metrics=metrics,
            CA_achieved=None, shots=None, total_cost=None, rigorous_enclosure=None)
        io.save(output/'candidates'/(r+'.json'), value)
        return r, value
    values = {}
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures = [pool.submit(candidate, r) for r in ROUTES]
        for f in as_completed(futures):
            try:
                k, value = f.result(); values[k] = value
            except BaseException:
                call.stop.set()
        if call.stop.is_set(): raise integrated.Stopped('strong baseline dependent STOP')
    b = baseline['metrics']; v = {r: values[r]['metrics'] for r in ROUTES}
    pairs = [(ROUTES[0]+'/B0_XDF', v[ROUTES[0]], b['XDF']), (ROUTES[1]+'/B0_CDF', v[ROUTES[1]], b['CDF']),
             (ROUTES[2]+'/B0_XDF', v[ROUTES[2]], b['XDF']), ('B0_CDF/B1_XDF', b['CDF'], v[ROUTES[0]]),
             ('B2_CDF/B2_XDF', v[ROUTES[1]], v[ROUTES[2]])]
    comparisons = [call('cost_comparison', label, lambda label=label, a=a, b=b: compare_metrics(a, b, label)) for label, a, b in pairs]
    return dict(status='H4_STRONG_BASELINE_S1_NUMERICAL_COMPLETE', candidates=values, comparisons=comparisons,
        B3='NOT_RUN', CA_achieved=None, shots=None, total_cost=None, mandatory_stop=True)


def run(root, path, input_path, output):
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve(); c, legacy = preflight(root, path, input_path)
    output = Path(output).resolve(); parts = output.relative_to(root/FAMILY).parts
    if len(parts) != 2 or parts[1] != 'run01' or date.fromisoformat(parts[0]).isoformat() != parts[0] or output.exists():
        raise ValueError('FRESH_RUN01_NO_RETRY_RESUME')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('circuits', 'operators', 'local', 'candidates'): (output/name).mkdir()
    for key, name in [('XDG_CACHE_HOME', ''), ('MPLCONFIGDIR', 'matplotlib'), ('TMPDIR', 'tmp')]:
        p = root/'.cache/h4-strong-baseline'/name; p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
    call = integrated.Calls(output, LIMITS, MAX_CALLS); plan = integrated.resource_plan(task_count=3, full_dimension=256)
    tick, cpu = time.monotonic(), time.process_time()
    try:
        io.save(output/'contract.json', c)
        io.save(output/'SOURCE.json', dict(io.read(root/c['source_manifest']), execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()))
        io.save(output/'resource_plan.json', plan)
        data = fair.load_data(root, legacy, input_path, call)
        verified = call('identity_gate', 'unchanged physical DF/PF/ground identity', lambda: fair.identity(data, legacy))
        io.save(output/'saved_identity.json', {k: v for k, v in verified.items() if k != 'rows'})
        baseline = call('baseline_gate', 'B0 provenance', lambda: baseline_gate(io.read(root/c['baseline_result']), io.read(root/c['baseline_SOURCE']), io.read(root/c['baseline_STOP']), c))
        with threadpool_limits(limits=1 if plan['workers'] > 1 else None):
            models, reports = prepare_models(data, legacy, call)
            io.save(output/'spin_basis.json', reports)
            result = execute(data, models, c, root, output, call, plan, baseline)
        io.save(output/'result.json', result)
        return result
    except BaseException as e:
        call.stop.set(); io.save(output/'failure.json', dict(error=str(e), first_failure=call.first_failure)); raise
    finally:
        io.save(output/'STOP.json', dict(mandatory_stop=True, failed=call.stop.is_set(), retry=0, resume=0,
            scientific_units=sum(call.counts.values()), call_counts=dict(call.counts)))
        io.save(output/'resources.json', dict(wall_seconds=time.monotonic()-tick, process_CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
