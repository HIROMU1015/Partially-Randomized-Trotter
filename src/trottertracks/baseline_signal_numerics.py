"""Supplied-context baseline means, reusing frozen PF/RTE numerical kernels.

No molecular input or scientific libraries on import. No PR signal dispatcher,
reference solver, sampler, circuit, N solver or price code. Real calls require
the separately permitted SOURCE/input/finite scientific scope.
"""
from copy import deepcopy
import ast
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace

CORE = 'e417ffc9f52b13c18f5493193ce510d858664259'
EXPM = 'dd24dd329949c255d5e37b90ff8993fa8df7159b'
SNAPSHOT = '155529e20a8fac39e00786c40b57830b71203c4f'
SHARED = '256caad3414a80de02fb68eca89c13c957ed6354'
RTE = '71ba49bb28b295aa28e51101bc3b1bf07d500767'
INDEX = dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
             path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')


@lru_cache(maxsize=None)
def kernels(root):
    import numpy as np
    from scipy.linalg import eigh
    from .r_price_phase1_science import functions
    from .baseline_signal_pipeline import blob
    scope = dict(np=np, eigh=eigh, math=math, SimpleNamespace=SimpleNamespace)
    tree = ast.parse(blob(root, EXPM, 'src/trottertracks/pf_power_stability.py'))
    scope['POWER_TOLERANCE'] = next(ast.literal_eval(n.value) for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'POWER_TOLERANCE' for t in n.targets))
    functions(root, CORE, 'scripts/h46_new2_core.py',
              ('spectrum', 'polynomial', 'occupation_one_body', 'matrix_only_tail_coefficients'), scope)
    functions(root, EXPM, 'src/trottertracks/pf_power_stability.py', ('expm_step', 'power_diagnostics'), scope)
    functions(root, SHARED, 'src/trottertracks/h6_df19_rpe_d2.py', ('make_context',), scope, strip=('h46_new2_core',))
    # The standalone pure-law module has no relative imports. Load its frozen
    # source without executing trotterlib.__init__ and unrelated circuit code.
    name = __name__ + '._rte_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    rte = types.ModuleType(name); sys.modules[name] = rte
    exec(compile(blob(root, RTE, 'src/trotterlib/rte.py'), RTE + ':src/trotterlib/rte.py', 'exec'), rte.__dict__)
    scope['finite_rte_distribution'] = rte.finite_rte_distribution
    return SimpleNamespace(**scope)


def load_shared_h4(root, binding):
    """Same frozen saved-snapshot loader/matrix constructor as A; no prepare/N.

    Only called by explicit execute, never by planning. Reuses array/hash/state
    checks, reads original inputs without writing or rebuilding the state/H.
    """
    from . import pf_pr_rpe_pipeline as p
    from .r_price_phase1_science import functions
    idx = p.SavedArtifacts(root).read(INDEX)
    inputs = idx['saved_inputs']['H4']['inputs']; spec = idx['T20_reference_files']['H4']
    for v in [*inputs.values(), spec]:
        if hashlib.sha256(Path(v['path']).read_bytes()).hexdigest() != v['sha256']:
            raise ValueError('Saved input/context SOURCE bytes differ')
    meta = json.loads(Path(spec['path']).read_bytes())
    for key in ('system', 'hamiltonian', 'state', 'snapshot', 'rank'):
        if meta['binding'].get(key) != binding.get(key):raise ValueError('Shared snapshot binding differs: ' + key)
    for key in set(meta['binding']) & set(binding):
        if not p.same(meta['binding'][key], binding[key]):raise ValueError('Shared input binding differs: ' + key)
    def require(value, reason):
        if not value:raise ValueError(reason)
    scope = dict(Path=Path, c=SimpleNamespace(
        sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        read=lambda path: json.loads(Path(path).read_bytes()), require=require))
    functions(root, SNAPSHOT, 'scripts/h46_m5_tol_saved_input.py', ('array_record', 'load'), scope)
    ham, sector, full, state, evidence = scope['load'](dict(mapping=dict(
        tol_snapshot=inputs['H4_tol_snapshot.npz']['path'], tol_receipt=inputs['H4_tol_receipt.json']['path'])))
    if ham.n_blocks != 10 or ham.n_qubits != 8 or len(state) != 36:
        raise ValueError('H4 DF10/sector saved layout required')
    saved = kernels(str(root)).make_context(ham, sector, state, {})
    return saved, deepcopy(binding), dict(prefixes=meta['prefixes'], source=dict(index=INDEX, inputs=inputs, context=spec))


class Context:
    """One immutable input context shared by requested baseline configurations.

    Each Context owns caches; no system-only/global molecular cache. A's shared
    saved context may be supplied directly to avoid duplicate matrix assembly.
    """
    def __init__(self, root, saved, binding, metadata=None):
        import numpy as np
        self.root = str(root); self.binding = deepcopy(binding); self.metadata = deepcopy(metadata or {})
        self.state = np.array(saved.state, dtype=np.complex128, copy=True)
        self.matrices = tuple(np.array(m, dtype=np.complex128, copy=True) for m in saved.matrices)
        self.constant = float(saved.ham.constant); self.ham = saved.ham
        if (len(self.matrices) != binding['rank'] + 1 or not math.isfinite(self.constant)
                or not np.isfinite(self.state).all() or abs(np.linalg.norm(self.state) - 1) > 1e-12):
            raise ValueError('Saved input matrices/state/constant layout unresolved')
        for m in self.matrices:
            if m.shape != (len(self.state), len(self.state)) or not np.isfinite(m).all() or np.linalg.norm(m-m.conj().T, 'fro') > 1e-9:
                raise ValueError('Finite Hermitian input matrices required; no projection')
            m.setflags(write=False)
        self.state.setflags(write=False)
        self.half = {}; self.steps = {}; self.spectra = {}; self.polynomials = {}; self.normalizers = {}
        self.tail_scalars = {int(k): float(v['scalar']) for k, v in self.metadata.get('prefixes', {}).items()}

    def tail_scalar(self, prefix):
        if prefix not in self.tail_scalars:
            # Same canonical identity coefficient extractor used by the prior
            # signal pipeline. This is future science, not a planning call.
            self.tail_scalars[prefix] = kernels(self.root).matrix_only_tail_coefficients(self.ham, prefix)[0]
        return self.tail_scalars[prefix]

    def pf_step(self, candidate, weights):
        import numpy as np
        c = candidate; times = tuple(c['T'] * w / c['q'] for w in weights)
        key = c['prefix'], c['formula'], tuple(v.hex() for v in times)
        if key not in self.steps:
            result = np.eye(len(self.state), dtype=np.complex128)
            for time in times:
                # Retain T*w/q operation order; reuse the direct-expm S2
                # constructor for each signed piece, sharing its half cache.
                result = kernels(self.root).expm_step(self.matrices[:c['prefix']+1], (1.,), time, self.half) @ result
            self.steps[key] = result
        return self.steps[key]


def evaluate(ctx, request, reference, *, source):
    """One requested baseline mean. Failure values remain in the returned record."""
    import numpy as np
    from . import pf_pr_rpe_pipeline as p
    record = dict(domain=deepcopy(request['domain']), source=deepcopy(source),
        kernel_sources=dict(PF=EXPM, finite_RTE=RTE, polynomial_spectrum=CORE, saved_context=SHARED),
        input_provenance=deepcopy(ctx.metadata.get('source')),
        physical_candidate=deepcopy(request['physical_candidate']), status='NUMERICAL_UNRESOLVED',
        signal=None, numerical=dict(gates_passed=False, certified=False), scientific_PASS_FAIL=None,
        N_X=None, N_Y=None, compiled_C=None, G=None)
    diag = record['numerical']; c = request['physical_candidate']; T = c['T']; pure = request['comparison_method'] == 'FULL_H_RTE'
    z = lambda v: complex(v['real'], v['imag'])
    zz = lambda v: dict(real=float(v.real), imag=float(v.imag))
    try:
        desc = request['domain']['reference']
        if (not p.same(reference['source'], desc['source']) or not p.same(reference['binding'], desc['binding'])
                or not p.same(reference['T'], desc['T'])):
            raise ValueError('Reference SOURCE/H/state/time mismatch')
        ref, other = z(reference['reference']), z(reference['independent_reference'])
        pair = reference['signal_difference']; diag['reference_pair_difference'] = pair
        if (not all(math.isfinite(v) for v in (ref.real, ref.imag, other.real, other.imag, pair))
                or not 0 <= pair <= 1e-10 or abs(ref-other) > 1e-10):
            raise ValueError('Saved reference pair gate')
        logB = 0.; sensitivity = 0.; closure = 0.; b = 1.
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if pure:
                prep = request['domain']['preparation']
                if (prep['one_body_policy'] != 'RANDOMIZED' or prep['deterministic_blocks'] != 0
                        or prep['scalar']['constant'] != ctx.constant or not p.same(prep['binding'], ctx.binding)):
                    raise ValueError('Full-H one-body/scalar/preparation differs')
                scalar = prep['scalar']['extracted_identity']; key = prep['preparation_hash'], float(scalar).hex()
                if key not in ctx.spectra:
                    non_scalar = sum(ctx.matrices, np.zeros_like(ctx.matrices[0])) - scalar * np.eye(len(ctx.state))
                    ctx.spectra[key] = kernels(ctx.root).spectrum(non_scalar)
                spec = ctx.spectra[key]; q, R = c['q'], c['R_outer']; dt = (T/q)/R
                tau = prep['lambda_R'] * dt; nk = key, tau.hex(), c['K']
                if nk not in ctx.normalizers:ctx.normalizers[nk] = kernels(ctx.root).finite_rte_distribution(tau, c['K']).exact_finite_distribution
                b = ctx.normalizers[nk]; diag['finite_normalizer'] = b
                if not math.isfinite(b) or b <= 1:raise ValueError('Normalizer precision unresolved')
                logB = q*R*math.log(b); diag['log_B'] = logB
                pk = key, dt.hex(), c['K']
                if pk not in ctx.polynomials:ctx.polynomials[pk] = kernels(ctx.root).polynomial(spec[0], dt, c['K'])
                one = ctx.polynomials[pk]**R
                factors = np.stack([one**q, (one/(b**R))**q], axis=1)
                vectors = spec[1] @ (factors*(spec[2] @ ctx.state)[:, None])
                vectors *= np.exp(-1j*T*(ctx.constant+scalar))
                corrected, raw = (complex(np.vdot(ctx.state, vectors[:, j])) for j in (0, 1))
                closure = float(np.linalg.norm(vectors[:, 1] - math.exp(-logB)*vectors[:, 0]))
                sensitivity = q*R*(math.log(math.nextafter(b, math.inf))-math.log(math.nextafter(b, 0.)))
                diag.update(backend='PURE_FULL_H_GROUPED_SPECTRAL_POWER', raw_corrected_independent_propagation=True,
                            raw_corrected_vector_closure=closure, norm_error=None, power_unitarity_error=None,
                            maximum_intermediate_norm=float(max(1.,np.linalg.norm(vectors,axis=0).max())),
                            maximum_norm_domain='POWER_OUTPUT_BOUNDARIES_NOT_EVERY_EVENT',
                            expectation_law='ideal_finite_RTE_law', sampler_law_distance=None)
            else:
                step = ctx.pf_step(c, request['structure']['signed_weights'])
                powered, metrics = kernels(ctx.root).power_diagnostics(step, c['q'], ctx.state)
                diag.update(metrics, backend='DIRECT_EXPM_SIGNED_PF_MATRIX_POWER', raw_corrected_independent_propagation=False)
                vector = powered @ ctx.state; vector *= np.exp(-1j*T*ctx.constant)
                diag['maximum_intermediate_norm'] = float(max(1.,np.linalg.norm(vector)))
                diag['maximum_norm_domain'] = 'POWER_OUTPUT_BOUNDARIES_NOT_EVERY_EVENT'
                corrected = complex(np.vdot(ctx.state, vector))
                if c['method'] == 'B0I':corrected *= np.exp(-1j*T*ctx.tail_scalar(c['prefix']))
                raw = corrected
        record['signal'] = dict(raw=zz(raw), corrected=zz(corrected), reference=deepcopy(reference['reference']),
            total_signed_bias=zz(corrected-ref), log_B=logB, B=None, attenuation=None)
        record['signal'].update(B=math.exp(logB), attenuation=math.exp(-logB))
        phase = lambda v: None if abs(v) == 0 or abs(ref) == 0 else float(np.angle(v*ref.conjugate()))
        record['local'] = dict(absolute_corrected_bias=abs(corrected-ref), corrected_phase_to_reference=phase(corrected),
            raw_phase_to_reference=phase(raw), corrected_radius=abs(corrected), raw_radius=abs(raw), reference_radius=abs(ref))
        diag.update(normalizer_one_ulp_log_sensitivity=sensitivity, scalar_once=True, normalization_closure=closure)
        vals = (raw.real, raw.imag, corrected.real, corrected.imag, logB, closure, sensitivity)
        if not all(math.isfinite(v) for v in vals) or closure > 1e-9 or math.exp(-logB) == 0:
            raise ValueError('Finite signal/normalization closure/range gate')
        if not pure and not diag['gate_pass']:raise ValueError('Deterministic power norm/unitarity gate')
        if max(abs(raw.real), abs(raw.imag)) > 1:raise ValueError('Raw outcome range unresolved; no clamp')
        diag['gates_passed'] = True; record['status'] = 'NUMERICAL_MEAN_AVAILABLE'
    except (ValueError, OverflowError, FloatingPointError) as error:
        # Keep finite diagnostics and failed mean, but never supply it as a B
        # probability. Invalid nonfinite values are represented by explicit null.
        record['reason'] = str(error)
    def finite(value):
        if isinstance(value, dict):return {k: finite(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):return [finite(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):return None
        return value
    return finite(record)
