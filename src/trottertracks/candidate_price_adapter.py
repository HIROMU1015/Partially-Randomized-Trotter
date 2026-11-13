"""PR candidate -> source-bound context -> existing C1 fast path acquisition.

Metadata helpers are passive. NativeRuntime.context/sample and Adapter.acquire
are explicit scientific entry points; no launcher, N solver, mean fit or E[C]/G.
Work in one thread per process (the existing reversible lowering/hash contexts
are process-local); parallel callers use separate Adapter/Runtime processes.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from . import pf_pr_rpe_pipeline as p
from . import compiled_price_fast as fast
from . import efficient_controlled_zz as zz
from . import p1_efficient_zz as pinned

AXES = {'X': 'cosine', 'Y': 'sine'}
IMPLEMENTATION = dict(carrier=pinned.CARRIER, provider=zz.PROVIDER,
    C1=pinned.C1_SCIENCE, fast_base='5a67cea7705e80ff295306606611d774724ae2e2',
    adapter='PR_CANDIDATE_PRICE_CONTEXT_V1')


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _real(value):
    x = p.accounting.exact_scalar(value)
    f = float(x)
    if not math.isfinite(f) or p.accounting.exact_scalar(f) != x:
        raise ValueError('Physical time must have the same exact binary64 value')
    return f


def from_b(config, m, physical, *, input_source, execution_source, axes=('X', 'Y'), preparation_hash=None, expected_scalar=None):
    """Bind an existing B configuration to its supplied physical row; no science.

    input_source identifies the restored inputs/loader; execution_source identifies
    the new adapter/runtime SOURCE. Neither is an approval artifact.
    """
    from . import pr_rpe_rmse_controller as B
    B._validate(config)
    if type(m) is not int or m not in range(config['spec']['M'] + 1):
        raise ValueError('Round outside supplied configuration')
    s, row = config['spec'], config['rounds'][m]
    axes = tuple(axes)
    if not axes or len(set(axes)) != len(axes) or any(a not in AXES for a in axes):
        raise ValueError('Explicit unique X/Y axes required')
    if not input_source or not execution_source:
        raise ValueError('Input and new execution SOURCE required')
    if s['method'] != 'PR' or s['binding']['system'] != 'H4':
        raise ValueError('OUTSIDE_DOMAIN: current Accounting uses eight H4 system wires')
    c = deepcopy(physical)
    rank = s['binding'].get('rank')
    if (type(rank) is not int or type(c['prefix']) is not int or not 1 <= c['prefix'] < rank
            or c['method'] != 'PR' or c['system'] != 'H4'
            or c['formula'] not in ('2nd', '4th(m5_best)')):
        raise ValueError('OUTSIDE_DOMAIN: nonempty ordered PR prefix/tail and existing S2/m5 required')
    if (any(type(c[k]) is not int for k in ('q', 'R_outer', 'K'))
            or c['q'] < 1 or c['R_outer'] < 1 or c['K'] < 0 or c['K'] % 2):
        raise ValueError('Physical q/R and even finite-K must be explicit integers')
    if c['id'] != p.candidate_id(c) or c['T'] != _real(row['T']) or c['delta'] != _real(row['actual_delta_pf']):
        raise ValueError('Physical ID/time differs from B')
    if (c['formula'], c['prefix'], c['K'], c['q'], c['R_outer']) != (
            s['formula'], s['L_D'], s['finite_K'], row['q_PF'], row['R_outer']):
        raise ValueError('Physical split/recipe/K/q/R differs from B')
    if row.get('physical_candidate') is not None and not p.same(row['physical_candidate'], c):
        raise ValueError('Source physical row differs from B')
    structure = p.structure(c)
    if not p.same(structure, row['structure']) or row['resolution'] != 'GENERATED':
        raise ValueError('Physical schedule/allocation unresolved or different')
    one_body = s['split'].get('one_body_policy')
    if one_body is None and s.get('schedule_adapter') == 'A_FIXED_GRID_510FBACD_V1':
        one_body = 'DETERMINISTIC'  # source-defined existing PR preparation, not a new split
    if (one_body != 'DETERMINISTIC'
            or s['split'].get('deterministic_scope') != 'PREFIX'
            or s['law']['model'] != 'ideal_finite_RTE_law'):
        raise ValueError('OUTSIDE_DOMAIN: ordered deterministic one-body/prefix and original finite-RTE law required')
    profile = deepcopy(s['profile'])
    if (any(profile.get(k) != v for k, v in p.PROFILE.items())
            or any(profile.get(k) is not None for k in ('backend', 'target', 'coupling_map'))):
        raise ValueError('Fixed compiler/control/full measured wrapper/P0 profile required')
    if profile.get('lowering_id', zz.LOWERING_ID) != zz.LOWERING_ID:
        raise ValueError('Efficient ZZ lowering required')
    profile['lowering_id'] = zz.LOWERING_ID
    context = dict(physical_candidate=c, structure=structure, binding=deepcopy(s['binding']),
        split=deepcopy(s['split']), one_body_policy=one_body, law=deepcopy(row['law_binding']), price_profile=profile,
        preparation_hash=preparation_hash, input_source=deepcopy(input_source),
        execution_source=deepcopy(execution_source), implementation=deepcopy(IMPLEMENTATION))
    if expected_scalar is not None:
        context['expected_scalar'] = _real(expected_scalar)
    result = dict(schema='PR_CANDIDATE_PRICE_REQUEST_V1', configuration_id=config['id'], m=m,
        identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
        axes=list(axes), physical_context=context, context_key=_key(context))
    result['request_id'] = _key(result)
    return result


def from_a_job(config, job, **kwargs):
    """Accept A821c's passive job; its numerical signal recipe is not a price."""
    m = job['m']; row = config['rounds'][m]; context = job['physical_context']
    if (job['B_configuration_id'] != config['id'] or job['B_candidate_id'] != row['candidate']['id']
            or not p.same(job['B_identity'], row['identity'])
            or not p.same(job['B_law_binding'], row['law_binding'])
            or any(not p.same(context[k], config['spec'][k]) for k in ('binding', 'profile', 'split', 'law'))
            or not p.same(context['structure'], row['structure'])):
        raise ValueError('Foreign A job/B physical context')
    return from_b(config, m, context['physical_candidate'], **kwargs)


def check_request(request):
    copy = deepcopy(request); identity = copy.pop('request_id')
    if identity != _key(copy) or request['context_key'] != _key(request['physical_context']):
        raise ValueError('Price request/context mutated')


class NativeRuntime:
    """Reuse pinned prepare/compose on an already restored, source-bound H4 H.

    The caller restores/verifies H/state/snapshot with its existing input loader.
    No input file reader, state solver, signal or historical Backend.context runs
    here. context() is scientific preparation, even before the first compile.
    """
    def __init__(self, root, ham, *, binding, input_source, execution_source):
        self.root, self.ham = Path(root), ham
        self.binding, self.input_source = deepcopy(binding), deepcopy(input_source)
        self.execution_source = deepcopy(execution_source)
        self.science = pinned.load_c1_backend(self.root)['r_price_phase1_science']
        self.preparations, self.contexts = {}, {}
        self.tools = None

    def _tools(self):
        if self.tools is not None:
            return self.tools
        import importlib
        import numpy as np
        import qiskit
        if (np.__version__, qiskit.__version__) != ('1.26.4', '1.3.0'):
            raise ValueError('Fixed NumPy1.26.4/Qiskit1.3.0 runtime required')
        # Require only the actual provider modules used; no shared settings changed.
        names = ('trotterlib.rte', 'trotterlib.df_partial_s2', 'trotterlib.df_rte_qiskit',
            'trotterlib.rte_compiled_cost', 'trotterlib.pr2_matched_accuracy_m1_execution',
            'trotterlib.df_trotter.ops', 'trotterlib.df_partial_randomized_pf', 'trotterlib.df_rte_circuit',
            'trottertracks.resource_applicability.ax2b_native_df_v5')
        for name in names:
            module = importlib.import_module(name)
            relative = 'src/' + name.replace('.', '/') + '.py'
            if Path(module.__file__).read_bytes() != pinned._blob(self.root, zz.PROVIDER, relative):
                raise ValueError('Provider SOURCE mismatch: ' + name)
        def require(value, reason):
            if not value: raise ValueError(reason)
        h = dict(np=np, math=math, json=json, hashlib=hashlib, require=require)
        s = self.science
        # Existing C1 extraction keeps original expressions and import bindings.
        import ast
        tree = ast.parse(pinned._blob(self.root, pinned.CARRIER, 'src/trotterlib/h46_m5_plan.py'))
        tail = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'TAIL' for t in n.targets))
        core = dict(math=math, TAIL=tail)
        s.functions(self.root, pinned.CARRIER, 'scripts/h46_new2_core.py', ['allocation'], core)
        s.functions(self.root, pinned.CARRIER, 'src/trotterlib/h46_m5_plan.py', ['weights', 'schedule'], core)
        h['core'] = SimpleNamespace(**{k: core[k] for k in ('weights', 'allocation', 'schedule')})
        s.functions(self.root, pinned.CARRIER, 'scripts/h46_new2_native.py', ['gaussian_operations', 'prepare'], h)
        s.functions(self.root, pinned.CARRIER, 'scripts/h46_pr_pilot_repair.py', ['canonical_tau'], h)
        s.functions(self.root, pinned.CARRIER, 'scripts/h46_long_time_native.py',
            ['trial_seed', 'sample_requests', 'compose'], h, ('h46_m5_contract', 'h46_pr_pilot_repair'))
        from trotterlib import df_rte_qiskit as dfq
        from trotterlib.rte import CompilerSettings
        from trotterlib.rte_compiled_cost import transpile_and_measure_cost
        from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
        from .h4_df10_price_v1 import Accounting
        from .resource_applicability import ax2b_native_df_v5 as native
        from .resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        self.tools = SimpleNamespace(h=h, dfq=dfq, np=np, Accounting=Accounting, native=native,
            CompilerSettings=CompilerSettings, compile=transpile_and_measure_cost,
            one=one_body_diagonal_primitives, squared=df_squared_diagonal_primitives,
            fingerprint_context=use_fingerprint_reuse)
        return self.tools

    def context(self, request):
        check_request(request)
        pc = request['physical_context']; c = pc['physical_candidate']; key = request['context_key']
        if (not p.same(pc['binding'], self.binding) or not p.same(pc['input_source'], self.input_source)
                or not p.same(pc['execution_source'], self.execution_source)):
            raise ValueError('Restored input/runtime SOURCE differs from request')
        if key in self.contexts:
            return self.contexts[key]
        t = self._tools()
        prep_key = _key(dict(binding=self.binding, input_source=self.input_source,
            execution_source=self.execution_source, split=pc['split'], implementation=pc['implementation']))
        if prep_key not in self.preparations:
            self.preparations[prep_key] = t.h['prepare'](self.ham, c['prefix'])
        prep = self.preparations[prep_key]
        if (prep.ld != c['prefix'] or prep.num_system_qubits != 8
                or prep.hamiltonian_hash != self.binding['hamiltonian']
                or tuple(prep.deterministic_fragment_indices) != tuple(range(c['prefix']))
                or tuple(prep.randomized_block_indices) != tuple(range(c['prefix'], self.ham.n_blocks))
                or self.ham.n_blocks != self.binding['rank']):
            raise ValueError('Prepared H4 split/head/tail differs from request')
        lam = prep.rte_preparation.symbolic_tail.lambda_r
        if (lam <= 0 or float(p.accounting.exact_scalar(pc['split']['lambda_R'])).hex() != float(lam).hex()
                or pc['preparation_hash'] not in (None, prep.preparation_hash)):
            raise ValueError('Preparation hash or exact tail lambda differs; empty tail outside domain')
        scalar = self.ham.constant + prep.extracted_identity_coefficient
        if pc.get('expected_scalar') is not None and float(scalar).hex() != float(pc['expected_scalar']).hex():
            raise ValueError('Prepared scalar differs from the requested total scalar')
        blocks = []
        for w in pc['structure']['signed_weights']:
            bs = []
            for b in prep.deterministic_blocks:
                # Match native piece time exactly; C1 delta*w/2 agrees on its dyadic grid.
                half = (c['T'] * w / c['q']) / 2
                diag = (t.one(t.np.array(b.diagonal_eigenvalues), half) if b.block_kind == 'one_body'
                        else t.squared(t.np.array(b.diagonal_eta), b.lam, half))
                basis = [dict(name=o.name, qubits=list(q), parameters=[str(v) for v in o.params])
                         for o, q in b.runtime_basis_operations]
                bs.append(dict(basis=basis, diagonal=diag))
            blocks.append(bs)
        ctx = SimpleNamespace(prep=prep, blocks=blocks, scalar=scalar,
            h=t.h, Accounting=t.Accounting, native=t.native, compile=t.compile,
            CompilerSettings=t.CompilerSettings, fingerprint_context=t.fingerprint_context, context_key=key)
        self.contexts[key] = ctx
        return ctx

    def sample(self, request, cohort, index):
        """Explicit normal-law sampling. No forced orders or automatic retries."""
        if not cohort or type(index) is not int or index < 0:
            raise ValueError('Explicit cohort and nonnegative index required')
        ctx = self.context(request)
        return ctx.h['sample_requests'](request['physical_context']['physical_candidate'], ctx.prep, cohort, index)[0]

    def lowering(self):
        return zz.efficient_controlled_zz(self._tools().dfq)


def _law_records(ctx, c, requests):
    seeds, laws = [], []
    weights = ctx.h['core'].weights(c['formula'])
    allocation = ctx.h['core'].allocation(weights, c['R_outer'])
    if len(requests) != c['q'] * len(weights):
        raise ValueError('Signed request grouping/count differs')
    for i, req in enumerate(requests):
        j = i % len(weights)
        if (req.preparation is not ctx.prep or req.step_time != c['T'] * weights[j] / c['q']
                or len(req.rte_occurrence.events) != allocation[j]
                or req.rte_config.finite_taylor_order != c['K']
                or req.rte_occurrence.tail_hash != ctx.prep.rte_preparation.symbolic_tail.tail_hash
                or req.controlled is not True or req.ancilla_qubit != 8):
            raise ValueError('Foreign preparation/K/time/grouping/control in event path')
        seeds.append(req.seed)
        laws.append(dict(steps=allocation[j], orders=list(req.rte_distribution.orders),
            probabilities=list(req.rte_distribution.order_probabilities)))
    return seeds, laws


class Adapter:
    """Per-process acquisition facade; delegates all pricing to existing fast C1."""
    def __init__(self, root, runtime, *, cache_root):
        self.root, self.runtime, self.cache_root = Path(root), runtime, Path(cache_root)

    def acquire(self, request, requests, *, cohort, index, directory, conditional=False):
        """Acquire one supplied path only; no expectation, fitting or G arithmetic."""
        check_request(request)
        if not cohort or type(index) is not int or index < 0 or type(conditional) is not bool:
            raise ValueError('Explicit cohort, index and conditional role required')
        ctx = self.runtime.context(request); pc = request['physical_context']; c = pc['physical_candidate']
        seeds, laws = _law_records(ctx, c, requests)
        science = self.runtime.science
        axes = tuple(AXES[a] for a in request['axes'])
        cls = fast.make_backend_class(science.Backend, science, axes=axes)
        cell = dict(id=request['request_id'], system='H4', candidate_id=c['id'], cohort=cohort)
        plan = dict(source=pc['execution_source'], menus=dict(H4=dict(candidates=[
            dict(candidate=deepcopy(c), structure=deepcopy(pc['structure']))])),
            runtime=dict(old_cache=str(self.cache_root)))
        backend = cls(self.root, plan)
        backend.contexts['H4'] = ctx  # only this owned, fresh backend; never historical context()
        with self.runtime.lowering():
            result = backend.price_requests(cell, index, Path(directory), requests, seeds, laws, conditional=conditional)
        records = []
        for i, axis in enumerate(request['axes']):
            raw = json.loads((Path(directory)/(AXES[axis] + '.json')).read_text())
            if any(type(raw['metrics'].get(k)) is not int or raw['metrics'][k] < 0
                   for k in ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')):
                raise ValueError('Invalid cached/compiled six metrics')
            records.append(dict(axis=axis, compiled_quantity='FULL_WRAPPER_PATH_COMPILED_RZ_COUNT',
                structural_quantity='FULL_WRAPPER_PATH_STRUCTURAL_RZ_COUNT', C=result['C'][i], S=result['S'][i],
                D=result['C'][i]-result['S'][i], metrics=raw['metrics'], fingerprint=raw['fingerprint'],
                cache_hit=raw['cache_hit'], cache_reference=str(self.cache_root/(raw['fingerprint']+'.json')) if raw['cache_hit'] else None))
        resolved = dict(preparation_hash=ctx.prep.preparation_hash,
            tail_hash=ctx.prep.rte_preparation.symbolic_tail.tail_hash, lambda_R=ctx.prep.rte_preparation.symbolic_tail.lambda_r,
            scalar=ctx.scalar, deterministic_fragments=list(ctx.prep.deterministic_fragment_indices),
            randomized_fragments=list(ctx.prep.randomized_block_indices))
        output = dict(quantity='PR_CANDIDATE_PAIRED_PATH_PRICE', request=deepcopy(request),
            resolved_context=resolved, resolved_context_key=_key(dict(context=pc, resolved=resolved)),
            cohort=cohort, index=index, conditional=conditional, records=records, seeds=seeds, laws=laws,
            event_digest=result['event_digest'], new_compiles=result['new_compiles'], cache_hits=result['cache_hits'],
            mean_compiled_price=None, confidence_interval=None, qualified_expectation=False,
            source=dict(execution=pc['execution_source'], input=pc['input_source'],
                private_result=str(Path(directory)/'result.json')))
        science.write(Path(directory)/'CANDIDATE_PATH.json', output)
        return output


def pending_price_records(request, *, reason='PATH_MEASUREMENTS_ARE_NOT_A_QUALIFIED_MEAN', path_source=None):
    """B-compatible unavailable responses. Never put S/path C into mean.rz_count."""
    check_request(request)
    row = dict(identity=request['identity'], structure=request['identity']['structure'])
    def measurement(axis):
        out = p.measurement(row, axis)
        out['law_context'].update(price_law=None, ideal_sampler_distance_qualified=None)
        return out
    return [dict(**measurement(axis), identity=deepcopy(request['identity']),
        law_binding=deepcopy(request['law_binding']), axis=axis, cohort='PENDING_CANDIDATE_MEAN',
        quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE', classification='UNAVAILABLE', expectation='UNAVAILABLE',
        mean=dict(rz_count=None), n_paths=None, model=None, price_status='NOT_EVALUATED', reason=reason,
        source=dict(execution=request['physical_context']['execution_source'], path_source=path_source),
        price_profile=deepcopy(request['physical_context']['price_profile']),
        uncertainty=dict(total_CI=None, law=None, rare=None, model=None),
        normalization_is_price_multiplier=False) for axis in request['axes']]
