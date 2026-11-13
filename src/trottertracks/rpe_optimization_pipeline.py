"""Thin A/C signal -> B allocation/cost/select connection, with supplied D prices.

Planning/import is passive. produce_signals/diagnose/compare on real data need
their own finite scientific scope. No optimizer, N qualifier or price builder.
"""
import ast
from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path
import sys
import types

from . import baseline_signal_pipeline as C

A_SOURCE = 'fefd548ecd56f60d72031b3d385f6f5e6ee95d06'
B_SOURCE = 'b9b9ffa6233db9dd2b413dc07b27e7039324c666'
C_SOURCE = '86f37d25ffd5eee3daf7f25ce001b090b1f2cf33'
D_SOURCE = '24b0a6eb1d806391a3c6ad454fb0c018148d8545'
D_ADAPTER_SOURCE = '0c6e16342b3c804fa0bd89a4028d6b5bea7b0322'
MODULE = 'src/trottertracks/rpe_optimization_pipeline.py'


@lru_cache(maxsize=None)
def providers(root):
    """Fixed Git modules, sharing C's pinned B API; import-only redirection.

    The controller is byte-identical at C's B reference and this B release.
    Two absolute imports are routed to the SAME pure RTE/decoder functions,
    avoiding trotterlib's eager circuit package import. Function arithmetic is
    unchanged. A's optional molecular factories remain external dependencies.
    """
    api = C.published_api(str(root)); name = api.B.__package__
    for short in ('pr_rpe_rmse_controller', 'pf_pr_rpe_pipeline', 'h4_df10_rpe_connection'):
        path = 'src/trottertracks/' + short + '.py'
        if C.blob(root, C.B_SOURCE, path) != C.blob(root, B_SOURCE, path):
            raise ValueError('Pinned B interface changed; integration needs a new binding')
    def module(short):
        m = types.ModuleType(name + '.' + short); m.__package__ = name
        m.__file__ = str(Path(root)/'src/trottertracks'/(short+'.py'))
        sys.modules[m.__name__] = m
        setattr(sys.modules[name], short, m)
        return m
    def load(short, commit, redirects=None):
        m = module(short); path = 'src/trottertracks/' + short + '.py'
        tree = ast.parse(C.blob(root, commit, path))
        class Imports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.module in (redirects or {}):
                    node.module = redirects[node.module]; node.level = 1
                return node
        exec(compile(ast.fix_missing_locations(Imports().visit(tree)), commit+':'+path, 'exec'), m.__dict__)
        return m
    rte = module('_signal_rte')
    def rte_attribute(key):
        if key not in ('finite_taylor_operator','finite_rte_distribution'):raise AttributeError(key)
        from .baseline_signal_numerics import kernels
        return kernels(str(root)).finite_rte_distribution.__globals__[key]
    rte.__getattr__ = rte_attribute
    phase = module('_signal_phase')
    @lru_cache(maxsize=None)
    def phase_attribute(key):
        if key!='reconstruct_rpe_phase':raise AttributeError(key)
        import numpy as np
        from .r_price_phase1_science import functions
        scope = dict(np=np, math=math)
        functions(root, B_SOURCE, 'src/trotterlib/rpe_four_round_phase_validation.py',
                  ('_wrap_phase', '_phase_distance', 'reconstruct_rpe_phase'), scope)
        return scope[key]
    phase.__getattr__ = phase_attribute
    load('pr_rpe_rmse', B_SOURCE, {'trotterlib.rpe_four_round_phase_validation':'_signal_phase'})
    load('pr_rpe_rmse_integration',B_SOURCE)
    S = load('rpe_decoder_statistics', B_SOURCE)
    F = load('rpe_decoder_fast', B_SOURCE)
    W = load('rpe_shot_allocation', B_SOURCE)
    load('pr_rpe_signal_plan', A_SOURCE)
    load('pf_power_stability', A_SOURCE)
    # Only the original allowance extractor is needed; don't import the old
    # scientific runner graph solely to obtain its SOURCE constant/blob reader.
    from .r_price_phase1_science import functions
    original = ast.parse(C.blob(root,A_SOURCE,'src/trottertracks/pr_rpe_rmse_s0_v2.py'))
    science = next(ast.literal_eval(n.value) for n in original.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='S0_SCIENCE' for t in n.targets))
    stub = module('h4_pr_m1_rmse'); stub.V = types.SimpleNamespace(S0_SCIENCE=science); stub.blob = C.blob
    allowance = module('h4_four_method_screen_mean')
    allowance.__dict__.update(lru_cache=lru_cache, ast=ast, math=math)
    functions(root,A_SOURCE,'src/trottertracks/h4_four_method_screen_mean.py',('allowance_formula',),allowance.__dict__)
    A = load('pr_rpe_signal_provider', A_SOURCE, {'trotterlib.rte':'_signal_rte'})
    for short in ('efficient_controlled_zz','p1_efficient_zz','compiled_price_fast'):
        load(short,D_ADAPTER_SOURCE)
    Price=load('candidate_price_adapter',D_ADAPTER_SOURCE)
    return types.SimpleNamespace(A=A,B=api.B,C=C,D=api.D,S=S,F=F,W=W,Price=Price)


def plan(root, config, *, target, signal_source, pr_physical=None, references=()):
    """Prepare exactly the supplied V2 configuration, not a search/menu rule."""
    api=providers(str(root)); api.B._validate(config)
    if not signal_source:raise ValueError('Explicit signal numerical SOURCE required')
    s=config['spec']
    if (target.get('quantity')!='ORIGINAL_DF_H_TARGET_ENERGY' or not target.get('source')
            or not C.p.same(target.get('binding'),s['binding']) or target.get('energy_shift_ha')!=0):
        raise ValueError('Original full-H target/binding/zero shift required')
    if s['method']=='PR':
        rows=pr_physical or {}
        pending=api.A.metadata.plan(config,rows,b_validate=api.B._validate,b_exact_scalar=api.B.a.exact_scalar,
            numerical_recipe=api.A.numerical_recipe(signal_source))
        kind='A_PR'; physical=[deepcopy(rows[m]) for m in range(s['M']+1)]
    else:
        if pr_physical is not None:raise ValueError('PR recipe is not a baseline recipe')
        pending=C.prepare(root,config,target=target,references=references,api=api)
        kind='C_BASELINE'; physical=None
    return dict(configuration=deepcopy(config),target=deepcopy(target),signal_source=deepcopy(signal_source),
        provider=kind,pending=pending,pr_physical=physical,science_calls=0,N=None,C=None,G=None)


def _validate(root, request):
    expected=plan(root,request['configuration'],target=request['target'],signal_source=request['signal_source'],
        pr_physical=None if request['pr_physical'] is None else dict(enumerate(request['pr_physical'])),
        references=() if request['provider']=='A_PR' else
            [r['domain']['reference'] for r in request['pending']['requests'] if r['domain']['reference']])
    if not C.p.same(expected,request):raise ValueError('Signal plan/source/physical schedule changed')


def pr_means(request, records):
    """A physical -> B canonical copy, retaining producer failures and unknowns."""
    jobs=request['pending']['jobs']; indexed={r['m']:r for r in records}
    if len(indexed)!=len(records) or not set(indexed)<=set(range(len(jobs))):raise ValueError('Foreign/duplicate PR means')
    means=[]
    for job in jobs:
        mean=dict(m=job['m'],identity=deepcopy(job['B_identity']),law_binding=deepcopy(job['B_law_binding']),
            target=deepcopy(request['target']),source=deepcopy(request['signal_source']),
            raw=None,corrected=None,normalization=None,bias=None,uncertainty=None,qualification='UNQUALIFIED',status='MISSING_MEAN')
        record=indexed.get(job['m'])
        if record:
            if any(not C.p.same(record.get(k),v) for k,v in dict(B_identity=job['B_identity'],
                    B_law_binding=job['B_law_binding'],physical_context=job['physical_context'],
                    candidate=job['physical_context']['physical_candidate'],source=request['signal_source']).items()):
                raise ValueError('PR mean physical/SOURCE/law/identity mismatch')
            mean['status']=record['status']
            if record['status']=='LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED' and record['numerical'].get('gate_pass') is True:
                sig=record['signal']; values=[sig[k][a] for k in ('raw','corrected','reference') for a in ('real','imag')]+[sig['log_B']]
                if not all(type(v) in (int,float) and math.isfinite(v) for v in values) or max(abs(sig['raw'][a]) for a in ('real','imag'))>1:
                    mean['status']='NUMERICAL_RANGE_UNRESOLVED'
                else:
                    mean.update(raw=deepcopy(sig['raw']),corrected=deepcopy(sig['corrected']),reference=deepcopy(sig['reference']),
                        normalization=dict(kind='FINITE_LOG_B_SUPPLIED_BY_A',log_B=sig['log_B'],B=sig['B'],attenuation=sig['attenuation']),
                        bias={k:deepcopy(sig[k]) for k in ('total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias')},
                        uncertainty=dict(numerical=deepcopy(record['numerical']),allowance=deepcopy(record['allowance']),
                            law_distance=None,state_error=None),physical_candidate=deepcopy(record['candidate']))
        means.append(mean)
    return means


def produce_signals(root, request, out, *, context_factory=None, reference_provider=None, saved=()):
    """Explicit finite scientific entry. Artificial tests supply toy factories."""
    _validate(root,request); api=providers(str(root))
    if request['provider']=='A_PR':
        if saved:raise ValueError('A saved evidence needs its published acceptance path; no relabel/retry')
        batch=api.A.SignalBatch(request['configuration'],dict(enumerate(request['pr_physical'])),
            b_validate=api.B._validate,b_exact_scalar=api.B.a.exact_scalar,source=request['signal_source'],root=root,
            context_factory=context_factory,reference_provider=reference_provider)
        produced=batch.run(out); means=pr_means(request,produced['records'])
    else:
        produced=C.execute(root,request['pending'],out,source=request['signal_source'],api=api,
            context_factory=context_factory,reference_reader=reference_provider,saved=saved)
        means=produced['means']
    indexed={r.get('m',r.get('physical_candidate',{}).get('m')):r for r in produced['records']}
    if request['provider']=='C_BASELINE':
        indexed={r['domain']['reference']['m']:r for r in produced['records'] if r['domain']['reference']}
    for mean in means:
        rec=indexed.get(mean['m'])
        if rec is not None:
            mean['provider_binding']=dict(physical_context=price_context(request,mean['m']),
                numerical_source=deepcopy(rec.get('source')),input_source=deepcopy(rec.get('input_source',rec.get('input_provenance'))),
                reference_source=deepcopy(rec.get('reference_source',rec.get('domain',{}).get('reference'))))
    return dict(configuration_id=request['configuration']['id'],target=deepcopy(request['target']),
        source=deepcopy(request['signal_source']),means=means,producer_result=produced)


def signal_input(packet):
    """The actual means/target consumed by a risk producer, including SOURCE."""
    return {k:deepcopy(packet[k]) for k in ('configuration_id','target','source','means')}


def price_context(request,m):
    s=request['configuration']['spec']
    if request['provider']=='A_PR':
        job=request['pending']['jobs'][m];physical=job['physical_context']['physical_candidate'];structure=job['physical_context']['structure']
    else:
        row=request['pending']['requests'][m];physical=row['physical_candidate'];structure=row['structure']
    return dict(binding=deepcopy(s['binding']),profile=deepcopy(s['profile']),physical_candidate=deepcopy(physical),
        structure=deepcopy(structure),split=deepcopy(s['split']),law=deepcopy(s['law']))


def price_requests(root,request,*,input_source=None,execution_source=None,axes=('X','Y'),preparation_hashes=None):
    """D's passive full-wrapper requests; no price/profile values are invented."""
    _validate(root,request);api=providers(str(root))
    if input_source is not None or execution_source is not None:
        from .rpe_d_price_connection import pending_requests
        return pending_requests(root,request,input_source=input_source,execution_source=execution_source,
                                axes=axes,preparation_hashes=preparation_hashes)
    axes=tuple(axes)
    if not axes or len(set(axes))!=len(axes) or any(a not in api.B.p.AXES for a in axes):
        raise ValueError('Unique explicit X/Y price axes required')
    requests=[]
    for row in request['configuration']['rounds']:
        for axis in axes:
            measurement=api.B.p.measurement(row,axis)
            measurement['law_context']['ideal_sampler_distance_qualified']=None
            requests.append(dict(configuration_id=request['configuration']['id'],m=row['m'],axis=axis,
                identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),
                **measurement,physical_context=price_context(request,row['m']),
                quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE',price_status='NOT_EVALUATED',
                profile_extension=None,mean=None,P=0,source=dict(module=MODULE,D_HANDOFF=D_SOURCE)))
    return requests


def compare(root, request, signal_packet, allocations, evaluations, prices, *, statistical_contract,
            cohorts, price_extension=None, price_bindings=(), interval_contract=None,
            price_policy='qualified_expectation',unexplored_scope='OUTSIDE_EXPLICIT_ALLOCATION_MENU'):
    """Existing W.prepare/compare. No diagnostic or provider runs implicitly.

    External risk evidence must name the actual consumed signal input. Missing
    numerical/physical qualification or missing D physical binding stays pending.
    No evidence is upgraded or synthesized here.
    """
    _validate(root,request);api=providers(str(root));config=request['configuration']
    if signal_packet['configuration_id']!=config['id'] or not C.p.same(signal_packet['target'],request['target']):
        raise ValueError('Signal packet configuration/target mismatch')
    if not C.p.same(signal_packet.get('source'),request['signal_source']):raise ValueError('Signal numerical SOURCE mismatch')
    bound=True
    for mean in signal_packet['means']:
        binding=mean.get('provider_binding')
        if mean.get('raw') is not None and binding is None:bound=False
        if binding and (not C.p.same(binding.get('physical_context'),price_context(request,mean['m']))
                or not C.p.same(binding.get('numerical_source'),mean.get('source'))):
            raise ValueError('Signal producer physical context or numerical SOURCE differs')
    entries=api.W.prepare(config,signal_packet['means'],allocations,target=request['target'],statistical_contract=statistical_contract)
    ready=bound and all(e['prepared']['input_complete'] and e['prepared']['physical_qualification_complete'] for e in entries)
    blockers=[] if ready else ['SIGNAL_INPUT_MISSING_OR_PHYSICALLY_UNQUALIFIED']
    accepted=[]
    for evaluation in evaluations:
        binding=evaluation.get('signal_input')
        if binding is None:
            blockers.append('N_SIGNAL_INPUT_BINDING_MISSING:'+evaluation['allocation']['allocation_id']);continue
        if not C.p.same(binding,signal_input(signal_packet)):raise ValueError('N evidence used different signal/SOURCE/target')
        accepted.append(evaluation)
    physical_ready=not any((p.get('mean') or {}).get('rz_count') is not None and 'quantity' not in p for p in prices)
    for binding in price_bindings:
        m=binding.get('identity',{}).get('m')
        if type(m) is not int or m not in range(config['spec']['M']+1):raise ValueError('Foreign price round identity')
        if binding.get('physical_context') is None:
            physical_ready=False
        elif not C.p.same(binding['physical_context'],price_context(request,m)):
            raise ValueError('Price physical candidate/structure/SOURCE/profile differs')
    if not physical_ready:blockers.append('D_PHYSICAL_PRICE_BINDING_MISSING')
    available_records=[p for p in prices if p.get('mean') is not None] if physical_ready else []
    result=api.W.compare(config,allocations,accepted if ready else [],available_records,
        cohorts=cohorts,statistical_contract=statistical_contract,price_extension=price_extension,
        price_bindings=price_bindings,interval_contract=interval_contract,price_policy=price_policy,
        unexplored_scope=unexplored_scope)
    return dict(request=deepcopy(request),signal_input=signal_input(signal_packet),shot_entries=entries,
        supplied_evaluations=deepcopy(evaluations),supplied_prices=deepcopy(prices),
        comparison=result,blockers=sorted(set(blockers)),
        source=dict(module=MODULE,A=A_SOURCE,B=B_SOURCE,C=C_SOURCE,D_HANDOFF=D_SOURCE),scientific_claim_adopted=False)


def select(root, results, *, formula_menu=None, unexplored_scope='OUTSIDE_EXPLICIT_CONFIGURATION_MENU'):
    """Existing B.select over one resolved allocation representative/config."""
    api=providers(str(root));configs=[r['request']['configuration'] for r in results]
    kinds={e['n_packet']['evidence_kind'] for r in results for e in r['supplied_evaluations'] if e.get('n_packet')}
    if len(kinds)>1:raise ValueError('Synthetic and scientific N evidence cannot share a menu')
    representatives=[r['comparison']['outer_evaluation'] for r in results if r['comparison']['outer_evaluation'] is not None]
    return api.B.select(configs,representatives,formula_menu=formula_menu,unexplored_scope=unexplored_scope)
