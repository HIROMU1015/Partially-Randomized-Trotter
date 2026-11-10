"""A RBatch + fixed C catalogs for B finite-R menus. No science on import.

prepare_menu is passive. RMenuBatch/run_missing are explicit future numerical
entries; current integration tests supply mock context/reference/action only.
No fixed-R N or C is relabeled for a different R.
"""
from copy import deepcopy

from . import pf_pr_rpe_pipeline as p
from . import pr_r_dependent_n as nmodel
from . import rpe_r_selection as r
from . import rpe_nc_coverage as saved

A_RELEASE='5ca2dce3234f3e4a971f4a8c852da676f050e315'
C_RELEASE='4cb61ef4633467446122954ed6cc43462e0306a9'
N_POLICY='R_AWARE_EMPIRICAL_ACTION_COUNT_V1'
LAW=dict(model='ideal_finite_RTE_law',sampling='IDEAL_FINITE_RTE_MEAN',
    source=dict(commit=A_RELEASE,path='src/trottertracks/pr_r_dependent_n.py'))


def _law(menu):
    if not p.same(menu.get('sampling_law'),LAW):raise ValueError('A finite-RTE implementation law binding required')


def bound_context_factory(root, system):
    """Bind existing checked H4/H6 loaders; loading occurs only on invocation."""
    if system not in ('H4','H6'):raise ValueError('H4/H6 context required')
    def load(task):
        base=nmodel.saved_task(root,system);nmodel.validate_task(task,base)
        if system=='H4':
            from . import h4_df10_rpe_batch_provider as loader
            packet=loader.prepare(root)
            if not p.same(packet['task']['binding'],task['binding']):raise ValueError('H4 loader binding differs')
            ctx,metadata=loader.load_saved_context(root,packet)
            state=dict(state_energy=metadata['state_energy'],state_residual=metadata['state_residual'])
        else:
            from . import h6_rpe_batch_provider as loader
            reuse=p.SavedArtifacts(root).read(dict(commit=saved.H6_RELEASE,
                path='docs/research/h6_df19_rpe_m7_preparation_20261110/PLAN.json'))
            if (not p.same(task['binding'],reuse['saved_tail_context']['binding'])
                    or not p.same(task['configuration'],reuse['configuration_reuse'])):
                raise ValueError('H6 loader binding/policy differs')
            ctx,metadata=loader.load_saved_context(root,reuse)
            state=dict(state_energy=metadata['solver_energy'],state_residual=metadata['solver_residual'])
        # Both existing loaders check input bytes, H/state and layout before
        # returning. This assignment declares their checked binding to A.
        ctx.scientific_binding=deepcopy(task['binding'])
        return ctx,state
    return load


def bind_saved_N(menu, records):
    """Copy exact identities and integer handoff values; exclude other R IDs."""
    _law(menu);rows={p.key(row):row for row in menu['candidates']}
    reuse,excluded=nmodel.partition_saved(menu,records)
    for n in reuse:
        row=rows[p.key(n['identity'])]
        n['law_binding']=r.request_context(menu,row)['law_binding']
        n['N_comparison_policy']=n.get('diagnostics',{}).get('allowance',{}).get('policy','UNSPECIFIED_SAVED_ALLOWANCE')
    return reuse,excluded


class FixedSavedPrices:
    """Only source-bound H4 R4 / H6 R16 catalogs; missing stays unavailable."""
    def __init__(self, read_json, system, *, selections=None):
        if system not in ('H4','H6'):raise ValueError('H4/H6 only')
        self.system=system
        factory=p.h4_saved_price_provider if system=='H4' else p.h6_saved_price_provider
        self.provider=factory(read_json,selections=selections)

    def bind(self, menu, record):
        _law(menu)
        if menu['binding']['system']!=self.system:raise ValueError('Foreign fixed price system')
        row=next(v for v in menu['candidates'] if p.key(v)==p.key(record['identity']))
        p._match(menu,row,record);value=deepcopy(record)
        value['law_binding']=r.request_context(menu,row)['law_binding']
        quote=value.get('provider_quote') or {}
        if value.get('model') and quote.get('value') is not None and quote.get('validation')!='OUTSIDE_DOMAIN':
            # This is the fixed provider's already checked candidate, not a new
            # R qualification. The original qualified flag remains unchanged.
            candidate=quote['conditions']['candidate']
            _,reason=self.provider._domain(quote['conditions'])
            if reason is not None:raise ValueError('Model domain differs from fixed provider')
            if not p.same(candidate,row['candidate']):raise ValueError('Price quote candidate mismatch')
            value['model']['domain']=deepcopy(candidate)
        return value

    def __call__(self, menu, rows):
        _law(menu)
        return [self.bind(menu,v) for v in self.provider(menu,rows)]


def prepare_menu(root, system, R_by_round, *, inputs=None):
    """Immutable N_HANDOFF/C scalar selection only; no aggregate or producers."""
    data=(saved.load_inputs(root,p.SavedArtifacts(root)) if inputs is None else inputs)[system]
    base=nmodel.saved_task(root,system)
    menu=r.build_menu(base,R_by_round,sampling_law=LAW)
    menu['N_comparison_policy']=N_POLICY
    shots,excluded=bind_saved_N(menu,data['shots'])
    catalog=FixedSavedPrices(p.SavedArtifacts(root).read,system)
    records,paths=saved.saved_catalog_records(catalog.provider,menu)
    lookup={p.key(row):row for row in menu['candidates']}
    legacy=[v for v in data['legacy_prices'] if p.key(v['identity']) in lookup]
    prices=[catalog.bind(menu,v) for v in legacy+records]
    p.plan(menu,shots,prices)
    return dict(base=base,menu=menu,shots=shots,prices=prices,excluded_old_R_N=excluded,
        path_sources=paths,N_handoff_source=deepcopy(data['N_handoff_source']),
        N_precision_note=data['N_precision_note'],total_resources=None,scientific_R_adopted=False)


class RMenuBatch:
    """Translate multiple alternatives into explicit one-R-per-round A batches.

    Shared context and bound reference cache last for this call only. Each
    candidate action is still distinct. A failures and STOPs remain visible.
    """
    def __init__(self, menu, base, source, context_factory, reference_factory=None):
        _law(menu);p.validate_task(menu)
        if not source:raise ValueError('Explicit numerical SOURCE required')
        self.menu=deepcopy(menu);self.base=deepcopy(base);self.source=deepcopy(source)
        self.context_factory=context_factory;self.reference_factory=reference_factory;self.last_run=None
        # A validates the pinned base and recipe at construction, not after
        # expensive context work. Every alternative must match its A recipe.
        for row in menu['candidates']:
            if p.arm(row['candidate'])=='PR' and row['m'] in menu['variable_R_rounds']:
                task=nmodel.with_round_R(base,{row['m']:row['candidate']['R_outer']},source)['task']
                target=next(v for v in task['candidates'] if p.key(v)==p.key(row))
                if not p.same(p.identity(task,target),p.identity(menu,row)):raise ValueError('A/B candidate identity differs')

    def __call__(self, menu, rows):
        if not p.same(menu,self.menu) or not rows or len({p.key(v) for v in rows})!=len(rows):raise ValueError('Explicit unchanged R menu required')
        if any(v not in menu['candidates'] or p.arm(v['candidate'])!='PR' or v['m'] not in menu['variable_R_rounds'] for v in rows):
            raise ValueError('Supply saved PF/early/fixed N; R provider accepts declared PR alternatives only')
        groups=[]
        for row in rows:
            group=next((v for v in groups if all(old['m']!=row['m'] for old in v)),None)
            if group is None:group=[];groups.append(group)
            group.append(row)
        context=[];refs={};records=[]
        self.last_run=dict(status='RUNNING',source=deepcopy(self.source),groups=[],records=records,automatic_retry=False,price_calls=0,G_calls=0)
        def context_factory(task):
            if not context:context.append(self.context_factory(task))
            return context[0]
        def reference_factory(ctx,T):
            if T not in refs:
                if self.reference_factory is not None:refs[T]=deepcopy(self.reference_factory(ctx,T))
                else:
                    from . import h6_df19_rpe_d2 as backend
                    refs[T]=dict(binding=deepcopy(menu['binding']),T=T,
                        source=dict(kind='NEW_TWO_PATH_REFERENCE',source=deepcopy(self.source)),evidence=backend.reference_pair(ctx,T))
            return deepcopy(refs[T])
        stopped=False
        try:
            for group in groups:
                task=nmodel.with_round_R(self.base,{v['m']:v['candidate']['R_outer'] for v in group},self.source)['task']
                selected=[next(v for v in task['candidates'] if p.key(v)==p.key(row)) for row in group]
                if stopped:
                    result=[nmodel.failure(task,row,self.source,'NOT_EXECUTED_AFTER_N_FAILURE','NOT_EXECUTED_AFTER_N_FAILURE') for row in selected]
                else:
                    provider=nmodel.RBatch(task,self.base,self.source,context_factory,reference_factory)
                    try:result=provider(task,selected)
                    finally:self.last_run['groups'].append(deepcopy(provider.last_run))
                    if len(result)!=len(selected) or {p.key(v['identity']) for v in result}!={p.key(v) for v in selected}:
                        raise ValueError('A N batch returned different candidate scope')
                    stopped=any(v['plan']['status']!='CONDITIONAL_ROUND_PASS' for v in result)
                for value,row in zip(result,group):
                    p._match(menu,row,value)
                    value['law_binding']=r.request_context(menu,row)['law_binding']
                    value['N_comparison_policy']=N_POLICY
                    records.append(value)
        except Exception as exc:
            self.last_run.update(status='STOP_N_PROVIDER_EXCEPTION',reason=str(exc));raise
        self.last_run.update(status='STOP_UNQUALIFIED_N' if stopped else 'BATCH_EVALUATED',reference_times=sorted(refs))
        p.plan(menu,records,[])
        return records


def run_missing(menu, shots, prices, *, n_provider=None, c_provider=None,
                cohort_selection=None, price_policy='qualified_expectation'):
    """Explicit batch entry, no N*C or R selection. Preserve returned cache."""
    _law(menu)
    result=p.run_missing(menu,shots,prices,n_provider=n_provider,c_provider=c_provider,price_policy=price_policy)
    if 'shots' in result:
        result['R_coverage']=r.coverage(menu,result['shots'],result['prices'],
            cohort_selection=cohort_selection or {},price_policy=price_policy)
    return result


def run_evaluation(menu, shots, prices, *, cohort_selection, n_provider=None, c_provider=None,
                   price_policy='qualified_expectation', simultaneous_bounds=None, schedule_success=None):
    """Future permitted N/C batch -> exact costs/R selection. Mock-only here."""
    batch=run_missing(menu,shots,prices,n_provider=n_provider,c_provider=c_provider,
        cohort_selection=cohort_selection,price_policy=price_policy)
    if batch['status']!='BATCH_PREPARED_NO_TOTALS':return dict(status=batch['status'],totals=None,batch=batch)
    result=r.evaluate_menu(menu,batch['shots'],batch['prices'],cohort_selection=cohort_selection,
        price_policy=price_policy,simultaneous_bounds=simultaneous_bounds,schedule_success=schedule_success)
    return dict(result,batch=batch)
