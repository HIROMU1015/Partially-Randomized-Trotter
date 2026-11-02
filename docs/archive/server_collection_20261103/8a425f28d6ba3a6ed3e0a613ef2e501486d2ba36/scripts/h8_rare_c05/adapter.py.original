"""Production trajectory bridge; exercised ONLY with symbolic providers in C0.5."""
from pathlib import Path
from contextlib import nullcontext
from .accounting import check_permit,atomic,IdentityCache,paired_pilot_summary,independent_pilot_moments
from .sampling import CELLS,requests,digest,require,POLICY,count_category,weights,IntegerCategorical,numerical_law_diagnostic
from .parallel import map_fork


class Adapter:
    def __init__(self,provider,permit,contract,budget,cache=None):
        check_permit(permit,provider.mock)
        require(permit.condition==contract['condition'],'PERMIT_CONDITION')
        require(contract['candidate'] in CELLS,'REGISTERED_CANDIDATE')
        require(contract['numeric_policy']==POLICY,'NUMERICAL_POLICY_CHANGED')
        if cache is not None:require(contract.get('cost_cache_enabled') is True and contract.get('live_session_library_verified') is True,'UNVERIFIED_COMPILE_CACHE')
        self.provider=provider;self.permit=permit;self.c=contract;self.budget=budget;self.cache=cache
        prefix,q,r,K=CELLS[contract['candidate']]
        self.prep=provider.context.preparations[prefix]
        require(self.prep.preparation_hash==contract['target_preparation_sha256'],'PREFIX_PREPARATION_HASH')
        require(provider.context.binding==contract['original_binding'],'ORIGINAL_CONTEXT_BINDING')
        self.cell=dict(prefix=prefix,q=q,r=r,K=K)
        self.api=provider.api()
        self.config=None;self.law=None
    def initialize_law(self,root_stream):
        require(self.law is None,'LAW_ALREADY_INITIALIZED')
        self.budget.take('law_formation',1,'law_formation')
        self.config,self.law=self.provider.law(self.prep,self.cell,root_stream)
        require(digest(self.law.to_dict())==self.c['original_law_sha256'],'ORIGINAL_UNCONDITIONAL_LAW_BINDING')
        if self.c.get('law_compatibility_probe') is True:
            self.budget.take('law_compatibility',1,'law_compatibility')
            diagnostic=numerical_law_diagnostic(self.law.order_probabilities,self.c['canonical_order_probability_reference'])
            atomic(self.budget.root/'law_diagnostic.json',diagnostic)
            require(diagnostic['engineering_gate_passed'],'NUMERICAL_LAW_DIFFERENCE_GATE')
        for stratum in self.c['strata']:
            self.budget.take('count_law_setup',1,'count_'+stratum)
            count_category(self.cell['q']*self.cell['r'],tuple(self.law.order_probabilities),stratum)
        self.budget.take('component_table_setup',1,'component_table_setup')
        components=self.prep.rte_preparation.symbolic_tail.components;total=self.api.validate_components(components)
        integer=weights([c.probability/total for c in components])
        require(all(integer),'COMPONENT_SUPPORT_ALREADY_LOST')
        self.api.prepared_components=(components,IntegerCategorical(tuple(enumerate(integer))))
    def trajectory(self,task,directory,axis_width,streams):
        check_permit(self.permit,self.provider.mock)
        require(task['candidate']==self.c['candidate'] and task['phase']==self.c['phase'],'TASK_COHORT')
        require(self.law is not None,'PARENT_LAW_NOT_INITIALIZED')
        root=Path(directory);prefix,q,r,K=CELLS[task['candidate']]
        for kind,n in [('trajectory',1),('occurrence',q),('RTE_event',q*r)]:self.budget.take(kind,n,task['id']+'_'+kind)
        steps,orders=requests(self.prep,task,self.law,self.config,streams,self.api)
        event_values=self.provider.event_values(steps);event_hash=digest(event_values)
        atomic(root/'trajectory.json',dict(task=task,orders=orders,event_digest=event_hash,events=event_values,
               numerical_policy=POLICY,cost_conditioning_only=True,quantum_weights_changed=False))
        self.budget.take('native_evolution',1,task['id']+'_evolution')
        evolution=self.provider.evolve(steps)
        if task['id'] in self.c['validation_task_ids']:
            for kind,n in [('control_probe',4),('event_oracle',1),('validation_wrapper_build',2)]:self.budget.take(kind,n,task['id']+'_'+kind)
            self.provider.validate(evolution,steps,root)
        def axis_unwrapped(axis_task,work,width):
            axis=axis_task['id'];key=task['id']+'_'+axis
            require(digest(self.provider.event_values(steps))==event_hash,'PAIRED_EVENT_MUTATION')
            self.budget.take('wrapper_build',1,key+'_wrapper')
            wrapper=self.provider.wrapper(evolution,axis);fingerprint=self.provider.fingerprint(wrapper)
            identity=dict(self.c['cache_binding'],candidate=task['candidate'],q=q,r=r,K=K,delta=repr(.8/q),
                event_digest=event_hash,axis=axis,measurement=True,fingerprint=fingerprint,numeric_policy=POLICY)
            def compute():
                self.budget.take('wrapper_compile',1,key+'_compile')
                return self.provider.compile(wrapper,fingerprint)
            if self.cache is None:value,hit=compute(),False
            else:value,hit=self.cache.get_or_compute(identity,compute)
            require(self.provider.fingerprint(wrapper)==fingerprint,'WRAPPER_MUTATED_DURING_COMPILE')
            require(digest(self.provider.event_values(steps))==event_hash,'PAIRED_EVENT_CHANGED_AFTER_COMPILE')
            return dict(axis=axis,event_digest=event_hash,fingerprint=fingerprint,compiler=self.c['original_binding']['compiler'],
                        metrics=value,cache_hit=hit)
        def axis_callback(axis_task,work,width):
            # Reuse the original P canonical fingerprint byte cache only inside
            # the disposable axis process. Compile-cache remains disabled.
            context=self.provider.fingerprint_session() if hasattr(self.provider,'fingerprint_session') else nullcontext()
            with context:return axis_unwrapped(axis_task,work,width)
        # Outer demand already reserves this many CPU/memory slots; no extra pool.
        cap=lambda:dict(cpu=axis_width,memory=0,io=axis_width)
        demand=lambda t,s:dict(cpu=1,memory=0,io=1,axis_width=1)
        receipt=map_fork([dict(id=a) for a in ('cosine','sine')],axis_callback,root/'axes',cap,demand,new_group=False,resources_pre_admitted=True)
        require(receipt['status']=='PASS' and receipt['all_reaped'],'AXIS_NATIVE_FAILURE')
        axes={h['id']:h['result'] for h in receipt['workers']}
        require(set(axes)=={'cosine','sine'},'PAIRED_AXES_REQUIRED')
        return dict(task_id=task['id'],candidate=task['candidate'],stratum=task['stratum'],phase=task['phase'],event_digest=event_hash,
                    axes=axes,orders=list(orders),normalization_changed=False,scientific_mock=self.provider.mock)


def run_registered(adapter,tasks,registry,root,capacity,demand,*,observe=lambda rows:None):
    """One fixed condition; grant consumption is a separate, explicit future gate."""
    check_permit(adapter.permit,adapter.provider.mock)
    require(set(registry)=={t['id'] for t in tasks},'FROZEN_TASK_REGISTRY')
    require(len(registry)==len(tasks),'DUPLICATE_TASK_REGISTRY')
    require(all(t['candidate']==adapter.c['candidate'] and t['phase']==adapter.c['phase'] and t['stratum'] in adapter.c['strata'] for t in tasks),'CONDITION_PHASE_STRATUM')
    if not adapter.provider.mock:
        require(digest(dict(tasks=tasks,streams=registry))==adapter.c['private_task_registry_digest'],'REGISTRY_NOT_SEALED')
        for st in adapter.c['strata']:
            selected=[t for t in tasks if t['stratum']==st]
            require(len(selected)==adapter.c['n_by_stratum'][st] and sorted(t['replica'] for t in selected)==list(range(len(selected))),'REGISTERED_SAMPLE_COUNT')
            require(adapter.c['phase']!='variance_load_pilot' or len(selected)>1,'PILOT_VARIANCE_REQUIRES_N_GREATER_THAN_ONE')
    require(set(adapter.c['validation_task_ids'])<={t['id'] for t in tasks},'VALIDATION_TASK_SCOPE')
    def callback(task,directory,width):return adapter.trajectory(task,directory,width,registry[task['id']])
    try:
        receipt=map_fork(tasks,callback,Path(root)/'tasks',capacity,demand,observe=observe)
        if receipt['status']=='PASS':
            n=len(tasks);v=len(adapter.c['validation_task_ids']);q=adapter.cell['q'];r=adapter.cell['r']
            expected=dict(law_formation=1,count_law_setup=len(adapter.c['strata']),component_table_setup=1,
                trajectory=n,occurrence=q*n,RTE_event=q*r*n,native_evolution=n,wrapper_build=2*n,
                wrapper_compile=sum(not item['cache_hit'] for h in receipt['workers'] for item in h['result']['axes'].values()),
                control_probe=4*v,event_oracle=v,validation_wrapper_build=2*v)
            if not adapter.provider.mock:expected.update(saved_preparation_decode=1,prefix_reconstruction=1)
            used=adapter.budget.used()
            require(all(used.get(k,0)==value for k,value in expected.items()),'EXACT_NATIVE_CALL_ACCOUNTING')
            if adapter.c.get('pilot_numeric_accounting') is True:
                adapter.budget.take('pilot_accounting',1,'pilot_accounting')
                values=[h['result'] for h in receipt['workers']]
                summary=paired_pilot_summary(values,adapter.c['shots'],adapter.c['candidate'],adapter.c['phase'])
                atomic(Path(root)/'pilot_summary.json',summary)
                adapter.budget.take('independent_numeric_validation',1,'independent_numeric_validation')
                atomic(Path(root)/'independent_numeric_audit.json',independent_pilot_moments(values,adapter.c['shots'],summary))
        adapter.provider.final_gate()
        atomic(Path(root)/'completion.json',dict(status=receipt['status'],final_byte_gate=True,one_shot=True))
        return receipt
    except BaseException as error:
        atomic(Path(root)/'STOP.json',dict(status='MANDATORY_STOP',reason=str(error),retry=False,resume=False))
        raise
