"""Fixed H4 one-preparation/two-law diagnostic, not a price evaluator.

The scientific leaf restores the existing descriptor and inspects native local
words. It never samples an event, invokes the full structural word/finish, or
evaluates a mean. Failure of a sufficient reset criterion means UNKNOWN.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from .r_price_phase1 import blob, digest

BASE='d458b5e4f596d628d923263f3b2f1678960f2ffa'
B_PUBLICATION='75b5c89bddf8a1ffa4a8ac99b2c06d6e05fa012d'
B_PATH='docs/research/pr_rpe_s0_v2_connection_20261113/metadata01/V2_CONFIGURATION.json'
REQUEST_PATH='docs/research/h4_s0_mean_price_plan_20261113/PRICE_REQUESTS.json'
SOURCE_FILES=('src/trottertracks/h4_pr_physical_closure.py','scripts/run_h4_pr_physical_closure.py',
              'src/trottertracks/s0a_structure_context.py','src/trottertracks/h4_df10_price_v1.py',
              'src/trottertracks/track_d_native_resources.py')


def read(root,commit,path):return json.loads(blob(root,commit,path))


def check_publication(root):
    plan=read(root,BASE,REQUEST_PATH);official=read(root,B_PUBLICATION,B_PATH)
    if official['id']!=plan['B_V2_configuration_id']:raise ValueError('B_CONFIGURATION_CHANGED')
    rows={r['m']:r for r in official['rounds']}
    for request in plan['focus_requests']:
        row=rows[request['m']]
        for key in ('identity','law_binding','physical_candidate'):
            if row[key]!=request[key]:raise ValueError('B_PHYSICAL_LAW_BINDING_CHANGED:'+key)
    return plan,dict(commit=B_PUBLICATION,path=B_PATH,configuration_id=official['id'],
                     status='EXACT_PUBLISHED_B_METADATA_MATCH')


def tracer(Accounting):
    """Reuse only original native expansion, without counting a full S word."""
    class Trace(Accounting):
        def __init__(self):
            super().__init__();self.word=[]
        def _native(self,name,wires,angle=None,role='other'):
            self.word.append(dict(name=name,wires=list(wires),angle=None if angle is None else float(angle)))
        def finish(self):raise RuntimeError('FULL_PATH_S_FINISH_FORBIDDEN')
    return Trace


def wire_effect(word,q):
    """Exact sequential suffix angle and reset/leading dependencies on a wire.

    If reset is true, the exit angle does not depend on the incoming pending Z.
    If false and z_only is true, this test does not prove a finite closure.
    """
    resets=0;prefix=[];suffix=[]
    for op in word:
        if q not in op['wires']:continue
        if op['name']=='rz':
            if not resets:prefix.append(op['angle'])
            suffix.append(op['angle'])
        else:resets+=1;suffix=[]
    value=None
    for angle in suffix:value=angle if value is None else value+angle
    return dict(reset=bool(resets),reset_count=resets,leading_z=prefix,
                exit_z=value,exit_z_hex=None if value is None else value.hex(),
                z_only=not resets and bool(suffix),untouched=not resets and not suffix)


def effects(word):return [wire_effect(word,q) for q in range(9)]


def basis_word(Trace,operations,inverse):
    t=Trace();t.basis(operations,inverse,'original_basis');t.flush_basis();return t.word


def application_word(Trace,support,role,unsigned=0.):
    t=Trace()
    if role=='product':
        for q in support:t.gate('cz',(8,q),role='tail_product')
    elif role=='rotation':t.gate('crz' if len(support)==1 else 'crzz',(8,*support),(2*unsigned,),'tail_rotation')
    else:raise ValueError('LEGAL_NONIDENTITY_ROLE_REQUIRED')
    return t.word


def diagnostic(context,plan,B_match):
    """Scientific metadata/alphabet analysis; one restored H4 context only."""
    prep=context.ctx.prep;tail=prep.rte_preparation.symbolic_tail
    if prep.num_system_qubits!=8 or prep.ld!=1 or prep.identity_policy!='extract_identity_phase':
        raise ValueError('FIXED_H4_PREPARATION')
    components=tail.components
    if not components or any(c.is_identity or not c.diagonal_pauli_support or len(c.diagonal_pauli_support)>2 for c in components):
        raise ValueError('LEGAL_EXTRACTED_NONIDENTITY_K2_SUPPORT')
    Trace=tracer(context.Accounting);registry={};component_rows=[]
    for c in components:
        # Exactly the representation in original RTEEventApplication.to_dict.
        ops=[dict(name=o.name,qubits=list(o.qubits),parameters=list(o.parameters)) for o in c.basis_change_operations]
        key=(c.basis_id,c.basis_hash)
        if key in registry and registry[key]!=ops:raise ValueError('BASIS_KEY_COLLISION')
        registry[key]=ops
        component_rows.append(dict(id=c.component_id,basis_id=c.basis_id,basis_hash=c.basis_hash,
            fragment=c.df_fragment_id,support=list(c.diagonal_pauli_support),sign=c.coefficient_sign,
            coefficient_abs=c.coefficient_abs,probability=c.probability,probability_hex=c.probability.hex()))
    basis=[]
    for (identity,sha),ops in registry.items():
        basis.append(dict(id=identity,hash=sha,operations=ops,
            forward_effects=effects(basis_word(Trace,ops,False)),inverse_effects=effects(basis_word(Trace,ops,True))))
    laws=[];letters=[]
    supports=sorted({tuple(c.diagonal_pauli_support) for c in components})
    for request in plan['focus_requests']:
        c=request['physical_candidate'];R=c['R_outer'];time=c['T']/c['q']
        tau=tail.lambda_r*(float(time)/R)
        config,distribution=context.rte.make_rte_config(tail,evolution_time=time,rte_steps=R,
            truncation_tolerance=context.tolerance(tau,2),finite_taylor_order=2,seed=None)
        if config.dimensionless_step_time!=tau or tuple(distribution.orders)!=(0,2):raise ValueError('FIXED_K2_LAW')
        laws.append(dict(m=request['m'],candidate_id=c['id'],q_PF=c['q'],R_outer=R,
            event_count=request['identity']['structure']['event_count'],time_per_outer=time,
            tau=tau,tau_hex=tau.hex(),orders=list(distribution.orders),
            order_weights=list(distribution.unnormalized_order_weights),
            order_probabilities=list(distribution.order_probabilities),
            event_normalization=distribution.exact_finite_distribution,
            rotation_angles={str(k):math.atan(tau/(k+1)) for k in distribution.orders},
            event_support_grammar='Choose rotation and k ordered products independently from the same component registry; apply products then rotation',
            event_probability_rule='weight[k]/Z * product(p_i / math.fsum(component_p))',
            runtime_sampler='PCG64_numpy_float64_CDF_saved_cohort',sampler_ideal_distance=None,
            phase_increments='order2 adds pi; every negative product adds pi; no identity rotation term',
            exact_phase_counter_max=3*R,phase_zero_iff_no_positive_increment=True))
        for support in supports:
            for order in (0,2):
                for sign in (-1,1):
                    word=application_word(Trace,support,'rotation',sign*math.atan(tau/(order+1)))
                    letters.append(dict(kind='rotation',R=R,order=order,sign=sign,support=list(support),effects=effects(word)))
    for support in supports:
        letters.append(dict(kind='product',support=list(support),effects=effects(application_word(Trace,support,'product'))))
    head=[]
    for j,blocks in enumerate(context.blocks):
        if j!=0:raise ValueError('S2_ONE_PIECE_ONLY')
        for i,b in enumerate(blocks):
            for directional in (False,True):
                t=Trace();t.diagonal(b['diagonal'],directional,'original_head')
                head.append(dict(block=i,directional=directional,basis=deepcopy(b['basis']),effects=effects(t.word)))
    failures=[]
    for b in basis:
        for direction in ('forward','inverse'):
            for q,effect in enumerate(b[direction+'_effects'][:8]):
                if effect['z_only']:failures.append(dict(basis=b['id'],direction=direction,wire=q,leading_z=effect['leading_z']))
    app_z_only=[dict(kind=a['kind'],wire=q) for a in letters for q,e in enumerate(a['effects']) if e['z_only']]
    control_rotation_reset=all(a['effects'][8]['reset'] and a['effects'][8]['exit_z'] is None
                               for a in letters if a['kind']=='rotation')
    # Check whether phase reaches a control reset without a preceding control RZ
    # in the first reverse-directional head diagonal. Basis ops use system wires.
    first=next(h for h in head if h['directional'] and h['block']==len(context.blocks[0])-1)['effects'][8]
    phase_barrier=control_rotation_reset and first['reset'] and not first['leading_z']
    return dict(status='LIMITED_PHYSICAL_STATE_DIAGNOSTIC_COMPLETE',B_match=B_match,
        preparation=dict(hash=prep.preparation_hash,H=prep.hamiltonian_hash,partition=prep.partition_hash,
            identity_policy=prep.identity_policy,lambda_R=tail.lambda_r,tail_id=tail.tail_id,tail_hash=tail.tail_hash,
            component_count=len(components),basis_count=len(basis),components=component_rows,basis_registry=basis),
        laws=laws,local_application_effects=letters,head_diagonal_effects=head,
        basis_reset_sufficient_criterion=dict(pass_for_all_system_wires=not failures,z_only_exceptions=failures,
            interpretation='Sufficient local reset check only; failure is not a reachable-history counterexample'),
        application_no_unreset_Z=not app_z_only,application_z_only_exceptions=app_z_only,
        phase_control_barrier=dict(last_rotation_resets_control=control_rotation_reset,
            reverse_head_resets_before_control_Z=phase_barrier,
            phase_zero_positive_quotient_conditionally_valid=phase_barrier,
            applicability='Legal nonidentity support; a legal tail is not split; bounded R2/R128 finite sums; full state quotient not inferred'),
        scalar=dict(value=context.scalar,hex=float(context.scalar).hex(),policy='-scalar*T once after q outers'),
        state_equivalence=dict(physical_small_closed_state=None,status='UNKNOWN',
            all_legal_continuation_reward_equivalence_proved=False,
            reachable_state_graph_constructed=False,
            reason='Local alphabet/reset information does not establish a tractable quotient for pending Z plus delayed basis and all head/outer/terminal continuations'),
        new_scientific_scope=dict(preparation_descriptors=1,law_descriptors=2,local_native_word_inspection=True,
            trajectory_sampling=0,full_path_S=0,mean_S=0,full_circuit_build_compile=0,fit=0,mean_compiled_C=0,N_G=0),
        mean_structural_price=None,mean_compiled_price=None,qualified_for_candidate=False,full_rpe_resources=None,
        mandatory_stop=True)


def execute(root,source):
    root=Path(root)
    for path in SOURCE_FILES:
        if (root/path).read_bytes()!=blob(root,source,path):raise ValueError('EXECUTION_SOURCE_CHANGED:'+path)
    plan,B_match=check_publication(root)
    from .s0a_saved_structure import saved
    from .s0a_structure_context import Context,no_circuits
    # Reuse C1 saved input/preparation restoration; no S0a paths are read.
    with no_circuits():
        context=Context(root,saved(root,'PLAN.json'),'H4')
        result=diagnostic(context,plan,B_match)
    result['execution_source']=source;result['baseline']=BASE
    return result
