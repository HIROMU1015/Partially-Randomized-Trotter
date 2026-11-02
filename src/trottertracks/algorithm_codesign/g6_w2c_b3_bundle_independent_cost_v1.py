"""Independent descriptive bundle cost boundaries; no science/CPU CI inference."""
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_offline_cost as C
from .g6_w2c_b3_bundle_independent_identity_v1 import equal

def aggregate(r,delivery,observation,pair_results,*,comparison_eligible=False):
    host=r['host_outer'];parent=V.nonnegative(host['parent_cpu_ns']);worker=V.nonnegative(host['worker_cpu_ns']);wall=V.nonnegative(host['wall_ns'])
    for k in ('parent_cpu_ns','worker_cpu_ns','wall_ns'):V.nonnegative(r[k])
    equal(r['total_cpu_ns'],r['parent_cpu_ns']+r['worker_cpu_ns'],'CPU_DOUBLE_ADD')
    V.require(parent>=r['parent_cpu_ns']and worker>=r['worker_cpu_ns']and wall>=r['wall_ns'],'HOST_OUTER_BOUNDARY')
    pairs=[];known=0;unknown=[];observed={p['task_key']:p for p in pair_results}
    for row in r['pairs']:
        key=row['task']['key']
        if row['cpu_ns']is None:unknown.append(key);continue
        total=[V.nonnegative(row['cpu_ns']),V.nonnegative(row['wall_ns'])]
        main=C.partition(row['coarse'],row['reconciliation']);equal(main['outer_total_ns'],total,'PAIR_COARSE_OUTER')
        fine=None
        if row['fine']is not None:fine=C.partition(row['fine'],row['fine_reconciliation']);equal(fine['outer_total_ns'],total,'PAIR_FINE_OUTER')
        equal(row['task']['pass_name']=='instrumented',fine is not None,'OBSERVER_PASS')
        allowed=C.ALLOWED|({'online.generator_and_audit'}if row['status']!='OK'else set())
        V.require(set(row['coarse'])<=allowed,'UNSUPPORTED_COARSE_STAGE')
        for k in ('cpu_ns','wall_ns'):V.require(0<=V.nonnegative(row['control_rpc'].get(k,0))<=row[k],'IPC_CONTAINED')
        if row['status']=='OK'and key in observed:
            pc=observed[key]
            for kind,count in pc['counts'].items():
                if count:
                    stage=row['coarse'].get('online.'+kind+'.generator_and_audit')
                    V.require(stage is not None and stage['calls']==count,'RAW_GENERATOR_COST_COUNT')
                    equal([stage['cpu_ns'],stage['wall_ns']],pc['generator_exclusive_ns_by_kind'][kind],'RAW_GENERATOR_COST_BINDING')
        known+=row['cpu_ns']
        pairs.append(dict(key=key,condition=row['task']['condition'],total_cpu_wall_ns=total,coarse=main,fine_alternative=fine,
                          setup=C.subtotal(row['coarse'],C.SETUP),online=C.subtotal(row['coarse'],C.ONLINE),
                          generator_and_audit=C.subtotal(row['coarse'],{s for s in C.ONLINE if s.endswith('generator_and_audit')}),
                          angle_and_requery=C.subtotal(row['coarse'],{'online.EVENT.angle_and_requery'}),
                          evidence=C.subtotal(row['coarse'],{s for s in C.ONLINE if s.endswith('.evidence')}),
                          serialize_hash_write_flush_finalize=C.subtotal(row['coarse'],C.OUTPUT),IPC_contained=row['control_rpc'],
                          memory=row['memory'],memory_baseline=row['memory_baseline'],io=row['io']))
    V.require(known<=worker,'PAIR_CPU_EXCEEDS_WORKER_OUTER')
    cond=r['conditions'];cp=sum(V.nonnegative(x['parent_cpu_ns'])for x in cond);cw=sum(V.nonnegative(x['worker_cpu_ns'])for x in cond)
    equal(cp,r['condition_parent_cpu_ns'],'CONDITION_PARENT_CPU_SUM');equal(cw,r['condition_worker_cpu_ns'],'CONDITION_WORKER_CPU_SUM')
    V.require(cp<=parent and cw<=worker and sum(V.nonnegative(x['wall_ns'])for x in cond)<=wall,'CONDITION_SHARED_COST_SCOPE')
    if delivery:
        V.require(delivery['outer_parent_cpu_ns']>=parent and delivery['outer_wall_ns']>=wall,'FINAL_PERSISTENCE_BOUNDARY')
        equal(delivery['worker_cpu_ns'],worker,'DELIVERY_WORKER_CPU')
        C.vector(delivery['final_persistence_cpu_wall_ns'])
    outer=host
    if observation:
        outer=observation['host_outer']
        V.require(outer['parent_cpu_ns']>=parent and outer['wall_ns']>=wall,'CALLER_OUTER_BOUNDARY')
        equal(outer['worker_cpu_ns'],worker,'CALLER_WORKER_CPU')
        if delivery:V.require(outer['parent_cpu_ns']>=delivery['outer_parent_cpu_ns']and outer['wall_ns']>=delivery['outer_wall_ns'],'CALLER_AFTER_FINAL_BOUNDARY')
    return dict(status='VERIFIED_ARTIFICIAL_COST_BOUNDARIES',parent_cpu_ns=outer['parent_cpu_ns'],all_worker_cpu_ns=worker,
                total_cpu_ns=outer['parent_cpu_ns']+worker,global_wall_ns=outer['wall_ns'],condition_costs=cond,
                shared_parent_unallocated_ns=outer['parent_cpu_ns']-cp,shared_worker_unallocated_ns=worker-cw,
                pairs=pairs,unknown_pair_costs=unknown,unknown_cost_is_zero=False,host_io=host['io'],
                global_peak_sampled_rss_bytes=r['peak_sampled_rss_bytes'],RSS_sum_is_unique_physical_memory=False,
                comparison_eligible=comparison_eligible,failed_prefix_in_normal_comparison=False,
                pair_IPC_observer_added_to_total=False,coarse_and_fine_added=False,parent_child_wall_added=False,
                isolated_observer_overhead_ns=None,parent_exclusive_individual_IO_ns=None,
                caller_observation_own_persistence_cost_ns=None,CPU_mean_CI=None,quantum_confidence_transferred=False,
                both_fused_gain_unidentifiable=True,fixed_S_then_P1_order_confound=True,scientific_extrapolation=False)
