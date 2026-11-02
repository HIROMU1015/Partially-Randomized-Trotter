"""Parallel cost accounting: process boundaries, not an estimand adoption."""
from . import g6_w2c_b3_offline_result as V
from . import g6_w2c_b3_offline_cost as C


def aggregate(verification):
    result=dict(schema='b3_parallel_cost_verification_v1',status='INSUFFICIENT_EVIDENCE',
                execution_condition='PARALLEL_CONCURRENT_ARTIFICIAL',serial_condition_is_different=True,
                pairs=[],comparison_eligible=verification.get('status')=='VERIFIED_COMPLETE',ranking=None,unknown_cost_is_zero=False,receipt_self_cost_ns=None,
                coarse_fine_added=False,pair_CPU_added_to_worker_CPU=False,IPC_added_to_outer=False,
                parent_worker_wall_summed=False,RSS_sum_is_unique_memory=False)
    report=verification.get('report')
    if report is None:return result
    if verification['status']not in('VERIFIED_COMPLETE','TECHNICAL_FAILURE'):return result
    try:
        host=report.get('host_outer')
        boundary=host if host is not None else report
        parent=V.nonnegative(boundary['parent_cpu_ns']);worker=V.nonnegative(boundary['worker_cpu_ns']);wall=V.nonnegative(boundary['wall_ns'])
        V.require(V.nonnegative(report['total_cpu_ns'])==report['parent_cpu_ns']+report['worker_cpu_ns'],'COORDINATOR_TOTAL_CPU_DOUBLE_COUNT')
        if host is not None:V.require(parent>=report['parent_cpu_ns']and wall>=report['wall_ns'],'HOST_COORDINATOR_BOUNDARY')
        known_pair_cpu=0;unknown_pairs=[]
        observations={p['task_key']:p for p in verification['pairs']}
        for r in report['pairs']:
            t=r['task'];k=t['key']
            if r.get('cpu_ns')is None:
                unknown_pairs.append(k);continue
            total=[V.nonnegative(r['cpu_ns']),V.nonnegative(r['wall_ns'])]
            part=C.partition(r['coarse'],r['reconciliation']);V.require(part['outer_total_ns']==total,'PAIR_OUTER_TOTAL')
            fine=C.partition(r['fine'],r['fine_reconciliation'])if r['fine']is not None else None
            if fine is not None:V.require(fine['outer_total_ns']==total,'FINE_OUTER_SCOPE')
            V.require((t['pass_name']=='instrumented')==(fine is not None),'PASS_OBSERVER_MIXING')
            rpc=r['control_rpc']
            for a,b in zip((rpc.get('cpu_ns',0),rpc.get('wall_ns',0)),total):V.require(0<=a<=b,'IPC_OUTER_SCOPE')
            for field in('memory','memory_baseline'):
                for value in r[field].values():
                    if value is not None:V.nonnegative(value)
            known_pair_cpu+=total[0]
            stages=r['coarse'];V.require(set(stages)<=C.ALLOWED|{'online.generator_and_audit'},'UNKNOWN_COARSE_STAGE')
            row=dict(task_key=k,phase=t['phase'],pass_name=t['pass_name'],arm=t['arm'],layer=t['layer'],
                     worker_pid=r['pid'],outer_cpu_wall_ns=total,coarse=part,fine_alternative=fine,
                     setup_cpu_wall_ns=C.subtotal(stages,C.SETUP),online_cpu_wall_ns=C.subtotal(stages,C.ONLINE|{'online.generator_and_audit'}),
                     generator_audit_cpu_wall_ns=C.subtotal(stages,{x for x in stages if x.endswith('.generator_and_audit')}),
                     angle_requery_cpu_wall_ns=C.subtotal(stages,{'online.EVENT.angle_and_requery'}),
                     evidence_cpu_wall_ns=C.subtotal(stages,{x for x in stages if x.endswith('.evidence')}),
                     output_cpu_wall_ns=C.subtotal(stages,C.OUTPUT),IPC_overlapping_diagnostic=rpc,
                     memory=r['memory'],memory_baseline=r['memory_baseline'],io=r['io'],
                     output_bytes=r['output_bytes'],output_rows=r['output_rows'],
                     logical_completed=observations.get(k,{}).get('global_logical_completed'),
                     failed_prefix=r['status']!='OK',shared_XY_setup_charged_once=True)
            result['pairs'].append(row)
        V.require(known_pair_cpu<=worker,'PAIR_PLUS_WORKER_CPU_DOUBLE_COUNT_OR_BOUNDARY')
        launch=report.get('worker_launch_cost',{})
        if launch.get('cpu_ns')is not None:V.require(0<=launch['cpu_ns']<=parent,'LAUNCH_ALREADY_IN_PARENT')
        final=report.get('finalization')
        if final is not None:
            coarse=final['coarse'];sums=[sum(x[k]for x in coarse.values())for k in('cpu_ns','wall_ns')]
            V.require(all(0<=s<=t for s,t in zip(sums,final['elapsed_ns'])),'FINALIZATION_SCOPE')
        result.update(status='VERIFIED_PARALLEL_COST_BOUNDARIES',measurement_boundary='host including outer STOP'if host else'coordinator snapshot; host outer missing',
                      parent_cpu_ns=parent,all_worker_cpu_ns=worker,total_cpu_ns=parent+worker,
                      global_wall_ns=wall,known_pair_cpu_in_worker_total_ns=known_pair_cpu,
                      worker_unattributed_including_lost_pair_cpu_ns=worker-known_pair_cpu,
                      pair_costs_unknown=unknown_pairs,parent_launch_diagnostic=launch,
                      coordinator_snapshot_cpu_wall=[report['parent_cpu_ns'],report['wall_ns']],
                      shared_finalization_in_parent=final,
                      post_stop_worker_CPU_estimate_ns=report.get('post_stop_worker_cpu_estimate_ns'),
                      post_stop_CPU_resolution_ns=report.get('post_stop_cpu_resolution_ns'),
                      peak_sampled_RSS_shared_page_sum=report['peak_sampled_rss_bytes'],
                      logical_output=verification.get('output_budget'),state_retained_bytes=verification['state_retained_bytes'],
                      state_cumulative_pwrite_traffic_bytes=verification.get('state_pwrite_traffic_bytes'))
    except(V.Invalid,KeyError,ValueError,TypeError)as e:result.update(status='COST_INTEGRITY_MISMATCH',issues=[str(e)])
    return result
