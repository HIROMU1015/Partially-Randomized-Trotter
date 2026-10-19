"""v3 boundary adapter. Does not rank workloads or adopt a scientific estimand."""
from . import g6_w2c_b3_parallel_offline_cost as C
from . import g6_w2c_b3_offline_result as V


def aggregate(verification):
    report=verification.get('report')
    if report is None or report.get('schema')=='b3_parallel_crash_observation_v3':
        return dict(schema='b3_v3_offline_cost_v1',status='INSUFFICIENT_EVIDENCE',
                    parent_cpu_ns=None,all_worker_cpu_ns=None,total_cpu_ns=None,global_wall_ns=None,
                    comparison_eligible=False,unknown_cost_is_zero=False,ranking=None,
                    reason='No complete process boundary observation; unknown is not zero')
    r=C.aggregate(verification);r['schema']='b3_v3_offline_cost_v1'
    try:
        if r['status']!='VERIFIED_PARALLEL_COST_BOUNDARIES':return r
        s=report['settings'];sequential=s['mode']=='S';fused=s['codec']=='fused'
        r.update(variant=('S' if sequential else 'P')+('F' if fused else ''),
                 execution_condition=dict(mode=s['mode'],workers=s['workers'],inflight_cap=s['inflight_cap'],
                    codec=s['codec'],workload=s['workload'],profile_n=s['profile_n'],flush_rows=s['flush_rows']),
                 measurement_condition=report.get('measurement_condition','sequential_pairs' if sequential else 'concurrent_pairs'),
                 measurement_condition_evidence='saved record' if 'measurement_condition' in report else 'fixed SOURCE definition applied to saved settings',
                 observer_cost_ns=None,observer_cost_unknown=True,
                 observer_boundary='fine partition is an alternative; no isolated cached-observer overhead observation',
                 host_io=report.get('host_outer',{}).get('io'),execution_io=report.get('execution_outer',{}).get('io'))
        V.require(r['measurement_condition']==('sequential_pairs' if sequential else 'concurrent_pairs'),'COST_VARIANT_CONDITION')
        outer=report.get('execution_outer');host=report.get('host_outer')
        if outer is not None and host is not None:
            for k in ('parent_cpu_ns','wall_ns','worker_cpu_ns'):
                V.require(V.nonnegative(report[k])<=V.nonnegative(outer[k])<=V.nonnegative(host[k]),'EXECUTION_HOST_BOUNDARY:'+k)
            r['execution_outer']=outer
        r['same_condition_comparison_key']=r['execution_condition']
        r['failed_prefix_excluded_from_comparison']=verification['status']!='VERIFIED_COMPLETE'
        r['comparison_eligible']=verification['status']=='VERIFIED_COMPLETE'
    except (V.Invalid,KeyError,TypeError,ValueError) as e:
        r.update(status='COST_INTEGRITY_MISMATCH',issues=[str(e)],comparison_eligible=False)
    return r
