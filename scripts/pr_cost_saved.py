"""Schema adapters for saved D, paired PR and precision rows; no fitting.

These functions are for the separately authorized scalar pass. They perform no
I/O and never read an NPZ, regenerate signals or infer a missing probability.
"""
from copy import deepcopy
import pr_cost_contract as c


def signal_accounting(records,binding,cohort_by_id):
    """Project existing precision rows. Missing factor10 or mode stays missing."""
    rows=[]
    for record in records:
        cid=record['cell']['id']
        for mode,field in [('USUAL_SIGNED','precision_rows'),('EMPIRICAL_NONCANCELLATION',
                'noncancellation_precision_rows' if 'noncancellation_precision_rows' in record
                else 'empirical_noncancellation_precision_rows')]:
            for e in c.core.EPS:
                for f in (1.,10.):
                    saved=record.get(field,{}).get(str(e),{}).get(str(f))
                    rows.append(dict(cell_id=cid,method=record['cell']['method'],bias_mode=mode,
                        epsilon=e,factor=f,binding=deepcopy(binding),accounting=deepcopy(saved),
                        cohort=cohort_by_id.get(cid),saved_signal_record_sha256=c.digest(record),
                        saved_accounting_field=field,missing_is_not_zero=True))
    return rows


def measured_axes(result,binding,proof):
    """H4 nested measurements / H6 RZ / H8 measured_RZ -> common actual prices."""
    c.require(proof['native_exit_code']==0 and proof['reaped'] is True and proof['final_gate']=='PASS',
              'D_FINAL_NATIVE_ACCEPTANCE')
    c.require(proof['SOURCE'] and proof['result_sha256'] and proof['preparation_identities'],
              'D_SOURCE_RESULT_PREPARATION_JOIN')
    body=result.get('result',result)
    c.require(body['status']=='COMPLETE_MANDATORY_STOP','D_COMPLETED_RESULT')
    groups=body.get('measurements')
    if groups is None:
        cid=body.get('cell_id',body.get('case',{}).get('id'))
        groups=[dict(cell_id=cid,rows=body['rows'])]
    rows=[]
    for group in groups:
        cid=group['cell_id'];seen=set()
        for row in group['rows']:
            axis=row['axis'];c.require(axis in c.core.AXES and axis not in seen,'D_UNIQUE_AXIS');seen.add(axis)
            c.require(row['compiler_hash']==binding['compiler'],'D_COMPILER')
            c.require(row.get('SOURCE',proof['SOURCE'])==proof['SOURCE'],'D_SOURCE_IDENTITY')
            c.require(row.get('profile_sha256',binding['profile'])==binding['profile'],'D_PROFILE_IDENTITY')
            before=row.get('fingerprint_before',row.get('fingerprint'))
            after=row.get('fingerprint_after',before)
            c.require(before and before==after,'D_FINGERPRINT')
            rows.append(dict(kind='DIRECT_MEASURED',cell_id=cid,axis=axis,
                RZ=row.get('measured_RZ',row.get('RZ')),binding=deepcopy(binding),
                SOURCE=row.get('SOURCE',proof['SOURCE']),result_sha256=proof['result_sha256'],
                preparation_identity=proof['preparation_identities'][cid],scalar_policy='original_plus_tail_once',
                native_exit_code=0,reaped=True,final_gate='PASS',fingerprint=before,
                previous_model_check=deepcopy(row.get('model_check'))))
        c.require(seen==set(c.core.AXES),'D_PAIRED_MEASURED_AXES')
    return rows


def paired_metrics(tasks,binding,*,cohort):
    """Normalize saved wrapper metrics without computing a sample statistic."""
    output=[]
    for task in tasks:
        axes={}
        for row in task.get('rows',[]):
            a=row['axis'];c.require(a in c.core.AXES and a not in axes,'UNIQUE_PAIRED_AXIS')
            metrics=row.get('metrics',row)
            c.require(row['compiler_hash']==binding['compiler'],'PAIRED_COMPILER')
            axes[a]=dict(RZ=metrics['RZ'],seed=task['seed'],event_digest=task['event_digest'])
        c.require(set(axes)==set(c.core.AXES),'PAIRED_AXES_REQUIRED')
        output.append(dict(cell_id=task['cell_id'],replica=task['replica'],seed=task['seed'],
            event_digest=task['event_digest'],axes=axes,binding=deepcopy(binding),cohort=cohort))
    return output
