"""Check the same Hoeffding formula at higher precision; no new signals.

Keep the original H4 batch immutable. Raise only N below the independently
checked ceiling for its stored binary64 epsilon/alpha. This is a numerical
integer check, not rigorous qualification of the supplied signal error disk.
"""
import argparse
from copy import deepcopy
from decimal import Decimal,localcontext,ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import subprocess

RAW_SOURCE='90dcc3ed20f30dfab6b22a68cc4adb9461485a98'


def integer_check(epsilon,alpha,nominal):
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (epsilon,alpha)) or epsilon<=0 or not 0<alpha<1 or type(nominal) is not int or nominal<=0:
        raise ValueError('Finite positive coordinate/alpha and integer N required')
    expected=math.ceil((2./(epsilon*epsilon))*math.log(2./alpha))
    if nominal!=expected:raise ValueError('Stored N differs from original binary64 formula; no silent replacement')
    ceilings=[]
    for precision in (140,180):
        with localcontext() as ctx:
            ctx.prec=precision
            e=Decimal.from_float(epsilon);a=Decimal.from_float(alpha)
            value=Decimal(2)/(e*e)*(Decimal(2)/a).ln()
            ceilings.append(int(value.to_integral_value(rounding=ROUND_CEILING)))
    if ceilings[0]!=ceilings[1]:raise ValueError('Independent integer ceiling precision unresolved')
    required=ceilings[0];selected=max(nominal,required)
    return dict(status='CHECKED_FOR_STORED_BINARY64_EPSILON_ALPHA',epsilon_coordinate=epsilon,alpha_axis=alpha,
        N_raw_decimal=str(nominal),N_high_precision_ceil_decimal=str(required),N_handoff_decimal=str(selected),
        increment_decimal=str(selected-nominal),changed=selected!=nominal,precisions=[140,180],
        no_downward_adjustment=True,signal_disk_rigor_qualified=False)


def checked_record(original,raw_reference,source):
    n=deepcopy(original);lane=n['plan'];checks={}
    if lane['status']!='CONDITIONAL_ROUND_PASS':raise ValueError('No recovery/retry of an unqualified round')
    if n['identity']['m']!=3:
        for axis in ('X','Y'):
            a=lane['axes'][axis]
            if a['shots']!=lane['N_'+axis]:raise ValueError('Stored axis N differs')
            check=integer_check(a['epsilon_coordinate'],a['alpha_axis'],a['shots']);checks[axis]=check
            lane['N_'+axis]=int(check['N_handoff_decimal']);a['shots']=lane['N_'+axis]
        n['source']=dict(commit=source,path='scripts/verify_h4_df10_rpe_shot_integers.py',raw_source=deepcopy(original['source']))
    # m3 is copied verbatim in the N/actual lanes, not reanalyzed.
    n['diagnostics']=dict(raw_record_reference=raw_reference,
        normalization=deepcopy(original['diagnostics'].get('normalization')),integer_checks=checks,
        saved_m3_re_evaluated=False,N_X_decimal=str(lane['N_X']),N_Y_decimal=str(lane['N_Y']))
    return n


def export(result,source,raw_path):
    if (result['scientific_source']!=RAW_SOURCE or result['status']!='H4_N_BATCH_EVALUATED'
            or not result['source_preserved'] or not result['input_preserved']):raise ValueError('Qualified preserved H4 raw batch required')
    if len(result['records'])!=26:raise ValueError('Complete fixed H4 batch required')
    records=[checked_record(n,dict(path=str(Path(raw_path).parent/(n['identity']['candidate']['id']+'.json')),scientific_source=RAW_SOURCE),source) for n in result['records']]
    changed=[dict(m=n['identity']['m'],candidate_id=n['identity']['candidate']['id'],axis=a,**check)
        for n in records for a,check in n['diagnostics']['integer_checks'].items() if check['changed']]
    return dict(status='CONDITIONAL_NUMERICAL_N_WITH_INTEGER_ROUNDING_CHECK',raw_scientific_source=RAW_SOURCE,
        integer_check_source=source,raw_result_sha256=hashlib.sha256(Path(raw_path).read_bytes()).hexdigest(),
        scope=dict(signal_re_evaluations=0,saved_m3_re_evaluations=0,new_candidate_conditions=0,new_integer_axis_checks=48),
        shots=records,changed_axes=changed,actual_N_qualified=False,full_RPE_qualified=False,
        total_resources=None,price_calls=0,G_evaluations=0,mandatory_stop=True,
        precision_note='Python integers retain all digits; decimal strings supplied for parsers using binary64 integers. This does not certify the signal disk or actual implementation/law.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw-result',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve prior handoff; no overwrite')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    relative=str(Path(__file__).resolve().relative_to(root))
    if Path(__file__).read_bytes()!=subprocess.check_output(['git','show',source+':'+relative],cwd=root):raise ValueError('Integer-check SOURCE differs')
    result=json.loads(Path(args.raw_result).read_bytes());value=export(result,source,args.raw_result)
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=value['status'],changed_axes=len(value['changed_axes']),signal_re_evaluations=0)))


if __name__=='__main__':main()
