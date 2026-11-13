"""Approved LD1/LD2 m0/m6 Pauli means, fixed-R and one full/sector check only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import df_head_pauli_tail_p1 as A

CALLER = 'scripts/run_df_head_pauli_tail_p1.py'
SUCCESS = 'P1_LIMITED_PAULI_MEANS_COMPLETE_NOT_RPE_QUALIFIED'


def sources(source):
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip() != source:
        raise ValueError('FIXED_CURRENT_SCIENCE_SOURCE_REQUIRED')
    out = []
    for commit,path in [(source,A.MODULE),(source,CALLER),(source,A.provider.MODULE),
        (A.P0,'src/trottertracks/df_head_pauli_tail_p0.py'),
        (A.P0,'src/trottertracks/pr_rpe_signal_plan.py'),
        (A.P0,'src/trottertracks/pf_power_stability.py'),
        (A.P0,'src/trottertracks/h4_four_method_screen_mean.py'),
        (A.P0,'src/trotterlib/rte.py')]:
        raw=A.p0.blob(ROOT,commit,path)
        if (ROOT/path).read_bytes()!=raw:
            raise ValueError('USED_SOURCE_CHANGED: '+path)
        out.append(dict(commit=commit,path=path,sha256=hashlib.sha256(raw).hexdigest()))
    out.extend([dict(commit=A.B,path='src/trottertracks/pr_rpe_rmse_controller.py'),
        dict(commit='fefd548ecd56f60d72031b3d385f6f5e6ee95d06',
            path='tests/check_pr_rpe_signal_plan.py',function='b_api')])
    return out


def execute(out,source):
    out=Path(out)
    if out.exists():raise ValueError('FRESH_OUTPUT_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    def save(name,value):
        with (out/name).open('w' if name=='CHECKPOINT.json' else 'x') as f:
            f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    start=time.monotonic();normalizers={}
    result=dict(status='STARTED',science_source=source,plan_source=A.p0.PLAN,P0_source=A.P0,
        B_source=A.B,started_UTC=datetime.now(timezone.utc).isoformat(),records=[],comparisons=[],
        saved_DF_acceptances=[],scope=dict(mean_calls=0,complex_outputs=0,unique_normalizers=0,
            saved_DF_rows_accepted=0,new_reference_actions=0,new_DF_means=0,operator_dictionary_recomputations=0,
            eigensolves=0,sampling_build_compile=0,RPE_N_MSE_C_G_evaluations=0),
        sufficient_N=None,global_RMSE_bound=None,C=None,G=None,winner=None,automatic_retry=False,mandatory_stop=True)
    save('EXECUTION.json',result)
    try:
        result['sources']=sources(source);meta=A.metadata(ROOT);b=A.b_api(ROOT)
        numerical=dict(commit=source,path=A.provider.MODULE,adapter=A.MODULE,recipe=A.provider.RECIPE)
        configs={p:A.generate_configuration(b,meta['original_DF_configurations'][p],meta['dictionaries'][p]) for p in (1,2)}
        jobs={(p,m):A.job(configs[p],m,numerical,b) for p in (1,2) for m in (0,6)}
        df_R=meta['original_DF_configurations'][1]['rounds'][0]['R_outer']
        duplicate=jobs[1,0]['physical_context']['physical_candidate']['R_outer']==df_R
        control_config=None if duplicate else A.generate_configuration(b,
            meta['original_DF_configurations'][1],meta['dictionaries'][1],fixed_R=df_R)
        control_job=jobs[1,0] if duplicate else A.job(control_config,0,numerical,b)
        plan=dict(common_kappa_configurations={str(p):c for p,c in configs.items()},
            fixed_R_control_configuration=control_config,fixed_R=df_R,control_deduplicated=duplicate,
            selected_common_kappa_rounds=[0,6],full_sector_control=dict(prefix=1,m=0,
                candidate_id=control_job['B_candidate_id']),maximum_mean_calls=6,maximum_unique_normalizers=5,
            new_reference_actions=0,original_DF_configurations=meta['original_DF_configurations'],
            original_exact_kappa=meta['saved']['original_exact_kappa'],P0_source=A.P0,B_source=A.B)
        save('PLAN.json',plan)
        result['common_kappa_configuration_ids']={str(p):c['id'] for p,c in configs.items()}
        result['fixed_R_control_configuration_id']=None if duplicate else control_config['id']
        result['control_deduplicated']=duplicate
        contexts=A.load_contexts(ROOT,meta,numerical,normalizers)
        accepted={}
        def evaluate(label,p,space,job):
            if result['scope']['mean_calls']>=6:raise ValueError('P1_MEAN_SCOPE_EXHAUSTED')
            print('START '+label,flush=True)
            ref=A.provider.saved_h4_reference(ROOT,job)
            A.provider._reference(job,ref)
            result['scope']['mean_calls']+=1
            record=A.provider.evaluate(contexts[p][space],job,ref,root=ROOT)
            item=dict(label=label,space=space,kind='FIXED_R_CONTROL' if 'fixedR' in label else 'COMMON_KAPPA',
                prefix=p,m=job['m'],record=record)
            result['records'].append(item);save(label+'.json',item)
            if record.get('signal') is not None:result['scope']['complex_outputs']+=3
            save('CHECKPOINT.json',result)
            if record['status']!='LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED':
                raise ValueError('P1_MEAN_NUMERICAL_FAILURE: '+label)
            print('PASS '+label,flush=True)
            return record
        def compare(left,right,kind,strict):
            value=A.compare_means(left,right,kind=kind,strict=strict)
            result['comparisons'].append(value);save('CHECKPOINT.json',result)
            if value['status']=='NUMERICAL_FAILURE':raise ValueError('P1_COMPARISON_GATE: '+kind)
        for p in (1,2):
            for m in (0,6):
                job=jobs[p,m];ref=A.provider.saved_h4_reference(ROOT,job)
                A.provider._reference(job,ref)
                old,old_source=A.accept_saved_DF(ROOT,p,m,meta['original_DF_configurations'][p],ref)
                accepted[p,m]=old
                result['saved_DF_acceptances'].append(dict(prefix=p,m=m,source=old_source,
                    original_DF_candidate_id=old['B_candidate_id'],saved_allowance=old['allowance'],
                    saved_numeric_gates=old['numerical'],saved_DF_signal_unchanged=old['signal']))
                result['scope']['saved_DF_rows_accepted']+=1
                record=evaluate('LD'+str(p)+'_m'+str(m)+'_kappa',p,'sector36',job)
                compare(record,old,'DF_SAME_LAYOUT_LD'+str(p)+'_m'+str(m),False)
                if (p,m)==(1,0):
                    fixed=record if duplicate else evaluate('LD1_m0_fixedR',1,'sector36',control_job)
                    if not duplicate:compare(fixed,old,'DF_FIXED_R_LD1_m0',False)
                    full=evaluate('LD1_m0_fixedR_full256',1,'full256',control_job)
                    compare(fixed,full,'PAULI_FULL256_VS_SECTOR36_LD1_m0_FIXED_R',True)
        result['scope']['unique_normalizers']=len(normalizers)
        if (len(normalizers)>5 or result['scope']['mean_calls']>6
                or result['scope']['complex_outputs']>18 or len(accepted)!=4):
            raise ValueError('P1_SCOPE_OR_COVERAGE')
        for spec in [*meta['saved']['inputs'].values(),meta['saved']['context_source']]:
            if A.p0.sha(spec['path'])!=spec['sha256']:raise ValueError('USED_INPUT_CHANGED')
        import numpy,scipy
        result['runtime']=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__)
        result['status']=SUCCESS
    except Exception as error:
        result['status']='STOP_P1_INPUT_SOURCE_OR_NUMERICAL_FAILURE'
        result['error']=dict(type=type(error).__name__,message=str(error))
    result['scope']['unique_normalizers']=len(normalizers)
    result['normalizer_cache']=[dict(tau_hex=k[0],K=k[1],b=v if math.isfinite(v) else None,
        nonfinite=None if math.isfinite(v) else repr(v)) for k,v in normalizers.items()]
    result['completed_UTC']=datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds']=time.monotonic()-start
    save('result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--source');parser.add_argument('--out');args=parser.parse_args()
    if args.preflight:
        meta=A.metadata(ROOT)
        print(json.dumps(dict(status='SOURCE_METADATA_ONLY_NO_NEW_SCHEDULE_OR_ARRAY_LOAD',
            P0_source=A.P0,B_source=A.B,prefixes=[1,2],rounds=[0,6],mean_limit=6,
            original_exact_kappa=meta['saved']['original_exact_kappa']),indent=2))
    else:
        if not args.source or not args.out:parser.error('--execute requires --source and --out')
        value=execute(args.out,args.source)
        print(json.dumps(dict(status=value['status'],scope=value['scope'],error=value.get('error'),
            elapsed_seconds=value['elapsed_seconds']),indent=2),flush=True)
        sys.exit(0 if value['status']==SUCCESS else 2)
