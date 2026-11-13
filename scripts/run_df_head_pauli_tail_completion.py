"""One approved run of 22 missing Pauli means, with all P1 conditions fixed."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import df_head_pauli_tail_completion as F
A=F.A
CALLER='scripts/run_df_head_pauli_tail_completion.py'


def sources(source):
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()!=source:
        raise ValueError('FIXED_CURRENT_SOURCE_REQUIRED')
    records=[]
    for commit,path in [(source,F.MODULE),(source,CALLER),
        (F.P1,A.MODULE),(F.P1,A.provider.MODULE),
        (F.P1,'src/trottertracks/df_head_pauli_tail_p0.py'),
        (F.P1,'src/trottertracks/pf_power_stability.py'),
        (F.P1,'src/trottertracks/h4_four_method_screen_mean.py'),
        (F.P1,'src/trotterlib/rte.py')]:
        raw=A.p0.blob(ROOT,commit,path)
        if (ROOT/path).read_bytes()!=raw:raise ValueError('USED_SCIENTIFIC_SOURCE_CHANGED: '+path)
        records.append(dict(commit=commit,path=path,sha256=hashlib.sha256(raw).hexdigest()))
    return records


def execute(out,source):
    out=Path(out)
    if out.exists():raise ValueError('FRESH_RUN_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    def save(name,value):
        with (out/name).open('w' if name=='CHECKPOINT.json' else 'x') as f:
            f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    start=time.monotonic();normalizers={}
    result=dict(status='STARTED',scientific_source=source,numerical_provider_source=F.P1,
        P1_result_source=F.P1,P0_source=A.P0,B_source=A.B,
        started_UTC=datetime.now(timezone.utc).isoformat(),records=[],
        scope=dict(new_mean_calls=0,new_complex_outputs=0,new_unique_normalizers=0,reused_P1_means=4,
            P1_mean_recomputations=0,new_full256_means=0,new_references=0,new_DF_means=0,
            operator_dictionary_recomputations=0,eigensolves=0,sampling_compile=0,N_MSE_C_G=0),
        sufficient_N=None,global_RMSE_bound=None,C=None,G=None,winner=None,
        automatic_retry=False,mandatory_stop=True)
    save('EXECUTION.json',result)
    try:
        result['code_sources']=sources(source)
        meta,b,configs,previous,reused=F.load_fixed(ROOT)
        numerical=dict(commit=source,path=A.provider.MODULE,adapter=F.MODULE,
            kernel_source=F.P1,recipe=A.provider.RECIPE)
        jobs={(p,m):A.job(configs[p],m,numerical,b) for p in F.PREFIXES for m in F.MISSING}
        save('PLAN.json',dict(configuration_sources=dict(commit=F.P1,path=F.P1DIR+'MEAN_HANDOFF.json'),
            configuration_ids={str(p):c['id'] for p,c in configs.items()},
            selected_rounds=list(F.MISSING),reused_rounds=list(F.REUSED),
            reused_mean_sources=[dict(prefix=p,m=m,source=s) for (p,m),s in reused.items()],
            jobs=[dict(prefix=p,m=m,candidate_id=j['B_candidate_id'],T=j['physical_context']['physical_candidate']['T'],
                qPF=j['q_PF'],qRPE=j['q_RPE'],R_outer=j['physical_context']['physical_candidate']['R_outer'],
                structure=j['physical_context']['structure']) for (p,m),j in jobs.items()],
            new_mean_limit=22,unique_normalizer_limit=22,source=source))
        contexts=A.load_contexts(ROOT,meta,numerical,normalizers)
        for p in F.PREFIXES:
            # A call-scope limit for this new 22-row batch, not a resource cap
            # or a relaxation of P1's frozen six-mean execution contract.
            contexts[p]['sector36'].normalizer_budget=22
        def evaluate(prefix,m,job):
            print('START LD'+str(prefix)+' m'+str(m),flush=True)
            ref=A.provider.saved_h4_reference(ROOT,job)
            result['scope']['new_mean_calls']+=1
            value=A.provider.evaluate(contexts[prefix]['sector36'],job,ref,root=ROOT)
            if value.get('signal') is not None:result['scope']['new_complex_outputs']+=3
            print(value['status']+' LD'+str(prefix)+' m'+str(m),flush=True)
            return value
        F.run_missing(jobs,evaluate,save,result)
        for spec in [*meta['saved']['inputs'].values(),meta['saved']['context_source']]:
            if A.p0.sha(spec['path'])!=spec['sha256']:raise ValueError('USED_INPUT_CHANGED')
        result['configuration_ids']={str(p):c['id'] for p,c in configs.items()}
        result['reused_P1_sources']=[dict(prefix=p,m=m,source=s) for (p,m),s in reused.items()]
        import numpy,scipy
        result['runtime']=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__)
    except Exception as error:
        result['status']='STOP_INPUT_SOURCE_OR_RUNTIME_FAILURE'
        result['error']=dict(type=type(error).__name__,message=str(error))
    attempted={(r['prefix'],r['m']) for r in result['records']}
    result['not_started']=[dict(prefix=p,m=m) for m in F.MISSING for p in F.PREFIXES if (p,m) not in attempted]
    result['scope']['new_unique_normalizers']=len(normalizers)
    result['normalizer_cache']=[dict(tau_hex=k[0],K=k[1],b=v if math.isfinite(v) else None,
        nonfinite=None if math.isfinite(v) else repr(v)) for k,v in normalizers.items()]
    result['completed_UTC']=datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds']=time.monotonic()-start
    save('result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight',action='store_true');modes.add_argument('--execute',action='store_true')
    parser.add_argument('--source');parser.add_argument('--out');args=parser.parse_args()
    if args.preflight:
        _,_,configs,_,reused=F.load_fixed(ROOT)
        print(json.dumps(dict(status='CANONICAL_METADATA_ONLY_NO_MEAN_OR_ARRAY_LOAD',
            configuration_ids={str(p):c['id'] for p,c in configs.items()},new_rounds=list(F.MISSING),
            reused_rounds=list(F.REUSED),new_mean_calls=0,reused_sources=list(reused.values())),indent=2))
    else:
        if not args.source or not args.out:parser.error('--execute requires --source and --out')
        value=execute(args.out,args.source)
        print(json.dumps(dict(status=value['status'],scope=value['scope'],error=value.get('error'),
            elapsed_seconds=value['elapsed_seconds']),indent=2),flush=True)
        sys.exit(0 if value['status']==F.SUCCESS else 2)
