"""Approved fixed LD1/LD2 signal scope; no N/MC/C/G, retries or fallback.

Numerical SOURCE remains fefd548e. This caller only loads the pinned menu,
delegates B metadata validation, and invokes the existing sequential batches.
"""
import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
PLAN='a9abf8de75bf3432fcd714e868e5c14c8884b8f8'
NUMERICAL='fefd548ecd56f60d72031b3d385f6f5e6ee95d06'
B_SOURCE='437ba4acdb82708b6fc98e3cd119a2a734f7cf64'
DIRECTORY='docs/research/pr_rpe_first_signal_menu_20261113/'
CALLER='scripts/run_pr_rpe_first_signal.py'
COMPLETE='FINITE_MEAN_SCHEDULE_COMPLETE_NOT_RMSE_QUALIFIED'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def load_metadata():
    from trottertracks import pr_rpe_rmse_s0_v2 as fixed
    from trottertracks import pr_rpe_signal_provider as provider
    changed=git('diff','--name-only',NUMERICAL,'--','src','scripts').decode().splitlines()
    if set(changed)-{CALLER}:raise ValueError('Numerical SOURCE differs: '+str(changed))
    if Path(__file__).read_bytes()!=git('show','HEAD:'+CALLER):
        raise ValueError('Commit caller before execution')
    store=fixed.p.SavedArtifacts(ROOT)
    menu=store.read(dict(commit=PLAN,path=DIRECTORY+'MENU.json'))
    if menu['signal_source']['commit']!=NUMERICAL or menu['B_source']!=B_SOURCE:
        raise ValueError('Fixed menu numerical/B SOURCE mismatch')
    node=next(n for n in ast.parse(git('show',NUMERICAL+':src/trottertracks/pr_rpe_rmse_s0_v2.py')).body
        if isinstance(n,ast.FunctionDef) and n.name=='b_metadata')
    scope=dict(vars(fixed),B_RELEASE=B_SOURCE)
    exec(compile(ast.Module(body=[node],type_ignores=[]),CALLER+':fixed_B_metadata','exec'),scope)
    b=scope['b_metadata'](ROOT)
    configs=[]
    for item in menu['candidates']:
        config=store.read(dict(commit=PLAN,path=DIRECTORY+item['configuration_file']))
        b.validate(config)
        if (config['id']!=item['configuration_id'] or config['spec']['L_D']!=item['L_D']
                or config['spec']['A_input']['kappa_paper']!=menu['conditions']['kappa_paper_exact']):
            raise ValueError('Candidate or exact kappa differs')
        configs.append(config)
    if [c['spec']['L_D'] for c in configs]!=[1,2] or any(c['spec']['M']!=12 for c in configs):
        raise ValueError('Only the fixed two 13-round configurations are permitted')
    exact=lambda v:Fraction(v['numerator'],v['denominator']) if isinstance(v,dict) else Fraction(v)
    return menu,configs,b,exact,store,provider


def run_batches(out,menu,configs,b,exact,store,provider,*,outcomes=None):
    """One menu, sequential starts so any failure prevents the next config."""
    outcomes=[] if outcomes is None else outcomes
    for config in configs:
        label='LD'+str(config['spec']['L_D'])
        print('START '+label+' '+config['id'],flush=True)
        batch=provider.SignalBatch(config,{r['m']:r['physical_candidate'] for r in config['rounds']},
            b_validate=b.validate,b_exact_scalar=exact,source=menu['signal_source'],root=ROOT,
            context_factory=lambda c:provider.load_h4_context(ROOT,c),
            reference_provider=lambda j:provider.saved_h4_reference(ROOT,j,store=store))
        value=batch.run(out/label)
        outcomes.append(dict(label=label,configuration_id=config['id'],result=value))
        print(label+' '+value['status']+' records='+str(len(value['records'])),flush=True)
        if value['status']!=COMPLETE:break
    return outcomes


def execute(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh directory required; no retry/resume')
    out.mkdir(parents=True)
    def save(name,value):
        with (out/name).open('x') as f:
            f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    result=dict(status='STARTED',plan_source=PLAN,numerical_source=NUMERICAL,B_source=B_SOURCE,
        caller_source=dict(commit=git('rev-parse','HEAD').decode().strip(),path=CALLER),
        started_UTC=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),outcomes=[],errors=[],
        new_reference_actions=0,old_saved_mean_reanalysis=0,
        sufficient_N=None,global_RMSE_bound=None,C=None,G=None,winner=None,
        automatic_retry=False,mandatory_stop=True)
    save('EXECUTION.json',result)
    try:
        menu,configs,b,exact,store,provider=load_metadata()
        save('FIXED_SCOPE.json',dict(plan_source=PLAN,menu=menu,
            configuration_sources=[dict(commit=PLAN,path=DIRECTORY+v['configuration_file']) for v in menu['candidates']]))
        run_batches(out,menu,configs,b,exact,store,provider,outcomes=result['outcomes'])
        result['status']=('MENU_FINITE_MEANS_COMPLETE_NOT_RMSE_QUALIFIED' if len(result['outcomes'])==2
            and all(r['result']['status']==COMPLETE for r in result['outcomes']) else 'STOP_PARTIAL_MENU')
        result['metadata_sources_read']=store.sources
        import numpy,scipy
        result['runtime']=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__)
    except Exception as error:
        result['status']='STOP_CALLER_INPUT_OR_RUNTIME_FAILURE'
        result['errors'].append(dict(type=type(error).__name__,message=str(error)))
    result['completed_UTC']=datetime.now(timezone.utc).isoformat()
    result['not_started_configurations']=[p for p in (1,2) if 'LD'+str(p) not in {r['label'] for r in result['outcomes']}]
    save('result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    value=execute(args.out)
    print(value['status'],flush=True)
    sys.exit(0 if value['status']=='MENU_FINITE_MEANS_COMPLETE_NOT_RMSE_QUALIFIED' else 2)
