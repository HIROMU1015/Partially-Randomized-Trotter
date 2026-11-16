"""Saved-only, four-method extension of A87d00ccba's public-Skill acceptance.

Uses A.load_skill and A.Program unchanged; never runs A's positive-budget cases.
The tested executable is the pinned public CLI. No LD/kappa/shot winners in input.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import acceptance_rpe_cost_compare_skill as A

METHODS=('FULL_PF','TRUNCATED_PF','PR_PAULI','PR_DF')
CATALOG_QUERY=r'''
import hashlib,json,sys
from pathlib import Path
from fractions import Fraction
from trottertracks import rpe_method_optimizer as O
root=Path.cwd();problem=O.load_json('configs/rpe_method_optimizer/h4_df10_problem.json')
reader,inputs,catalog=O.load_catalog(root,problem)
def f(x):return Fraction(int(x['numerator']),int(x['denominator'])) if isinstance(x,dict) else Fraction(x)
expected={}
for method in ('FULL_PF','TRUNCATED_PF','PR_PAULI','PR_DF'):
    rows=[p for p in catalog['candidates'] if p['method']==method]
    feasible=[p for p in rows if p['status']=='FEASIBLE' and p.get('conditional_cost') is not None]
    least=min((f(p['conditional_cost']) for p in feasible),default=None)
    expected[method]=dict(minimum=None if least is None else O.jsonable(least),
      minima=[p for p in feasible if f(p['conditional_cost'])==least],
      states={s:sum(p['status']==s for p in rows) for s in sorted({p['status'] for p in rows})},
      excluded=[dict(configuration_id=p['configuration_id'],status=p['status'],conditional_cost=p.get('conditional_cost')) for p in rows if p['status']!='FEASIBLE'])
print(json.dumps(O.jsonable(dict(problem_sha256=hashlib.sha256(Path('configs/rpe_method_optimizer/h4_df10_problem.json').read_bytes()).hexdigest(),
 catalog_input=problem['evidence']['saved_catalog_inputs'],catalog_sources=inputs['sources'],comparison=inputs['comparison'],expected=expected)),ensure_ascii=False))
'''


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=1,default=str)+'\n')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invariant(result):
    return {m:dict(best=result['methods'][m]['best'],stop_reason=result['methods'][m]['stop_reason'])for m in METHODS}


def observe(run):
    run=Path(run)
    result=json.loads((run/'ROUND_RESULT.json').read_text())
    needs=json.loads((run/'NEEDS.json').read_text())
    inv=[json.loads(s)for s in (run/'INVOCATIONS.jsonl').read_text().splitlines()if s.strip()]
    jobs=sorted(p for p in (run/'jobs').glob('*/*')if p.is_dir())
    return dict(result=result,needs=needs,invocations=inv,jobs=[str(p)for p in jobs],
      launch_records=[str(p)for p in (run/'jobs').glob('*/*/LAUNCH.json')],
      outcomes=[str(p)for p in (run/'jobs').glob('*/*/OUTCOME.json')])


def same_point(best,row):
    if best is None:return False
    exact_keys=(('configuration_id','configuration_id'),('N_X','N_X'),('N_Y','N_Y'),('status','q3_status'))
    if not all(best[k]==row[r]for k,r in exact_keys):return False
    if A.frac(best['cost'])!=A.frac(row['conditional_cost']):return False
    lo,hi=A.frac(best['MSE_lower']),A.frac(best['MSE_upper'])
    return lo<=A.frac(row['MSE_lower'])<=A.frac(row['MSE_upper'])<=hi


def validate(expected,first,second):
    checks=[]
    for name in METHODS:
        item=first['result']['methods'][name];target=expected['expected'][name]
        matches=[p for p in target['minima']if same_point(item['best'],p)]
        checks.append(A.check(name+'_fixed_catalog_minimum',bool(matches),expected=target['minimum'],
           observed=item['best']and item['best']['cost'],configuration_id=item['best']and item['best']['configuration_id']))
        notready={c['configuration_id']for c in first['needs']['candidates']if c['method']==name and not c['ready']}
        checks.append(A.check(name+'_waiting_candidates_not_best',item['best']is not None and item['best']['configuration_id']not in notready,
            not_ready_count=len(notready),catalog_states=target['states']))
        resumed=second['result']['methods'][name]
        checks.append(A.check(name+'_same_result_and_stop_after_resume',item['best']==resumed['best'] and item['stop_reason']==resumed['stop_reason'],stop=item['stop_reason']))
        for phase,data in [('first',first),('resume',second)]:
            inv=data['invocations'][-1]
            checks.append(A.check(name+'_'+phase+'_zero_calls',data['result']['methods'][name]['new_calls']==0 and inv['launched'][name]==0 and inv['new_calls'][name]==0,
               budget=inv['budget'][name],launched=inv['launched'][name],new_calls=inv['new_calls'][name]))
    checks.append(A.check('four_methods_in_one_round',set(first['result']['methods'])==set(METHODS)))
    checks.append(A.check('no_job_or_scientific_worker_started',not first['jobs'] and not second['jobs'] and not second['launch_records'] and not second['outcomes']))
    checks.append(A.check('same_run_two_invocations',len(first['invocations'])==1 and len(second['invocations'])==2))
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skill-commit',default='19252baf39698570bd80a98bf10142cc8f215517');p.add_argument('--program-root',required=True);p.add_argument('--python',required=True);p.add_argument('--output',required=True);p.add_argument('--program-source');p.add_argument('--phase',default='current');p.add_argument('--search-overrides',type=Path)
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    skill=A.load_skill(args.skill_commit);program=Path(args.program_root).resolve();sha=A.git(program,'rev-parse','HEAD').decode().strip()
    required=args.program_source or skill['program_source']
    if sha!=required:raise ValueError('Program checkout is not the fixed source')
    if A.git(program,'status','--porcelain').strip():raise ValueError('Use a clean dedicated program checkout')
    spec=deepcopy(skill['template']);spec.update(round_id='FOUR_METHOD_SAVED_ACCEPTANCE_'+args.phase.upper(),methods=list(METHODS),max_new_q3_per_method={m:0 for m in METHODS},initial='all',method_overrides={},resume_from=[],science_approved=False)
    if args.search_overrides:spec['search_overrides']=json.loads(args.search_overrides.read_text())
    rp=out/'ROUND_INPUT.json';dump(rp,spec)
    # Query the very program version's own fixed catalog, before selecting any
    # expectedwinner; the CLI still receives only method/range/budget choices.
    catalog=subprocess.run([args.python,'-B','-c',CATALOG_QUERY],cwd=program,env=dict(A.os.environ,PYTHONPATH='src',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
    (out/'CATALOG_QUERY.log').write_text(catalog.stderr)
    if catalog.returncode:raise ValueError('Actual fixed catalog read failed: '+catalog.stderr[-1000:])
    expected=json.loads(catalog.stdout);dump(out/'EXPECTED_FROM_ACTUAL_CATALOG.json',expected)
    prog=A.Program(program,args.python,out);error=None;checks=[];first=second=None;before={}
    try:
        prog.run(rp,out/'run');first=observe(out/'run')
        snap=out/'first_snapshot';snap.mkdir()
        for name in ('ROUND_RESULT.json','NEEDS.json','INVOCATIONS.jsonl','PER_METHOD.csv','ROUND.json'):shutil.copyfile(out/'run'/name,snap/name)
        before={str(f.relative_to(out/'run')):digest(f)for f in (out/'run'/'jobs').glob('**/*')if f.is_file()}
        prog.run(rp,out/'run');second=observe(out/'run');checks=validate(expected,first,second)
        after={str(f.relative_to(out/'run')):digest(f)for f in (out/'run'/'jobs').glob('**/*')if f.is_file()}
        checks.append(A.check('completed_jobs_not_restarted_or_modified',before==after))
        checks.append(A.check('identical_resume_command',prog.calls[0]['cmd']==prog.calls[1]['cmd']and prog.calls[0]['cwd']==prog.calls[1]['cwd']))
        checks.append(A.check('program_checkout_unchanged',not A.git(program,'status','--porcelain').strip()))
    except Exception as exc:
        import traceback
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    result=dict(schema='FOUR_METHOD_SKILL_SAVED_ACCEPTANCE_V1',phase=args.phase,skill=dict(commit=skill['commit'],declared_program_source=skill['program_source'],files={k:v['sha256']for k,v in skill['files'].items()}),program_source=sha,
       program_root=str(program),actual_entry='Skill documentation -> unchanged public CLI (no native Skill tool available)',
       reused_A_source='87d00ccba',round_input=str(rp),input_has_manual_LD_kappa_shots=False,expected_catalog_input=expected['catalog_input'],
       commands=prog.calls,checks=checks,counts=dict(PASS=sum(c['status']=='PASS'for c in checks),FAIL=sum(c['status']=='FAIL'for c in checks)),error=error,
       observed_cli_launches=sum(sum(i['launched'].values())for i in(second or first or{}).get('invocations',[])),
       scientific_calls=dict(signal=0,sampling=0,compile=0,fit=0,real_Q3=0),science_evidence='Pinned prepare path is saved metadata only; four zero budgets + INVOCATIONS/LAUNCH/OUTCOME observations; no producer callback enabled',
       results=None if first is None else {m:dict(best=first['result']['methods'][m]['best'],stop_reason=first['result']['methods'][m]['stop_reason'])for m in METHODS})
    dump(out/'ACCEPTANCE_RESULT.json',result)
    if first:
        lines=['|方式|configuration|X/Y配分|MSE区間・判定|条件付きRZ費用|停止理由|','|---|---|---|---|---|---|']
        for m in METHODS:
            r=first['result']['methods'][m];b=r['best'];lines.append('|%s|%s|%s|%s|%s|%s|'%(m,b and b['configuration_id'],b and ('X='+str(b['N_X'])+'<br>Y='+str(b['N_Y'])),b and ('[%.12e, %.12e] %s'%(float(A.frac(b['MSE_lower'])),float(A.frac(b['MSE_upper'])),b['status'])),b and str(A.frac(b['cost'])),r['stop_reason']))
        (out/'RESULT_TABLE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(phase=args.phase,counts=result['counts'],error=error and error['message']),ensure_ascii=False))
    return int(error is not None or bool(result['counts']['FAIL']))


if __name__=='__main__':raise SystemExit(main())
