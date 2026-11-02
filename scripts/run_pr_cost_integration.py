"""A/B/D independent one-shot entry. Preparation emits INVALID grants only."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pr_cost_contract as c


def account(data,permit):
    rows=c.core.account_rows(data,permit)
    by=defaultdict(list)
    for row in rows:by[(row['bias_mode'],row['epsilon'],row['factor'])].append(row)
    comparisons=[]
    for endpoint,items in sorted(by.items()):
        for name,ids in data['deterministic_sets'].items():
            selected=[r for r in items if r['cell_id'] in ids+data['PR_ids']]
            comparisons.append(dict(bias_mode=endpoint[0],epsilon=endpoint[1],factor=endpoint[2],
                comparison_scope=name,**c.core.compare(selected,ids,data['PR_ids'])))
    return dict(status='COMPLETE_MANDATORY_STOP',stage='D_ACCOUNTING',rows=rows,comparisons=comparisons,
        new_signal=0,compile=0,fit=0,sampling=0,shot_recalculation=0,next_stage_authorized=False)


def execute(args):
    # Deliberately authorize before touching config/input paths or native imports.
    grant=c.read(args.grant);c.authorize(grant,args.system,args.stage)
    config=c.read(args.config);fresh=c.verify(config,grant,args.system,args.stage)
    output=c.home_path(config['output']);lock=c.home_path(config['one_shot_lock'])
    lock.parent.mkdir(parents=True,exist_ok=True)
    c.claim_attempt(lock,grant['SOURCE'],grant['scope'])
    output.mkdir(parents=True,exist_ok=False)
    try:
        c.save(output/'startup_gate.json',fresh)
        if args.stage=='A_COST':
            import pr_cost_native
            result=pr_cost_native.execute(args.system,config,grant,output)
        else:
            # Input schema is a source/input/profile-bound projection of saved
            # records. It cannot be replaced by a different cost law or new fit.
            row=next(x for x in config['inputs'] if x['role']=='accounting_payload')
            data=c.read(row['path']);c.core.same_binding(data['binding'],config['binding'])
            c.require(data['kind']=='SAVED_REAL_VALUES','REAL_INPUT_KIND')
            budget=c.Budget(output,dict(saved_scalar_pass=1));budget.take('saved_scalar_pass')
            permit=c.core.Permit(args.stage,synthetic=False,authorization=grant)
            result=account(data,permit) if args.stage=='D_ACCOUNTING' else c.core.conditional_diagnostic(data,permit)
        c.save(output/'result.json',result)
        c.save(output/'final_gate.json',c.verify(config,grant,args.system,args.stage,final=True))
        c.save(output/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',exit_code=0,
            SOURCE=grant['SOURCE'],result_sha256=c.sha(output/'result.json'),next_stage_authorized=False,retry=False))
    except BaseException as error:
        c.first_stop(output,str(error) or type(error).__name__)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=('H4','H6','H8'),required=True)
    p.add_argument('--stage',choices=c.STAGES,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',type=Path)
    mode.add_argument('--run',action='store_true')
    p.add_argument('--grant',type=Path);p.add_argument('--config',type=Path)
    args=p.parse_args()
    try:
        if args.prepare:
            args.prepare.mkdir(parents=True,exist_ok=False)
            c.save(args.prepare/'plan_PROPOSED.json',c.plan(args.system,args.stage))
            c.save(args.prepare/'grant_INVALID.json',c.invalid_grant(args.system,args.stage))
            print('PREPARATION_ONLY; INVALID_GRANT; SCIENTIFIC_CALLS=0')
        else:
            c.require(args.grant is not None and args.config is not None,'GRANT_AND_CONFIG_REQUIRED')
            execute(args)
    except (Exception,KeyboardInterrupt) as error:
        print('STOP: '+str(error),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
