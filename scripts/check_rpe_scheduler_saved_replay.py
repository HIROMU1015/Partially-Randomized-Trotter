"""Compare adoption from the fixed C catalog; no generated kappa grid or new Q3."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_method_optimizer as O,rpe_optimizer_concurrent as C
spec=importlib.util.spec_from_file_location('_replay_fixtures',ROOT/'tests/test_rpe_optimizer_concurrent.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start=time.perf_counter();problem=O.load_json(ROOT/'configs/rpe_method_optimizer/h4_df10_problem.json')
    problem.pop('price_rules',None)  # saved own prices only; do not launch the external quote helper
    context=O.prepare(ROOT,problem,generate_configurations=False)
    prepared=time.perf_counter()
    methods=list(problem['methods']);initial={m:[]for m in methods};seen=set()
    by_id={c['configuration_id']:c for c in context['candidates']}
    for(cid,_),r in context['saved'].table.items():
        if cid not in by_id or cid in seen or r['status']!=O.PASS:continue
        seen.add(cid);initial[by_id[cid]['method']].append(r['label'])
    problem['search']['incumbent_scope']='CONFIGURATION'
    budgets={m:0 for m in methods};B=F.baseline();rows=[]
    with tempfile.TemporaryDirectory()as tmp:
        kwargs=dict(context=context,methods=methods,initial=initial,max_new=budgets,science_approved=False)
        before=time.perf_counter();old=B.optimize(ROOT,problem,**kwargs);oldwall=time.perf_counter()-before
        ex=C.ConcurrentExecutor(ROOT,Path(tmp)/'fast',problem['execution_routes'],context['expected'],problem,
            max_concurrent=18,budgets=budgets,science_approved=False)
        before=time.perf_counter();new=C.optimize(ROOT,problem,executor=ex,**kwargs);newwall=time.perf_counter()-before
        same=F.canonical(old)==F.canonical(new)
        assert same and not list(Path(tmp).glob('**/LAUNCH.json'))
        for m in methods:
            rows.append(dict(method=m,same_best_adoption_order_stop=True,
                best=O.jsonable(F.canonical(new)[m]['best']),stop_reason=new[m]['stop_reason'],
                adoption_order=O.jsonable(F.canonical(new)[m]['trace'])))
        data=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            scope='FIXED_C_CATALOG_ONLY_NOT_RUNNING_KAPPA_SEARCH',baseline='1e88cd85cc8d2739a4dce43d138b94b153309333',
            configuration_generation=False,new_price_quotes=False,initial_labels=initial,rows=rows,
            preparation_seconds=prepared-start,legacy_replay_seconds=oldwall,concurrent_replay_seconds=newwall,
            real_Q3_calls=0,metrics=ex.metrics(),source_inputs=context['catalog'].get('inputs'))
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(dict(all_match=same,adopted={m:len(r['trace'])for m,r in new.items()},new_real_Q3_calls=0)))


if __name__=='__main__':main()
