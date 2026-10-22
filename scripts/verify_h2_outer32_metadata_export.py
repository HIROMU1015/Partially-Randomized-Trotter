"""Static receipt/provenance checks; no scientific imports or arithmetic."""
import argparse
import ast
from collections import Counter
from pathlib import Path
import re
import export_h2_outer32_metadata as e


def verify(dest, private_map):
    dest=Path(dest);mapping=e.loads(Path(private_map).read_text());repo=Path(__file__).resolve().parents[1]
    x=e.Export(repo,dest,private_map);x.roots=dict(mapping['roots'])
    origins=e.loads((dest/'origin_inventory.json').read_text())
    def untoken(locator):
        for root,tag in mapping['roots'].items():
            prefix='${'+tag+'}'
            if locator==prefix or locator.startswith(prefix+'/'):return Path(root+locator[len(prefix):])
        raise ValueError('UNRESOLVED_TOKEN')
    source_index={r['id']:r for r in origins['originals']}
    checks=[]; cache={};projection_count=0;number_tokens=0
    def check(ok,name):
        e.require(ok,name);checks.append(name)
    def walk(node):
        nonlocal projection_count,number_tokens
        if isinstance(node,e.Number):number_tokens+=1
        elif isinstance(node,dict):
            if {'origin','pointer','saved'}<=node.keys():
                origin=source_index[node['origin']];p=untoken(origin['locator'])
                if str(p) not in cache:
                    check(e.sha_file(p)==origin['original_sha256'],'ORIGIN_SHA:'+origin['id'])
                    cache[str(p)]=e.loads(p.read_text())
                expected=x.public(x.pointer(cache[str(p)],node['pointer']))
                check(expected==node['saved'],'ORIGINAL_POINTER_EXACT_COPY:'+str(projection_count))
                projection_count+=1
            for value in node.values():walk(value)
        elif isinstance(node,list):
            for value in node:walk(value)
    files=sorted(p for p in dest.rglob('*') if p.is_file() and p.name not in ('final_static_verification.json','file_inventory.json'))
    for path in files:
        text=path.read_text()
        check(not re.search(r'/home/[^\s"<>]+|(?<![\w}])/tmp/',text),'NO_INTERNAL_ABSOLUTE_PATH:'+path.name)
        check(not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|Bearer\s+[A-Za-z0-9]|gh[pousr]_[A-Za-z0-9]{20}|sk-[A-Za-z0-9]{20}',text),'NO_CREDENTIAL_PATTERN:'+path.name)
        if path.suffix=='.json':walk(e.loads(text))
    taskdata=e.loads((dest/'task_acceptance_67.json').read_text());tasks=taskdata['tasks']
    check(len(tasks)==67 and len({r['task']['id'] for r in tasks})==67,'EXACT_67_TASKS')
    check(Counter(r['task']['candidate']['id'] for r in tasks)==Counter(e.TARGETS),'EXACT_33_34_COVERAGE')
    for row in tasks:
        r=row['result']['saved'];t=row['task'];tid=t['id']
        check(r['task']==t and r['SOURCE']==e.SOURCE and r['plan_sha256']==e.PLAN and r['seed_cohort']=='EXPLORATION','IDENTITY:'+tid)
        check(row['acceptance']['saved']['result_sha256']==source_index[row['result']['origin']]['original_sha256'],'ACCEPTED_SHA:'+tid)
        check([a['axis'] for a in row['axes']]==['cosine','sine'],'TWO_AXES:'+tid)
        for i,a in enumerate(row['axes']):
            check(a['cost']['saved']==r['axis_costs'][i],'COST_COPY:'+tid+':'+str(i))
            guard=e.loads((dest/a['guard_projection']).read_text())['saved']
            check(guard['policy']==e.GUARD and guard['unchanged_gate_stream'] and not guard['state_fit'],'GUARD:'+tid+':'+str(i))
    for raw,digest in mapping['original_sha256'].items():
        check(Path(raw).is_file() and e.sha_file(raw)==digest,'PRESERVATION:'+x.token(raw))
    stored=e.loads((dest/'runtime_provider_manifest.json').read_text())
    check(len(stored['historical_sealed_files'])==2012,'RUNTIME_SEAL_COVERAGE')
    # These are metadata inventories, not scientific distributions/statistics.
    runtime_counts=Counter('MATCH' if r['matches_saved'] else 'DIFFERENT_OR_MISSING' for r in stored['historical_sealed_files'])
    check(all(r['root_token'].startswith('${') for r in origins['roots']),'TOKEN_DICTIONARY_PRESENT')
    tree=ast.parse((repo/'scripts/export_h2_outer32_metadata.py').read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    check(all(n in ('collections','concurrent.futures','pathlib') for n in imports),'STDLIB_EXPORTER')
    return dict(status='PASS_STATIC_EXPORT_NOT_SCIENTIFIC_GO',checks_PASS=len(checks),
        source_pointer_projections=projection_count,unchanged_numeric_token_occurrences=number_tokens,
        original_files_rehashed=len(mapping['original_sha256']),runtime_inventory_counts=dict(runtime_counts),
        target_tasks=67,target_axes=134,compiler_guard_files=134,new_scientific_calls=0,
        numerical_analysis=0,new_compile=0,shot_recalculation=0,old_audit_rerun=False,
        limitation='Static identity/copy verification by author, not independent scientific review or restored native validation.',
        checks=checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--private-map',required=True)
    args=p.parse_args();report=verify(args.bundle,args.private_map)
    output=Path(args.bundle)/'final_static_verification.json'
    output.write_text(e.encode(report)+'\n')
    print(e.encode({k:v for k,v in report.items() if k!='checks'}))


if __name__=='__main__':main()
