#!/usr/bin/env python3
"""Read-only publication metadata/preservation audit; no scientific calculation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
BASE='ba4530678df9d665d2e6005fb08179b0f7a7921f'
TAG='n1_decode_recovery_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root):
    checks=0
    def req(ok,msg):
        nonlocal checks
        if not ok:raise ValueError(msg)
        checks+=1
    path='artifacts/validation_manifest.json';oldraw=subprocess.check_output(['git','show',BASE+':'+path],cwd=root,text=True);raw=(root/path).read_text();old=json.loads(oldraw);current=json.loads(raw)
    req(len(old['result_sets'])==89 and current['result_sets'][:89]==old['result_sets'] and len(current['result_sets'])==90,'89 prior entries preserved plus one')
    req(all(current[k]==v for k,v in old.items() if k!='result_sets'),'all other manifest fields preserved')
    start=oldraw.index('[',oldraw.index('"result_sets"'));_,end=json.JSONDecoder().raw_decode(oldraw,start);trim=end-1
    while oldraw[trim-1].isspace():trim-=1
    req(raw[:trim]==oldraw[:trim] and raw.endswith(oldraw[trim:]),'manifest raw prefix/suffix preserved')
    entry=current['result_sets'][-1];req(set(entry)=={'id','status','summary','counts','completion_criteria'} and entry['id']==TAG+'_run1_20261011' and entry['status']=='source_present_no_current_ci','new entry schema')
    req(bool(entry['summary']) and all(type(v)==int and v>=0 for v in entry['counts'].values()) and all(isinstance(x,str) and x for x in entry['completion_criteria']),'new entry fields')
    protected=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--','src/trottertracks/','tests/','artifacts/n13_redesign/','artifacts/hamiltonian_algorithm_design/'],cwd=root,text=True).splitlines()
    protected+=['scripts/run_n13_redesign.py','scripts/verify_n13_redesign.py','scripts/audit_n13_redesign_saved_diagnostics.py','docs/research/n13_redesign_execution_plan_20261011.json','docs/research/n13_redesign_execution_20261011.md','docs/research/n13_redesign_results_20261011.md']
    old_protected=len(set(protected))
    protected+=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--','artifacts/n3_generator_n1_recovery/'],cwd=root,text=True).splitlines()
    protected+=['docs/research/n3_generator_transfer_contract_v1_20261011.md','docs/research/n3_generator_transfer_contract_v1_20261011.json','docs/research/n1_decode_recovery_contract_v1_20261011.md','docs/research/n1_decode_recovery_contract_v1_20261011.json','docs/research/n3_generator_n1_recovery_preparation_20261011.md']
    for path in sorted(set(protected)):req((root/path).read_bytes()==subprocess.check_output(['git','show',BASE+':'+path],cwd=root),'protected original blob '+path)
    out=root/('artifacts/'+TAG+'/2026-10-11/run1');plan=json.loads((root/('docs/research/'+TAG+'_execution_plan_20261011.json')).read_bytes())
    for path,s in plan['source_input_sha256'].items():req(sha(root/path)==s,'fixed source/input binding '+path)
    req(plan['this_batch_authorized'] and not plan['next_scientific_stage_authorized'],'separate current authorization')
    stop=json.loads((out/'MANDATORY_STOP.json').read_bytes());req(stop['mandatory_stop'] and not stop['next_scientific_stage_authorized'] and stop['central_theme'] is None and stop['scientific_retries']==0,'terminal STOP')
    files=[p for p in (root/('artifacts/'+TAG)).rglob('*') if p.is_file()];req(sum(p.stat().st_size for p in files)<=512*1024**2,'aggregate output cap')
    req(all(p.stat().st_size<=32*1024**2 for p in files),'JSON/IR shard cap');req(all(p.stat().st_size<=16*1024**2 for p in files if p.suffix=='.log'),'log cap')
    max_native_ops=0;IR=0;max_serialized_integer_bits=0
    def integers(value):
        nonlocal max_serialized_integer_bits
        if type(value)==int:max_serialized_integer_bits=max(max_serialized_integer_bits,abs(value).bit_length())
        elif isinstance(value,dict):
            for x in value.values():integers(x)
        elif isinstance(value,list):
            for x in value:integers(x)
    for p in files:
        if p.name.endswith('.json.gz'):
            item=json.loads(gzip.decompress(p.read_bytes()));IR+=1;max_native_ops=max(max_native_ops,len(item['operations']));req(len(item['operations'])<=20000,'saved native operation cap')
        elif p.suffix=='.json' and p.name not in ('file_inventory.json','publication_receipt.json'):integers(json.loads(p.read_bytes()))
    req(max_serialized_integer_bits<=4096,'serialized integer/rational evidence bits')
    result={'status':'LOCAL_PUBLICATION_METADATA_PASS','checks':checks,'original_manifest_entries_preserved':89,'prior_frozen_blobs_preserved':old_protected,'all_protected_blobs_preserved':len(set(protected)),'new_entry_local_schema':'PASS','global_manifest_status':'FAIL_PRESERVED_OLD_RELEVANT_COMMITS_7_DESCRIPTION_MISSING','files':len(files),'output_bytes':sum(p.stat().st_size for p in files),'native_IR':IR,'maximum_native_operations':max_native_ops,'maximum_serialized_integer_bits':max_serialized_integer_bits,'serialized_bits_are_all_intermediate_arithmetic_measurement':False,'new_scientific_runs':0,'compilations':0,'solver_calls':0,'mandatory_stop':True}
    inventory=root/('artifacts/'+TAG+'/2026-10-11/file_inventory.json')
    if inventory.exists():
        saved=json.loads(inventory.read_bytes())
        for path,entry in saved['files'].items():p=root/path;req(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'publication file inventory '+path)
        result['checks']=checks;result['inventory_blobs_checked']=len(saved['files'])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--report',type=Path);args=p.parse_args();result=audit(args.root.resolve());data=json.dumps(result,indent=2)+'\n'
    if args.report:args.report.write_text(data)
    print(data,end='')
if __name__=='__main__':main()
