"""Read the selected original D producer before making a task-owned clone.

This is metadata/preflight only. It never calls signals/acquire/freeze/finish.
Unregistered partial results and consumed STOPs remain at their original owner.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from run_pf_preparation_stage import cache_mean_pool


PRICE_PLAN_HOLDS = {
    'At least two distinct-q calibration cells and one later holdout required',
    'One saved axis only: resolve per-axis native execution before compiling',
}


def inspect(root,request,observations,signal_index=None):
    root=Path(root).resolve();sys.path[:0]=[str(root/'scripts'),str(root/'src')]
    spec=importlib.util.spec_from_file_location('original_pf_provider',root/'scripts/prepare_pf_candidate.py')
    provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
    source=provider.source_gate();rows=[]
    for item in request['selected']:
        configuration=provider.selected_configuration(item);cid=configuration['id']
        registry=Path(request['registry_file']);entry=None
        if registry.is_file():entry=json.loads(registry.read_text()).get('configurations',{}).get(cid)
        if entry and all(entry.get(k)for k in ['signal_packet','prices','q3_inputs']):
            rows.append(dict(configuration_id=cid,status='REUSE_COMPLETE_ORIGINAL_OWNER',entry=entry));continue
        owners={Path(request['output_root'])/cid,root/provider.STUDY/'selected'/cid,
                root/'.pf_evidence_private'/cid}
        owners.update(Path(p)for p in request.get('prior_owner_directories',[]))
        found=[]
        for directory in sorted(owners):
            if not directory.is_dir():continue
            # Explicit selected directories only, bounded depth; no project or
            # private-circuit archive scan and no process signalling.
            for pattern in ['*.json','*/*.json','*/*/*.json','*/*/*/*.json']:
                for path in directory.glob(pattern):
                    if any(word in path.name.upper()for word in ['STARTED','STARTING','LAUNCH','STOP','RESULT','STATUS']):
                        found.append(str(path))
            completed=directory/'COMPLETION.json'
            if completed.is_file():
                saved=json.loads(completed.read_text());candidate=saved.get('registry_entry')
                if saved.get('configuration_id')==cid and candidate and all(candidate.get(k)for k in ['signal_packet','prices','q3_inputs']):
                    entry=candidate;break
        if entry:
            rows.append(dict(configuration_id=cid,status='REUSE_COMPLETE_ORIGINAL_OWNER',entry=entry));continue
        if found:
            rows.append(dict(configuration_id=cid,status='PRIOR_OWNER_HOLD',
                reason='Selected original owner has running, UNKNOWN, STOP or unregistered partial scientific artifacts; recover/admit those before another execution',
                original_paths=sorted(set(found))));continue
        # Preserve D's exact selected-cell duplicate check at the ORIGINAL
        # root, where its live/private owner markers can still be observed.
        from trottertracks import pf_evidence_preparation as E
        if signal_index is not None:
            cache_mean_pool(root,signal_index)
            signal_index=None
        try:
            plan=E.make_plan(root,configuration,source)
        except ValueError as error:
            if str(error) not in PRICE_PLAN_HOLDS:
                raise
            rows.append(dict(configuration_id=cid,status='PRICE_EVIDENCE_REQUIRED',
                             reason=str(error),signal_deferred_for_price=True))
            continue
        destination=Path(observations)/cid;destination.mkdir(parents=True,exist_ok=True)
        try:
            provider.preflight(plan,destination)
            rows.append(dict(configuration_id=cid,status='ORIGINAL_OWNER_PREFLIGHT_PASS'))
        except Exception as error:
            rows.append(dict(configuration_id=cid,status='PRIOR_OWNER_HOLD',reason=type(error).__name__+': '+str(error)))
    return dict(schema='PF_ORIGINAL_OWNER_METADATA_V1',source=source,rows=rows,scientific_calls=0,compile_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--provider-root',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--signal-index',type=Path)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    result=inspect(args.provider_root,request,args.output.parent/'original_preflight',args.signal_index)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
