"""Prepare new private inputs from fixed H6 receipts; never launch science."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import h6_dba_contract as c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-config',type=Path,required=True)
    p.add_argument('--private',type=Path,required=True)
    a=p.parse_args();a.private.mkdir(parents=True,exist_ok=False)
    old=c.read(a.old_config);dest=a.private/'inputs';dest.mkdir()
    provider=a.private/'provider'
    subprocess.run(['git','clone','--shared',old['provider_root'],str(provider)],check=True)
    subprocess.run(['git','checkout','--detach',old['provider_source']],cwd=provider,check=True)
    inputs=[];mapping={}
    def add(role,path,target_name=None):
        path=Path(path);target=dest/(target_name or role)
        shutil.copyfile(path,target)
        c.require(c.sha(path)==c.sha(target),'INPUT_COPY')
        inputs.append(dict(role=role,path=str(target),bytes=target.stat().st_size,sha256=c.sha(target)))
        mapping[role]=str(target)
    for role in ('h6_input_snapshot.npz','snapshot_receipt.json','df_receipt.json'):
        original=next(x for x in old['files'] if x['logical_name']==role)
        c.require(c.sha(original['path'])==original['sha256'],'ORIGINAL_INPUT_RECEIPT')
        add(role,original['path'])
    archive=deepcopy(old['archive']);c.require(c.sha(archive['path'])==archive['sha256'],'OLD_ARCHIVE_HASH')
    add('preparations.json',archive['path']);archive['path']=mapping['preparations.json']
    add('font_seed',old['font_seed']['path'],old['font_seed']['name'])
    font=deepcopy(old['font_seed']);font['path']=mapping['font_seed']
    artifact=c.ROOT/'artifacts/resource_applicability'
    raw=gzip.decompress((artifact/'track_a_v1_phase_a_run02/2026-10-16/signal_results_v1.json.gz').read_bytes())
    export=c.read(artifact/'track_a_v1_phase_a_run02/2026-10-16/signal_export_receipt_v1.json')
    c.require(__import__('hashlib').sha256(raw).hexdigest()==export['uncompressed_sha256'],'SAVED_SIGNAL_EXPORT')
    signals=json.loads(raw)
    saved=next(r for r in signals['signals'] if r['cell']['id']==c.plan()['D']['cell_id'])
    path=a.private/'D_saved_signal.json';c.save(path,saved);add('D_saved_signal',path,'D_saved_signal.json')
    costs_path=artifact/'track_a_v1_reaccount_result/2026-10-16/axis_costs.json'
    add('saved_axis_costs',costs_path,'saved_axis_costs.json')
    costs=c.read(costs_path)['rows']
    predictions={axis:next(r['RZ'] for r in costs if (r['cell_id'],r['axis'])==
                          (c.plan()['D']['cell_id'],axis)) for axis in ('cosine','sine')}
    config=dict(schema='H6_DBA_config_v1',provider_root=str(provider),provider_source=old['provider_source'],
        provider_hashes=old['provider_hashes'],python=old['python'],runtime_file_hashes=old['runtime_file_hashes'],
        archive=archive,binding=old['binding'],compiler_options=old['compiler_options'],
        profile_reference=old['profile_reference'],font_seed=font,inputs=inputs,mapping=mapping,
        D_predictions=predictions,source_hashes={name:c.sha(c.ROOT/name) for name in
        ('scripts/h6_dba_contract.py','scripts/h6_dba_native.py','scripts/run_h6_dba.py')})
    c.save(a.private/'config_D.json',config)
    public=c.ROOT/'artifacts/resource_applicability/track_a_h6_dba/2026-10-18'
    public.mkdir(parents=True,exist_ok=False)
    c.save(public/'plan_v1.json',c.plan())
    c.save(public/'input_binding_v1.json',dict(binding=config['binding'],profile=config['profile_reference'],
        D_predictions=predictions,inputs=[{k:v for k,v in x.items() if k!='path'} for x in inputs],
        provider_SOURCE=old['provider_source'],runtime_files=len(config['runtime_file_hashes']),
        grant_issued=False))
    print(json.dumps(dict(config=str(a.private/'config_D.json'),predictions=predictions)))


if __name__=='__main__':main()
