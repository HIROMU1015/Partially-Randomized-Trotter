"""Metadata-only preparation of new H8 D inputs; never reconstruct or fit costs."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import h8_dba_contract as c


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-config',type=Path,required=True)
    p.add_argument('--private',type=Path,required=True);a=p.parse_args()
    a.private.mkdir(parents=True,exist_ok=False);inputs=a.private/'inputs';inputs.mkdir()
    old=c.read(a.old_config);provider=a.private/'provider'
    subprocess.run(['git','clone','--shared',old['provider_root'],str(provider)],check=True)
    subprocess.run(['git','checkout','--detach',c.PROVIDER],cwd=provider,check=True)
    mapping={};receipts=[]
    for row in old['files']:
        role=row['logical_name'];source=Path(row['path'])
        c.require(source.stat().st_size==row['bytes'] and c.sha(source)==row['sha256'],'ORIGINAL_INPUT:'+role)
        target=inputs/(role if '.' in role else role+'.json');shutil.copyfile(source,target)
        mapping[role]=str(target);receipts.append(dict(role=role,path=str(target),bytes=target.stat().st_size,sha256=c.sha(target)))
    raw=gzip.decompress(Path(mapping['H8_phase_A_signals']).read_bytes());signals=json.loads(raw)['signals']
    costs_path=c.ROOT/'artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/accounting_rows.jsonl.gz'
    costs=[json.loads(row)['updated'] for row in gzip.decompress(costs_path.read_bytes()).splitlines()]
    target=inputs/'saved_cost_rows.jsonl.gz';shutil.copyfile(costs_path,target)
    receipts.append(dict(role='saved_cost_rows',path=str(target),bytes=target.stat().st_size,sha256=c.sha(target)))
    predictions={};targets={}
    for case in c.plan()['D']:
        cid=case['id'];signal=next(row for row in signals if row['cell']['id']==cid)
        c.require(any(v['1.0']['cost_acquisition_eligible'] for v in signal['precision_rows'].values()),'SAVED_SIGNAL_ELIGIBLE')
        target=dict(case_id=cid,corrected=signal['signals']['corrected'],
            independent_forward=signal['independent_forward_signals']['corrected'],reference=signal['reference'],
            scalar_receipt=signal.get('scalar_receipt'))
        if case['method']=='B0':c.require(target['scalar_receipt'] is None,'B0_SCALAR_POLICY')
        else:c.require(target['scalar_receipt']['policy']=='EXACT_STORED_BINARY64_TAIL_IDENTITY','B0I_SCALAR_POLICY')
        targets[cid]=target;selected=[r for r in costs if r['cell_id']==cid and r['eligible']]
        c.require(bool(selected),'SAVED_COST_REQUIRED')
        predictions[cid]={axis:selected[0]['axis_costs'][axis]['RZ'] for axis in ['cosine','sine']}
        c.require(all(all(r['axis_costs'][axis]['RZ']==predictions[cid][axis] for axis in predictions[cid]) for r in selected),
                  'SAVED_COST_AXIS_UNIT_CONSISTENCY')
    binding=c.read(c.ROOT/'references/h8_key_compile_inputs_v1.json')
    config=dict(schema='H8_DBA_D_config_v1',provider_source=c.PROVIDER,provider_root=str(provider),
        provider_hashes=old['provider_hashes'],runtime_file_hashes=old['runtime_file_hashes'],python=old['python'],
        profile_sha256=c.PROFILE,compiler_hash=c.COMPILER,
        compiler_options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17),
        mapping=mapping,inputs=receipts,predictions=predictions,targets=targets,
        binding=binding['scientific_binding'],expected_preparation_hashes=binding['expected_preparation_hashes'],
        font_seed=dict(old['font_seed'],path=mapping['font_seed']))
    source_names=set(old['source_hashes'])|{'scripts/h6_dba_contract.py','scripts/h8_dba_contract.py',
        'scripts/h8_dba_direct.py','scripts/run_h8_dba_direct.py','scripts/h8_dba_witness.py','scripts/prepare_h8_dba_direct.py'}
    config['source_hashes']={name:c.sha(c.ROOT/name) for name in sorted(source_names)}
    c.save(a.private/'config_prepared.json',config)
    public=c.ROOT/'artifacts/resource_applicability/track_a_h8_dba/2026-10-18';public.mkdir(parents=True)
    c.save(public/'plan_v1.json',c.plan())
    c.save(public/'saved_binding_v1.json',dict(binding=config['binding'],profile=c.PROFILE,compiler=c.COMPILER,
        predictions=predictions,targets=targets,provider_SOURCE=c.PROVIDER,
        input_receipts=[{k:v for k,v in r.items() if k!='path'} for r in receipts],
        numerical_fit=False,new_cost_prediction=False,new_signal=False))
    print(json.dumps(dict(config=str(a.private/'config_prepared.json'),predictions=predictions)))


if __name__=='__main__':main()
