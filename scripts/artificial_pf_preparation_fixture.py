"""ARTIFICIAL PF completion fixture, called only after normal candidate selection.

No operators, signals, circuits, compiled prices or scientific Q3 are computed.
The output is explicitly artificial and is rejected by scientific C consumers.
Neither prefix selection nor shot selection is implemented in this provider.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pf_optimizer_connection as PF


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text()!=text:
        raise ValueError('Preserve previous artificial fixture output: '+str(path))
    path.write_text(text)


def enc(value):return dict(numerator=int(value),denominator=1)


def complete(request):
    if request.get('artificial_fixture') is not True:
        raise ValueError('This entry is an explicit artificial fixture only')
    selected=request.get('selected')or[]
    if len(selected)!=1 or not selected[0].get('configuration_body'):
        raise ValueError('The normal selector must supply one canonical configuration')
    cfg=deepcopy(selected[0]['configuration_body']);meta=PF.configuration_metadata(cfg)
    if cfg['spec']['method']!='PF' or len(cfg['rounds'])!=13:
        raise ValueError('Canonical deterministic PF metadata required')
    physical=PF.physical_ids(ROOT,cfg);cid=cfg['id']
    directory=Path(request['output_root'])/cid
    packet=dict(artificial=True,configuration_id=cid,source=dict(kind='ARTIFICIAL_METADATA_NO_SIGNAL_EVALUATION'),
        means=[dict(m=r['m'],identity=deepcopy(r['identity']),law_binding=deepcopy(r['law_binding']),
                    physical_candidate=dict(id=physical[r['m']]),status='SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED',
                    artificial=True)for r in cfg['rounds']])
    axes=[]
    for r in cfg['rounds']:
        for axis in ('X','Y'):
            # Distinct own-prefix/round/axis fixture units; no circuit accounting.
            unit=(meta['p']+1)*(r['q_PF']+1)+(r['m']+1)*(1 if axis=='X'else 2)
            axes.append(dict(m=r['m'],axis=axis,physical_id=physical[r['m']],configuration_id=cid,
                unit_price_RZ=enc(unit),kind='ARTIFICIAL_ONLY_NOT_COMPILED',artificial=True,
                price_evidence=dict(kind='ARTIFICIAL_METADATA_NOT_A_DIRECT_RECEIPT',configuration_id=cid)))
    prices=dict(artificial=True,quantity='ARTIFICIAL_FIXTURE_UNITS_NOT_COMPILED_RZ',axes=axes)
    total='ARTIFICIAL_TOTAL_'+hashlib.sha256(cid.encode()).hexdigest()
    binding='ARTIFICIAL_BINDING_'+hashlib.sha256((cid+'/binding').encode()).hexdigest()
    inputs=dict(artificial=True,cases=[dict(configuration_id=cid,configuration=cfg,signal_packet=packet,
        total_view_id=total,binding_id=binding)])
    entry=dict(method=meta['method'],p=meta['p'],formula=meta['formula'],delta_over_h=str(meta['delta_over_h']),
        configuration=cfg,signal_packet=packet,prices=prices,q3_inputs=inputs,
        total_configuration_id=total,binding_id=binding,price_scope='ARTIFICIAL_FIXTURE_ONLY_NOT_SCIENTIFIC_PRICE')
    registry_path=Path(request['registry_file'])
    registry=json.loads(registry_path.read_text())if registry_path.exists()else dict(schema=PF.SCHEMA,artificial=True,configurations={})
    if registry.get('artificial')is not True:
        raise ValueError('Do not write an artificial entry into a production registry')
    previous=registry['configurations'].get(cid)
    if previous is not None and previous!=entry:raise ValueError('Artificial candidate evidence changed')
    registry['configurations'][cid]=entry
    for name,value in [('CONFIGURATION.json',cfg),('SIGNAL.json',packet),('PRICES.json',prices),('Q3_INPUTS.json',inputs)]:
        put(directory/name,value)
    put(registry_path,registry)
    with (directory/'PROVIDER_CALLS.jsonl').open('a')as stream:
        stream.write(json.dumps(dict(artificial=True,configuration_id=cid,mode='complete',real_scientific_calls=0))+'\n')
    return dict(artificial=True,status='ARTIFICIAL_PREPARED',items=[dict(configuration_id=cid,output=str(directory))],
                registry_file=str(registry_path),real_scientific_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['complete'],required=True)
    args=parser.parse_args()
    response=complete(json.loads(args.request.read_text()));put(args.output,response)
    print(json.dumps(dict(status=response['status'],artificial=True,real_scientific_calls=0)))


if __name__=='__main__':main()
