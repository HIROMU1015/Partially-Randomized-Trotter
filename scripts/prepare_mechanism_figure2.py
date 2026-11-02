#!/usr/bin/env python3
"""Transfer fixed saved fields to Figure 2 tables; no scientific arithmetic."""
import argparse
import csv
import hashlib
import io
import importlib.util
import json
from pathlib import Path
_spec=importlib.util.spec_from_file_location('mechanism_paper_transfer_schema',
    Path(__file__).resolve().parents[1]/'src/trotterlib/mechanism_paper.py')
_schema=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schema)
build_tables=_schema.build_tables


def read_sources(manifest):
    spec=json.loads(Path(manifest).read_text())
    if spec.get('mode')!='SAVED_VALUE_TRANSCRIPTION_ONLY':
        raise ValueError('TRANSCRIPTION_MODE_REQUIRED')
    loaded={}
    for role,entry in spec['inputs'].items():
        data=Path(entry['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError('INPUT_SHA_MISMATCH_'+role)
        loaded[role]=list(csv.DictReader(io.StringIO(data.decode()))) if entry['format']=='csv' else json.loads(data)
    return spec,loaded


def prepare(manifest,output):
    spec,inputs=read_sources(manifest)
    winners=[]
    for original in inputs.pop('h4_D_result')['winners']:
        saved=dict(original)
        saved['epsilon']=str(saved['epsilon'])  # saved endpoint label only
        winners.append(saved)
    bundle=build_tables(**inputs,h4_winners=winners)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for key in ('main_table','core_axes','explanation_factors','display_plan'):
        rows=bundle[key]
        with (output/(key+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
    receipt=dict(status='SAVED_FIELD_TRANSFER_COMPLETE_NO_SCIENTIFIC_REANALYSIS',
        counts=bundle['counts'],numeric_cells=bundle['numeric_cells'],
        scientific_arithmetic_performed=False,source_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.csv'))},
        core_reanalysis=0,explanation_reanalysis=0,SE_reanalysis=0,statistics=0,sampling=0,compile=0,ranking=0)
    (output/'transfer_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.saved_sources,args.output)['counts']))


if __name__=='__main__':
    main()
