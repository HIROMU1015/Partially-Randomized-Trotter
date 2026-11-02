#!/usr/bin/env python3
"""Static C/L proof connection only; no execution, acceptance or accounting."""
import argparse,csv,importlib.util,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cl_transfer_cli',ROOT/'src/trotterlib/h4_cl_binding.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--verify-record',type=Path)
    p.add_argument('--mode',default='VERIFY_SAVED_ONLY')
    a=p.parse_args()
    try:
        m.no_execution(a.mode)
        m.require(a.manifest is not None,'MANIFEST_REQUIRED')
        store=m.TransferStore(ROOT,m.T.read_json_bytes(a.manifest.read_bytes()))
        output=m.produce(store)
        if a.verify_record:m.verify(store,m.T.read_json_bytes(a.verify_record.read_bytes()))
        status=m.verify(store,output)
        if a.output:
            a.output.mkdir(parents=True,exist_ok=False)
            def save(name,data):
                with (a.output/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
            save('gate_connections.json',output);save('verification.json',status)
            save('primitive_coverage.json',output['primitive_coverage']);save('formal_lemmas.json',output['formal_lemmas'])
            rows=[]
            for row in output['rows']:
                for name,gate in row['gates'].items():
                    rows.append(dict(case=row['target']['case'],epsilon_sig=row['target']['epsilon_sig'],axis=row['target']['axis'],gate=name,scope=gate['scope'],category=gate['category'],existing_proof_status=gate['existing_proof_status'],conditions=json.dumps(gate['conditions']),target_digest=row['target_digest'],scientific_gate_closed=False,formal_certificate_accepted=False))
            with (a.output/'gate_matrix.csv').open('x',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps(status));return 0
    except (ValueError,KeyError,OSError) as error:
        print('CL_STOP: '+str(error),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
