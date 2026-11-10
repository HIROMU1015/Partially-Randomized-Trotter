"""Phase B only: reuse frozen V1 scalars; exactly five development fits."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.h6_df19_price_v2 import V1,reuse,campaign
OUT=ROOT/'docs/research/h6_df19_price_v2_20261110'


def main():
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=['src/trottertracks/h6_df19_price_v2.py','scripts/run_h6_df19_price_v2_saved.py'];source_hashes={}
    for path in files:
        b=(ROOT/path).read_bytes()
        if b!=subprocess.check_output(['git','show',source+':'+path],cwd=ROOT):raise ValueError('Commit source before real fits')
        source_hashes[path]=hashlib.sha256(b).hexdigest()
    inputs={};data=[]
    for name in ('DEVELOPMENT.json','HOLDOUT.json','AUXILIARY.json'):
        path='docs/research/h6_df19_price_v1_20261110/'+name;b=subprocess.check_output(['git','show',V1+':'+path],cwd=ROOT)
        inputs[path]=hashlib.sha256(b).hexdigest();data.append(json.loads(b)['rows'])
    result,rows=campaign(*data,reuse(ROOT));result.update(scientific_source=source,input_carrier=V1,input_sha256=inputs,source_sha256=source_hashes,scope=dict(fits=5,V1_recounts=0,circuit_build_compile=0,sampling=0,signal_N_G=0))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'SAVED_CORRECTIONS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    fields=list(rows[0])
    with (OUT/'SAVED_PATH_RESIDUALS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(source=source,selected=result['selected'],calibrations=result['calibrations']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
