"""Metadata/SHA only; never call a scientific or shot evaluator."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    old=ROOT/'artifacts/non_df_h4_conditional_cng_results/2026-10-24'
    axes=ROOT/'artifacts/non_df_h4_a_axes/2026-10-24'
    plan=ROOT/'artifacts/non_df_h4_conditional_cng_plan/2026-10-24'
    files=dict(read(plan/'inputs.json')['files'])
    required=[plan/'inputs.json',plan/'axis_conditions.json']
    required += [old/p for p in ('active_contract.json','seal.json','verification.json',
        'run01/conditional_functions.json','run01/conditional_results.json','run01/comparison_regions.json',
        'run01/cost_summary.json','run01/STOP.json','run01/native/STOP.json','run01/result_summary.json')]
    required += [axes/p for p in ('seal.json','verification.json','conditional_CNG_readiness.json',
        'run01/paired_costs.json','run01/statistics.json','run01/STOP.json','run01/native/STOP.json')]
    for p in required:files[str(p.relative_to(ROOT))]=sha(p)
    for p,h in files.items():assert sha(ROOT/p)==h, 'OLD_INPUT_HASH:'+p
    save(BASE/'inputs.json',dict(kind='H4_CENTRAL_SAVED_INPUTS_V1',files=files,
        previous_accounting_SOURCE='b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8',
        paired_price_SOURCE='a37232f85812688c3b9ab957c5073c5ce143b17e',
        roles='original plan/inputs.json roles retained; central paired rows from A_AXES',
        historical_full_input_manifest_reused=True,scientific_recalculation=False))
    preservation=read(axes/'preservation_before.json')
    for p,h in preservation.items():assert sha(ROOT/p)==h, 'OLD_PRESERVATION_HASH:'+p
    paths=subprocess.check_output(['git','ls-files','--',str(axes.relative_to(ROOT)),str(old.relative_to(ROOT))],cwd=ROOT,text=True).splitlines()
    preservation.update({p:sha(ROOT/p) for p in paths})
    save(BASE/'preservation_before.json',preservation)
    save(BASE/'manifest_before.json',read(ROOT/'artifacts/validation_manifest.json'))
    for src,dst in [('/tmp/h4_central_legacy_artificial01.log','artificial_legacy16.log'),
                    ('/tmp/h4_central_artificial01.log','artificial_adapter14.log')]:
        shutil.copyfile(src,BASE/dst)
    shutil.copyfile('/home/AbeHiromu/.codex/attachments/8dba663e-08f6-4929-bef5-ab5a3fbc87f1/貼り付けたテキスト.txt',BASE/'approval.txt')
    print(json.dumps(dict(inputs=len(files),preserved=len(preservation),real_H4_scalar_calls=0)))
