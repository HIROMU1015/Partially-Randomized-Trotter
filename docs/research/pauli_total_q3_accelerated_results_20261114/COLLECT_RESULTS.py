"""Reuse the fixed result checker/reporter; no scientific evaluation entry."""
import json
from pathlib import Path
import subprocess
from types import ModuleType

ROOT=Path(__file__).resolve().parents[3]
DIRECTORY='docs/research/pauli_total_q3_accelerated_results_20261114'
OUTPUT=ROOT/DIRECTORY
REFERENCE='68dbba6711a2828f83d6085c104525659144ee65'
PATH='scripts/collect_pauli_total_q3_fixed.py'
module=ModuleType('_reference_pauli_result_collector');module.__file__=str(ROOT/PATH)
exec(compile(subprocess.check_output(['git','show',REFERENCE+':'+PATH],cwd=ROOT),REFERENCE+':'+PATH,'exec'),module.__dict__)
module.ROOT=ROOT;module.DIRECTORY=DIRECTORY;module.OUTPUT=OUTPUT
summary=module.collect()
inputs=json.loads((OUTPUT/'INPUTS.json').read_bytes())
summary['acceleration']=inputs['acceleration'];summary['stopped_original_run']=inputs['stopped_original_run']
original=inputs['stopped_original_run']['commit']
checks=[]
for row in summary['cases']:
    label=row['label'];result=json.loads((OUTPUT/'run01'/('RESULT_'+label+'.json')).read_bytes())
    row['acceleration']=result.get('acceleration')
    # The original run returned these reference components before it was
    # stopped. Same raw/law/target/vector must reproduce the same values.
    for kind in ('COARSE','LEMMA'):
        path='docs/research/pauli_total_q3_fixed_results_20261114/run01/PARTIAL_'+kind+'_'+label+'.json'
        old=json.loads(subprocess.check_output(['git','show',original+':'+path],cwd=ROOT))
        new=json.loads((OUTPUT/'run01'/('PARTIAL_'+kind+'_'+label+'.json')).read_bytes())
        for obj in (old,new):
            for name in ('all_event_loss_contract','unresolved_loss'):
                if isinstance(obj.get(name),dict):obj[name].pop('source',None)
        if old!=new:raise ValueError('Original reference component changed: '+label+' '+kind)
        checks.append(dict(label=label,component=kind,all_non_SOURCE_fields_identical=True))
summary['stopped_original_reference_checks']=checks
(OUTPUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
with (OUTPUT/'REPORT.md').open('a') as stream:
    stream.write('\n## 高速版の実行来歴\n\n')
    stream.write('ユーザーが旧run01の停止・部分成果保全と、高速版での2件の再実行を明示許可した。旧runは未完了として別branchに保全しており、完了したことにしていない。\n\n')
    stream.write('元停止runのcoarse/lemmaと、両構成の全非SOURCE fieldが一致。新backendはexact整数集約・process並列化・遅延約分。\n\n')
    stream.write('各構成8 worker、2構成並列。CPUはparent＋child合計を別fieldに記録し、RSSはparent peak・最大単独child・2秒サンプリングのprocess-tree合計を区別する。\n')
print(json.dumps(dict(status='RESULT_CHECKS_AND_REPORT_SAVED',reference_components_checked=len(checks),
    cases=[dict(label=r['label'],status=r['status'],performance=r.get('performance')) for r in summary['cases']],mandatory_stop=True)),flush=True)
