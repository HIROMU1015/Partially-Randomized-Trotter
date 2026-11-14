"""Report the two completed approved diagnostics; never evaluate a risk kernel.

Optional wait/publish completes only this namespace. Returned exact intervals
are checked and displayed, without a target-precision filter, N search or G.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY='docs/research/pauli_total_q3_fixed_results_20261114'
OUTPUT=ROOT/DIRECTORY
BRANCH='codex/b-rpe-pauli-integration-20261114'
BASE='eecbcdaed88b626b00858c7e94b15f598258d5ec'


def read(path):return json.loads(path.read_bytes())


def write_new(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def exact(value):
    return Fraction(value['numerator'],value['denominator']) if isinstance(value,dict) else Fraction(value)


def rational(value):return dict(numerator=value.numerator,denominator=value.denominator)


def display(value):
    if value is None:return 'UNKNOWN'
    f=exact(value)
    with localcontext() as ctx:
        ctx.prec=14
        return format(Decimal(f.numerator)/Decimal(f.denominator),'.10E')


def finite(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Non-finite returned evidence')
    if isinstance(value,dict):
        for item in value.values():finite(item)
    if isinstance(value,list):
        for item in value:finite(item)


def collect():
    run=OUTPUT/'run01';completion=read(run/'COMPLETION.json');inputs=read(OUTPUT/'INPUTS.json')
    summaries=[];checks=[];execution=read(run/'EXECUTION.json')
    for case in inputs['cases']:
        label=case['label'];path=run/('RESULT_'+label+'.json')
        if not path.exists():
            summaries.append(dict(label=label,status='UNKNOWN_STOP_NO_RESULT',reason='Worker process failure; see completion'))
            continue
        result=read(path);finite(result)
        summary={k:result.get(k) for k in ('label','status','reason','performance','scientific_evaluation_calls')}
        for key in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'):
            if result.get(key) is not None:raise ValueError('Forbidden qualification or cost output')
        if result.get('scientific_evaluation_calls',0)>1:raise ValueError('More than one scientific evaluation')
        record=result.get('record')
        if record is not None:
            if record['configuration_id']!=case['total_configuration_id']:raise ValueError('Canonical result ID changed')
            if record['allocation']['shots']!=case['allocation']['shots']:raise ValueError('Shot vector changed')
            risk=record['diagnostic']['deterministic_risk'];dp=risk['bounds'].get('total_state_dp') or {}
            if risk['estimator']!=inputs['options']['estimator']:raise ValueError('Estimator changed')
            if risk.get('q3',{}).get('limit') is not None:raise ValueError('Unexpected target precision filter')
            if not 0<=exact(risk['MSE_lower'])<=exact(risk['MSE_upper']):raise ValueError('Invalid MSE interval')
            for key in ('MSE_lower','MSE_upper','MSE_upper_method','moments','transition_allowance'):
                summary[key]=risk.get(key)
            summary['configuration_id']=record['configuration_id']
            summary['original_configuration_id']=case['original_configuration_id']
            summary['binding_id']=case['binding_id']
            summary['lemma']=risk['bounds'].get('lemma')
            summary['upper_candidates']=risk['bounds'].get('upper_candidates')
            summary['partial_decoder_reference']=record['diagnostic']['partial_decoder_risk']
            if dp:
                for key in ('status','MSE_lower','MSE_upper','MSE_exact','retained_MSE_contribution',
                    'residual_probability','residual_loss_contribution_upper','residual_dropped_states',
                    'residual_principal_guard_failed_probability','transition_pairs','any_zero','partial_decoder',
                    'max_observed_transition_excess_rad','rounds'):
                    summary.setdefault('total_state_dp',{})[key]=dp.get(key)
                residual=exact(dp['residual_probability']);summary['tracked_mass']=rational(1-residual)
                if not 0<=residual<=1:raise ValueError('Residual outside probability range')
                if exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])!=exact(dp['MSE_upper']):
                    raise ValueError('State residual loss not conserved')
                if dp['transition_pairs']>141559036:raise ValueError('Finite transition scope exceeded')
                moments=dp['moments'];blo,bhi=map(exact,moments['bias_squared']);vlo,vhi=map(exact,moments['variance'])
                lower,upper=exact(dp['MSE_lower']),exact(dp['MSE_upper'])
                if not (0<=blo<=bhi and 0<=vlo<=vhi and max(lower,blo+vlo)<=min(upper,bhi+vhi)):
                    raise ValueError('Bias squared + variance intervals inconsistent with MSE')
                identity='EXACT_EQUALITY' if dp['status']=='COMPLETE_TOTAL_STATE_DP' else 'INTERVAL_CONSISTENCY_ONLY_DEPENDENT_TERMS'
                if dp['status']=='COMPLETE_TOTAL_STATE_DP' and not lower==upper==blo+vlo:
                    raise ValueError('Exact MSE != bias squared + variance')
                summary['bias_variance_consistency']=identity
            summary.update(sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)
            checks.append(dict(label=label,canonical_vector_loss_intervals='PASS',
                               sufficient_N_and_cost_remain_null=True,precision_filter=False))
        summaries.append(summary)
    protected=('src/trottertracks/rpe_total_decoder.py','src/trottertracks/rpe_total_risk.py',
        'src/trottertracks/rpe_total_binding.py','src/trottertracks/rpe_deterministic_risk.py',
        'src/trottertracks/rpe_decoder_fast.py','src/trottertracks/pr_rpe_rmse_controller.py','scripts/run_h4_dp_risk_fixed.py')
    for path in protected:
        fixed=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
        if (ROOT/path).read_bytes()!=fixed:raise ValueError('Protected scientific SOURCE changed: '+path)
    if subprocess.check_output(['git','diff','e7ce23803b5644bd9df0e83c877b390aeb9405e0',
        '--name-only','--','docs/research/h4_dp_risk_fixed_results_20261114/'],cwd=ROOT).strip():
        raise ValueError('Old DP results changed')
    summary=dict(status=completion['status'],source=execution['source'],B_source=BASE,
        cases=summaries,checks=checks,completion=completion,old_DP_source_preserved=True,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    write_new(OUTPUT/'SUMMARY.json',summary)
    text=['# 保存Pauli LD1/LD2のZ1＋T1/Q3直接モデルMSE診断','',
        '承認されたH4/1.00 Å/STO-3G/DF10・Pauli LD1/LD2、各1 control vectorのみ。',
        '固定SOURCE・条件は[INPUTS](INPUTS.json)と[EXECUTION_SCOPE](EXECUTION_SCOPE.md)。新しい探索・MC・価格/Gなし。','',
        '| 構成 | status | モデルMSE下界 Ha² | モデルMSE上界 Ha² | 採用方法 |',
        '|---|---|---:|---:|---|']
    for row in summaries:
        text.append('| '+row['label']+' | '+row['status']+' | '+display(row.get('MSE_lower'))+' | '+
            display(row.get('MSE_upper'))+' | '+str(row.get('MSE_upper_method',row.get('reason','UNKNOWN')))+' |')
    for row in summaries:
        text+=['','## '+row['label'],'']
        if 'MSE_upper' not in row:
            text+=['UNKNOWN理由：'+str(row.get('reason'))+'。部分証拠はrun01を参照。無断retryはしない。'];continue
        dp=row.get('total_state_dp',{});mom=row.get('moments') or {}
        text+=['- canonical：`'+row['configuration_id']+'`。元/新binding：`'+row['binding_id']+'`。',
            '- bias²区間：'+str([display(v) for v in mom.get('bias_squared',[])])+' Ha²。',
            '- variance区間：'+str([display(v) for v in mom.get('variance',[])])+' Ha²。',
            '- bias²＋Var整合：'+str(row.get('bias_variance_consistency'))+'。partial区間の端点和を等号で同定しない。',
            '- lemma：'+str((row.get('lemma') or {}).get('status'))+'。各first-unsafeの主値guard/capはSUMMARYのlemmaに保存。',
            '- total DP：'+str(dp.get('status'))+'。遷移組数 '+str(dp.get('transition_pairs'))+'。',
            '- 追跡mass '+display(row.get('tracked_mass'))+'、残余mass '+display(dp.get('residual_probability'))+'。',
            '- 解決終端の追跡loss '+display(dp.get('retained_MSE_contribution'))+'、残余loss上界 '+display(dp.get('residual_loss_contribution_upper'))+' Ha²。',
            '- any-zero確率 '+display(dp.get('any_zero'))+'。各roundのzero保持/tie argminのtracked確率はDP.roundsに保存。',
            '- total decoderはzero/tieにもenergyを返す。旧部分decoderの未解決確率診断は別fieldであり、total推定量の未解決eventとして扱わない。',
            '- zero/tieのeventに独立のloss寄与を割り当てて合算する分解は固定kernelにない。tracked event確率・旧部分decoder unresolved boundsとtotal lossを分けて報告する。',
            '- 性能：'+json.dumps(row.get('performance'),ensure_ascii=False)+'。']
    text+=['','## 資格・限界とSTOP','',
        '保存solver energy −2.1663874486347634 Haを条件付きproxyとし、保存ideal finite-K raw・IID・EXACT_AFFINEの下で評価したモデル診断だけである。',
        '元packetのground未認証・実sampler law移送未認証・数値certificateなしを保持。binary64 transition allowanceは未認証engineering allowance。',
        '元Minkowski契約は参照として保全し、PF/finite-Kの位相ずれを含むraw直接riskへAのδEを追加していない。',
        '目標energy精度に対する追加filterを行っていない。control vectorはnominal/十分N/最適Nではない。sufficient_N/global_RMSE_bound/G/G_qualified/winnerはnull。','',
        '[SUMMARY](SUMMARY.json)にexact有理数と診断、[run01](run01/)に元record・部分証拠・資源観測を保存。',
        '次：ユーザーが結果に基づく研究方針と追加の有限範囲を判断し、Codex Bは別途明示許可後だけ進める。新しいshot探索・価格/G・正式採択へ自動進行しない。',
        '**mandatory STOP**。']
    with (OUTPUT/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    with (OUTPUT/'HANDOFF.md').open('x') as f:
        f.write('# 固定2診断HANDOFF\n\n[REPORT](REPORT.md)／[exact SUMMARY](SUMMARY.json)／[INPUTS](INPUTS.json)。\n\n')
        f.write('対象はPauli LD1/LD2各1vector、total推定量opt-in。N producerや資格証拠は作っていない。\n')
        f.write('C/Dへ新しい価格取得を依頼せず、Gの重みとして十分Nを発行しない。元receipt/価格/SOURCEは不変。\n')
        f.write('次はユーザーが必要な研究判断・科学範囲を指定する。Codex Bはmandatory STOP。\n')
    return summary


def publish(summary):
    current=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if current!=BRANCH:raise ValueError('Own publication branch changed')
    relative=Path(DIRECTORY)/'REPORT.md'
    for path,target in [('PROJECT_MAP.md',str(relative)),('docs/README.md',str(relative).removeprefix('docs/')),
                        ('docs/research/研究概要・現状.md',str(relative).removeprefix('docs/research/'))]:
        file=ROOT/path;text=file.read_text();head,_,rest=text.partition('\n')
        line='[固定Pauli total/Q3 2診断]('+target+')：'+summary['status']+'。1vector×LD1/LD2のみ、モデル内上下界・moments・残余を保存。十分N/G未認定、旧DP不変・STOP。'
        file.write_text(head+'\n\n'+line+'\n'+rest)
    manifest=ROOT/'artifacts/validation_manifest.json';original=manifest.read_text()
    entry=dict(status=summary['status'],report_path=str(relative),summary=DIRECTORY+'/SUMMARY.json',
        execution_source=summary['source'],evaluations=2,claim='CONDITIONAL_OPT_IN_RAW_MODEL_ONLY',
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,mandatory_stop=True)
    fragment=json.dumps({'pauli_total_q3_fixed_results_20261114':entry},ensure_ascii=False,indent=2)
    manifest.write_text('{\n'+fragment[2:-2]+',\n'+original[2:]);json.loads(manifest.read_bytes())
    note='docs/research/研究ノート/2026-11-14_pauli_total_q3_fixed.md'
    with (ROOT/note).open('a') as stream:
        stream.write('\n固定2評価終了：'+summary['status']+'。[REPORT](../pauli_total_q3_fixed_results_20261114/REPORT.md)とexact SUMMARYを保存。十分N/価格/G/精度filterは実施せず、公開後mandatory STOP。\n')
    paths=[DIRECTORY,'PROJECT_MAP.md','docs/README.md','docs/research/研究概要・現状.md','artifacts/validation_manifest.json',note]
    subprocess.run(['git','add','--sparse','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','user.name=HIROMU1015','-c','user.email=u411612g@ecs.osaka-u.ac.jp',
        'commit','-m','Report two fixed Pauli total-Q3 model diagnostics and stop [skip ci]'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=ROOT,check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=ROOT,text=True).split()[0]
    if sha!=remote:raise ValueError('Remote SHA mismatch')
    subprocess.run(['git','fetch','--no-tags','origin','refs/heads/'+BRANCH],cwd=ROOT,check=True)
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    if tree!=subprocess.check_output(['git','rev-parse','FETCH_HEAD^{tree}'],cwd=ROOT,text=True).strip():
        raise ValueError('Remote tree mismatch')
    for path in (str(relative),DIRECTORY+'/SUMMARY.json',DIRECTORY+'/INPUTS.json',
                 DIRECTORY+'/run01/RESULT_LD1.json',DIRECTORY+'/run01/RESULT_LD2.json'):
        local=subprocess.check_output(['git','rev-parse','HEAD:'+path],cwd=ROOT)
        remote=subprocess.check_output(['git','rev-parse','FETCH_HEAD:'+path],cwd=ROOT)
        if local!=remote:raise ValueError('Remote result blob mismatch')
    print(json.dumps(dict(status='PUBLISHED_AND_REMOTE_VERIFIED',sha=sha,tree=tree,mandatory_stop=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true')
    parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    while args.wait and not (OUTPUT/'run01/COMPLETION.json').exists():time.sleep(30)
    summary=collect()
    if args.publish:publish(summary)
