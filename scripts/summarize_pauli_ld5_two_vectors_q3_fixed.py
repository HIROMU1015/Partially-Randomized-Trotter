"""Quote LD3 and join the two saved LD5 Q3 returns to exact own prices."""
import argparse
import csv
from decimal import Decimal,localcontext
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_ld5_four_point_summary',ROOT/'scripts/run_pauli_ld5_two_vectors_q3_fixed.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
Q=C.C


def read(p):return json.loads(p.read_text())


def display(value,precision=28):
    q=value if isinstance(value,Fraction) else Q.rational(value)
    with localcontext() as ctx:
        ctx.prec=precision
        return str(Decimal(q.numerator)/Decimal(q.denominator))


def summarize(snapshot):
    snapshot=subprocess.check_output(['git','rev-parse',snapshot],cwd=ROOT,text=True).strip()
    p=ROOT/C.DIRECTORY;run=p/'run01';inputs=read(p/'INPUTS.json');history=read(run/'RESULT_INDEX.json')
    source=read(run/'EXECUTION.json')['source'];binding=read(p/'TOTAL_BINDING.json');view=read(p/'TOTAL_SIGNAL_VIEW.json')
    tau=Q.rational(inputs['adopted_evaluation']['MSE_upper']);rows=[];joins=[]
    if read(p/'PROCESS_RETURN.json')['returncode']!=0:raise ValueError('Caller exit abnormal; preserve evidence')
    def published(path):
        relative=str(path.relative_to(ROOT))
        if C.R.blob(snapshot,relative)!=path.read_bytes():raise ValueError('Raw snapshot differs')
        return dict(commit=snapshot,path=relative)
    for i,case in enumerate(inputs['cases'],1):
        prior=next((r for r in history['history'] if r['label']==case['label']),None)
        pair=[]
        for prefix in (3,5):
            if prefix==3:
                ref=case['LD3_comparison']['API_source'];record=Q.document(ref)[0]
                cost=case['LD3_comparison']['conditional_cost'];risk=record['diagnostic']['deterministic_risk']
                status=C.L.H.assess(risk,inputs['adopted_evaluation'],ref)['status']
                if (record['configuration_id']!=case['LD3_comparison']['total_configuration_id']
                        or record['allocation']['shots']!=case['allocation']['shots']
                        or record['evaluation_target']!=case['evaluation_target']):raise ValueError('LD3 comparison contract/vector differs')
            else:
                cost=case['conditional_cost'];ref=None;record=None;risk=None
                status='NOT_EVALUATED' if prior is None else prior['status']
                if prior is not None:
                    folder=run/f'{i:02}'/case['label'];api=folder/'API_RETURN.json'
                    ref=published(api) if api.exists() else None
                    if ref is not None:
                        record=Q.document(ref)[0]
                        if record['configuration_id']!=case['total_configuration_id'] or record['allocation']['shots']!=case['allocation']['shots']:
                            raise ValueError('LD5 result canonical/vector differs')
                        if status!='UNKNOWN':
                            risk=record['diagnostic']['deterministic_risk']
                            if C.L.H.assess(risk,inputs['adopted_evaluation'],source)['status']!=status:
                                raise ValueError('Saved classification differs')
            row=dict(prefix=prefix,allocation_name=case['comparison_name'],status=status,result_source=ref,
                N_X=[r['N_X'] for r in case['allocation']['shots']],N_Y=[r['N_Y'] for r in case['allocation']['shots']],
                conditional_cost=cost,conditional_cost_display_RZ=display(cost),MSE_lower=None,MSE_upper=None,
                threshold_headroom=None,qualified_for_candidate=False,E_C=None,G=None,G_qualified=None,sufficient_N=None)
            if risk is not None:
                lo,hi=Q.rational(risk['MSE_lower']),Q.rational(risk['MSE_upper'])
                row.update(MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],
                    MSE_lower_display_Ha2=display(lo),MSE_upper_display_Ha2=display(hi),
                    MSE_minus_threshold=[Q.exact(lo-tau),Q.exact(hi-tau)],threshold_headroom=Q.exact(tau-hi),
                    threshold_headroom_display_Ha2=display(tau-hi))
            if prefix==5:
                row['new_scientific_calls']=0 if prior is None else prior['scientific_evaluation_calls']
                row['reason']=history['stop_reason'] if prior is None else prior.get('reason')
                if prior is not None:
                    row['resources']=read(folder/'RESOURCE_FINAL.json')
                    sup=read(folder.parent/'SUPERVISION.json')
                    row['resources'].update(process_exit_gate=sup['process_exit_gate'],
                        sampled_process_tree_peak_RSS_KiB=sup['sampled_process_tree_peak_RSS_KiB'][case['label']])
                    row['operational_status']=read(folder/'PERSISTED.json')['operational_status']
                if risk is not None:
                    dp=risk['bounds']['total_state_dp']
                    row['diagnostics']=dict(moments=risk['moments'],residual_mass=dp['residual_probability'],
                        residual_loss_upper=dp['residual_loss_contribution_upper'],any_zero=dp['any_zero'],
                        rounds=dp['rounds'],transition_pairs=dp['transition_pairs'],transition_allowance=dp['transition_allowance'])
                terms=[]
                for term in case['price_terms']:
                    m=term['m'];old=binding['original_configuration']['rounds'][m];new=binding['total_configuration']['rounds'][m]
                    terms.append(dict(**term,original_law_binding=old['law_binding'],total_law_binding=new['law_binding'],
                        physical_context_source=dict(commit=source['commit'],path=C.DIRECTORY+'/TOTAL_SIGNAL_VIEW.json',
                            selector=['means',m,'canonical_origin','provider_binding','physical_context']),
                        signal_source=view['means'][m]['source'],implementation_source=view['means'][m]['implementation_source']))
                if sum(Q.rational(t['N_times_C_ref']) for t in terms)!=Q.rational(cost):raise ValueError('Exact 26-axis total differs')
                joins.append(dict(original_configuration_id=C.CID,total_configuration_id=case['total_configuration_id'],
                    binding_id=case['binding_id'],configuration_source=case['configuration'],signal_packet_source=case['signal_packet'],
                    dictionary_identity=view['dictionary_identity'],allocation=case['allocation'],
                    allocation_source=dict(commit=source['commit'],path=C.INPUT,selector=['cases',i-1,'allocation']),
                    axes=terms,conditional_total=cost,A_comparison_source=case['A_comparison_source'],
                    result_source=ref,result_record_selector=[0],status=status,reason=row['reason'],
                    threshold=inputs['adopted_evaluation']['MSE_upper'],threshold_headroom=row['threshold_headroom'],
                    price_source=C.PRICE,price_kind='MODEL_REFERENCE_PRICE_NOT_E_C',model_axes=26,
                    qualified_for_candidate=False,E_C=None,CI=None,G=None,G_qualified=None,sufficient_N=None))
            pair.append(row);rows.append(row)
    summary=dict(execution_source=source,result_snapshot_commit=snapshot,rows=rows,
        new_scientific_calls=history['new_scientific_calls'],stop_reason=history['stop_reason'],
        exact_threshold=inputs['adopted_evaluation']['MSE_upper'],threshold_headroom_definition='tau - valid MSE upper; positive means model threshold met',
        global_RMSE_bound=None,sufficient_N=None,E_C=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    C.R.write_new(p/'SUMMARY.json',summary)
    C.R.write_new(p/'C_COMPARISON_JOIN.json',dict(execution_source=source,result_snapshot_commit=snapshot,
        records=joins,kind='LD5_FIXED_ALLOCATION_MODEL_COST_AND_Q3_NOT_QUALIFIED_G',mandatory_stop=True))
    fields=['prefix','allocation_name','conditional_cost_display_RZ','MSE_lower_display_Ha2','MSE_upper_display_Ha2',
            'threshold_headroom_display_Ha2','status']
    with (p/'SUMMARY.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in rows)
    lines='\n'.join(f"| LD{r['prefix']} | {r['allocation_name']} | {display(r['conditional_cost'],16)} | "
        f"{r.get('MSE_lower_display_Ha2','—')} | {r.get('MSE_upper_display_Ha2','—')} | {r.get('threshold_headroom_display_Ha2','—')} | {r['status']} |" for r in rows)
    outcomes='、'.join(f"{r['allocation_name']}={r['status']}" for r in rows if r['prefix']==5)
    report=f'''# PR-Pauli LD5：LD3と共通の固定2配分・total/Q3

LD5のtotal viewを作成し、新規Q3 {history['new_scientific_calls']}件。{outcomes}。停止理由`{history['stop_reason']}`。LD3の2点は保存引用のみ。新規signal/sampling/MC/compile/価格再fit/新配分生成/retryは各0件。mandatory STOP。

## 4点比較

H4／1.00 Å／STO-3G／tol-only DF10／sector(2,2)。LD3/LD5それぞれ自身のhead/tail辞書、ideal finite-K raw、κ/scheduleを使用。共通Z1+T1 total decoder、保存full DF10 solver proxy −2.1663874486347634 Ha、全事象二乗loss、shots/X/Y/round IIDモデル。exact τ=(0.00159360010199040 Ha)²=0.00000253956128506381328204169216 Ha²。

| 構成 | 配分 | 条件付きモデル費用 RZ | MSE下界 Ha² | MSE上界 Ha² | 閾値余裕 τ−上界 Ha² | 判定 |
|---|---|---:|---:|---:|---:|---|
{lines}

exact判定は原有理数を使用。余裕の正/負を明示し、UNKNOWN（gate不成立）とモデル不適合（有効下界>τ）、INCONCLUSIVE（有効区間がまたぐ）を区別する。上表と[SUMMARY](SUMMARY.json)に原APIの解決可能な固定commit/pathを保存した。

- FULL_PF__B_PRICE：X/Y各`[48,45,42,39,36,33,29,26,23,19,16,8,2]`。
- VP：X/Y各`[48,45,42,39,36,33,29,26,23,19,16,13,9]`。

固定順序で①を評価後、適合/不適合を理由に早期終了せず②を評価した。新しい配分は生成していない。これらの列を十分N、最小N、方式全体の優劣とは認定しない。

## total view・SOURCE・会計

元LD5 ID `{C.CID}` → total ID `{inputs['cases'][0]['total_configuration_id']}`、binding `{inputs['cases'][0]['binding_id']}`。元A packet ac46d100e、config39a1fc7df、辞書6255fbc42、数値SOURCE c67469ad4/原carrier1e3985e7eを保全。Aの8c8ce542e::b_intake_dictionary_identityと既存P1 jobを使い、同じbind_packet/C metadata/total binding経路をLD5の固定SOURCEへ最小拡張した。辞書hash・bit順・残余locator・13 physical ID・各round SOURCE・law・scalar onceを照合し、rawやnormalizationは変更/再適用していない。Aの元total_view=nullは変更せずBのviewとして保存した。

価格SOURCEはb8bcdf97b693fa7e99d3d35f44f5ce6d3a691201のPRICE_26AXIS.json。元rowsのNと積はVP用である。①は①自身のNで26軸積を作り直し、exact総額3792207339/7 RZ、②は6105690627/7 RZとなり、各A.comparison[name].T_LD5とexact一致した。新しい価格予測やfitはしていない。全26軸はモデル適用、E[C]/CI/価格資格は未取得。較正cellの直接receiptをround receiptへ付け替えていない。

実行SOURCE `{source['commit']}` でcaller/INPUTS/total viewを実行前commit/push固定。原科学結果snapshot `{snapshot}`。kernel710aa69b・高速backend9e294778・既存worker自身のAPI/partials/resources保存・修正済み観測contextを再利用。各配分のcount-onlyを確認し、LD5自身のρ1閉包・質量・残余・bounds/moments・principal guard・数値allowanceは各call内で検査した。数値判定と保存状態・CPU/RSS/wall/子exitcodeはSUMMARY/原記録で別々に保全した。人工/回帰testは[TESTS](TESTS.txt)。作成者自身の検査であり独立レビューではない。

保存solverは条件付きproxyで、ground認証falseを保持。ideal IIDから実samplerへの移送は未認証。binary64 allowanceは厳密certificateではない。物理的全RMSE、十分N、正式E[C]/G/winnerへ昇格せず、A δEは二重加算していない。価格差もモデル参照値の差である。元SOURCE/raw/旧STOPとLD3の保存2点は変更せず、追加評価へ進まない。

[C_COMPARISON_JOIN](C_COMPARISON_JOIN.json)に元/total configuration・allocation ID・26軸N/単価/積・exact総額・Q3判定・原API SOURCEを結合した。ロードマップ本体は更新せず[HANDOFF](HANDOFF.md)で差分を渡す。
'''
    (p/'REPORT.md').write_text(report)
    (p/'HANDOFF.md').write_text(f'''# A/C・統合担当HANDOFF：LD5 total viewと固定2配分

新規Q3 {history['new_scientific_calls']}件、{outcomes}。停止理由`{history['stop_reason']}`。LD3は再評価していない。4点のMSE上下界・τとの差・条件付きモデル費用は[REPORT](REPORT.md)、exact正本は[SUMMARY](SUMMARY.json)。

## 受入用locator

- 科学実行SOURCE `{source['commit']}` / {C.INPUT}。2配分と評価順・target/loss/law/options・価格参照・exact会計を実行前固定。
- 原API固定snapshot `{snapshot}` / `{C.DIRECTORY}/run01/01/LD5__FULL_PF__B_PRICE/API_RETURN.json` と同`run01/02/LD5__VP/API_RETURN.json`（未開始の場合はnullと理由）。
- `C_COMPARISON_JOIN.json::records`：LD5元ID `{C.CID}`、total ID `{inputs['cases'][0]['total_configuration_id']}`、binding、allocation/source、全26軸N/C_ref/積、original/total candidateとlaw、signal/implementation SOURCE、A exact総額の出所、Q3 statusと原APIの固定commit/path。価格kindはMODEL_REFERENCE、qualified_for_candidate=false、E_C/CI/G/G_qualified/sufficient_Nはnull。Cの正式G guardを緩めないこと。
- `TOTAL_BINDING.json`、`TOTAL_SIGNAL_VIEW.json`、`C_SIGNAL_INPUT.json`：A保存packetを保全したmetadata view。新decoder用IDと元physical IDを区別し、元receiptの単純付替えを行わない。
- `run01/RESULT_INDEX.json`、各SELECTION/OUTCOME/API/partials/RESOURCE/PERSISTED/SUPERVISION/PROCESS_EXITS：実行順・数値判定・保存状態・負荷を追跡できる。

価格はb8bcdf97bの26モデル軸。元のN/積はVPなので、FULL_PF__B_PRICEでは新配分のNを使った。両exact総額をA.comparisonに照合済み。LD3の価格・Q3をLD5へ流用していない。

## ロードマップへの影響

LD5の全13signal/26モデル価格に、total decoder接続と共通2配分のQ3証拠が加わる。4点比較は同一decoder/proxy/IID/loss/τの固定配分モデルに限る。各列のmodel適否と、物理的RMSE保証・十分N・実期待価格の資格は別。価格差から正式winnerを認定しない。別配分・価格domain検証・追加科学実行が必要なら別指示。ロードマップ本体は統合担当が更新する。

新規信号/sampling/MC/compile/価格再fit/配分生成/retry各0件。旧成果・旧STOPを保全し、mandatory STOP。
''')
    for row in rows:print('LD'+str(row['prefix']),row['allocation_name'],row['status'],row.get('MSE_upper_display_Ha2'),row['conditional_cost_display_RZ'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result-commit',required=True)
    summarize(parser.parse_args().result_commit)
