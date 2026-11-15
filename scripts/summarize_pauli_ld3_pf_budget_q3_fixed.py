"""Publish saved finite-menu results and reference-cost joins; no Q3 kernel."""
import argparse
import csv
from decimal import Decimal,localcontext
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_ld3_budget_summary',ROOT/'scripts/run_pauli_ld3_pf_budget_q3_fixed.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)


def display(q,precision=28):
    q=C.rational(q)
    with localcontext() as ctx:
        ctx.prec=precision
        return str(Decimal(q.numerator)/Decimal(q.denominator))


def read(p):return json.loads(p.read_text())


def summarize(snapshot):
    p=ROOT/C.DIRECTORY;run=p/'run01';inputs=read(p/'INPUTS.json');history=read(run/'SEARCH_HISTORY.json')
    source=read(run/'EXECUTION.json')['source'];returned=read(p/'PROCESS_RETURN.json')
    if returned['returncode']!=0:raise ValueError('Caller process failure; preserve evidence')
    def published(path):
        relative=str(path.relative_to(ROOT))
        if C.R.blob(snapshot,relative)!=path.read_bytes():raise ValueError('Result snapshot differs: '+relative)
        return dict(commit=snapshot,path=relative)
    rows=[];joins=[]
    old_binding=C.document(dict(commit=C.BASE,path=C.L.DIRECTORY+'/TOTAL_BINDING.json'))
    for i,case in enumerate(inputs['cases'],1):
        prior=next((r for r in history['history'] if r['label']==case['label']),None)
        row=dict(label=case['label'],aliases=case['aliases'],budget_name=case['budget_name'],budget=case['budget'],
            N_X=[s['N_X'] for s in case['allocation']['shots']],N_Y=[s['N_Y'] for s in case['allocation']['shots']],
            allocation=case['allocation'],conditional_cost=case['conditional_cost'],unused_budget=case['unused_budget'],
            conditional_cost_display_RZ=display(case['conditional_cost']),unused_budget_display_RZ=display(case['unused_budget']),
            difference_T_minus_P7=C.exact(C.rational(case['conditional_cost'])-C.BUDGETS[0][1]),
            difference_T_minus_Full_PF=C.exact(C.rational(case['conditional_cost'])-C.BUDGETS[1][1]),
            status='NOT_EVALUATED',reason=history['stop_reason'],MSE_lower=None,MSE_upper=None,
            result_source=None,new_scientific_calls=0,price_model_axes=26,price_qualified=False)
        if prior is not None:
            row.update(status=prior['status'],reason=prior.get('reason'),MSE_lower=prior.get('MSE_lower'),MSE_upper=prior.get('MSE_upper'),
                new_scientific_calls=prior['scientific_evaluation_calls'],disposition=prior.get('input_disposition','UNKNOWN'))
            if prior.get('input_disposition')=='REUSED':
                row['result_source']=prior['API_source']
            else:
                folder=run/f'{i:02}'/case['label'];api=folder/'API_RETURN.json'
                row['result_source']=published(api) if api.exists() else None
                row['result_record_source']=published(folder/'RESULT.json')
                row['resources']=read(folder/'RESOURCE_FINAL.json')
                row['operational_status']=read(folder/'PERSISTED.json')['operational_status']
                supervision=read(folder.parent/'SUPERVISION.json')
                row['resources'].update(sampled_process_tree_peak_RSS_KiB=supervision['sampled_process_tree_peak_RSS_KiB'][case['label']],
                    process_exit_gate=supervision['process_exit_gate'])
            if row['MSE_upper'] is not None and row['status']!='UNKNOWN':
                row['MSE_lower_display_Ha2']=display(row['MSE_lower']);row['MSE_upper_display_Ha2']=display(row['MSE_upper'])
                record=C.document(row['result_source'])[0]
                if record['allocation']['shots']!=case['allocation']['shots'] or record['configuration_id']!=case['total_configuration_id']:
                    raise ValueError('Saved result vector/canonical join mismatch')
                risk=record['diagnostic']['deterministic_risk']
                if C.L.H.assess(risk,inputs['adopted_evaluation'],source)['status']!=row['status']:
                    raise ValueError('Saved threshold classification differs')
                dp=risk['bounds']['total_state_dp']
                row['diagnostics']=dict(moments=risk['moments'],residual_mass=dp['residual_probability'],
                    residual_loss_upper=dp['residual_loss_contribution_upper'],any_zero=dp['any_zero'],
                    transition_pairs=dp['transition_pairs'],transition_allowance=dp['transition_allowance'])
        rows.append(row)
        axes=[]
        for term in case['price_terms']:
            m=term['m'];axis=term['axis']
            axes.append(dict(**term,physical_context_source=dict(commit=C.BASE,
                path=C.L.DIRECTORY+'/C_RESULT_JOIN.json',selector=['rows',2*m+(axis=='Y')]),
                original_law_binding=old_binding['original_configuration']['rounds'][m]['law_binding'],
                total_law_binding=old_binding['total_configuration']['rounds'][m]['law_binding']))
        joins.append(dict(label=case['label'],original_configuration_id=case['original_configuration_id'],
            total_configuration_id=case['total_configuration_id'],binding_id=case['binding_id'],
            original_configuration_source=case['configuration'],signal_source=case['signal_packet'],
            allocation=case['allocation'],allocation_source=dict(commit=source['commit'],path=C.INPUT,selector=['cases',i-1,'allocation']),
            evaluation_target=case['evaluation_target'],evaluation_contract_source=dict(commit=source['commit'],path=C.INPUT),
            status=row['status'],result_source=row['result_source'],result_record_selector=[0],
            unevaluated_or_unknown_reason=row['reason'],
            axes=axes,conditional_reference_total=case['conditional_cost'],price_source=C.PRICE,
            price_qualified_for_candidate=False,E_C=None,CI=None,G=None,G_qualified=None,sufficient_N=None))
    best=next((r for r in rows if r['label']==history['best_confirmed']),None)
    vp=C.cached_record(inputs['seed_case'],inputs)
    preserved_vp=dict(allocation=inputs['seed_case']['allocation'],source=vp['API_source'],status=vp['status'],
        MSE_lower=vp['MSE_lower'],MSE_upper=vp['MSE_upper'],conditional_cost=inputs['seed_cost'],
        difference_T_minus_P7=C.exact(C.rational(inputs['seed_cost'])-C.BUDGETS[0][1]),
        difference_T_minus_Full_PF=C.exact(C.rational(inputs['seed_cost'])-C.BUDGETS[1][1]),new_scientific_calls=0)
    summary=dict(execution_source=source,result_snapshot_commit=snapshot,budgets=inputs['budgets'],
        proposals=inputs['proposals'],rows=rows,stop_reason=history['stop_reason'],
        new_scientific_calls=history['new_scientific_calls'],best_confirmed=history['best_confirmed'],
        minimal_within_generated_set=history['minimal_within_generated_set'],cheaper_unresolved=history['cheaper_unresolved'],
        achieved_conditional_cost=history['achieved_conditional_cost'],seed_cost=inputs['seed_cost'],
        exact_threshold=inputs['adopted_evaluation']['MSE_upper'],price_source=C.PRICE,preserved_VP=preserved_vp,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    C.R.write_new(p/'SUMMARY.json',summary)
    C.R.write_new(p/'C_COMPARISON_JOIN.json',dict(execution_source=source,result_snapshot_commit=snapshot,
        records=joins,kind='CONDITIONAL_MODEL_REFERENCE_COST_AND_Q3_NOT_QUALIFIED_G',mandatory_stop=True))
    fields=['label','N_X','N_Y','conditional_cost_exact','conditional_cost_display_RZ','unused_budget_exact',
            'MSE_lower_display_Ha2','MSE_upper_display_Ha2','status','new_scientific_calls','reason']
    with (p/'SUMMARY.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow({k:r.get(k,'') for k in fields if k not in ('conditional_cost_exact','unused_budget_exact')} |
                dict(conditional_cost_exact=str(C.rational(r['conditional_cost'])),unused_budget_exact=str(C.rational(r['unused_budget']))))
    lines=[]
    for r in rows:
        lines.append(f"| {r['label']} | {display(r['conditional_cost'],16)} | {display(r['unused_budget'],16)} | "
            f"{r.get('MSE_lower_display_Ha2','—')} | {r.get('MSE_upper_display_Ha2','—')} | {r['status']} |")
    vectors='\n'.join(f"- `{r['label']}`：X = `{r['N_X']}`、Y = `{r['N_Y']}`。exact費用 `{C.rational(r['conditional_cost'])}`、予算残額 `{C.rational(r['unused_budget'])}` RZ。" for r in rows)
    answer=('適合点なし。保存VPの参照達成費用173557305044/165（1,051,862,454.81 RZ表示）を維持する。' if best is None else
        f"{best['label']}が適合。条件付き参照費用は{display(best['conditional_cost'],16)} RZ（exact {C.rational(best['conditional_cost'])}）。")
    if best is not None:
        answer+='今回生成した4列内の最安適合点。' if history['minimal_within_generated_set'] else '安い判定保留列があるため確認済み最安。'
        answer+=f" p7基準との差T−395,962,678は{display(best['difference_T_minus_P7'],16)} RZ、Full PF基準との差T−559,457,386は{display(best['difference_T_minus_Full_PF'],16)} RZ。"
    text=f'''# PR-Pauli LD3：PF達成費用予算での有限shot配分評価

{answer}

新規Q3は{history['new_scientific_calls']}件。停止理由：`{history['stop_reason']}`。追加信号、sampling、MC、compile、価格再fit、retryはいずれも0件。mandatory STOP。

## 固定4列と結果

H4／1.00 Å／STO-3G／tol-only DF10／sector(2,2)、PR-Pauli LD3。元ID dd5b881d…、total ID 6757998a…。保存H/state/snapshot、LD3辞書・law・共通κ・全13round・raw/normalization/scalarは不変。モデル参照価格は全26軸で適用されている。費用はP=0のfull controlled Hadamard wrapperの参照価格による条件付き算術であり、E[C]や正式Gではない。

| 順序・配分 | 条件付き費用 RZ（表示） | 予算残額 RZ（表示） | MSE下界 Ha² | MSE上界 Ha² | 判定 |
|---|---:|---:|---:|---:|---|
{chr(10).join(lines)}

未評価列の理由はSUMMARYに保存。上界<=exact τのみ適合、下界>τはモデル不適合、区間がまたげばINCONCLUSIVE、gate不成立はUNKNOWN。安い保留列：{history['cheaper_unresolved']}。不適合の結論は評価した各配分に限り、MSEのshot単調性やLD3全体の不適合を主張しない。

### X/Yの具体列・exact会計

{vectors}

保存VP（新規0call）のモデルMSE区間は[{display(vp['MSE_lower'])}, {display(vp['MSE_upper'])}] Ha²で適合。保存VP費用は173557305044/165 RZ、p7基準との差T−395,962,678は{display(preserved_vp['difference_T_minus_P7'],16)} RZ、Full PF基準との差T−559,457,386は{display(preserved_vp['difference_T_minus_Full_PF'],16)} RZ。元のSOURCE・結果・旧STOPは保全した。

VPを各軸上限として、各予算で既存A_SHAPE（重みVP）とB_PRICE（重みVP/C_ref）の正の有理数breakpoint規則を各1本適用した。4列とも別vectorで、SOURCE固定前に生成・順序・exact費用・残額を保存した。個別roundのriskを独立最適化せず、各列を全13round一組でQ3評価した。結果による配分変更はしていない。

## SOURCE・資格・保全

科学実行SOURCE `{source['commit']}`。元価格SOURCE f25a05f8a92aab43cfbe30fc80b974054156aac1のHANDOFF_BC.json、信号・評価契約・total view・保存VPは2de33a4f4。具体的な入力・26軸積・生成元は[INPUTS](INPUTS.json)、重複照合・finite scopeは[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。有理数生成器e094b2d60のpure関数、修正済みLD3 metadata/observer、既存generic workerと高速DPを再利用した。作成者自身の人工検査であり独立レビューではない。

原API/partials/RESULT/RESOURCEはworker自身で保存し、親monitorから独立。科学結果snapshot `{snapshot}` は再実行せず保全した。各callのclosure/mass/residual/bounds/moments/scopeを既存validatorで検査し、runtime失敗はUNKNOWNで停止する。CPU・RSS・wall・子OS終了code/signalと、数値判定・保存状態をSUMMARY/原記録で区別した。

Z1+T1 total decoder、保存solver energy −2.1663874486347634 Haへの全事象二乗loss、保存ideal finite-K rawを条件とするIID、EXACT_AFFINE_OF_BINARY64_RAW、exact τ=(0.00159360010199040 Ha)²を維持。binary64 allowanceは厳密certificateではない。物理ground資格、実sampler law移送、十分N/global RMSE/G_qualifiedは未認定。AのδEは加算していない。

全26軸はMODEL_REFERENCE_PRICE_NOT_E_C、qualified_for_candidate=false、CIなしを保持。較正q8/R660から各roundへの適用はAのモデルの範囲・未検証外挿を引き継ぐ。Q3適合は価格資格を与えない。指定PF基準も保存固定配分での条件付き達成費用であり、方式を最適化した場合の優劣・大域最適・正式winnerへ一般化しない。旧VP・旧予算・旧STOPは変更していない。

[C_COMPARISON_JOIN](C_COMPARISON_JOIN.json)にconfiguration・allocation・Q3原API・26軸価格・exact総額を結合した。ロードマップ本体は編集せず、影響は[HANDOFF](HANDOFF.md)。
'''
    (p/'REPORT.md').write_text(text)
    handoff=f'''# C・統合担当へのHANDOFF：LD3 PF予算の有限配分

{answer}

固定4列の表・X/Y・exact予算残額・MSE上下界は[REPORT](REPORT.md)と[SUMMARY](SUMMARY.json)。新規Q3 {history['new_scientific_calls']}件、終了理由`{history['stop_reason']}`、安い保留列{history['cheaper_unresolved']}。未評価列は{history['unevaluated']}。

## 機械可読引渡し

- `C_COMPARISON_JOIN.json::records`：各configuration（元dd5b881d…/total6757998a…）・binding・allocation/source・全26軸N/C_ref/積・元/total candidate/law・Q3結果SOURCE・exact総額。各recordの`result_source`は公開snapshot `{snapshot}`内の原APIへ解決可能。未評価はnullと理由を保持する。
- `INPUTS.json`：科学実行SOURCE `{source['commit']}`で全候補・評価順・価格参照・具体配分・exact費用を実行前固定。
- `run01/SEARCH_HISTORY.json`：実際の順序と判定・再利用/新規・停止。`run01/*/*/API_RETURN.json`、partials、RESOURCE_FINAL、PERSISTED、SUPERVISION、PROCESS_EXITSを保全。
- 固定価格入力：f25a05f8a92aab43cfbe30fc80b974054156aac1 / docs/research/a_pauli_ld3_cal_price_20261116/HANDOFF_BC.json。

価格は全26軸がモデル参照値。`E_C/CI/G/G_qualified/sufficient_N`はnull、価格qualified_for_candidate=false。Cの正式G guardを緩めず、条件付き参照費用とriskを別の資格で受け入れること。価格cell receiptをround receiptへ付け替えない。

## ロードマップへの差分

LD3に、現在PF達成費用を予算としたA_SHAPE/B_PRICEの有限4列のQ3・条件付き費用証拠が追加された。結論は今回の生成集合・採択Q3モデル・A参照価格の範囲。新結果が適合でも正式十分N、物理的精度保証、E[C]、G、方式全体のwinnerは認定しない。旧VPやLD2の既評価配分を再取得する必要はない。価格domainの追加検証や別配分・別候補探索が必要なら別指示。ロードマップ本体は統合担当が更新する。

新signal/sampling/MC/compile/fit/retry各0件。mandatory STOP。
'''
    (p/'HANDOFF.md').write_text(handoff)
    print(answer)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result-commit',required=True)
    summarize(parser.parse_args().result_commit)
