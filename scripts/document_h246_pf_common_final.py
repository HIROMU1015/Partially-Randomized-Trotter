"""Document audited saved H246 common-grid results; never launches science."""
from pathlib import Path
import json
from finalize_h6_pf_common_results import ROOT,OUTPUT,META,H6,NAMES,doc


def read(p):return json.loads(Path(p).read_text())


def build():
    out=ROOT/OUTPUT;summary=read(out/'summary.json');comparisons=read(out/'comparisons_60.json')
    meta=read(ROOT/META/'ending_audit.json');h6=read(ROOT/H6/'end_audit.json');receipts=read(ROOT/H6/'stage_receipts.json')
    global_table,within=doc(comparisons)
    changes=read(out/'old_vs_common_15.json');breakdown=read(out/'nested_PF_cost_change_decomposition_10.json')
    oldtable=[]
    for r in changes:
        if r['system']!='H6':continue
        o=r['old'];n=r['new'];oldtable.append(f"| {r['epsilon']:g} | {o['m5_det_subset_method']} p{o['m5_det_subset_prefix']}/q{o['m5_det_subset_q']} | {float(o['best_PR_improvement_vs_m5_det_subset_percent']):+.4f}% | {n['PF_id']} | {n['PR_reduction_percent']:+.4f}% |")
    factors=[]
    for r in breakdown:
        if r['system']!='H6':continue
        factors.append(f"| {r['epsilon']:g} | {float(__import__('fractions').Fraction(r['policy_and_saved_arithmetic_factor_exact'])):.8f} | {float(__import__('fractions').Fraction(r['m5_prefix_expansion_factor_exact'])):.8f} | {float(__import__('fractions').Fraction(r['formula_expansion_factor_exact'])):.8f} |")
    sources=read(out/'source_result_references.json')
    report=f'''# H2/H4/H6共通決定論PF探索：588候補の固定PR/PF費用比較完了

2026-10-22 JST。**H6の不足63 metadata取得と固定276候補の取得・適格性/shot・必要なcompile・会計・終了監査PASS。H2/H4を再計算せず参照結合し、共通588候補すべての評価が揃った。local commit後mandatory STOP、push0。**

以下は保存PR探索標本と有限共通格子PFの点費用比較であり、母平均順位の保証ではない。H6各精度の方向・exact費用は表とmachine-readable比較行を正とする。

## 固定条件・SOURCE

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only DF threshold1e-8、保存rank3/10/19、T=.8、P=0、q=1/2/4/8、δ=.8/.4/.2/.1。S2/Yoshida4/m5_bestの元binary64係数・signed scheduleを維持。full cosine/sine Hadamard wrapper、symmetric-directional control、元scalar一回、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17。backend/coupling/layout未指定。H2既存scalar guardは旧系列のまま、H6に新たに移していない。

| 系 | B0/B0I L_D | Full | 登録数 | 結果SOURCE |
|---|---|---:|---:|---|
| H2 rank3 | 0..2 | 3 | 84 | {sources['H2']['SOURCE']} |
| H4 rank10 | 1..9 | 10 | 228 | {sources['H4']['SOURCE']} |
| H6 rank19 | 5..15 | 19 | 276 | {sources['H6']['SOURCE']} |

H4 DF12、q16、未登録prefix、追加公式は含めない。最良は今回の固定共通格子内の最良。H2 prefix0/Full aliasを推測統合しない。保存PRは各系outer32/K4/prefix1/3/5のEXPLORATION 3公式×4qだけで、pilot・旧cohort・118確認標本を混合しない。新しいPR sampling・統計・単価・shot変更は0。

H6 metadata科学SOURCEは `{meta['SOURCE']}`、共通科学SOURCEは `{h6['SOURCE']}`。起点0b1886f53、専用branch `pf-common-h6-metadata-execution-20261022`。旧H2/H4のSOURCE/raw/grant/ledger/marker/STOPはbyte同一のまま参照結合。旧H6 INPUT_STOPと失敗stageも正式履歴として保全した。

H6原snapshot SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005` がreceipt/bindingと一致。Hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、state hash `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3` を原配列から検査し、保存reference・full preparation・runtime/compiler/provider sealと接続した。H6は元protocolで採用した保存weighted-Hermitian completionのDF Hamiltonianをそのまま使用し、化学入力を再生成した系列ではない。今回SCF/DF/state/reference新生成0。H2/H4は以前に科学配列hash一致を確認した再生成容器であり、原snapshot復元とは区別する。

## 不足63 metadataの正当な取得

旧B0I serializerはscalar信号のみ回転し、途中norm/closureを保存しなかった。保存B0 scalarから中間binary64 vectorを推測せず、旧STOPの63 IDだけを別契約で再観測。元merged S2/Yoshida4 schedule・compact_signal・独立recurrence_pathsを使い、native corrected/rawを別実行、各primitiveのsector leakage<1e-12を検査した。scalar/作用順序/referenceは元のまま。最大normは途中の原観測点を保持し最終B0I位相作用後も観測。closureは保存vectorからnorm(raw−corrected)を計算した実測値で、0補完ではない。

52候補は旧corrected/reference/signed biasをそのまま保存し、新観測だけ結合。原rawが未回収だった11候補は新signalとして明記し、原本復元とは呼ばない。保存u/marginを下回らない。独立監査では全63 archiveの原SHA復元、保存vector closure、旧52値のliteral不変を確認した。

- 旧corrected最大差：{meta['maxima']['corrected_error']:.7e}。
- reference最大差：{meta['maxima']['reference_error']:.7e}。
- signed bias最大差：{meta['maxima']['signed_bias_error']:.7e}。
- 独立forward vector最大差：{meta['maxima']['forward_vector_error']:.7e}。
- 最大途中norm：{meta['maxima']['maximum_intermediate_norm']:.17g}。
- raw/corrected vector closure最大：{meta['maxima']['raw_corrected_vector_closure']}（別実行vectorの実測）。

metadata実使用はnative corrected63/raw63、独立forward63、native primitive build/作用各18736、forward作用9368、native outer scalar476/forward238、最終vector位相作用189、closure63、旧値比較52、独立sector matrix/eigh各16、tail scalar11、snapshot/reference/oracle/full preparation各1。63 worker全exit0・回収、wall約78.99秒。これらは従来102 signal予算とは完全別枠。

## H6取得・科学call・再利用

| 項目 | H6実使用 | 旧固定上限 |
|---|---:|---:|
| 旧観測の再利用 | 111候補 | 保存値のみ |
| 追加metadata | 63候補（旧52+新11） | 別承認63 |
| 本来の新signal | 102候補 | 102 |
| 基礎signal / B0I overlay | 51 / 51 | 51 / 51 |
| 独立forward | 51 | 51 |
| 適格性/shot | 1,380 endpoint | 1,380 |
| 新compile候補/軸 | {h6['new_compile_candidates']} / {h6['new_compile_axes']} | 224 / 448 |
| 同native保存price使用 | {h6['used_saved_compiled_pairs']}候補 | identity照合済みのみ |
| 新PR sampling | 0 | 0 |

H6全276候補・5精度の適格性が確定し、少なくとも1精度で適格な239候補の必要費用が揃った。全精度不適格37候補のcompileは不要。全588候補の評価完了は、不適格候補までcompileした意味ではない。共通段階の新signal全系計{summary['new_common_signal_candidates']}、新compile全系計{summary['new_common_compile_candidates']}候補/{summary['new_common_compile_axes']}軸、適格性2940 endpoint。別metadata予算の新11signalと52再観測は別記録。

H6 SIGNALはsigned S2 piece1419、spectral/forward block作用各30588、prefix preparation9、expm構築448。同一行列/timeのexpm再利用で重複計算だけ削減、作用数・signed順序不変。COSTはsigned piece3334≤3625、native evolution/low/high/独立occupation各209、wrapper/compile/各vector作用418、prefix preparation10、context各1。ledger=reserved counter=receipt、すべてgrant上限内。

旧legacy217候補の価格は今回も同native identityを証明できず再利用0。H6同native保存36 pairのうち30適格pairを使い、残る6は今回不適格。H2/H4保存22を含む使用再利用は全系{summary['same_native_price_pairs_used']} pair。旧profile/model推定を新しい直接実測へ代用しない。

## 共通政策と費用の定義

通常signed bias、ε=.05/.01/.005/.001/.0005、allowance倍率1、α_axis=.025。u=max(floor1e-9,10closure+64eps(1+norm)(1+stages),saved u)。log_B marginはfloor1e-12とroundoff式・saved margin以上、u/headroom≤.01。凍結binary64 shotと基本演算を段階ごとbinary64へ丸めるDecimal経路で照合。libm値は共有し、独立超越関数実装の照合とはしない。経験的allowanceはreference条件付きで厳密certificateではない。

G=N_cos×RZ_cos+N_sin×RZ_sin、状態準備費用P=0。PFは整数の決定論的会計、PRは保存exact有理数mean、不偏分散、paired cosine/sine共分散込みSEとnをそのまま参照。表の±は1 SEで母平均CIではない。量子測定shotを実行した結果ではなく、有限T複素signalのanalytic shot込みcompiled RZ予測。全energy/QPE/RPE round最適化の最終費用ではない。

## 比較B：3公式全体のPF最良対保存PR最良

PR削減率=100×(1−PR平均/PF費用)。負値はPF低費用。

{global_table}

## 比較A：同じ公式内

{within}

各精度の6者（各公式のPF最良とPR点最良）の全順位は[15精度順位](../../{OUTPUT}/precision_rankings_15.json)、exact費用・保存PRの分散/SE/nは[60比較CSV](../../{OUTPUT}/comparisons_60.csv)と[完全JSON](../../{OUTPUT}/comparisons_60.json)。欠測候補を除いたsubset最良への置換なし。

## 旧比較との順位変化・内訳

旧保存PR同士の選択・費用は不変。以前のH2約47%、H4約53%、H6約63%削減は高次PR対S2 PRであり、最良決定論PFに対する主張ではない。旧m5疎格子PFだけを分母にした比較とも区別する。

H6旧m5部分集合に対する比較と、今回の全公式共通PF分母を比較すると次の通り。

| ε | 旧m5疎格子PF最良 | 旧PR削減率 | 新全公式PF最良 | 新PR削減率 |
|---|---|---:|---|---:|
{chr(10).join(oldtable)}

分母変化を、旧m5疎格子を共通政策で評価した費用/旧保存費用、m5 prefix拡張後/旧m5疎格子、全公式最良/m5格子最良の3比へexact有理数で分解した。これらの積が新PF/旧PFに完全一致する。旧m5には同一native価格を使い、旧profile価格を移送しない。H4分も機械表へ保存。有限保存値の入れ子集合比較であり、探索範囲だけの単独因果効果や母平均保証は主張しない。

| ε | 政策・保存算術比 | m5 prefix拡張比 | S2/Yoshida4を含む公式拡張比 |
|---|---:|---:|---:|
{chr(10).join(factors)}

[旧比較15行](../../{OUTPUT}/old_vs_common_15.json)、[分母分解10行](../../{OUTPUT}/nested_PF_cost_change_decomposition_10.json)。

## 技術停止・資源・監査

前回のmetadata stageは0次元scalar shape検査の不具合でsnapshot_restore1後にSTOP。その消費・grant/marker/ledger/STOPを保全し返却0。今回の明示継続指示を新SOURCE/pack/grantへ結合した。修正後の63取得は新grantの一回実行。後続pack準備では汎用係数helperにm5がなく停止したが、科学grant前のため科学call0。元m5 signed scheduleを参照して新packへ封印した。消費済みgrantのretry/resumeなし。

metadata監査readerの来歴修正を分離し、実行SOURCEはGitからarchive化した。作者による別数値経路チェックであり、独立GPT/Claude科学レビューや外部再現ではない。旧STOPを成功へ書き換えていない。

実affinity128、idle約127、空きRAM約1TB・disk約125GBのfresh観測からmetadata63、signal51、COST127 workerを選択。内側BLAS/compiler1、共有immutable context/preparationを再利用。固定の低いworker/CPU/RAM上限や共有設定変更・他ユーザー操作は0。

H6 wall秒：SIGNAL {receipts['SIGNAL']['wall_seconds']:.3f}、QUALIFY {receipts['QUALIFY']['wall_seconds']:.3f}、COST {receipts['COST']['wall_seconds']:.3f}、ACCOUNT {receipts['ACCOUNT']['wall_seconds']:.3f}。COST全worker exit0・同親join/reap・残存0。新raw archive {h6['archives_verified']}件の原byte SHA復元、全native gate/phase alignmentなし、保存shot integer-ceiling、別Fraction整数費用、80桁Decimal比、完全格子最小値、ledger上限を監査PASS。

H6 native誤差最大値は[終了監査](../../{H6}/end_audit.json)に各check別で保存。compiled vectorは元CT01<1e-7、他native gate<1e-9を保持し、allowanceへ流用しない。

## 統計的限界と終了

有限標本の探索選択効果は補正していない。保存SEが小さくても母平均順位を保証しない。全PR群でS4未観測、rare event費用/分散/tail上界はnullで0ではない。PFにはMonte Carlo SEがないが、reference/bias/normalizationの診断は経験的。今回の新PF分母との独立確認samplingは0、前回118確認はPR対PRであり新PF比較の確認ではない。

[metadata取得・旧52照合](../../{META}/ending_audit.json)、[H6全候補](../../{H6}/candidate_status.json)、[588登録](../../{OUTPUT}/candidate_status_588.csv)、[全契約・SOURCE・入力SHA・grant/ledger/marker](../../{H6}/execution_contract_projection.json)、[統合監査](../../{OUTPUT}/independent_saved_result_checks.json)を保存した。原private raw/vector/QPY/NPZ archiveと旧grant・STOPを保持。H2/H4はbyte参照結合のみ。

**588候補の固定比較・監査・保存を完了してローカルcommit後mandatory STOP。GitHub push、q16、候補追加、新規PR sampling、詳細科学レビュー、production、論文本文作成は0。**
'''
    path=ROOT/'docs/research/track_a_h246_pf_common_final_20261022.md'
    with path.open('x') as f:f.write(report)
    for folder,title in [(META,'H6不足metadata63取得'),(H6,'H6共通PF276取得'),(OUTPUT,'H246共通PF588最終統合')]:
        p=ROOT/folder/'README.md'
        with p.open('x') as f:f.write(f'# {title}\n\n[結果・条件・限界](../../../../docs/research/track_a_h246_pf_common_final_20261022.md)。metadata63・H6全276・統合588評価完了。旧H2/H4は再計算せず参照、PR保存統計不変、経験的allowance・有限探索条件付き。SOURCE/input/grant/ledger/marker/raw原SHAを保存。消費済みgrant再利用禁止、local commit後mandatory STOP、push0。\n')
    return path


if __name__=='__main__':print(build())
