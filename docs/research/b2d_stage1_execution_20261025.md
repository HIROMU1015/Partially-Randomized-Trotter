# B2-D Stage 1：限定科学検証・技術復旧記録

## 最終状態：COMPLETE、A4対A3の優劣はINCONCLUSIVE

**固定Stage 1を完了した。** 入力検査・独立A4数学参照4件・published列規約4件を全てPASSし、その後に36行比較を完了。登録941 DAG dispatch・448 exponential key・入力10境界・signal expectation144を全件照合した。event3108件、scope超過・未完了・保存失敗なし。
ローカルの科学検証結果であり、CIや独立した外部再現ではない。科学的GO／NO-GO・総資源優位・高次法の一般的優位は認定しない。完了後の科学再実行は0件でmandatory STOP。

| 対象 | 固定SOURCE・保存先 |
|---|---|
| 初回科学SOURCE | `54008c697e686a42ec48941d25219d523bafd454`、技術的中断 |
| 完走した科学SOURCE | **`758a05ed6aac58a5945af5a5330c792a842b174d`**、Input packageの2参照のみ修正 |
| 保存後のCLI表示修正 | `5bef9c9cbcd7cfd595ec82874531ad3c7982d014`。この版での科学再実行は行っていない |
| 完走した実行原本 | `/tmp/prt-b2d-stage1-recovery-science-20261025/artifacts/research_b_b2d_stage1/20261025/technical-r1` |
| 公開用の同一bytes | [technical-r1](../../artifacts/research_b_b2d_stage1/20261025/technical-r1/) と [初回停止記録](../../artifacts/research_b_b2d_stage1/20261025/simple-once/) |
| 全結果・集計 | [result.json](../../artifacts/research_b_b2d_stage1/20261025/technical-r1/result.json)、[36行CSV](../../artifacts/research_b_b2d_stage1/20261025/COMPARISON.csv)、[要約JSON](../../artifacts/research_b_b2d_stage1/20261025/STAGE1_SUMMARY.json) |

### 入力とcorrectness

Synthetic 2-mode one-body＋3 DF因子（分子geometry／AO basis／分子DF rank探索は非該当）、N=1／mode0 occupation、L_D=2 weight-ranked prefixを使用。選択因子は0,1、残差は2、sector basisは[1,2]。実入力・array・preparation・state identityは[inputs.json](../../artifacts/research_b_b2d_stage1/20261025/technical-r1/inputs.json)へ保存した。
H=D+R+cIのoperator残差は **6.16645e-17**。D内部3対のcommutator normは0.023895、0.003920、0.01607445、D/Rは **0.00428043** で、固定atol1e-12に対して必要な非可換性を確認した。tail leakageは0。
A4独立数学参照の最大operator誤差 **1.89162e-15**、actual inverse最大 **2.44250e-15**、published full列参照最大 **6.69489e-16**。全4件がatol1e-12以内。
全gateがoperator／signal比較より先に完了したことを保存eventsでも確認した。旧validatorの直接append operatorとの同値性を主張する検査ではない。

### A0〜A4の全36行比較

以下は同じ固定syntheticモデル・N=1／mode0・L_D=2 weight-ranked、δ=±0.09、q=2,3、r_RTE=2／K=2、complex128、phase無効の全実行窓でのmin–max。A1〜A4のn_D=1,2を全て含む。correctness atol1e-12は改善認定や誤差差分の信頼区間ではない。

| 方式 | n_D | D half operator誤差 | Full physical operator誤差 | Corrected physical signal誤差 |
|---|---:|---:|---:|---:|
| A0 | — | — | 4.007258e-06–5.974889e-06 | 1.807640e-06–2.694640e-06 |
| A1 | 1 | 2.083856e-07 | 1.339230e-06–1.996814e-06 | 5.999438e-07–8.944059e-07 |
| A1 | 2 | 5.209165e-08 | 7.169124e-07–1.068928e-06 | 3.199561e-07–4.770184e-07 |
| A2 | 1 | 6.406136e-11 | 5.097543e-07–7.600520e-07 | 2.267377e-07–3.380549e-07 |
| A2 | 2 | 4.003566e-12 | 5.095157e-07–7.596962e-07 | 2.266449e-07–3.379162e-07 |
| A3 | 1 | 1.110794e-15 | 5.094998e-07–7.596725e-07 | 2.266387e-07–3.379070e-07 |
| A3 | 2 | 1.221451e-15 | 5.094998e-07–7.596724e-07 | 2.266387e-07–3.379070e-07 |
| A4 | 1 | 2.888666e-15 | 5.094998e-07–7.596724e-07 | 2.266387e-07–3.379070e-07 |
| A4 | 2 | 1.337234e-15 | 5.094998e-07–7.596724e-07 | 2.266387e-07–3.379070e-07 |

A4のD内部誤差は約1.3–2.9e-15まで低下し、full physical coherent signal誤差もA0比 **87.460–87.462%**、A1 D-S2比ではn_D=1で **62.220–62.223%**、n_D=2で **29.163–29.166%** 小さかった。局所D近似の改善がfull signalに残ることは、この限定比較で観測された。
一方、A2 Yoshida4との差はsignal誤差でn_D=1に **0.04369–0.04377%**、n_D=2に **0.002731–0.002735%** の小さな記述差にとどまる。精度保証付きの優位認定ではない。
A3 Y8m10bとA4 YP8m8はどちらもD内部で丸め誤差水準であり、full signal誤差の差は最大約 **1.46e-15**。この差の符号・順位は解釈せず、**A4対A3の優劣はINCONCLUSIVE** とする。高精度や追加gridでの自動追試は行わない。
高次3方式のfull operator誤差は約5.1–7.6e-7、signal誤差は約2.27–3.38e-7に集まり、D内部精度の違いがfull signalで大きな独自利得を生むとは確認できなかった。観測されたouter PF誤差と同程度の残差が残っている。

Normalization productは **1.000055855727–1.000083784761**、raw meanからcorrected meanへのrescaling整合残差最大 **1.13494e-15**。
有限RTEのoperator誤差は約3.25–4.87e-11で、full physical誤差とは分けて保存した。X/Y・半径・raw／corrected信号をCSVとraw JSONへ保持しており、attenuationを消した架空の生信号として扱わない。収束次数・別モデルへの一般化・RPE統計誤差・総資源費用は未評価。

### 技術的中断・修正・再実行の履歴

1. Attempt 1：旧contract参照によるAttributeError、入力9境界完了・DAG0・比較0行。旧原本・STOPを保全し、2参照の修正を独立stubで確認して新SOURCE758a05edへ固定した。
2. Attempt 2：科学計算・結果9ファイル・status／STOPまで全件完了。その**後**、CLI標準出力の`json.dumps`が科学provenance中のFractionを扱えずexit 1となった。保存結果は既存科学encoderで正しくserializeされている。
3. CLIは同じ既存encoderを使う2行差分で修正。旧CLIの失敗と修正版の正常終了をstubで確認し、人工6ファイルは15＋14＋16＋15＋4＋8＝**72件PASS**。保存済み結果を変更せず、科学計算の再実行も行っていない。

[各attemptのstdout／stderr／time](../../artifacts/research_b_b2d_stage1/20261025/execution_logs/)を公開した。Attempt 2のscience完了とprocess exit 1を混同しない。
CPython3.11.9と既存50 packageの環境を再利用し、version・interpreter／import元・環境変数をbindingへ記録した。共有環境・他研究・OS設定は変更なし。
単一processで既存2×2 sector・vectorized演算・共有cacheを利用し、CPU／thread上限は追加していない。Attempt 2はwall5.65秒、user CPU11.06秒／system CPU3.06秒、max RSS366832 KiB。初回とのprocess wall合計は11.29秒で、調査・人工テスト・文書化を含む作業全体の時間ではない。

### 次に必要な研究判断とSTOP

研究Bのユーザー／研究判断担当が、A2・A3に対する独自性と、内部D近似の利得が残る条件・費用をさらに調べる価値を判断する。
今回の固定fixtureではA0・A1からの改善は見えたが、既存高次baselineとの差別化、誤差あたりcost、processorを含むnative full wrapper費用は未確認。Stage 2・新条件・高精度・追加gridは別途の研究判断と実行許可が必要。
Codexは結果公開・remote照合を完了してmandatory STOPする。通常の研究B進行として統括報告は行わない。

---

## 以下は初回停止と再実行前に固定した記録

2026-10-25 JST。実行条件は[固定科学契約](b2d_stage1_simple_execution_20261025.md)を継承する。
Synthetic 2-mode one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、A0〜A4、δ=±0.09、q=2,3、n_D=1,2、r_RTE=2／K=2、complex128、correctness atol=1e-12、phase無効。分子geometry／AO basis／分子DF rank探索は非該当。
比較は36行・941 DAG dispatch・448 exponential key、記述的診断のみ。科学条件・係数・scope・参照・許容差は変更しない。

## 今回の実行範囲と技術復旧の例外

ユーザーは固定Stage 1の科学実行と、予期しない技術的中断に限る原因調査・最小修正・同一条件での新attemptを明示許可した。
各attemptはfresh outputとし、旧runを上書き・resumeしない。科学的FAIL・非可換性不成立・実際の参照不一致・精度不足・改善なしを、技術エラーとして再試行しない。
旧契約・SOURCE・raw・STOPは遡及変更しない。承認証明・grant等は作成しない。Stage 2、別fixture／sector／state／精度／grid、native compile、総資源比較へ進まない。

## Attempt 1：技術的中断

固定SOURCE `54008c697e686a42ec48941d25219d523bafd454`、指定の `simple-once` へ初回実行した。
preflightはPASS。入力fixture・split・preparation・sector・full H・D block3・tailの9境界は完了したが、Input packageで `AttributeError` となった。
承認管理撤去後も `permit.contract["numerical_atol"]` と `permit.contract["source_commit"]` の2参照が残っていたことが原因。
状態 `FAILED_MANDATORY_STOP`、science DAG未開始・比較0行。STOP・events・binding・partial result・statusを保全した。
wall5.64秒、user CPU10.69秒／system CPU2.95秒、max RSS364156 KiB。Matplotlibは非書込の既定cache先から専用tmpへfallbackし、計算停止の原因ではなかった。

## 復旧差分と独立した人工確認

変更は2参照だけ：atolを科学scopeのgraph settingsから、SOURCEをrun contextのsource_commitから取得する。
承認contractを再導入せず、科学値・入力recipe・数値algorithm・許容差を維持する。
全科学import／関数をopaque stubへ置換した実loaderテストを追加し、正常なleakageと漏洩拒否の両branchを確認した。
旧SOURCEの関数を同じstubで実行すると両branchともmissing-contract AttributeError、新SOURCEでは正常完了／所定のSECTOR_BASIS_LEAKAGEとなることを独立確認した。
人工6ファイルは15＋14＋16＋15＋4＋7＝**71件PASS**。実科学量を使った原因探索・係数変更・閾値緩和は行っていない。

## 次のattemptの起動条件（この記録の固定時点は実行前）

本修正とこの記録を含むcommitを新科学SOURCEとして固定し、clean detached checkout `/tmp/prt-b2d-stage1-recovery-science-20261025` を用いる。
新出力は同checkout内 `artifacts/research_b_b2d_stage1/20261025/technical-r1`。旧出力・STOP・生成済み入力を再利用せず、同じ固定recipeを最初から実行する。
SOURCE変更理由は上記の明確な接続バグのみで、固定scope・科学条件は不変。予期しない技術中断だけがユーザーの今回の再実行許可の対象である。
32 CPU、空きRAM約51 GiB、GPU利用不可を確認した。2×2 sector・vectorized演算・448 key共有cacheを再利用し、共通入力／gate順序を守る単一processとする。CPU／thread／RAMの人工上限や共有環境変更なし。
結果・全gate・call実績を確認後、正常／科学的STOP／INCONCLUSIVE／解決不能な技術停止のいずれでもmandatory STOPする。
