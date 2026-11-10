# H4 DF10 Mq：同一δ・別qの独立検証 完了結果

2026-11-10 JST。**S1で有望だったMqは、未使用のPF q32とPR q16の双方でM0より正確だった。** PFの絶対相対誤差は0.10525%。PR33本の相対RMSEは1.31743%、最大絶対相対誤差は4.98547%。固定係数の有限q移送を支持するが、長q・全law期待価格・係数/系統不確かさと全RPE費用への影響は未資格である。

正本：[RESULT.json](RESULT.json)／[全136 model・path・axis行のCSV](PATH_RESIDUALS.csv)／[実際の入力・trajectory binding](PROVENANCE.json)。取得前の[固定68軸予測](../PREDICTIONS.json)、[入力provenance](../PROVENANCE.json)、[PLAN](../PLAN.md)、[STOP後の継続計画](../TECHNICAL_COMPLETION_PLAN.md)を保持する。表は表示だけ丸め、JSON/CSVを数値正本とする。

## 1. 固定条件・SOURCE

- Linear H4/1Å/STO-3G、保存tol-only1e−8 DF10/sector(2,2)、保存DF-H ground target/数値state条件付き/P=0。S2、δ=.1、元symmetric_directional・測定付きfull Hadamard wrapper、X/Yのcompiled RZ数。
- PF：B0/prefix7/qPF32/T3.2、candidate `f01949581a5fab7c0ae1`、決定的1回路。PR：prefix1/R_outer4/K2/qPF16/T1.6、candidate `64e4f7e9ec7fba28815e`、新33 trajectory。PF B0をFull PFへ変更しない。
- 元Qiskit1.3.0、rz/sx/x/cx、opt1/seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。元H/state/snapshot/fragment順/scalar/lawを保全。
- **モデル・cohort・新予測とPF実行SOURCE：`55eccab8865684fadefa5bd79875b1154518b410`。PR継続実行SOURCE：`f8bb11fe6be3965fbc4c4bea9376c0bb238705d2`。** 継続はユーザーの明示許可後に行った。旧SOURCE・STOP・PF価格・予測を変更せず、新しい出力へ完成させた。
- M0は[V1](../../h4_df10_price_v1_20261110/REPORT.md)のAccounting本体を再利用。新qの入口scopeだけを拡張し、gate展開・binary64 zero・basis共有・scalar・wrapper会計ruleは不変。
- Mqは[S1](../../h4_df10_price_v2_s1_20261110/REPORT.md) carrier `cefc01622a3cd75dd16d30de7b9fb68274f3eddd`のbPF=606.75、bPR=176.88888888888894を両軸で凍結。\(\widehat C=C^{(1)}+b_s q_{\rm PF}\)。新fit、MqB、特徴探索なし。
- H `c1a6c428…`、state `9bcc0b0e…`、snapshot `12219b7d…`。PR preparation `b99720d6…`、PF p7 preparation `fc83f26f…`。完全hashは両PROVENANCEと各trial bindingに保存。元入力binding内T=.8はreference metadataで、今回のTは上記candidateを正本とする。

元price/builder carrier155529e2、provider c31ef424、V2設計df1dcd7bf、S1と旧資料の入口は[最初のREPORT](../REPORT.md)と[設計](../../h4_df10_price_v2_design_20261110/REPORT.md)。旧sourceを今回の科学実行SOURCEへ付け替えない。RPEのεE=.00159360010199040 Ha、αtotal=.05/方式、βtotal=.4、h=.1、m0…12は不変で、N/signal/energy資格を再計算しない。

元rawの`conditional=False`はTaylor orderを条件付けて抽出していないことを表し、保存数値stateによる条件付き評価とは区別する。

## 2. 保存資産・独立cohort・停止の引継ぎ

元の既知6価格表に今回の同δ・同q価格はなく、PR q32/T1.6/δ.05を代用しなかった。実wrapperのcanonical fingerprint・compilerで照合した元cache hitも0。**科学対象の新compileはPF2＋PR66＝計68 wrapper**。元cache・他worktree/processへ書き込まず、専用cacheを使用した。

cohort `H4_DF10_V2_INDEPENDENT_Q_20261110`の33本・528 seedは初回SOURCEで一回生成し、既知のregistry/exclusion/cohort seed集合との衝突0を確認。元V1/S1の9/33本と混合しない。全履歴の完全被覆や有限PRNGの数学的独立性は主張しない。全68軸のM0/Mqを直接価格の取得前に固定した。

初回PRはcallbackのprep引数欠落でbuild前にSTOPした。PFは独立taskとして完了し回収した。ユーザーの継続許可後、既存`h46_same_event.decode_event`と元request型で**保存33本をRNGなしで復元**し、正しい4引数callbackで残66軸だけを取得した。継続で新sampling・新M0予測・PF再compileは0。元STOPは保持し、独立検証データを校正へ回していない。

各compiled trialの実event/application順のcanonical hashが固定予測と一致することを全34 circuit/pathで確認し、private resultのdigest・preparation hashを本directoryのPROVENANCEへ記録した。raw event・seed・QPY・NPZは公開していない。

## 3. 新qでの絶対価格・予測誤差

以下はRZ/軸、e=compiled−prediction。X/Yの値・指標は同じであり、PR標本数は33（66ではない）。「相対RMSE」は各pathのe/Cから計算した値で、RMSEを平均価格で割った値とは区別する。

| 対象 | モデル | 実価格（PRは平均） | 予測（PRは平均） | 平均残差 | MAE | RMSE | 最大絶対残差 |
|---|---|---:|---:|---:|---:|---:|---:|
| PF q32、n1 | M0 | 176723 | 157493 | +19230 | 19230 | 19230 | 19230 |
| 同上 | **Mq** | **176723** | **176909** | **−186** | **186** | **186** | **186** |
| PR q16、n33 | M0 | 33527.242 | 30767.697 | +2759.545 | 2759.545 | 2792.326 | 3357 |
| 同上 | **Mq** | **33527.242** | **33597.919** | **−70.677** | **302.663** | **432.422** | **1605.222** |

| 対象 | モデル | 相対RMSE | 最大絶対相対誤差 | 残差範囲 |
|---|---|---:|---:|---:|
| PF q32 | M0 → Mq | 10.88144% → **0.10525%** | 10.88144% → **0.10525%** | +19230 → −186 |
| PR q16 | M0 | 8.32533% | 10.20799% | +1225…+3357 |
| 同上 | Mq | **1.31743%** | **4.98547%** | −1605.222…+526.778 |

PFはq8で校正したbを変更せず、S1で既知だったq16を超える新q32でも改善した。PRはq8だけで校正したbを新q16へ移し、同じ33本でM0よりbias・MAE・RMSE・最大絶対誤差が小さくなった。これは固定した2つの新qに対する有限の検証結果であり、一般compilerの再現や全qの上界ではない。観測した整数一致0は診断に留め、採用必須条件や一律1/2/5%閾値を置かない。

## 4. PR：平均価格とpath変動、q8との比較

新q16の実価格SDは1208.885 RZ、IID仮定下のcohort平均の記述的SEは210.440 RZ。Mqの残差SDは433.222 RZ、残差分散は187681.068 RZ²、平均残差の同SEは75.414 RZ。予測と実価格は同じpathなので、平均誤差のSEは対応する残差から求め、両価格のSEを独立と仮定して合成しない。

Mqは同q内では一定補正のため、残差SDはM0と変わらず、主要biasを除いたことがRMSE改善の中心である。平均残差−70.677 RZと、個別pathの最大1605.222 RZを同じ精度指標として扱わない。最大絶対誤差はpath16（実32198、Mq33803.222）で、元path/axisをCSVから追跡できる。

| 固定Mq、PR各n33 | q8/T.8・S1使用済み確認 | **q16/T1.6・今回の独立cohort** |
|---|---:|---:|
| 平均残差 | +18.131 | **−70.677** |
| RMSE | 252.244 | **432.422** |
| 相対RMSE | 1.46636% | **1.31743%** |
| 最大絶対相対誤差 | 4.00230% | **4.98547%** |
| 残差SD | 255.493 | **433.222** |

q8側はS1保存summaryの引用で、再fit・再集計しない。別seed・別長さのcohortなのでpath同士を人工的にpaired比較しない。q増加で絶対誤差は増えた一方、相対RMSEは増えず、最大相対誤差は増えた。どれか一つだけで一般化を認定しない。

今回order2は7 pathで各1 event、計7 eventを観測したが、複数order2を含むpathは観測していない。希少eventの全law寄与や未観測構造を十分に評価したとはしない。条件付き係数や新特徴をfitしない。

## 5. correctness・不確かさ・全RPEへの境界

元の代表PF/PR path0の低control・高control・独立occupation vector・Hadamard readout・compiled-vectorの各7 gateは全通過。PRの独立vector差約1.80e−13、compiled X/Y差約3.53e−12/3.81e−12で、元tol1e−9/compiled1e−7・位相alignmentなしを維持した。残32 PRの全入力作用や全law correctnessを新たに証明したとはしない。

平均誤差SEはfrozen fitとIIDを仮定した記述診断。S1の開発9本から得たbの不確かさ、未知のsystematic、有限PRNGと理想IIDとの差は含まない。平均biasがSEより小さいことからsystematic=0や95%保証を認定せず、全体CIはnullのままとする。αtotal=.05はRPE設定であり価格CIではない。

今回の結果は固定δ・S2/prefix/R/K/compiler/full-wrapperのq16PR/q32PFでMqの移送を支持する。未検証q、q4096、全13roundのlaw期待価格とcross-round covarianceは未資格。PF/PRで今回のq/Tが異なるので、両価格を直接比較して方式の総費用順位を出さない。N・N×C/Gを計算せず、Main Claim・全RPE providerの採用は変更しない。

長時間価格への使用には、より大きいqでの外挿検証と、cohort平均の誤差・校正不確かさ・系統差の分離が残る。次の最小候補は、同δの大きい共通q（例：qPF64/T6.4）に限定した独立点。PFは各軸1回路、PRは独立cohort一つとし、本数・必要精度・SOURCE/STOPを事前に固定する。少数点の良好な結果だけで4096まで保証せず、今回データへの再fitで成功扱いしない。これは未採択・未実行の候補である。

## 6. 完了と次の担当

人工検証は事前4＋継続準備2 method PASS。科学対象は新33 trajectory、取得前予測68軸、最終direct68軸、CSV136 model/axis行を完了。実H4の新compileは計68、継続の新sampling/PF compile/予測0、fit/N/G0。最初の技術STOPと生成済み資産を保全し、承認された未完了部分だけを完成させた。

V1/V2設計/S1のSOURCE/input/raw/結果/不一致/旧grant/ledger/marker/KEEP/STOP、他worktree/process、共有設定は不変。公開後にremote SHAと主要成果物を照合し、**mandatory STOP**。q64・追加cohort・モデル更新・全RPE費用・詳細科学レビューへ自動進行しない。

次の担当：**ユーザー**が長q検証案を進めるか判断する。進める場合、**Codex**が上記最小候補の本数・必要精度・SOURCE・範囲を具体化し、新しい科学実行許可後に行う。
