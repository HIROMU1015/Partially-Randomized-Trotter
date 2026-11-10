# H4 DF10価格V2 S1：少数補正の保存データ評価

2026-11-10 JST。**承認された3 fitと既知データ評価を完了。単純なq比例補正Mqが有望で、basis-group項の追加MqBは確認cohortのRMSEを改善しなかった。** 既知データでの探索的結果であり、長q・独立汎化・全RPE価格providerの採用を資格付けない。

正本：[RESULT.json](RESULT.json)／全path・axis・model別[PATH_RESIDUALS.csv](PATH_RESIDUALS.csv)／実行前固定[PLAN](PLAN.md)／[V2事前設計][DESIGN]。表は表示だけ丸め、JSON/CSVの予測・残差は丸めず保存した。残差eはcompiled−prediction。

## 1. 固定範囲・SOURCE・再利用

- Linear H4/1Å/STO-3G、保存tol-only DF10/sector(2,2)、保存DF-H ground target、保存数値state条件付き、P=0。S2、δ=.1、PF B0/p7とPR p1/R4/K2。元symmetric_directional・測定付きfull Hadamard wrapperのcompiled RZ価格。
- 元compilerはQiskit1.3.0、rz/sx/x/cx、opt1/seed17。εE=.00159360010199040 Ha、αtotal=.05/方式、βtotal=.4 rad、h=.1、m0…12は不変。N/Gは再計算しない。
- 科学実行SOURCE：**`cc6b42f12f928fd051831549f5f86509dad6942b`**。候補・実装・PLAN・人工testを実H4評価前にcommitし、数値処理後にモデルを変更していない。
- 入力carrier：`eec5bdf9cc391fe49b55b5673f3e385deeb20d42`のV1 DEVELOPMENT/HOLDOUT。Git objectから読んだ二つの公開JSONだけを使用。元trajectory/NPZ/QPY/provider・V1 runnerは呼ばない。入力digestとV1 model SOURCEはRESULTに保存。
- 校正はPF q8/T.8の1 pathとPR q8/T.8探索9本のみ。PF q16/T1.6の1 path、PR q8確認33本は**既知のV1使用済みデータ**での照合。cohortをpoolせず、未使用独立holdoutと呼ばない。
- `track_d_c7_4_models.fit`の固定関数本文をASTで読み、equal-q重み・rank検査・multiresponse最小二乗をそのまま再利用。D固有labels/feature抽出/import chainは使わない。追加部分は保存rowの対応付け、少数design列、S1集計だけ。

全44 circuit/pathのX/Yを対応付け、独立nを倍にしない。RESULTのscientific_bindingは元V1のcohort基準metadataを継承し、個別T/qはCSVのrowを正本とする。

## 2. 固定モデルと校正係数

\[
M0:\ \widehat C=C^{(1)},\qquad
Mq:\ \widehat C=C^{(1)}+bq_{\rm PF},\qquad
MqB:\ \widehat C=C^{(1)}+bq_{\rm PF}+cB.
\]

\(C^{(1)}\)は凍結V1の`structural_rz`、Bは保存`tail_basis_groups`。MqはPF/PRで別fit、MqBはPRだけ。追加intercept・希少order係数・特徴探索・regularizationはない。

| 方式・モデル | 校正path数 | rank | b（両軸同値） | c（両軸同値） |
|---|---:|---:|---:|---:|
| PF Mq | 1 | 1 | 606.75 | — |
| PR Mq | 9 | 1 | 176.88888888888894 | — |
| PR MqB | 9 | 2 | 351.9050480769233 | −59.72115384615389 |

PRはq8しかなく、bはq方向の傾きと端点offsetを識別しない。cの符号もbasis変換の因果的な単価ではない。係数不確かさは未評価のnull。

## 3. 既知データでの確認結果

全てRZ/軸、X/Yで同じ指標。PR n=33であり66ではない。相対RMSEはpathごとのe/Cの二乗平均平方根で、平均価格に対するRMSE比ではない。

| 対象 | モデル | 平均残差 | RMSE | 最大絶対残差 | 相対RMSE | 最大絶対相対誤差 |
|---|---|---:|---:|---:|---:|---:|
| PF q16、n1 | M0 | +9646 | 9646 | 9646 | 10.8976% | 10.8976% |
| 同上 | **Mq** | **−62** | **62** | **62** | **0.0700%** | **0.0700%** |
| PR確認 q8、n33 | M0 | +1433.242 | 1455.157 | 1960 | 8.6104% | 10.8034% |
| 同上 | **Mq** | **+18.131** | **252.244** | **679.111** | **1.4664%** | **4.0023%** |
| 同上 | MqB | +33.212 | 276.936 | 697.510 | 1.5890% | 3.8331% |

PF Mqはq8だけから校正し、q16で88577を予測、保存価格88515に対して−62の残差。有限の一つのq方向確認として改善したが、独立な新qによる検証ではない。

PR Mqの確認平均予測は16929.65656565657、観測平均16947.78787878788。大きな正biasは減ったが、残差範囲は−679.111…+544.889 RZ。q8で一定補正を加えるため、path間残差SDはM0と同じ約255.493のままで、path変動の説明を改善したわけではない。

MqBは最大絶対**相対**誤差だけならMqより小さいが、RMSE・MAE・最大絶対RZ誤差・biasは悪化。確認MAEはMq188.044に対してMqB215.792。新しい一律閾値や単一指標で採択せず、特徴追加の利益は現在の確認結果から支持されないと判断する。

## 4. 開発と層別診断

| 開発データ | モデル | RMSE/軸 | 最大絶対残差/軸 |
|---|---|---:|---:|
| PF q8、n1 | M0 → Mq | 4854 → 0 | 4854 → 0 |
| PR探索 q8、n9 | M0 | 1445.838 | 1869 |
| 同上 | Mq | 296.494 | 659.111 |
| 同上 | MqB | 263.801 | 446.769 |

PF1点の校正一致とPR training平均残差ほぼ0は最小二乗の校正結果で、予測性能の根拠ではない。MqBはtraining RMSEを減らしたが、確認RMSEではMqより悪化。basis特徴の一般的無用性や統計的に確定した過学習の証明とはしない。

RESULTの`basis_strata`と`order_2_strata`に、固定cohort/model/axisごとの保存特徴別n・誤差を収録。order2は開発・確認とも少数で、層別値を全lawの寄与推定や専用係数fitへ拡張していない。CSVで個別pathを追跡できる。

## 5. 不確かさと支持される範囲

既知確認PRのMq平均残差の記述的SEは約44.476 RZ、MqBは約48.602 RZ。IIDを仮定しfit済み係数を条件付けた残差SD/√nにすぎず、開発9本由来の係数不確かさ、未知の系統差、finite-bit lawの理想IIDとの差を含むCIではない。平均biasが小さいことから系統誤差0を認定しない。

支持されるのは、**固定DF10・同じcompiler/full-wrapper・保存範囲で、少数補正によりV1より小さい価格誤差を得たこと**。PF/PRの絶対価格を別々に予測できた。Mqを次の検証候補として残すが、確認値で候補を比べた結果は探索的推薦とする。

未資格なのはPRのq方向、独立cohortへの汎化、law期待価格・希少orderの十分な評価、q4096を含む長q、全13roundの費用・順位・CI。観測最大誤差は一様上界ではない。RZ残差0を採用必須条件にせず、1/2/5%等の新しい閾値も採択しない。αtotal=.05を価格の95%保証へ読み替えない。

## 6. 実装確認・成果保全・STOP

人工テスト4 method PASS：既存fitの二応答係数/rank、3 fitのtraining限定と既知価格混入防止、完了順に依存しないaxis/feature binding、zero/負値/非有限値診断を確認。実H4の回路作用や独立科学レビューのPASSではない。

固定SOURCEで一回完了。fit3回、CSV260 path/axis/model行、rank不足0・負予測0・非有限予測0。新構造予測、原trajectory読取、新sampling、実回路build/compile、bootstrap、signal/N/N×C/G、長q価格、候補最適化、詳細科学レビューは0。V1モデル・結果・不一致判定、旧SOURCE/input/raw/grant/ledger/marker/KEEP/STOPと他worktree/processは保全。完了済みrunnerを再実行して結果を生成し直さない。

保存結果・入力対応・差分形式・V1無変更を確認し、GitHub公開後にremote SHAと主要成果物を照合して**mandatory STOP**。Mqを全RPE providerへ自動採用しない。

次の作業：**ユーザー**が、Mqの同δ・別qでの独立検証へ進むか判断する。進む場合は**Codex**がPR q16/T1.6等の不足点についてSOURCE・cohort本数・校正/検証役割・STOPを具体化し、**ユーザーの新しい科学実行許可後**に実施する。今回は追加compile枠や新qを採択・実行していない。

[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df1dcd7bf0469a724493e0e4193b107ad3a157f0/docs/research/h4_df10_price_v2_design_20261110/REPORT.md
