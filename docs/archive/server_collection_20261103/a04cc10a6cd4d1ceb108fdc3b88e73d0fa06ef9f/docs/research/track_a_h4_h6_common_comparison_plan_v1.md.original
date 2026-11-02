# H4/H6：既存証拠の採用範囲と最小共通panel案

2026-10-13実行追記：[固定16 anchorの補完結果](../../artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/README.md)。旧H4 rank12/stateを保持し、同compiler/primary symmetric control、S2/S4 baseline・PR32標本の532compileを取得済み。点PR/PF比.507/.472/.453/.511。DF tol-policyの一致とfull-grid optimumは依然未認定。以下の未取得記載は補完前の判断履歴。

2026-10-13 JST。前の[対応表](track_a_h4_h6_h8_comparison_readiness_20261013.md)を実施判断へ具体化する。今回の科学再計算0。旧H4 source/結果/稼働runを保全する。

## 保存系列ごとの採用判断

| 保存系列 | そのまま使える情報 | 共通H4/H6費用としての扱い | 不足 |
|---|---|---|---|
| M1-B1 H4/1.00 Å/rank12、210候補、194×32 random | 旧契約内のfull-wrapper RZ/CX/depth、paired SE、整数shots、候補境界 | H4内部の独立case studyとして採用。共通size curveには未採用 | tol-only入力との差、S4、symmetric native/control、uと候補選択の統一 |
| PM-1 H4/1.00 Å、B0 rank4/5×q4、8追加候補 | 補完済みB0比較、bestB0 G=229718060 | M1へ同条件で追加できる。旧B0 bestを使った48.388%削減を最新43.07%へ置換 | H6 controlへの費用移植不能 |
| PM-2 H4218候補・302精度点 | 保存値でのprecision/P envelope、元epsilon.05再現、定義説明 | 旧H4条件内で再利用。302独立計算・H6の厳精度結果とはしない | H6 epsilon.001まで外挿しない |
| M2 H4/1.30 Å・固定5構成 | 旧設定のgeometry transfer証拠、196 wrapper | 別geometryの限定transferとして採用、系別最良ではない | 候補集合・geometryが異なる |
| AX2B H4 limited/native/supplement | 入力・primitive・S4/control semantics・MP診断の保存証拠 | 共通builderを検証するscope付き再利用候補。新native費用Gは未取得 | 実cost compile/同一panelのshot map |
| 稼働中H4 Gaussian geometry/resume | run来歴と将来のgeometry研究 | 今回は保留。未完了出力を横断集計しない | geometry/profile/回路構成がH6契約と異なる |

M1-B1の`RPE_HADAMARD_CONTROL_CONVENTION=ordinary_controlled_diag_I_U_m`とconstruction_policy=`boundary_optimized`はsourceで確認した。Qiskit1.3.0/opt1/seed17が共通でも、H6のsymmetric_directionalへgate数を換算する係数はない。H6のordinary感度2 anchorで観測した約2倍の比をH4へ一律適用しない。

出典：[M1-B1検証](../pr2_matched_accuracy_m1_b1_result_validation.md)、[PM-2](../pr2_pm2_precision_resource_result_validation.md)、[AX2B補完](track_a_ax2b_h4_supplement_execution_v1.md)、[旧本文](../manuscripts/track_a_resource_study_v0_1.md)。limited launch_v2のPHASE_WALL_CAPは成功へ昇格せず、その後のnative receiptとsupplementの対象範囲だけを採用する。

## 再利用gate

1. 固定H4 input bytes・DF順序・state・projectionの同一性を確認する。固定rank12とtol-only1e-8が偶然同じかは未確認。異なる場合に旧signalを新Hamiltonianの結果として再利用しない。
2. native演算・control・paired axes・S4 signed scheduleの意味論を保存proofと新sourceで照合する。費用のidentityは意味論だけでは保証されない。
3. scalar shot再計算はbias/u/normalization/headroomの契約が一致した場合だけ可能。旧uの値を新policyの証明にしない。
4. 新cost必要時だけ新grant・別namespace・carry・source closureを固定する。旧transpile cacheやold compiler bytesは同一bindingが証明された対象に限る。

## 推奨：最適化結果ではなく16構成の共通anchorから

科学判断前の提案。H4/H6/H8で各系のDF rank Lに対しp=ceil(L/4)を用いる。H6の事前選択形を保持し、PRは(q,r)=(1,4),(2,4),(4,4),(8,2)、各K2/K4の8構成。full PFはq1/2/4/8×S2/S4の8構成、計16 signal候補である。H6のこのpanelは保存済みなので再計算不要。H4旧rank3のprefixと新tol-onlyのquarter prefixを同一と仮定しない。

- H4全8PFが適格な保守案：PF paired16 compile＋PR8×32×2=512、計528 compile/256 trajectory。
- B0の3prefix×4qも追加するなら12 signal/24 compileを加え、28 signal/552 compile。
- このpanelは固定設定transferの比較であり、H4の同条件full-grid最良を保証しない。
- 科学目的が各系のresource-map最良比較なら、H4も同一generator全gridと独立選択を別枠で計画する。旧210/218候補から都合のよい一部をH6と混ぜない。

元source/inputが一致する部分のsignalやprimitive proofは上のgate後に再利用できるため、無条件の全H4再計算は要求しない。追加が必要なのは、共通DF/native/control下の不足signal・S4 baseline・paired full-wrapper costとそのshot会計である。新計算の実行範囲・入力生成・CPU/RAMはまだ未確定である。

## GPTへ戻す判断

共通DF政策をtol-onlyにするか、旧H4内部結果を独立case studyとして残すか。比較対象を固定anchor transferとするか、各系で同じgridから選んだ最良とするか。これを先に決め、H4補完528/552件の必要性を判断する。計算実行許可と科学比較契約の採用は区別する。
