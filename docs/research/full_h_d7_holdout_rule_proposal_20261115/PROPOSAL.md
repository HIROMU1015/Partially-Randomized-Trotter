# Full-H κ2 D7：独立holdoutの観測前比較規則案

**規則案公開・ユーザー採択待ち。第一案は記述的比較とし、モデルPASS/FAILを付けない。**
本書の公開・採択は、holdout取得・開封・比較計算の実行許可を意味しない。
Cのfit完了を待たずに規則を提案する。Cの予測値は今回取得・計算していない。

## 1. 採択済み規則の確認と、この提案の根拠

確認範囲は下記固定資料、関連Full-H文書履歴・remote ref名、今回の明示指示。
**Full-H固有の採択済み数値許容差・比較規則は確認できなかった。**
同等のFull-H規則案も確認範囲に見つからず、既存の記述式を参照してFull-H向け条件だけを整理した。

| 固定資料 | 確認内容 |
|---|---|
| A `7a0c9468b2499fac9fa1cd279ae2980c9dab048e` / `docs/research/a_full_h_p2_pauli_b1_price_design_20261114/REPORT.md`、`A1_FULL_H_P2_PLAN.md` | D7とH-dは当時の提案。人工SEは実価格の分散推定ではない。H-d用m6 n2は追加案 |
| D `a43050c786c741d267ddf322232d888780efc03c` / `docs/research/d_full_h_p2_caller_preparation_20261114/NEXT_SCIENCE_PLAN.md` | 当時はD6/D7/D7_H2とも未採択。許容差・confidence・区間contractはUNKNOWN |
| D1 `e628c0d69c7da559369d99f323e3656a96c0728a` / `docs/research/d_full_h_p2_calibration_resume_20261115/HANDOFF_C.md`、`FROZEN_JOBS.json` | 後のD7採択・較正5 cell/10 record引渡し。plan.adopted=true。一方、較正実行scopeのholdout_compiles=0で、後続6件は実行対象外 |
| C `c628e9e8447c11c984064b922cbce6c52dc72d48` / `docs/research/paper_g_price_routes_20261114/PRICE_VALIDATION_PLAN.md` §7 | A8は許容差の採択事項、H-dは区間契約を要する選択肢。提案を採択済みとは扱わない |
| Pauli先例 `f4480ee3f831a7b106d287c401ee78fbd62f8615` / `docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/HOLDOUT_RULE_PROPOSAL.md` | 記述式・欠測と精度の分離を参照。Pauliの条件・予測・採択状態をFull-Hへ移送しない |

古いSOURCEに残る「D7未採択」の表示と、後のD7設計採択を区別する。
**D7の設計採択・較正実行許可は、A8/H-dの採択でも後続holdoutの実行許可でもない。**
今回の明示指示も規則の提案までであり、新しい閾値を採択していない。

用途は、固定3係数モデルをFull-H κ2の指定m5/m6へ適用した際の有限cohortでの差を調べる診断。
モデルの特徴量は`FULL_H_GROUP_V1=[one,event_count,outer_boundary_count]=[1,qR,q−1]`、
形式は`a+b(qR)+c(q−1)`。期待価格の資格を意味しない。
現在の資料には用途に結び付いた数値誤差予算・有効な区間contractがなく、較正残差・人工SEから
許容%は定まらない。決定論PFの2%を流用せず、数値閾値なしの記述的比較を第一案とする。

## 2. 比較単位と固定条件

対象は **m5-X、m5-Y、m6-X、m6-Y** の4件。Pは、Cがholdout観測前に固定した
同じround・同じ軸の保存予測値を読む。係数からの再予測・再fitはこの比較に含めない。
Cの固定commit・path・4件のselectorは[HANDOFF](HANDOFF_D1.md)の記入欄で後から結合する。

| round / cell | q | R | T | cohort内のfresh path | O（軸aごと） | receipt |
|---|---:|---:|---:|---:|---|---|
| m5 / M5_ROUND | 32 | 283 | 3.2 | 2 | 同じ軸のpath0・path1の算術平均 | p0X、p1X、p0Y、p1Y |
| m6 / M6_ROUND | 64 | 566 | 6.4 | 1 | 同じ軸のpath0の観測値 | p0X、p0Y |

固定metadataはD `a43050c786c741d267ddf322232d888780efc03c` の
`docs/research/d_full_h_p2_caller_preparation_20261114/PROPOSALS.json`、
`proposals.D7.plan.cells`と`proposals.D7.preflight.cells`。
後者の配列index 5/6がM5_ROUND/M6_ROUNDで、固定row・cohort・step_seedsを保持する。
元row.idはm5が`FULL_H_bc8aab34256b73b68025`、m6が`FULL_H_4ec8d73571760a6830b8`。
cohortはそれぞれ`D_FULL_H_P2_D7_M5_ROUND_PROPOSED_V1`、`D_FULL_H_P2_D7_M6_ROUND_PROPOSED_V1`。
C/D1で別形式のphysical IDを使う場合も、同じ固定row・physical bindingとの対応を確認する。

H4 linear1Å・STO-3G・保存tol-only DF10・one-body込みFull-H/L_D=0/K=2/κ2、group時間0.1。
route=`FULL_H_RTE`、lowering=`CONTROLLED_ZZ_CX_CRZ_CX_V1`、
scalar=`HAMILTONIAN_CONSTANT_PLUS_EXTRACTED_FULL_H_IDENTITY`を総Tで一回。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、transpiler seed17、P0、symmetric_directional、
測定付きfull Hadamard wrapperのcompiled RZ数（状態準備費除外）。law・SOURCE・profileは固定metadataを継承する。

X/Yは同一pathの別測定軸で、標本数を合算しない。3 fresh path・6 receipt・4比較単位を区別する。
**後続範囲はD7の6 compile（m5の4、m6の2）を維持**する。m6 n2のD7_H2は別設計であり追加しない。
P-1配線標本・較正path・paired親子をfresh holdoutへ差し替えない。

## 3. 採択後・別途実行許可後に使う比較式と報告

Z[m,a,i]を固定path i・軸aの有効なcompiled RZ数とする。

```text
O[5,a] = (Z[5,a,0] + Z[5,a,1]) / 2
O[6,a] = Z[6,a,0]
signed_error[m,a] = P[m,a] - O[m,a]
signed_relative_error[m,a] = (P[m,a] - O[m,a]) / O[m,a]   # O > 0
```

各4行にP、O、符号付き誤差、符号付き相対誤差、単位の状態・理由、予測/receipt locatorを報告する。
正の誤差は過大予測、負は過小予測。整数／有理数のexact値を正本とし、百分率だけ表示時に丸める。
欠測・無効行のOと誤差はnullとし、Pは固定予測locatorとともに残す。
O=0は相対誤差をnull、理由`ZERO_OBSERVED_PRICE`とする。0/0も0とはしない。
P−Oは報告できるが、分母をP・1・epsilonへ変更しない。exact一致も合格とは呼ばない。

6個別receiptに実行SOURCE・公開SOURCE、round/軸/physical ID、cohort/path index/seed、
fingerprint・元compile record・既存作用/実装gateを結び付ける。
平均・誤差の算術より先に、必要なpath/receiptと固定physical/law/SOURCE/profile/scalar/loweringの対応を確認する。
gateの適用範囲は採択された実行契約どおりとし、別pathのPASSを未検査pathのPASSへ転用しない。

| 状態 | 比較・報告での扱い |
|---|---|
| 必要なpath/receiptが欠ける | 当該単位は欠測。m5の片方だけを2 path平均として使わない。0や予測値で補完しない |
| 軸・SOURCE・physical binding等が不一致、取得/実装gate不成立 | 当該単位は無効／保留。価格の精度FAILへ読み替えず、理由と部分receiptを保全してSTOP |
| 4単位が有効、全O>0 | campaign=`DESCRIPTIVE_COMPLETE`。4行の観測差を記述する |
| 4単位が有効、O=0を含む | campaign=`DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED`。未定義理由を残す |
| 1単位以上が欠測／無効 | campaign=`INCOMPLETE_OR_INVALID`。確認済み有効行を保全し、4件完了としない |

いずれも`model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE`とし、モデルPASS/FAILを付けない。
取得・gateのPASSやcampaign完備性はモデルの精度認定ではない。X/Yやround間の平均で4行を置き換えない。
閾値が未採択なので「許容差超過」の判定も行わない。

## 4. 主張範囲・STOP

記述できるのは、凍結予測と指定fresh有限cohortの4件の観測差まで。
m6の1 pathから標本分散・SE・CIを推定せず、X/Yを独立標本として代用しない。
有限cohortの差から全lawのE[C]、CI/confidence、sampler/ideal law距離、rare寄与、
境界一般化、m7以降の長roundへの移送資格、N/Gや方式順位を認定しない。モデル資格は自動昇格しない。

同じ4観測を見て閾値を選ぶ場合は事後的・探索的と明記し、その閾値での判定を事前固定の独立検証と呼ばない。
数値的なPASS/FAILが必要なら、観測前に用途・分母・閾値・境界・campaign規則を別途採択する。

D1は未解決のidentity・receipt・必要gate、取得失敗時に部分証拠を保全してSTOPし、
無断retry・追加path・seed変更・再fit・feature変更をしない。比較報告後もmandatory STOP。
今回C2は文書と既存metadataの照合だけを行い、holdout観測価格の探索・開封、P/O/誤差の計算、
fit、sampling、compileを行わない。**規則案公開・ユーザー採択待ちでmandatory STOP。**
