# H4 DF10 Mq：独立q検証のPF結果とPR技術STOP

2026-11-10 JST。**未使用PF q32ではMqがM0より正確だった。PR q16はadapterの引数不一致でbuild前に停止し、移送精度は未判定。長時間RPEへの使用には、未取得PR価格、さらに長q・law期待値・係数/系統不確かさの検証が残る。** 本campaignは部分完了であり、全指定範囲の成功とはしない。

[RESULT.json](RESULT.json)／[PF残差CSV](PATH_RESIDUALS.csv)／[取得前予測68軸](PREDICTIONS.json)／[PROVENANCE](PROVENANCE.json)／[事前PLAN](PLAN.md)。PR未取得価格・残差はnullで保持した。[継続修正案](TECHNICAL_COMPLETION_PLAN.md)は準備済み・未実行。

## 1. 固定条件と来歴

- Linear H4/1Å/STO-3G、保存tol-only1e−8 DF10/sector(2,2)、保存DF-H ground target/数値state条件付き/P=0。S2、δ=.1、PF B0/p7、PR p1/R4/K2、元symmetric_directional・測定付きfull Hadamard wrapper、X/Y。
- 元Qiskit1.3.0、rz/sx/x/cx、opt1/seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。S1/元priceと同じ設定。
- 実行SOURCE：**`55eccab8865684fadefa5bd79875b1154518b410`**。モデル・cohort規則・PLAN・4人工testを新価格取得前にcommit。元builder/sampler/compileはcarrier155529e2、実providerはc31ef424の固定版と主要使用fileを照合した。
- M0は変更なしのV1 Accounting。V1 `predict`先頭の2つの旧q scope guardだけを今回用adapterに置き換え、計数本体をそのまま再利用。旧PF q8/q16・PR q8の人工入力で元関数との全出力一致を確認した。元V1ファイルは変更しない。
- MqはS1 carrier `cefc01622a3cd75dd16d30de7b9fb68274f3eddd`の係数bPF=606.75、bPR=176.88888888888894を固定。両軸同値。新fit・MqB・特徴探索なし。
- H/state/snapshotはPROVENANCEの保存hashへbinding。H `c1a6c428…`、state `9bcc0b0e…`、snapshot `12219b7d…`。PR preparationは元の`b99720d6…`と一致。PF p7 preparationは`fc83f26f…`。完全hashは保存provenance/RESULTを正本とする。
- PROVENANCEのscientific_binding.T=.8は元入力のreference metadata。今回の時刻はcandidateのPF T3.2/q32、PR T1.6/q16を正本とする。既存εE=.00159360010199040 Ha、αtotal=.05/方式、βtotal=.4、h=.1、13roundは変更せず、N/signal/energy資格を再計算しない。

## 2. 資産・cacheと今回の取得範囲

元155529e2の既知6価格表に、対象PR q16/T1.6/δ.1、PF q32/T3.2/δ.1の価格はなかった。PR q32/T1.6はδ.05なので代用しなかった。元cacheは読取専用とし、実wrapperのcanonical numeric fingerprint・compiler一致時だけ再利用する設計とした。

実際に取得できたPF X/Yのfingerprintは`803bc9b3…`と`f778f84a…`。元cache hitは0、新compileは2。自分の専用cacheへ保存し、元cacheを更新しなかった。trialとwrapper/axisのbindingはRESULTに保存。

| 対象 | 固定candidate ID | sampling/予測 | 直接価格 |
|---|---|---|---|
| PF B0/p7/q32/T3.2 | RESULTのPF candidate | 決定的1回路、両軸予測固定 | **X/Y各1、完了** |
| PR p1/R4/K2/q16/T1.6 | `64e4f7e9ec7fba28815e` | 新33 trajectory、両軸予測固定 | **0/66軸、未取得** |

cohortは`H4_DF10_V2_INDEPENDENT_Q_20261110`、path0…32。元seed関数のnamespace/cohort/candidate/index/outer/pieceから528 seedを固定し、明記した既存registry/exclusion/cohort seed集合との衝突0を確認。V1の探索9/確認33本を再利用していない。全履歴被覆・PRNGの数学的独立性は主張しない。全event/application順とseedは専用private領域へ保存した。

68軸のM0/MqをPREDICTIONSへ保存してからcompileを開始し、その時点の観測価格参照数は0。並列予測は34 worker、compileは元代表gate後に残PRを解放する方式。PR代表が停止したため残32本はbuild/compileへ解放していない。shared設定・他worktree/processは変更しなかった。

## 3. PF q32：独立価格への移送

値はRZ/軸、残差はcompiled−prediction。X/Yで同値だが独立標本数は倍にしない。

| モデル | 予測 | 直接compiled | 残差 | 絶対相対誤差 |
|---|---:|---:|---:|---:|
| M0 | 157493 | 176723 | +19230 | 10.88144% |
| **Mq** | **176909** | **176723** | **−186** | **0.10525%** |

PFはn1の決定的回路なのでMAE/RMSE/最大絶対誤差はいずれもM0で19230、Mqで186。標本分散・SD/SEは付けない。旧S1の既知q16ではMq残差−62、絶対相対誤差約0.0700%だった。未使用q32でも改善したが、残差は0でなく、q4096への一様誤差保証ではない。新しい1/2/5%閾値を採択しない。

元gateは全7項目通過。低control誤差約5.89e−14、独立occupation vector差約2.16e−13、compiled-vector X/Y差約2.04e−11/1.87e−11で元1e−7以内、位相alignmentなし。入力stateでの有限correctness検査であり、全入力の数学的証明や新しいRPE signal資格ではない。

## 4. PR停止：モデルの否定と区別する

こちらのadapterの`sampled(row, cohort, index)`でprep引数を落とし、元`cost_trial`の`sample_requests(row, prep, cohort, index)`と不一致になった。代表path0は`TypeError: sampled() takes 3 positional arguments but 4 were given`で停止。PRのsamplingは親processで33本完了・保存済みで、このworkerはbuild/compile前に停止した。

PFは有効な独立taskとして最後まで完了し、controllerはその結果を回収してcampaign STOPを保存した。自動retry/resumeは0。今回指示§11の「新しい科学campaignで停止条件に達した場合、無断で再試行しない」に従い、PRを再実行しなかった。

PR q16のcohort平均・path別誤差・残差variance/SD/有限平均SEは**未評価**。S1 q8のMq確認RMSE252.244 RZ/軸、平均残差+18.131 RZは旧条件の保存結果であり、q16へ流用して成功・失敗を判定しない。

## 5. 残る不確かさと次の最小作業

PF q32の一点は、固定bのq方向延長の有効性を支持する有限証拠。PRのq方向は直接価格がなく未判定。係数不確かさ・model systematic・未観測event寄与を0扱いせず、全law期待値・長q・全13roundの価格/総費用/順位/CIは未資格のまま。N/N×C/G、再fit、候補最適化、詳細科学レビュー、Main Claim変更は0。

最小の継続は、新trajectoryやPF再compileを増やさず、保存したPR33本と固定予測から未取得66軸を完成させること。[継続adapter](../../../scripts/complete_h4_df10_price_v2_independent.py)は既存`h46_same_event.decode_event`を再利用し、正しい4引数callbackでrequestを復元する。RNGなしの人工roundtripとcallback regressionの2 methodはPASS。新sampling・新M0・PF再compile・fitは含まない。

**継続案は未実行。ユーザーがSTOP後の残PR66軸の継続を明示許可した場合にだけ、Codexが別の識別可能な出力へ実施する。** 元run01のSOURCE/STOP/raw/予測/PF結果を保全する。追加qやtop-up、全RPEへ自動進行しない。

## 6. 保全・公開

実行前人工4 method PASS（うち元compilerの人工X/Y2 wrapper）、停止後の継続準備人工2 method PASS。実H4はsampling33、予測68軸、新wrapper compile2/PFのみ、PR compile0。V1/V2設計/S1のSOURCE・結果・不一致判定と旧grant/ledger/marker/KEEP/STOPは変更しない。

REPORT・部分RESULT/CSV・PREDICTIONS/PROVENANCE・未実行継続案を公開し、remote SHAと主要成果物を照合後mandatory STOP。**担当ユーザー：残PR66軸の継続可否を判断。担当Codex：明示許可後に保存cohortを再利用して完成させる。**
