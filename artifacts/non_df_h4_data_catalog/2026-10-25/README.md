# H4 Pauli／DF比較研究：統合データカタログ

2026-10-25。基点 `9fd3084b52c7f6c9cc72b5ee04d1c13eb820f196`、専用branch `codex/non-df-h4-data-catalog-20261025`。
**保存データの所在・条件・SOURCE・保証範囲を整理した索引と転記表。新しい科学計算、数値再解析、統計集計、順位判定、作図は行っていない。** 論文本文や新しいMethodsも作成していない。

## 使い方

1. [catalog.csv](catalog.csv)で評価量・方式・状態を検索し、詳細は[catalog.json](catalog.json)の同じ`dataset_id`を読む。61項目、6実装系列を収録する。
2. [comparison_circuit_prices.csv](comparison_circuit_prices.csv)で回路価格を比較する。6系列×2軸の12行。PRは保存256本平均、PFはsingletonである。
3. CX/depth等の選択回路比較には[comparison_selected_metrics.csv](comparison_selected_metrics.csv)を使う。6行すべてcosine、PRはindex0だけであり、256本平均ではない。
4. [comparison_conditional_resources.csv](comparison_conditional_resources.csv)は既存の6系列×5精度、仮想ρ=0の30行引用表を再利用したもの。新しいN/Gは計算していない。
5. 詳しい精度・局所保証・診断は[comparison_data.json](comparison_data.json)。各`values`は保存fieldの転記で、`source`のpath・JSON pointer／CSV selectorを辿れる。`catalog_metadata`は識別・注意事項であり、新しい科学的評価値ではない。
6. 追加取得の要否は[gaps.csv](gaps.csv)／[gaps.json](gaps.json)。現在の限定比較と、母平均・実精度保証を求める場合だけ必要な証拠を区別した。候補は未採択・未実行。

CSVの空欄は**null／その要約では未収録**で、0を意味しない。`missing_metric_status`と元資料を確認する。元CSVの数値文字列、exact有理数、外向きdecimalはそのまま保持し、新しい丸め・区間縮約・換算はしない。PFの保存0やempty wrapperの実測0はそのまま残す。

## 収録内容と入口

| データ | カタログID／比較JSONのsection | 状態・範囲 |
|---|---|---|
| 科学条件・Hamiltonian・DF rank・prefix・state・compiler | `I01`、`I02_*`、`series`、`shared_conditions` | 元receipt/model/rawへの参照。Hamiltonian・state再生成なし |
| Pauli generic／central、DF native各PF/PRのRZ | `COST_*`／`circuit_price_rows` | 2軸の保存価格と不確かさ。母平均upperはPRで未保証 |
| PR価格標本、variance/covariance、両軸対応 | `RAW_*`、`CENTRAL_PRICE_STATS` | 元256本を複製せず参照。centralの257行ファイルはPF1＋PR256なので`case`を選ぶ |
| D1初回停止、後続SOURCE-Bの9構成 | `D1_INITIAL`、`D1_NINE`／`D1_configurations` | 初回FAILと後続9/9成功は別の来歴 |
| 保持/event/emptyと非加法性 | `D1_INTERACTION`／`D1_interactions` | 保存Δを引用。再減算・因果的費用配賦なし |
| D2制御方式対照 | `D2_CONTROL`／`D2_comparisons`、`selected_full_metrics` | 同じbodyのPFとPR0。control構成＋compiler応答 |
| 保存bias、u_enc、B/B_up、guard | `PRECISION_*`／`precision_axes`、`scale_rows` | 60軸の既存会計field。大きい補助scalarは元ファイルを参照 |
| C encoded-reference包含 | `C_AXES40`、`C_RESULT_*`／`C_axis_outcomes` | 40軸の保存TARGET_MET。compiled全体のcertificateではない |
| C normalization、有限grid law | `C_NORMALIZATION`／`normalization_evidence` | Pauli/DFのflagと前提を分離。formal acceptanceはfalse |
| DF L局所gate/basis/support保証 | `L_DF_LOCAL`、`L_*`／`DF_local_summary` | 128bitの局所保証と保存伝播値。C/L結合はBLOCKED |
| R0/R1の経験的残差 | `R0_*`、`R1_*` | binary64経験値。保証付き区間へ格上げしない |
| Pauli central Arb 6局所結果 | `ARB_SIX`、`ARB_*`／`local_Arb` | PF両軸、PR0/1両軸のみ。全law・order4・DFを被覆しない |
| 保存QPY・raw ray・C-targetへの接続 | `INPUT_BINDING_*`、`binary_inputs` | 保存input manifest／SHAと実在を索引化。binaryを複製・decode・作用しない |
| 5精度の条件付きN/G・比較領域 | `G_*`、`GUARD_*`／`conditional_resource_rows`、`guard_comparisons` | 仮想ρ=0とguard boxを区別。nominal CIとguard範囲も別 |
| order0/2/4、未観測component、費用tail | `RTE_RARE`／`rare_event_rows` | nominal per-event確率とimplemented trajectory質量を分離。未知費用はnull |
| 未解決の精度接続・正式受理・元終了証拠 | `CL_*`、`R2C_STATUS`、`ORIGINAL_COMPLETION`、`gaps` | 元N/G null・native UNKNOWN・R2-C未完を維持 |

元PRのevent記録は`trajectory_layout`のpath templateから、保存`trajectory_id`で`events.json`と`price_join.json`を辿れる。新しい抽選やRNG replayは不要。既存図は[前段の図表カタログ](../../non_df_h4_paper_plan/2026-10-25/figure_table_plan.json)を参照し、画像を重複保存していない。

## 比較条件を混合しないための規則

- 固定linear H4／隣接1.00 Å／STO-3G、system8＋ancilla8、S2／T=.8／q8／PR r4/K4。Pauli184非identity項・PR prefix27、DF tol-only1e-8/rank10・PR prefix1＋one-body。原本のconfigも`series`に保存した。
- Pauli genericとcentralは別の`series_id`。同じPauli trajectoryの両軸・generic/centralは対応するが、PauliとDFの同じindexをpaired標本にしない。
- Pauliのprefix27はZ/ZZ。DFはGaussian basis付きfragment/involutionを用いる。prefix選択weight・law・制御方式・compiler入力が異なるため、表現単独の因果効果ではない。
- PF raw configの`cost_samples=256`や`r/K/randomization_law`は共通configの保存値であり、PF256本やrandom law適用を意味しない。実価格標本数はsingleton。PF全保持をrawの`L_D=null`だけから未取得と誤読しない。
- 費用はstate準備を除くfull Hadamard wrapperのcompiled RZが主指標。CX/depthをRZへ換算せず、選択PR0の値をcohort平均へ移さない。
- D1隔離回路は診断用で、有効なPF/PR代替ではない。保存Δをbasisだけ・compiler passだけの費用と解釈しない。D1はgeneric系列なのでcentral内部費用へ転用しない。
- `scientific_SOURCE_commits`は実行／処理codeの来歴。`artifact_snapshot_commit`は結果ファイルを含む固定基点で、GitHubのデータURLは後者を使う。中央系列の混在した取得SOURCEは、`paired_costs.json`内の`cosine_SOURCE`／`sine_SOURCE`を優先する。
- `original_evidence_kinds`は元結果の種類、`catalog_action`は今回の転記・参照作業。実測価格、経験的標本推定、条件付き会計、範囲限定の数値包含、静的命題、終了証拠を区別する。

## 保証水準の注意

保存CのPR normalizationでは、`saved_log_B_upper_encloses`は両表現でtrueだが、`range_plus_minus_saved_B_certified`はPauliでtrue、DFでfalse。いずれも**保存encoded scopeのflag**で、compiled回路全体のcertificate受理ではない。元rawの`normalization_certificate_verified=false`も書き換えない。

Arb6局所は同じraw ray/exact norm、phase/control/+Im、B_saved一回の尺度。全PR lawのρ・DF compiled残差・正式certificateは未解決である。DF Lの局所operator上界をmeasurement残差と読み替えず、C totalへ単純加算しない。

条件付きGは保存bias/normalization・保存価格平均を用いる。ρ=0は仮想断面で、実測0ではない。guard比較も実ρ所属やPR母平均tailを証明しない。量子shot Nと価格のMC256本は別で、一条件のfailure budget=.05を全条件同時95%保証へ変えない。Pauli/DFは各表現の保存full-H referenceを使い、新しい共通厳密Hへの精度転送は認定していない。

未観測order4やDF componentの費用はunknownであって0ではない。観測最大・nominal Student-t区間は母平均費用の厳密上界ではない。DF未観測集合Uは同じ標本から選んだ記述的集合で、独立coverage試験ではない。

## 既存資料との関係と追加取得

前段の[統合報告](../../../docs/non_df_h4_comparison_synthesis_20261025.md)が科学的解釈の入口である。[論文構成案](../../../docs/non_df_h4_paper_plan_20261025.md)・[Methods草案](../../../docs/non_df_h4_paper_methods_DRAFT_20261025.md)は今回改訂していない。`catalog.json.documents_checked`に指定8資料の固定参照を収めた。

現在の固定実装・有限cohortの価格比較と条件付きGの報告に必要な保存データは利用できる。限定的報告のために新しい科学計算を必須とはしない。より強い主張を選ぶ場合の価値が高い追加証拠は、**PR母平均費用のbounds**と、**Pauli PR全law／DFのmeasurement残差接続**である。共通厳密target保証が必要な場合は、その未被覆転送も独立に扱う。

最小候補は、まず既存law・structural counter・C/L・選択Arbの未被覆範囲を設計し、その結論を左右するscalar／費用／残差診断だけを限定取得すること。対象・call数は別途固定し、**新数値解析・compile/sampling・科学計算・正式受理は今回の許可に含めない**。全cohort再計算、大量order4 sampling、R2-Cの自動継続を既定工程にはしない。詳細は`gaps`の各行を参照。

次の担当は**ユーザー**：限定比較の利用で十分か、どの強い主張に追加証拠が必要かを選ぶ。**Codex**：明示指示後に限り必要最小の準備・許可された作業を行う。研究方針の正式採択・GPT詳細レビューは今回開始していない。

## 照合・保全・公開

既存の6系列価格・30行N/G引用表、図表map、C/L/Arbの保存要約を優先して再利用した。標準libraryのJSON/CSV field抽出とpath／SHA照合のみを使用し、科学entrypointは呼ばない。新しい汎用管理・承認証明・科学grantの仕組みは作成していない。

[verification.json](verification.json)は引用fieldの等値、参照、既存SHA、差分範囲の**主agent自身による構造確認**で、独立科学レビュー・最終evidence freezeではない。大きいraw標本・QPY・stateは参照し、複製しない。QPYの保存SHAは入力manifestから引用し、今回そのbinaryを再hash/decodeしたとは記録しない。

旧SOURCE/raw/結果/grant/ledger/marker/STOP、初回D1 FAIL/原因UNKNOWN、元native UNKNOWN、旧actual N/G null、PR全lawρ/null、正式受理falseを維持。後続native正常終了で失われた元終了証拠を代替しない。今回のデータ索引追加は、科学証拠statusの変更ではないため旧validation manifestは更新しない。

最新版共通指示2026-10-25の§8.4/§10/§10.3を適用。共有設定・他ユーザー・他jobへの操作なし。[publication_status.json](publication_status.json)に公開状態を記録し、専用branchのcommit・GitHub公開・remote SHA照合後、**mandatory STOP**。新科学実行・研究方針採択へ自動移行しない。
