# M1-S：次期適用条件検証の契約候補（未採択）

2026-10-17 JST。`DRAFT_WITH_OPEN_SCIENTIFIC_CHOICES / NOT_EXECUTABLE / MANDATORY_STOP`。

## 1. この候補の役割

[Decision Packet](m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md)のD1–D11を、ユーザーが採否判断できる契約のfieldへ整理する。**完成した実行契約ではない**。数値の未決値はnull、方式選択は未選択のまま保持する。採択・実行・coverageの認定を代行しない。

最新レビューの「限定継続」は科学的推薦であり、今回の依頼は技術整理の承認に限る。既存pilot条件・結果・precision・authorization・消費済markerは変更しない。

## 2. 承認範囲

今回可能：保存文書・SOURCEの静的調査、既存数式の整理、ファイルidentity、schema・guard・保全検査、文書の公開。

今回不可：保存signalからの新shot予測・bias再採点・cellランキング、native回路生成／合成／compile、RPE simulation、追加signal/Hamiltonian/Pauli/W/v/ρ評価、科学的correctness test、新実行authorization・grant・marker。新計算用のadapterやrunnerも今回準備しない。

## 3. 継承する証拠scope

H4 linear1.00 Å／STO-3G／8 modes／保存DF rank12、G3/G6/G9、δ1/8・1/16 Ha^-1、R2・4、T1・4 Ha^-1、保存oracle、108行を**過去の証拠の範囲**として参照する。これを新RPE gridの採択や追加roundの実施許可と解釈しない。G9の絶対差unresolved、旧v5の未保存full-H/HD比較UNDETERMINEDを維持する。

## 4. 採択前に埋めるfield

|field|入力すべき内容|
|---|---|
|research_direction|限定継続／ノートとして縮小・区切る等のユーザー判断|
|target_quantity|raw単一時刻位相／特定energyのRPE、参照modelと単位|
|accuracy_semantics|高確率絶対誤差／RMSE等。失敗時損失の扱いも明示|
|accuracy_target, confidence_policy|精度値、α総額と配分。旧pilot精度を代用しない|
|time_and_branch_policy|t_0、全round/horizon、energy prior、decoder、符号規約|
|bias_and_information_policy|raw/split route、state品質、補正の可否・取得費用|
|candidate_and_baseline_policy|評価cell、同情報・同精度の対照、事後選択の扱い|
|cost_and_implementation_policy|metric、native profile、全wrapper、HD近似、合成精度|
|state_preparation_and_classical_cost_policy|包含/除外、感度域、品質と費用の分離|
|statistical_policy|IID/reuse、軸別shots、confidence、整数化とfloor|
|decision_and_coverage_policy|materiality、不確かさ、条件付きcoverage、結果分類|
|future_verification_scope|V1/V2/V3のうち必要な最小部分だけを指定|
|future_resource_caps|CPU/GPU/aggregate memory/calls/result/log等の上限。新規契約はwall/CPU time cap既定なし|
|future_source_identity|必要なSOURCE完成後にhash/commit・入力・契約へ結合|

未決fieldの値を結果への一致や過去コードのdefaultから推測して入れない。新しいscientific thresholdの数値候補も今回は設定しない。

## 5. 条件の候補

Decision PacketのP1–P3、R1、S1、C1、G1/G2を**数式の候補**として参照する。特に、b+q≤γ<π/3と全roundのbranch/最終精度、正のvisibility下界、IIDな測定結果と統計予算、同じmetricでの総費用の分離が必要になる。

これらの式が成立すること、現H4入力が仮定を満たすこと、採択済みplatform保証があることは主張しない。十分条件・Hoeffding設計上界・連続proxyを、最適性・下界・科学的GOへ読み替えない。

## 6. 実行までの順序

1. ユーザーが研究方針とtask・科学条件の採否を判断する。重要な新科学問題が生じた場合のみ、その論点について別途レビュー開始承認を得る。
2. 採択された最小範囲で、Codexが必要なSOURCE・schema・guard・非科学的検査を準備し、固定契約候補を作る。
3. 必要な契約採択と、source-boundの別途明示的な科学実施承認を受ける。資料・契約の採択だけでは実行しない。
4. 承認範囲を一度実施し、結果・不一致・未決を保全、mandatory STOP。失敗時に自動retry・threshold変更をしない。

今回この順序を自動進行しない。machineの`adopted`、`coverage_model_adopted`、`contract_grants_execution`、`scientific_execution_authorized`、`run_ready`はすべてfalse。実行entrypointは存在しない。旧one-shotを再利用することもない。

## 7. 成果物

machine候補：`artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1/next_verification_contract_candidate.json`。
本候補は既存結果を再分類せず、次のユーザー判断に必要な条件を明示する資料として公開する。

**新規科学計算0。数値forecast0。正式採択0。新execution authorization／grant／marker0。Mandatory STOP。**
