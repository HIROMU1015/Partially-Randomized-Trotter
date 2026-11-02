# Track A AX-1a：結果前契約

2026-10-09、version 1。**契約作成・静的監査の成果物。解析結果ではなく、AX-1bの実行認可でもない。**

## 1. 正本、対象、承認範囲

対象は `HIROMU1015/Partially-Randomized-Trotter`、branch `pr2-v4-s2-parallelization-20260928`、開始HEAD `dd859f591d69c5acbd3bd2007f9176acf5412fd9`。主checkoutではなく、同名の専用worktreeで作成した。原稿v0.1の証拠commitは `4c23453c541700c6a41ba71fc5ec9323b53858d6`。

上位正本はAX-0の[研究契約](track_a_ax0_research_contract.md)、[原論文・モデル対応](track_a_ax0_model_correspondence.md)、[証拠目録](track_a_ax0_evidence_inventory.md)、[比較実験仕様](track_a_ax0_benchmark_protocol.md)、[計算予算](track_a_ax0_compute_budget.md)。それらのbytes・既存科学結果・statusを変更しない。本契約は利用者による `APPROVE_AX0_WITH_MINOR_AMENDMENTS` と今回のAX-1a指示に基づく追加仕様である。

今回許可されるのはsourceのテキスト読取、保存JSON/CSVのschema・field・identity・hashの確認、新規契約3文書とJSON2件の作成・検証、それらだけのcommit/push。モデルfit、保存科学値の再解析、誤差/regret実測、科学runner/test、Hamiltonian/state生成、NPZ/NPY/pickle/runtime/cache/registryの科学データ読込、sampling/seed再生成、回路build/compile、GPU、H6/H8 pilotは禁止。Track B、未コミット資料整理、AX-0、M1〜PM-2、原稿v0.1を保存する。

## 2. 目的と達成可能な範囲

RQ-R（方式の資源と適用範囲）とRQ-P（資源予測の信頼性）を維持する。AX-1は既存H4における**モデル比較の開発診断**であり、サイズ一般性・独立確認・PRの普遍的優位を示す段階ではない。

| 問い | 入出力 | AX-1bの予定範囲 |
|---|---|---|
| RQ-P1 回路費用 | `C_pred → C_compiled`、axis別one-shot measured wrapper RZ | 単一係数・少数parameterモデルを同じ210候補で校正し、group診断。action indexは別単位。構造モデルは必要入力不足ならN/A |
| RQ-P2 精度・shot・総資源 | `(bias_pred, B_pred, C_pred) → (eligible_pred, N_pred, G_pred)` | Bのsource会計は可能。運用可能なaxis-bias predictorは現allowlistにないため、operationalなN/G/eligibility/selectionはN/A |
| 条件付き資源 | `G_conditional = Σ_a N_ref,a C_pred,a` | 参照bias/shotを使用する **CONDITIONAL_ORACLE** 診断。RQ-P2達成と呼ばない |

保存mean compile costはrandom候補で有限標本の推定値、deterministic候補で単一compile結果である。量子shot NはHoeffding会計上の必要数であり実機で取得したshotではない。32 cost samplesをN shotsと同一視しない。

## 3. 独立レビューに対するAX-1a amendments

AX-0の内容を上書きするamendmentではなく、結果前の適用条件を追加する。競合が判明した場合は本契約を `STOP_CONTRACT_CONFLICT` とし、旧正本を黙って修正しない。

| 修正点 | 固定する追加条件 | 記載先 |
|---|---|---|
| A 原論文と実装 | AX-M0/1/2を維持。QPEとfinite-time signal、FT費用とnative RZを分離。上界・heuristic・会計を別分類。旧paper_d6/F6未解決を維持 | 本書§4、[モデル仕様](track_a_ax1a_model_comparison_and_fit.md)§2、allowlistの欠測一覧 |
| B RQ-P分割 | RQ-P1とRQ-P2、四つのfactor cases、oracle flagとN/Aを全output schemaで区別 | 本書§2、[評価仕様](track_a_ax1a_evaluation_protocol.md)§2/4/7、execution draft |
| C DF target接続 | rank12 H4はlegacy。共通DF policyのH4 bridgeをAX-2要件に追加。truncation_valueのnorm保証未確認 | 本書§5、execution draftのfuture-stage prerequisites |
| D 単純baseline | 方式B0〜B3と別名のPRED_BASE_*、NNLS、従属項削除、欠測、complexity gateを固定 | [モデル仕様](track_a_ax1a_model_comparison_and_fit.md)§3〜7、execution draftのmodel configuration |

## 4. 文献と予測モデルの境界

一次文献の版・式対応はAX-0の対応表を継承する。PR原論文は [Günther et al., arXiv:2503.05647v2](https://arxiv.org/pdf/2503.05647v2)。

1. **原論文式の再現可能部分**：App. A.2のfinite paired LCU weight/probability/normalization（A23〜A29）に対応する保存入力。解析不等式の適用仮定とslackを別評価する。full QPE schedule・FT gate synthesisの同条件再現は今回の保存値ではできない。
2. **原論文の考え方の対応付け**：normalizationによるB² shot増加、deterministic/random会計の分離、basis/control/boundaryの重要性。原論文のE21をnative wrapperの数値予測へ無条件に置換しない。
3. **source由来の独立会計**：`n_det`、`E_rand`、`n_fixed`、q、finite B、既存shot式。`W_action`はaction indexでありRZではない。
4. **同単位の直接比較**：保存wrapper mean RZと単一係数・少数parameter予測のaxis別比較。構造RZモデルはbasis/event/category入力が揃った場合のみ。local compiled U_opsを使う既存helperは純解析モデルではない。
5. **N/A**：original QPE総費用対signal RZ-work、T/Toffoli対RZの直接誤差、未確認PF式からのaxis-bias保証、現allowlistによるfull structural RZ、operational shot/G予測。

`paper_d6`という旧名称と別原稿Appendix F6の対応・符号・版はU01のまま残す。原論文由来の保証、bias predictor、正当化済みPF上界には使わない。

## 5. データ層と後続サイズ検証

| 層 | 条件 | 使用目的 |
|---|---|---|
| Legacy development | H4線形1.00 Å、STO-3G、DF rank12、T=.8、二次PF、finite K=2/4、prefix0/3/6/9/12。PM-1はdiscard prefix4/5 | 主校正210、追加8の診断。既観測H4内の性能 |
| Legacy geometry diagnosis | H4線形1.30 Å、同basis/rank/T、M2の固定5構成 | 既観測geometryへの固定構成診断。新held-out、全構成最適性ではない |
| Common-policy H4 bridge | AX-2で新DF policyを登録して作成するH4 | 旧rank12 target/fragment/identityと新policyの同等性を検査。今回生成しない |
| New size validation | 同じpolicyのH6/H8。actual rankはpolicyの結果 | 初回H4モデル移送、改良後の未使用H8確認。未実行・未認可 |

新policyで旧H4のHamiltonian、fragment ordering/identity、one-body/scalar、λ_R、prefix意味論が同等と証明できない場合、**新H4 bridgeの作用/資源接続検証をAX-2の必須要件**とする。DF truncation valueと要求operator-norm errorの関係は未確認であり「DF精度保証」と呼ばない。policy決定、target生成、bridge計算は別認可が必要。

H6 technical pilotが参照bias/Cをモデル設計へ露出させた範囲はdevelopment。隔離できなければH6全体をdevelopmentとする。H8は参照値を開く前にmodel version/source/calibration hashes、feature、候補集合、selector、予測record、metricを凍結する。結果後の変更は別versionのexploratory分析となる。

## 6. 結果前仮説

期待と逆の結果も正常な結論。仮説ごとのN/Aを残し、成功例を探して候補・modelを変更しない。

| 仮説 | 必要入力・観測量 | 反証条件 | 判定不能・追加検証 |
|---|---|---|---|
| H1 構造による説明改善 | basis/relative transition、boundary、control、random event/category会計、同scope RZ。単一係数と構造モデルのgroup誤差差 | 入力取得費用を含めても構造モデルの改善がない、または悪化 | 詳細basis/eventがallowlistにないためfull H1はN/A。n_det/E_rand/q等の集約比較だけでは個別構造機序を実証しない。後続でcompile-free構造入力を登録する必要 |
| H2 複雑さの価値 | 単一係数と少数parameterの同coverage leave-one-q-out誤差、条件付きregret、収集費用 | §7のcomplexity gate不通過。単純モデル同等なら単純を採用 | fold不足・従属性・coverage不一致なら判定不能。H4 groupは内部診断、独立評価は将来H6/H8 |
| H3 cost/shotの影響 | 四factor cases、同direct setのselection、false acceptance、N/G誤差 | cost改善が条件付きselectionを改善しない。将来shotモデルを用意できればshot改善も同様に検査 | 現状predicted-shot二caseがN/Aなのでcost対shotの因果的優劣は判定不能。参照N固定のcost感度まで |
| H4 サイズ移送 | 凍結H4モデル、共通DF policyとH4 bridge、未使用H6/H8の同scope direct C、予測取得費用 | 凍結モデルが10%以上の過小評価/条件付きregret等のmaterialityを満たさない範囲を特定。PR優位を前提にしない | AX-1bではH6/H8なし。新サイズの構造がtraining span外ならextrapolationを表示。H6更新後の検査は未使用H8のみ |

「10%」はAX-0の予算・選択materialityであって、理論保証や論文成立の二値閾値ではない。H2の採用規則は別途固定する。

## 7. AX-1a完了、AX-1bのGO/STOP

AX-1a完了は、5新規成果物、追跡可能なallowlist/hash/join、fit/metric/missing規則、未解決範囲、明示的な禁止状態、静的検証、既存変更の保存、安全なcommit/pushで判定する。terminalは `AX1A_COMPLETE_STATIC_CONTRACT_ONLY`。解析値・fit係数・実測regretは成果物に含めない。

AX-1bへ進むためには、本契約の独立レビューに加え、明示的なuser launch、分析実装のsource commit/hash固定、許可入力を越えないread gate、実際のPython/package環境、割当CPU/RAM/wall/output-disk上限、出力namespace、静的schema検証の準備を別途満たす必要がある。新規解析実装は今回作成していない。予算未確認をobserved host性能で埋めない。

| 状態 | 動作 |
|---|---|
| hash/schema/candidate identity/join不一致、許可外input要求 | STOP。seed・cache等で補完しない |
| operational bias predictorなし | RQ-P2とpredicted-shot casesをN/A。cost/conditional評価までなら契約上の比較範囲は成立 |
| basis/event入力なし | STRUCT_ACCOUNT/full H1をN/A。action・simple/few比較まで |
| 単純モデルと同等、complexity gate不通過 | 単純モデルを残す。新feature、method別係数、探索gridを増やさない |
| budget/env/implementation/user launch未確定 | `DRAFT_BLOCKED_NOT_AUTHORIZED`。解析を開始しない |
| 解析完了またはnegative result | `mandatory_stop=true`、`next_stage_authorized=false`。AX-2を自動開始しない |

## 8. 成果物と追跡

本書、[モデル・fit仕様](track_a_ax1a_model_comparison_and_fit.md)、[評価仕様](track_a_ax1a_evaluation_protocol.md)、[入力allowlist](../../artifacts/resource_applicability/track_a_ax1a/2026-10-09/input_allowlist_v1.json)、[実行draft](../../artifacts/resource_applicability/track_a_ax1a/2026-10-09/execution_plan_draft_v1.json)を一組とする。最後二件のrepository-relative pathは：

`artifacts/resource_applicability/track_a_ax1a/2026-10-09/{input_allowlist_v1,execution_plan_draft_v1}.json`。

機械設定と文書が競合したらSTOPし、reviewした新versionを作る。未確定項目はnullとblocking conditionで記録する。draftは `ax1b_analysis_authorized=false`、`science_authorized=false`、`explicit_user_launch_required=true`、`mandatory_stop=true`、`next_stage_authorized=false` を保持する。
