## Track B R1一回結果・mandatory STOP（2026-10-06）

固定S `d43d64a821a0249a0dfab12a2472bd3a72fdee74` →直接子authorization-only A
`411f08f768244fe87b600d82308c3851847fe9e4`からrun1/retry0。
[結果照合](r1_one_shot_result_validation_20261006.md)：126 keys / 264 rows / 132 controlled tasks完了、全task適格。
terminal R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW、[保存field専用監査](../../../scripts/tracks/algorithm_codesign/audit_r1_saved_result.py) PASS。
primary distinct-basis controlledではB²改善とnative/shot資源のtrade-offを保存し、自動研究GOはない。
[全264 rows CSV](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/resource_rows_display_v1.csv)、[evidence manifest](../../../artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)、
[GPT判断への入口](r1_post_run_gpt_review_request_20261006.md)。原result/marker/source、既存証拠・共通API・Track A保持。
science終了後mandatory STOP、追加合成/target/grid/分子/DF/trajectory/GPUは行わない。
研究方針・RQ・新規性・着地点・追加検証の必要性/範囲はGPT側。
以下のpending/最新記述は当時の履歴として本文をそのまま保持する。

# Track B — Algorithm Co-design

## Latest: R1 v2 source preparation / GPT source review

[Preregistration v2](rte_reallocation_r1_preregistration_v2.md)、
[native semantics](rte_reallocation_r1_native_semantics_v1.md)、
[GPT source review](rte_reallocation_r1_source_review_request_20261006.md)。
ordinary/PTSC-K0/A、PauliのみCTS control。canonical、distinct-basis controlled primary、resource vector。
27 focused tests、42角度/126 keys、264予定resource rows。実synthesis/cost取得0。
authorization pending、RUN_READY=false、旧証拠/STOP保持、mandatory STOP。以下は判断履歴。


## Latest: R0.5 result-prior equivalence audit — 2026-10-06

[監査](rte_reallocation_r05_equivalence_novelty_audit_v1.md)、
[一次資料表](rte_reallocation_r05_primary_source_locator_table_v1.md)、
[symbolic readout](rte_reallocation_r05_symbolic_comparison_readout_v1.md)、
[GPT handoff](rte_reallocation_r05_gpt_handoff_20261006.md)。
指定P0–P3内の限定 `METHOD_DELTA_CANDIDATE` / `CONDITIONAL-R1`。
同I0 ordinary/PTSC K0のnorm差とI1 collected CTSの優位を分離。R1必要性はGPTへ返す。
科学/synthesis/compile/分子/DF/NPZ/GPUなし、旧証拠/STOP保持、R1未認可、mandatory STOP。


## Current: R0 finite-mean RTE reallocation audit (2026-10-06)

- [Review packet](rte_reallocation_r0_review_packet_20261006.md): claim status, independent proof, primary passages, fixed exact fixtures and source identity.
- Broad reallocation/common-angle principle is known; specific class theorem priority and resource value remain unresolved. R1 proposal is conditional only.
- Existing BF/BM/FR/BS/SP conclusions stay fixed. RUN_READY=false; no science authorization; mandatory STOP and GPT research review.


最新は[BS-0.5設計監査](bs05_method_target_design_audit_v1.md)（2026-10-06）。
[ordinary有限RTE形式仕様](bs05_ordinary_finite_rte_baseline_v1.md)、[pilot案v2](block_synthesis_pilot_amendment_v2.md)。
branch `track-b-bs05-design-audit-20261006`、base `2c8c022db39c3582d6175fcf25a6752037727a21`。
現candidateのnew-method deltaなし、専用armを外す。O/C別、多資源Pareto、control4/performance8。
[監査manifest](../../../artifacts/track_b_bs05_design_audit/2026-10-06/audit_manifest_v1.json)。
実装/science/tests0、RUN_READY=false、mandatory STOP。application/closureはGPT reviewへ戻す。旧v1と証拠は維持。
以下は判断履歴。

最新は[SP-1後GPT reviewのblock合成・数学/API仕様](block_synthesis_design_review_20261006.md)（2026-10-06）。
[claim/対照表](block_synthesis_claim_and_baseline_matrix_v1.md)、[小型pilot未承認案](block_synthesis_small_pilot_proposal_v1.md)。
branch/worktree `track-b-post-sp1-block-synthesis-design-20261006`、base `9d2bb1fa439748b02084bd9fbc9b10a705328f8a`。
有限精度coherent対象とphase/誤差/資源をそろえる設計。短いfinite-RTE平均Mは未実証の構成候補。
[設計manifest](../../../artifacts/track_b_block_synthesis_design/2026-10-06/preparation_manifest_v1.json)はauthorizationではない。
実装/science/tests0、RUN_READY=false、mandatory STOP。具体domain/辞書/metric/予算採否はGPT reviewへ戻す。
既存SP結果・BF/BM closure・A証拠境界を維持。以下は判断履歴。

最新は[SP-1一回結果・保存値監査](sp1_one_shot_result_validation_20261006.md)（2026-10-06）。
S `0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`→direct authorization-only A `9630e06122172af238ac5219bec6dfc8b01ca837`。
独立branch/worktree `track-b-sp1-one-shot-execution-20261006`でrun1、48 rows／96 axes完了。
SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW、適格42／モデルshot cap6、technical failureなし。
[raw result／consumed marker／audit／manifest](../../../artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/)。
CのDRはn=8/16でgain、32でloss、64でcap。CのD-only/R-only material gainは0。
mandatory STOP、retry0、追加science未認可。[GPT研究再評価依頼](sp1_post_run_gpt_review_request_20261006.md)へ戻す。
actual DF/RTE／compiled費用／新規性は未実証。以下のsource準備・未実行statusと旧closuresを履歴として保持する。

最新は[SP-1採用済み結果前契約・source最終review](sp1_wrapper_preregistration_v1.md)（2026-10-06）。
GPT reviewの必須2修正を反映：role/mask非依存pre-placement fusionと、加法的保存primitive T費用。
12 wrappers/16 pathsのstatic fusion監査は全0。実adapter/runner・59 focused testsはlocal pass。
[source manifest](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)。
branch `track-b-sp1-wrapper-source-review-20261006`、base `f974f8e7caa5a2769255c0483d7e40caae0a3930`。
science resource/signal sweep0、RUN_READY=false、別SP-1実行authorization未認可。
新source Sを公開後、GPTの最終source reviewへSTOP。review commitから実行しない。
以下の提案・未完了記述とSP-0.5 consumed marker/closuresを履歴として保持する。

最新は[SP-1 wrapper累積・placementの契約案](sp1_wrapper_preregistration_proposal_v1.md)（2026-10-06）。
受領review `PROCEED_TO_WRAPPER_ACCUMULATION_AND_PLACEMENT_PILOT`に沿った準備。
12 synthetic wrapper／4 mask／一π/4 catalogueの具体案、共通Fraction会計／17 focused testsのlocal pass。
Sparse PS本文ではwhole-circuit T費用とcrossoverも既知。SP-1結果・新規性・actual DF/RTE利益は未取得。
[準備manifest](../../../artifacts/track_b_sp1_wrapper_preparation/2026-10-06/preparation_manifest_v1.json)。
独立branch/worktree `track-b-sp1-wrapper-preparation-20261006`、base `e57c1fdd28589422e9c973e34e53f6c725b921e9`。
common fusion／数値interval・ordered channel adapter／runnerは未完了。science execution unauthorized、RUN_READY=false。
必要資料をGitHub公開後STOPしてGPTの具体契約案reviewへ戻す。SP-0.5一回結果・markerと過去closuresは保持する。
以下は受領前の履歴。

最新は[SP-0.5 one-shot結果・保存値監査・GPT handoff](sp05_one_shot_result_validation_20261006.md)（2026-10-06）。
source `65f6fcdb3dc1ad8bfccfaee6e1413336aef91184`、direct authorization-only child `9477cd2fcfca69f3f24b801770a1f02805907eac`。
別実行branch/worktree `track-b-sp05-one-shot-execution-20261006`でrun1。PRIMITIVE_TRADEOFF_EXISTS。
23 keys・16 rows、ordinary/control各strict7、zero-cost ordinaryとJ=1 controlledを保持。
[元result／marker／logs／監査manifest](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/)。
cap／runtime／numeric failureなし、retry0、科学run後mandatory STOP、追加scienceなし。
known PAIのprimitive交換関係の実装確認。新規性／DF-native／D-R placement／wrapper GOを意味しない。
結果・方針判断はGPTへ返す。source review準備 `b0fa5af`は実行HEADに使用せず保存した。
以下の未実行・authorization待ちは結果前の履歴として保持する。


現段階は[SP-0.5経済性gate・source review](sp05_synthesis_economics_preregistration_v1.md)（2026-10-06）。
GPT review `PROCEED_TO_SYNTHESIS_ECONOMICS_GATE_BEFORE_PRIMITIVE_PILOT`を受領。
pygridsynth 2.0.0、exact π/4 catalogue一つ、8 target／23 keys、strict J<1の存在可能性判定を結果前固定。
新namespace `src/trottertracks/algorithm_codesign/synthesis_placement/`、専用runner／34 focused testsがlocal passed。
登録target合成／J採点0、science authorization=false、wrapper pilot未認可。
[preparation manifest](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/preparation_manifest_v1.json)、
[dated note](../../research/研究ノート/2026-10-06_track_b_sp05_economics_preparation.md)。
独立branch/worktree `track-b-sp05-economics-preparation-20261006`。公開後STOPしてsource reviewへ戻す。
B-F／BM現route closure、旧BM-1未実行と過去STOPは維持。以下は受領前の履歴として読む。


最新は[BM-0.5後review・合成placement設計仕様案](synthesis_placement_design_review_20261006.md)（2026-10-06）。
GPT側の方針reviewを受領し、B-F現仮説／B-M現adapterのnew-method路線を閉じる。旧BM-1は未実行を維持する。
新候補は固定DF/PF/finite-RTE wrapperの合成cost・測定負担を含むrandomization placement設計。
四mask、ancilla込みmean、weight／bias／shot／T、限定inventory、原始pilot案を一文書へ集約する。
一般原理は既知、利益・独立新規性は未実証、実装／pilot未認可、science操作・testsは0、STOP。
[設計状態・provenance JSON](../../../artifacts/track_b_synthesis_placement_design/2026-10-06/design_contract_v1.json)、
[dated note](../../research/研究ノート/2026-10-06_track_b_synthesis_placement_design.md)。
独立branch/worktreeは `track-b-synthesis-placement-design-20261006`。以下は元の固定結果・判断履歴として保持する。

最新は[BM-0.5同値性監査・GPT packet](bm05_review_packet_20261005.md)。
compact再帰=BM floor/internal三次係数。同backend／同集約のscoreは同じ、共通項reuseも強い対照に可能。
現adapterのnew-method gateは不通過。BM-1は実行せず、application／別deltaの採否はGPTへ戻してSTOP。
[BM-1 v2 amendment](bm1_pilot_scope_amendment_v2.md)はleading heuristic／I2、primary count／secondary人工costを記録する。
独立branch/worktree `track-b-bm05-equivalence-20261005`。以下のBM-0原文・manifestは固定commitの履歴として保持する。

最新は[BF-A後GPT review・BM-0設計packet](bm0_review_packet_20261005.md)。
[B-F現行仮説を限定negativeとして閉じる](bf_current_hypothesis_closure_20261005.md)review判断を記録する。
B-Mはnative列・DF評価・小型pilotの**未実証の候補**。新scienceは0、BM-1実行は未認可、mandatory STOP。
文書作業は独立branch/worktree `track-b-bm0-design-20261005`。以下は過去のhandoff・実行履歴として保持する。

研究Bの全面再設計をGPTへ渡す入口は[固定資料handoff index](research_redesign_handoff_20261005.md)。
最新R0・過去STOP・構造結果・最新Aの別commit・既存提案入力を辿る。docs-only公開、研究方針の採否はGPT側、mandatory STOP。

最新は[BF1-R0復元結果・GPT handoff](bf1_read_only_recovery_result_validation_20261005.md)。
`BF1_READ_ONLY_RECOVERY_COMPLETE`、事後のpreregistered primary復元はBF-A。F/L有限最小値は同一。
新science data/rerunは0、bridgeは不足cellを記録し、mandatory STOP。原resultはINCONCLUSIVEのまま保持する。
result/audit: `artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1/`。以下のR0準備記述は履歴。

新たな利用者指示は[BF1-R0契約](bf1_read_only_recovery_contract_v1.md)。別branch `track-b-bf1-read-only-recovery`で
既存1269 cellだけの一回のdeterministic replayを準備し、全outcome後STOPする。science rerunは未認可。
module: `src/trottertracks/algorithm_codesign/recovery.py`、runner: `scripts/tracks/algorithm_codesign/run_bf1_read_only_recovery.py`、
test: `tests/tracks/algorithm_codesign/test_bf1_recovery.py`、contract: `artifacts/track_b_bf1_read_only_recovery_contract/2026-10-05/`。

B worktree: `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign`。
branch: `track-b-algorithm-codesign`。Aの最新worktreeとは分離する。
このcheckoutのA文書はM2 result commitのsnapshotであり、並行して進むAの現在状態を更新する場所ではない。

原BF-1は一回実行後の`BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY`、`INCONCLUSIVE`を保持する。
source `e59344a`、authorization-only child `cc971e4`を固定して実行したが、NumPy整数のJSON保存例外で中断した。
入口は[BF-1結果照合](bf1_one_shot_result_validation_20261005.md)。retry、BF-2は未認可。
利用者指示により、研究方針全体の修正はGPT側、細かな検証はCodex側で扱う。
別branch `track-b-bf1-serialization-repair`の[JSON保存修正・限定検証](bf1_serialization_repair_review_20261005.md)は
実装上の修正のみで、旧source/authorizationによる科学再実行を認可しない。
`296ec7e`の[旧最終source review](bf1_final_source_review_296ec7e.md)と以下の契約・準備packetは結果前履歴。
[BF-1 preregistration v1](bf1_preregistration_v1.md)、[execution gate amendment v2](bf1_execution_gate_revision_v2.md)、
[assembly guard amendment v3](bf1_assembly_guard_revision_v3.md)、
`artifacts/track_b_bf1_preparation/2026-10-05/v3/`を一組として読む。
v1/v2準備packetと[旧実装review packet](bf1_implementation_and_review_20261005.md)は履歴として保持する。
one-shot markerはconsumedのまま保持する。全outcomeでmandatory STOP後、研究Bの方針を全面再評価する。

BF-0のreviewされた履歴は[数学契約](bf0_mathematical_contract.md)、[claim監査](bf0_prior_art_claim_matrix.md)、
[pilot proposal](bf1_minimal_pilot_proposal.md)、[review request](bf0_external_review_request_20261004.md)。
BF-1の具体化は新preregistrationへ記録し、review対象の本文を無言で書き換えない。

code: `src/trottertracks/algorithm_codesign/`、runner: `scripts/tracks/algorithm_codesign/`、
test: `tests/tracks/algorithm_codesign/`。`src/trotterlib`の共有API・Aのartifact/status/contractは変更しない。
repair reports/audit: `artifacts/track_b_bf1_serialization_repair/2026-10-05/`。
