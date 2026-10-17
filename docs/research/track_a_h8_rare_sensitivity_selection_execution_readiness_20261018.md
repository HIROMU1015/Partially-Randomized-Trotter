# H8：RARE_SENSITIVITY・SELECTION_SAVEDの実行実装・数値契約

2026-10-18 JST。準備正本 `0b68a2a20730518700de43fc89cfc90089e1df29` / SOURCE `432b76e0…`を継承し、数値解析用module・read-only Git adapter・one-shot runner・人工検査を追加した。旧sampler/旧CLIのproduction-dispatch拒否は変更していない。

**その後、利用者が実行まで承認し、Stage別の具体条件にも回答したため、新しいprivate grantで各一回を完了した。** 準備案UNAPPROVED/INVALIDと実行済み契約を混同しない。[結果報告](track_a_h8_rare_sensitivity_selection_run01_results_20261018.md)が実行状態の正本。新SOURCEは `3c0c9546701fd953de3b299be561344375c1a78b`（初版74948aa2に固定eligibility maskの実行前照合を加えた）。

## 固定範囲

| Stage | 候補・endpoint | 方法 | 未知・保証 |
|---|---|---|---|
| RARE_SENSITIVITY | q4/K4とq8/K4、prefix7/S2/r4、通常/非相殺、5精度、倍率1/10 | 未知費用を変数のまま残す2変数境界 | multi/s4の仮定値なし。μhat0・μhat_singleは保存標本の点平均。母平均保証なし |
| SELECTION_SAVED | 元union6×n32、元4精度、USUAL_SIGNED_SAVED、倍率1 | paired trajectory記述統計＋候補別product empirical bootstrap2000回 | .025/.975経験分位。候補間独立性はresamplingの仮定。厳密CI/coverageなし |

科学条件：linear H8、1.00Å、STO-3G、tol-only DF28、16 modes・sector4900、T=.8、P0、full-wrapper RZ。q4 R16 δ.2 / q8 R32 δ.1、比較対照は6軸反映後 `96e69a01…` の強化184保存最良と費用来歴。候補・適格性・保存shot・modelを変更しない。q2は選択解析union内のみ。

RAREの式：`π2*w*μ2_multi+π4*μ4 = Gdet−π0*μhat0−π2*(1−w)*μhat_single`。元law/count vectorと保存πを固定、wは保存正のmulti cellを加算し、single係数は1−wとする。保存massの丸め残差は診断欄へ残し、再正規化しない。各μは`Ncos*Ccos+Nsin*Csin`で重み付け、未知層の両軸同値は仮定しない。C<0を非負費用領域で説明する。pi4=0/w=0時は除算しない。欠測・不適格はstatus/nullとし、未知費用0補完はしない。

未知費用を別途仮定した単変数閾値の数式と人工テストも用意したが、今回の認可は2変数境界のみ。その追加数値案は未採択。元PR費用・削減率・winner・shotを上書きしない。

## 実装・入力schema

- [数値module](../../src/trotterlib/h8_saved_analysis.py)：Fraction exact平均・分散、Decimal80 SD/SE・層係数・境界、paired shot-weighting、bootstrapのempirical interval/差分/選択頻度。
- [adapter](../../scripts/h8_saved_analysis_adapter.py)：公開Git objectを44 roleのcommit/path/blob/bytes/SHA256で読み、元の監査kernelを再利用。保存n32・pilot・D0 shot・非相殺shot・強化184対照を結合。量子packageをimportせず、旧SOURCEの数値runnerを実行しない。
- [runner](../../scripts/run_h8_saved_analysis.py)：Stage別grant/config/SOURCE/approval/runtimeを照合してから固定入力を読む。未承認draftは実科学入力・不足approval/closureを開く前に拒否。
- [追加人工テスト](../../tests/test_h8_saved_analysis.py)：既存73＋追加62＝135 PASS。両Stageの人工e2e、first STOP・drift・mask・namespace/one-shot再使用拒否を含む。著者自身の技術検査であり独立レビューではない。
- [入力manifest](../../references/h8_saved_analysis_inputs_v1.json)・[構造説明schema](../../artifacts/resource_applicability/track_a_h8_saved_analysis_readiness/2026-10-18/schemas_v1.json)。schemaとID/hashの実検査はadapter/runnerで行う。JSON Schema engineを別途使用したとはしない。

trajectoryはcandidate/replica/phase/law/seed、両軸RZ整数、両軸event digestを持つ。confirmation/unconditionalだけ、各候補32、両軸event/seed/compiler/SOURCE一致、探索seedとの重複なしが必要。candidate間replica一致をpairingとはしない。endpointはbias/epsilon/factor、各候補の保存eligible/shot/費用、保存184対照、保存PR winnerを持つ。入力fileは全て固定公開blobから得られ、private科学入力は不要。元selection receipt原bytesだけは未照合で、その限界を維持する。

## 選択解析の統計契約

推定対象を、探索事前固定候補の確認平均、確認後点最良、有限union母平均最小、選択手続の母集団性能に分離。後2者と選択候補母平均の保証値はnull。ε=.0005を元探索winnerへ遡及対応させない。元固定候補は .05:q2/K2、.01/.005:q4/K2、.001:q8/K2。source支持の選択履歴と確認後K4のpoint selectionを分ける。

最小の解析familyはunion6×元4精度、通常bias/倍率1、保存eligibility mask固定。4精度で同じ候補のtrajectoryとbootstrap indexを共有し、独立標本数をendpoint数倍にしない。各replicateでcandidate別のpaired両軸合計を一度だけ計算し、全endpointへ再利用。resampling seed namespaceは`H8_SAVED_SELECTION_MINIMAL_20261018_RUN01_V1`で、candidate-keyed SHA256からstdlib MT19937へ渡す。bootstrap2000回、quantileはtype7のexact rational補間。

`statistical_method=EMPIRICAL_PRODUCT_BOOTSTRAP`、`dependence_assumption=PRODUCT_OF_CANDIDATE_EMPIRICAL_DISTRIBUTIONS_NOT_MEASURED_INDEPENDENCE`。探索n8や条件付きpilotを混合しない。候補間共分散は未推定。候補間paired構造なし、数学的独立性の証明なし。bootstrap winner頻度・empirical optimismは診断だけ。未観測tailの回収もcoverage保証もしない。

`family_alpha=null`、`diagnostic_tail_fraction=.025`。既存案α=.05は厳密保証の根拠として採用しない。正規/SEだけの厳密marginal interval、all-support費用上界なしのHoeffding、妥当なmarginal intervalなしのBonferroniを拒否する。拡張案（5精度・両bias・倍率1/10）は未実行であり、SOURCEの実行gateも今回最小範囲だけを認める。

## 認可・one-shot・STOP

[公開draft](../../artifacts/resource_applicability/track_a_h8_saved_analysis_readiness/2026-10-18/stage_contracts_UNAPPROVED_v1.json)はauthorization_status UNAPPROVED/runtime_authorization false/science_dispatch_count0のまま保持。[INVALID grant](../../artifacts/resource_applicability/track_a_h8_saved_analysis_readiness/2026-10-18/grants_INVALID_v1.json)も変更しない。旧draft digestは流用せず、新closure/plan digestを発行した。実行時は利用者の新たな実行承認とStage条件を、別private config/grantへ結合した。[消費済契約receipt](../../artifacts/resource_applicability/track_a_h8_saved_analysis_run01/2026-10-18/consumed_execution_contracts_receipt_v1.json)に再構成方法とdigestを保存。これは再実行承認ではなく、暗号署名付きgrantとも主張しない。

SOURCE closure9 file、44入力、Python実行file/使用stdlib・native module、config/grant/approval bytesを実行前後に照合。組込みmoduleは封印Python executableへ結合。保存eligible maskも実行前に固定digestと一致させる。one-shot registryはO_EXCLで予約し返却しない。既存output衝突・one-shot再使用は拒否。最初のSTOP理由を保存し、失敗でも上書きせず自動retryしない。

STOPはSOURCE/input/runtime/grant drift、pair/seed/schema不一致、欠測/非finite、保存平均不一致、自分のprocess異常・SIGTERM/OOM等。PRに不利な値・負の境界余裕自体は異常ではない。無関係worktree更新・一時PSI/CPU競合で強制停止しない。各Stage終了時にmandatory STOPを出し、もう一方へrunner自身は遷移しない。今回は両方の明示承認に基づき、primaryが別プロセス・別grantとして一度ずつ呼んだ。

共有OS/cgroup/scheduler・既存venv/package/cache・他processの優先度/affinity/ファイルは不変。新テスト専用venvだけを用意した。native control376/376は旧receiptとして保持し、保存値解析にnative予算を流用・返却/resetしない。

**次の担当：ユーザー・GPT**が実行済み結果を読んで、追加仮定・解析拡張・新測定が必要か判断する。Codexは結果保存・公開確認後mandatory STOP。新しい科学計算の承認をこの記録から推定しない。
