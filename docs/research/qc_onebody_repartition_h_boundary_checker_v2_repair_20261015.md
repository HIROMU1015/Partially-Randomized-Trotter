# H4 h境界checker v2：技術修正と再監査準備

ユーザーの「修正まで行って」に基づくchecker修正・人工検査・新scopeの実行前準備。基準は `3dd8acecbe3fe4e18d11b7e462906706921c8152`。旧収集source `5e3341c12bd84f30f19072cefa590f4499c36c38`、契約、一次データ、失敗記録、消費済み認可は不変。実H4の数値再監査は今回行っていない。

## 修正した不整合と確認範囲

旧checkerはproducerの保存行列 $U_s$ に基づく `numerical_allowance` を、独立Padé再構成 $U_p$ の残差・unitarity defectから計算した値と等しいとして比較した。同じPFの近似でも両行列の丸め誤差は異なり、$1/h$ による拡大のため保存幅を再現する検査として不適切。円上spreadの表式にも演算順序の差があった。

小さい登録外人工例では、保存行列と再構成行列の差が既存の $10^{-9}$ 行列一致条件を満たしても旧checkerはNUMERICAL_ALLOWANCEで停止することを再現した。修正版は保存行列自身から残差・unitarity・orthogonality・円上spread・幅を再計算し、保存フィールドの自己整合性を検査する。mean位相・spreadは保存定義と同じ演算表現、cluster識別は独立した円上connected-component実装。production checkerはproducerをimportしない。

独立Padé照合は別検査として保持し、行列差 $<10^{-9}$、独立スペクトル欠陥、三角不等式による残差・unitarity整合性を検査する。独立側が保存側の数値正常範囲を超えれば停止する。三角不等式の浮動小数slackは診断整合性だけで、eligibilityを広げない。保存幅の一致条件 $<10^{-10}$、floor $10^{-10}$ Ha、branch最大欠陥 $10^{-9}$、overlap、価格、10%基準は保持し、保存幅や判定を上書きしない。

29人工テストでは旧不整合の再現、両行列幅の分離、改竄した幅/残差/unitarity/orthogonality、nonfinite/missing、PF差、独立欠陥、円上wrap/alias、完全な小系audit、CS2幅伝播・価格・合成予算・coverage改竄の拒否、expm cap/例外時counter、旧認可/未発行/identity改竄拒否を確認した。旧27件・旧35件等のsuiteは再実行していない。人工fixtureの演算は分子結果でない。

実H4の旧失敗cell/双方の幅/partial countersは旧handlerが保存しておらず、正確な停止原因は **UNDETERMINED** のまま。修正した静的不整合が唯一の実H4原因だとは主張しない。新checkerが実H4でPASSするかも未確認。

## 新しい停止診断

v2はstage、cell key/config/order/h、保存報告値、保存行列から再計算した値、独立行列の値、spread、行列差、PF/expm開始・完了回数、array load数、確認済みrecord/cost数、経過秒、peak RSSを保存する。exceptionでもpartial countersを保持し、nonfinite診断値は文字列化してJSON保存の失敗を避ける。旧失敗記録を補筆・置換しない。

## 保存データ再監査の固定scope

linear H4/1.00 Å/STO-3G/保存DF rank12・generation順全12density、Nα=Nβ=2/36dim、先頭pairの固定X、v1 seal/state/frame/leadingを再利用。PR L_D/delta探索ではない。既存762追跡h（381共通比較h、support含む）/4572 primitive NPZ/4572 method records/18288 cost rowsをそのまま監査する。target $10^{-4},10^{-5}$ Ha、controlled価格P1/P2、予算配分、uniform10%改善基準は旧契約のまま。E0は保存済み値をbias検証だけへ使用、候補再選択はない。

新scope `QC_ONEBODY_REPARTITION_H_BOUNDARY_REAUDIT_V2_20261015` は収集ではなく保存結果の独立再監査。別出力directory、attempt1/retry0、1worker/1thread、RSS512MiB、output/log各2MiB、時間capなし。最大4572独立PF再構成、571500 expm（13fragment S4の125factor/recordを全recordへ適用した保守的上限）、親array pack load1、cell/projector loads最大13716。新PF収集/Schur・H eig・RDM・leading/selector/optimizer・分子生成・compile/T合成0。ここでのPadé再構成は新規科学入力の収集と分けて計数する。

新runnerはstdlibの認可/identity/thread/environment/親NPY-header gateを数値import・payload decodeの前に実施し、旧認可を拒否する。byte/SHA preflightは保存値の数値集計をしない。実行時はclean isolated worktreeと固定source祖先を要求し、新attempt markerを排他的に作成する。途中停止でも再試行せず結果を保存してMandatory STOP。

## 証拠・限界

[checker](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_h_boundary_checker_v2.py)、[認可gate](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_h_boundary_reaudit_v2.py)、[runner](../../scripts/verify_qc_onebody_repartition_h_boundary_v2.py)、[29人工tests](../../tests/test_qc_onebody_repartition_h_boundary_checker_v2.py)、[test record](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/synthetic_test_record.json)、[contract](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/contract.json)、[source freeze](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/source_manifest.json)、[input identity](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/input_identity.json)、[性能準備](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/performance_plan.json)、[byte/header preflight](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/preflight.json)、[未発行target](../../artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15/authorization_target.json)。

旧性能値は引き続き **UNVALIDATED_PRODUCER_ONLY**。double精度・sampled branch・登録static費用モデルの技術検査であり、有理区間保証/continuous optimum/compile/QPE総費用/独立新規性を示さない。source固定・人工PASS・GitHub byte再取得は科学的数値再現PASSと区別する。

N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、Track A/B・C1–C4・第二候補、既存global manifest description FAILを保持。新GPT科学レビュー/研究採択・終了判断は行わない。公開・再取得後Mandatory STOP、実H4再監査は別明示実行承認待ち。
