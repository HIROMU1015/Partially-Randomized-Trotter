# H4数値精度B＋保証設計C：引継ぎ

branch：`codex/non-df-h4-precision-bc-20261023`。worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-precision-bc-20261023`。

基点ebe20403bcb5c4439162614f64d991a769d7ed68、旧科学SOURCE0c329b30a81accb35c212be8defa66fee0dd0f12、B専用技術SOURCE2bd212288e8530e519c4a8135257df37f7608662。後続結果commitはこのbranchのgit logで確認する。新規pushなし。

## 完了範囲

Bは保存scalar40軸だけを一回実行し、CSV/JSON/Markdown・exact rational・独立Decimal・SHA・新B自身のnative exit0/waitを保存した。B grantは消費済みで再使用不可。R22軸はuの約1.53〜90.11倍、H18軸は約107.04〜2823.40倍の改善が必要。必要予算は全体で約2.45e-6〜3.50e-4だが、達成済みの数値精度ではない。

Cは設計書、40軸の手法対応、人工prototype、private FLINT接続13検査、承認待ち次回契約・実装計画まで完了。先行人工/回帰33検査もPASS。実H4の高精度/区間/certificate生成・新shot/G再会計は0。

## 保持する未解決状態

- 全20精度行のN/G=null、全40軸UNDETERMINED。保存biasの不確かさ・数値包含・PR normalization certificateは未取得。
- 元H4 caseのnative exit/reap/正常STOP証拠は未回収、監視消失原因UNKNOWN。新Bの正常終了や将来Cの証拠で代替しない。
- 実H4 production assembler/certificate runnerは未実装。B用SOURCEや人工prototypeをC実行SOURCEへ読み替えない。
- 旧SOURCE/raw/input/grant/ledger/marker/STOPを保持。H2 S2/Yoshida4費用逆転研究とは統合しない。

## 次の担当と作業（未実行）

**ユーザー**：C設計と次回契約を確認し、production実装・回帰を進める承認範囲を指定する。実H4での証拠生成も希望する場合は、契約の対象/予算と新SOURCE固定後の実行承認を明示する。N/G再会計と公開pushはさらに別の承認対象。

**Codex（承認後）**：C_implementation_plan.jsonのproduction assembler/controller/runner/testを実装し、人工/回帰gateを完了、実装commit・private backend・旧準備runtime・入力を新SOURCEへ封印する。実H4実行の明示承認がある場合に限り、新namespace・新grantで契約内の包含診断をone-shot実行する。

候補最大予算：operator復元2 batches、確率対応2 batches、reference6、nominal mean12、実装PR mean6、PR normalization6、dense exp39 requests。精度64→96→128bitsは必要対象だけ進める。SCF/FCI/新DF/状態最適化/sampling/compile/shot/Gは0。実行時は保存準備/hash・model/law対応を先に検証し、未解決ならSTOP。

## 現在の停止状態

execution_authorized/grant_valid/dispatch_enabled=false。C科学grant未発行・未消費、launch commandなし。B＋C保存・監査後mandatory STOP。ここから自動で実装・実H4証拠生成・certificate・N/G再会計に移行しない。
