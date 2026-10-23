# F1実行結果の引継ぎとSTOP

**F1_TECHNICAL_FAILURE_PARTIAL_RESULTS_MANDATORY_STOP**。F1全体は未完了。

Repository：HIROMU1015/Partially-Randomized-Trotter。
Branch：`research/pr-algorithm-improvement-phase2d-f1-execution-20261024`。
科学SOURCE／公開起点：`31e8391a56fbb1a0e64f477f78e242655329de99`。
新SOURCE修正はなく、結果・索引・研究statusだけを追加／更新する。

4armのH-IDはPASS、SSのfinite meanとM0 raw／optimized／actual adjoint／X信号、完了したcomponent／catalogのoperator検査は固定許容差内。最初のSS compileで `normalized.metadata=None` をQiskit1.3.0が拒否し、transpile前に停止した。その他は協調停止。詳細は[実行記録](EXECUTION_REPORT.md)と[機械可読要約](../../../artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/F1_RESULT_SUMMARY.json)。

compile0completed、meanはSS以外未実行、LAW最終mass gate未実行、価格DP／physical期待価格／selector／fallback未実行。未資格intervalを認証扱いしない。科学的不一致は実行済み範囲で未検出だが、F1の一般的な正しさ、新規性、資源優位性のGOはない。

新grantを一度消費し、retry／resumeなし。raw369fileはprivateで保全し、公開は要約とhash台帳のみ。旧B原本・grant／marker／STOP、F1-PRE環境・固定SOURCE・過去結果、他jobを変更していない。SOURCE、入力、env lockのidentityは要約と固定F1-PRE manifestを参照。

**次の担当はユーザー／GPT**：この結果を読み、metadata処理の修正と新しいF1実行を承認するか判断する。**Codex**：その新しい明示指示後だけ、旧SOURCEを保全した修正版を固定し、必要なら新規grantと実行範囲を準備する。使用済みgrantは使わない。F2には残るF1検証と比較条件の判断、別の実行承認が必要。

結果の公開commit／照合記録は公開後に末尾へ追加する。成否によらないmandatory STOPを保持し、F2・追加科学計算・重要科学レビューを開始しない。


## 公開・照合記録

結果payload commitは `4d592cf97c66b8a4eccac85f6789b866cf1c641f`、treeは `d593df91e8f9bdc0e111f4750779a1d879183dce`。
[固定結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/4d592cf97c66b8a4eccac85f6789b866cf1c641f/docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024)の13fileをGitHubから全件読み戻し、内容とGit blobを照合した。独立結果cloneへのgit fetchでもbytes／SHA256／treeが一致した。

科学SOURCEはF1-PRE固定commitの96fileすべて同一。結果起点1,119fileのうち1,112fileはbyte同一、更新した7fileは研究状況／証拠台帳／索引のみで、既存の科学結果・SOURCEを置換していない。raw369file、旧B／F1-PREの保護対象50file、runtime17,617fileと消費済みgrant／markerをread-only照合した。元workspaceのdirty/untrackedファイルと既存refs／worktreeを本作業で変更していない。

既存のvalidation manifest checkerはPASS（status／path／schemaの技術検査のみ）。公開資料のリンクと非公開path・run ID・承認情報の混入を確認した。新しい科学計算、QPY／NPYによる再評価、retry／resumeは0件。ローカル技術照合をimmutable CIや外部科学再現と呼ばない。

本receiptはhandoff.mdだけを追記する。最終receipt commit／remote HEAD／全13fileの照合はprivate receiptと最終チャットへ固定し、自己hashを再帰的に埋め込まない。**F1_TECHNICAL_FAILURE_PARTIAL_RESULTS_MANDATORY_STOP**を保持する。
