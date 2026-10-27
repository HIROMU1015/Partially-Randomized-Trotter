# P2採用評価と選択統合の結論

LOCAL / 2026-10-28 JST。固定P2 `04f4cd9cdc8e348a9f0a182ead9cd07f1ee69696`、直接base `00958ae1945c780718b2deef07d5dab1096a5cd8`。
**第一推奨：既存 `codex/pauli-jw-research-summary-20261026` をPauli-JWのコード開発代表として選び、P2の13ファイル相当＋既存H6 adapter1ファイルを選択統合する。**
保存成果は同branchの固定summary `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b` と後続H6の固定handoffを版別入口として保持する。
branchを今後の開発に使っても、元の科学SOURCEや過去のsummaryを新しい科学結論へ書き換えない。

## 採用判定

|区分|18ファイル中の件数|内容|
|---|---:|---|
|ADOPT_AS_IS|5|package入口/context/adapter/人工fixture/CLI|
|ADOPT_WITH_CHANGES|8|composer/core binding/test＋既存案内5|
|DEFER|5|固定P2の履歴文書。公開原本を保全しURL参照|
|ALREADY_PRESENT|0|選択先summaryにP2差分は未採用|
|DO_NOT_ADOPT|0|ファイル単位の永久不採用はない|

[18行の判定表](ADOPTION_DECISION_MATRIX.csv)は機能範囲・依存・修正・test・科学未検証・保護条件を持つ。
別途必要なH6 adapterはP2の18行には混ぜず、既存donor `00958ae1945c780718b2deef07d5dab1096a5cd8:src/trotterlib/h6_hybrid_fairness_extension.py` を同じpathへ1回だけ追加する。
blobは `0f09c36f1b191abba75e39f56746a1828f4419f8`、5,567 bytes。新しいH6 kernelを作り直さない。

## なぜsummaryを選ぶか

summary→P2は直線7 commit。先行6はH6研究/実行/結果の履歴、最後の1だけがP2の18変更。
選定コードclosure57 pathのうちsummaryにある56はbaseと同blob、欠損はH6 adapter1だけだった。
完了済みH6実行branchを今後の汎用開発で更新する必要はない。
P2専用branchをそのまま恒久代表へ昇格する案も比較したが、採用修正と役割決定を省略せず、既存成果入口から新共通入口を辿れるAを推奨する。
[代表比較](REPRESENTATIVE_BRANCH_DECISION.md)、[履歴・依存](DEPENDENCY_AND_HISTORY_REVIEW.md)。

## 取り込み前に直す点

新規の限定技術検証は **17 PASS / 0 FAIL / 0 SKIP**。
このうち13件はAPI/静的/人工契約確認、4件は**現版の不足を再現する観測test**であり、望ましい拒否が成立したPASSではない。
method・K・PF_orderの誤表示を同bindingのstageが受理する3例、Providers.core=NoneでAttributeErrorになる1例を確認した。
composerの明示照合とcore欠損拒否を追加し、callback属性欠損の診断もcore bindingで整える。
研究コードは今回変更せず、[次回実装計画](SELECTIVE_INTEGRATION_PLAN.md)へ具体化した。

固定P2の109 PASS/1 SKIPは過去記録を再利用し、今回再実行/合算していない。
通常名での実core importは`matplotlib`不足で停止。全optional stack/実atomic writer/数値同等性は未確認。
[新規test範囲](TECHNICAL_TEST_RESULTS.md)。

## 実行可能な次段階

統合先branchの着手時remote tip/計画baseは `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`。
次回は独立checkoutのこのtipから、P2採用13 path＋H6 adapter1 pathの計14 pathを準備する。
H6のrunner/成果/契約/STOPファイルを取り込まず、既存固定GitHub locatorへリンクする。
全branch merge/rebase/cherry-pickを使わず、旧initializer・既存kernel・旧runner・SOURCE/result/cohort/STOPを保つ。
新拒否testと限定mock回帰後に、明示許可された代表branchへ通常fast-forward公開する案である。

**ユーザーが次に許可する範囲**：この14 path相当の技術移植・列挙した最小修正・限定人工/mock回帰・最小案内更新・commit/push。
Codexはその次の明示指示後だけ実施する。real provider/loader、PF/B0/H2の新接続、科学数値検証、P3実装は含めない。
[P3引継ぎ](P3_HANDOFF.md)は既存B2-D入口と責務設計だけで、P3を開始していない。

今回は7資料だけを既存inventory branchへ追加公開する。研究branch/main/P2/P1/元dirty checkoutは変更しない。
新科学計算/sampling/build/compile/再解析0、B3本体と保護5worktree/9refの内容アクセス0、A/B接続/転送0、他job/共有設定変更0。
正式2026-10-25指示の実ファイル・同版hashを確認し§8.1/8.4/10を適用。新しい承認証明/grant制度は作らない。
SOURCE/結果/negative/未観測rare event/STOP/N1-N3未解決は保持する。
公開照合後mandatory STOP。代表branchへの実統合は未実施。
