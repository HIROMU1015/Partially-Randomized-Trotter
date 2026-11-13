# 2026-11-14：PRリソース評価のリポジトリ横断・実装総合監査

## 確認済み（技術）

- 対象：A `878000d8`、B `06db9aef`、C `367dc98d`、D `01e068af`（テストは直前のheadで実行）と、固定SOURCE（provider `c31ef424`、carrier `155529e2`、C1 `1a87e533`等）。
- 有限K RTEのevent lawとpaired Taylor多項式の恒等式を、人工成分の全event列挙で確認した（正負の時間、差 ≤ 2.9e−14）。
- PRの有限平均kernel（carrierのspectral版、Aのdirect-expm版）を、独立の総当たり実装と人工行列で比較し、一致した（差 ≤ 7.6e−15）。
- 私的provider rootの252 fileはcommit `c31ef424`と全file一致。現行codeが固定する107 commitは、すべてremoteから到達できる。
- 現行系列の回帰test：A・Cのheadは全件PASS。B・Dのheadは5 test fileが失敗（共有moduleのbyte/AST固定が原因）。

## 所見

- **科学**：計算の誤りは見つからなかった。Full-H RTEの保存価格は旧loweringで、新lowering価格のPRとは直接比べられない。決定論的risk上界（約0.49 Ha）は一律の最大lossのため緩い。
- **効率**：`candidate_id`の毎回parseが、遅いtest 11本の時間の77%を占める。m6の1 pathではY軸側が37%、fingerprintの繰返しが約17%。人工回路2000個で、X/Yのcompile結果は最後のancilla runだけが違った。
- **統合**：11/13の1日に、初回実行の配線ミス等によるSTOPが7件あった。Bの`h4_s0_v2_intake`は、B自身の35分後の共有module変更で動かなくなっている。
- **構造**：Pythonは3,064 file・56万行。現行系列から到達するのは非test codeの26%。

## 判断

- 修正・科学計算は行っていない。改善の採否と担当はユーザー／GPTが決める。
- 作業の最初に`git fetch --prune`で、remoteで削除済みのbranchのremote-tracking refを消してしまった。削除済み1,128 branchの最終commitがすべてlocalで到達可能であることを確認した。
- 詳細は[REPORT](../pr_resource_repo_wide_audit_20261114/REPORT.md)と[FINDINGS](../pr_resource_repo_wide_audit_20261114/FINDINGS.md)。mandatory STOP。
