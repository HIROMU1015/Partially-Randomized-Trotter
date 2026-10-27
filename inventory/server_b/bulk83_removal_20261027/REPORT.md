# SERVER_B 固定83 worktree一括削除結果

**固定83件すべて削除成功、保留0、失敗0。SERVER_Bの登録checkoutは379件から296件へ減少した。** Git branch・重要commit・SOURCE・研究資料・private原本・保全archive/bundle・B3保護対象は保持した。

実行日時：2026-10-27 12:11:14〜12:11:40 JST。今回の明示許可は固定83件の物理worktreeだけで、旧準備資料の承認未取得fieldや履歴を書き換えていない。新しい結果をこのdirectoryへ追記する。

## ID別と研究単位別の結果

[83件のID別結果](FIXED83_REMOVAL_RESULTS.csv) / [研究単位別CSV](RESEARCH_UNIT_RESULTS.csv)。対象は[固定準備commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50612696d8ef7db322729fd3946b2033d3d39e01/inventory/server_b/bulk83_preservation171_20261027/REPORT.md)のREADY83_IDS.csvと完全一致し、追加や置換は0。

| 研究単位 | 削除成功 | 保留 | 失敗 |
| --- | ---: | ---: | ---: |
| BASELINE_OR_OTHER_UNRESOLVED | 1 | 0 | 0 |
| DF_H2 | 1 | 0 | 0 |
| DF_H46_PF_PR | 5 | 0 | 0 |
| DF_H5_H7_OPEN_SHELL | 7 | 0 | 0 |
| DF_H6 | 10 | 0 | 0 |
| DF_H8 | 28 | 0 | 0 |
| DF_NATIVE_TRACK_A | 18 | 0 | 0 |
| PAPER_EVIDENCE_PRESENTATION | 4 | 0 | 0 |
| TRACK_B_G6_HISTORY | 5 | 0 | 0 |
| TRACK_B_OTHER_HISTORY | 4 | 0 | 0 |
| **合計** | **83** | **0** | **0** |

83件を1つの論理batchとして固定sequence順に通常git worktree removeで処理した。全件preflightに合格し、削除直前にも個別状態を再確認した。Git通常removeの拒否や失敗は0、forceや回避手段は使っていない。

## 保全監査

[Git履歴・科学資料・B3保護の監査報告](PRESERVATION_AND_B3_AUDIT.md)。主な結果は以下のとおり。

- 全83 HEADの保持ref/commit/treeとtracked SOURCE・文書・結果・図表・失敗・STOPの取得可能性を確認。削除前後のlocal branch/tag ref変更0。
- 4 common Git/object storeと既知alternateを保持。削除対象外296件は全て登録・物理directoryを保持し、登録metadata変化0。
- 検証済み94 archive・3 Git bundleと必要なmanifest/復旧情報を保持。追加86件・依存85件を含む171件、他環境待ち63件、既存KEEP群は今回削除していない。
- B3保護4 IDを保持し、B3 process/service/scheduler/runtimeの操作や科学実行は0。B3の現在稼働・科学的完了は判定していない。
- 既知の外部private原本/input/raw/result/checkpointを保持。保全copyは同じfilesystem上で、独立バックアップとは表現しない。

予期しない登録の消失・並行追加はともに0で、今回の83件削除が379→296の差分に一致する。容量の新規全件計測はしておらず、過去の容量を実際の解放量として提示しない。

## 公開と次の判断

詳細な実path/branch/完全HEAD・復旧情報は本人専用実行領域に保持し、公開版へ内部path/host/IP/ユーザー名/private原本は含めない。整理branchの最新remote履歴へこの新directoryだけを追加し、通常fast-forward更新とGitHub読戻し・local/remote SHA照合を行う。既存SERVER_A/B報告を上書きしない。

次の担当は**ユーザー/研究担当**。追加86件について今後の利用・再開予定の終了を判断し、依存85件は固定SOURCE/control/runtimeの具体的役割を解消するか判断する。旧63件の他環境条件もその担当へ引き継ぐ。今回の許可をこれらの削除やGit branch整理へ拡張しない。

**今回の削除・監査・結果公開を終えたらCodexはmandatory STOP。後続の削除や科学処理を自動開始しない。**
