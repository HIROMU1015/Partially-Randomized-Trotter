# SERVER_B 追加86 worktree一括削除結果

**86件すべて削除成功。通常remove17件、単一--force 69件、保留0・失敗0。登録worktreeは296件から210件へ減少した。**

対象は[保全・分類commit 50612696d8efの固定86 ID](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50612696d8ef7db322729fd3946b2033d3d39e01/inventory/server_b/bulk83_preservation171_20261027/NEW_TECHNICALLY_READY_IDS.csv)だけ。利用終了と実削除の今回の明示指示をWORKSPACE_END_APPROVEDとして新しい記録へ反映した。元CSVのPENDING_OWNER_DECISIONは履歴として保持する。前回の固定83件削除に続く作業であり、85件・63件・KEEP群やB3へ削除許可を拡張していない。

## 実行結果

[86件のID別結果](ADDITIONAL86_REMOVAL_RESULTS.csv) / [研究単位別集計](RESEARCH_UNIT_RESULTS.csv) / [remove方法と理由](REMOVAL_METHODS.csv)。固定CSVと非公開ID/path/branch/完全HEAD対応を照合し、対象追加・置換0。1つの論理batchとして同じcommon Gitの更新を競合させず固定順序で実行した。全件の物理directoryと登録の消失を確認した。

| 研究単位 | 成功 | 通常remove | 単一force |
| --- | ---: | ---: | ---: |
| BASELINE_OR_OTHER_UNRESOLVED | 1 | 1 | 0 |
| DF_H2 | 4 | 0 | 4 |
| DF_H246_HIGHER_PF | 4 | 1 | 3 |
| DF_H4 | 20 | 0 | 20 |
| DF_H46_PF_PR | 3 | 1 | 2 |
| DF_H5_H7_OPEN_SHELL | 2 | 2 | 0 |
| DF_H6 | 16 | 6 | 10 |
| DF_H8 | 15 | 4 | 11 |
| DF_NATIVE_TRACK_A | 12 | 1 | 11 |
| PAPER_EVIDENCE_PRESENTATION | 4 | 1 | 3 |
| PAULI_AND_REPRESENTATION | 3 | 0 | 3 |
| TRACK_B_G6_HISTORY | 2 | 0 | 2 |
| **合計** | **86** | **17** | **69** |

保留・失敗IDはない。preflight時点から新しい重要file変更がなく、追加保存の必要は0。69件のforceは保全済みのdirty/untracked/ignored状態の除去にだけ使い、lock回避や未確認fileの破棄には使っていない。禁止されたdouble force、clean、reset --hard、手動再帰削除、prune/gc、branch削除、force push、研究branchのmerge/rebaseは0。

## 保全と非干渉

[固有情報・Git履歴保全監査](PRESERVATION_AUDIT.md) / [B3・対象外210件の保持](B3_AND_UNTARGETED_RETENTION.md)。

- 重要branch/tag/commit/SOURCEとtracked研究資料は保持Gitから取得可能。local branch/tag ref変更0。
- 既存94 archive・3 full-history Git bundle・manifest・復旧索引を保持し、作成/全展開/171件の全面監査をやり直していない。同じfilesystem上の保全であり独立バックアップではない。
- 単一forceを使ったIDは、各固有fileの保全と必要なstage/working差分復元を個別確認。再生成可能cacheと科学的原本を区別し、数値解析・科学SOURCE実行は0。
- B3保護4 ID、依存85件、他環境待ち63件、KEEP_REQUIRED59件と主checkout/common Gitを保持。B3のprocess/service/科学実行を操作していない。

登録の減少86は成功した86 pathと一致し、今回scopeに予期しない登録消失・並行追加はない。固定83件の先行削除と合わせて当初379登録から169 checkoutを削減し、現在210登録。解放容量は新規に計測しておらず、旧容量から推定しない。

## 公開と残る整理課題

整理branchの最新remote履歴を開始時と更新直前に確認し、この新directoryだけを通常fast-forwardで追加する。SERVER_A等の後続履歴と既存報告を上書きせず、GitHub読戻しとlocal/remote SHA照合後に完了する。内部path/host/IP/ユーザー名/private原本や秘密の復旧情報は公開しない。

次の担当は**ユーザー/該当研究担当**。依存85件の固定SOURCE/control/runtimeの役割と物理path保持要否、旧63件の他環境条件、既存KEEP群の保持理由を必要に応じて判断する。今回それらの解除や削除、GitHub branch整理、B3や追加科学計算へは進まない。

**今回の削除・保全監査・匿名化公開を完了したらCodexはmandatory STOP。**
