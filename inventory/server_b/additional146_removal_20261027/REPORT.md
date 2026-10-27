# SERVER_B追加146 worktree一括削除結果

ユーザーが固定146件の利用終了と物理checkout削除を明示許可したため、1つの論理batchとして実施した。対象は[固定候補CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1668b7a7943445e3b7104cab7ac2ab412ad7f22a/inventory/server_b/dependency_review_85_63_20261027/NEW_CLEANUP_CANDIDATES.csv)と完全一致する。元85件から84件、元63件から62件。対象追加・置換0件。今回の許可を保持対象64件・他環境・branch削除・科学実行に転用していない。

| 項目 | 実測結果 |
| --- | ---: |
| 固定対象 | 146 |
| 削除成功 | 146 |
| 保留 | 0 |
| 失敗 | 0 |
| 通常remove成功 | 93 |
| 単一force成功 | 53 |
| 登録worktree・削除前 | 210 |
| 登録worktree・削除後 | 64 |
| 追加保全が必要だったID | 0 |
| 保持archive | 94 |
| 保持full-history bundle | 3 |

## 研究単位別結果

| 研究単位 | 対象 | 成功 | 保留 | 失敗 | 通常 | 単一force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BASELINE_OR_OTHER_UNRESOLVED | 3 | 3 | 0 | 0 | 2 | 1 |
| DF_H2 | 16 | 16 | 0 | 0 | 11 | 5 |
| DF_H246_HIGHER_PF | 1 | 1 | 0 | 0 | 1 | 0 |
| DF_H4 | 70 | 70 | 0 | 0 | 52 | 18 |
| DF_H46_PF_PR | 3 | 3 | 0 | 0 | 2 | 1 |
| DF_H6 | 14 | 14 | 0 | 0 | 2 | 12 |
| DF_H8 | 4 | 4 | 0 | 0 | 1 | 3 |
| DF_NATIVE_TRACK_A | 5 | 5 | 0 | 0 | 2 | 3 |
| PAPER_EVIDENCE_PRESENTATION | 3 | 3 | 0 | 0 | 3 | 0 |
| PAULI_AND_REPRESENTATION | 7 | 7 | 0 | 0 | 7 | 0 |
| TRACK_B_G6_HISTORY | 10 | 10 | 0 | 0 | 0 | 10 |
| TRACK_B_OTHER_HISTORY | 10 | 10 | 0 | 0 | 10 | 0 |

## 保留・失敗

なし。

## 方法と保全

各IDの直前確認を通過してから、同じcommon Gitの操作が競合しない逐次順序でremoveした。通常removeはCLEAN対象、単一forceは固定53件のうち保全・差分/index復旧・依存・非利用条件を満たした対象だけに使用した。二重force、clean/reset/prune/gc、手動強制再帰削除、unlock、stash、branch削除、研究branchのmerge/rebaseは0件。

重要branch・tag・commit・SOURCE ref・Git object storeを保持し、元の全ref変更0件。94 archive・3 bundle・manifest・stage/working/index復旧資料・小型copy・復旧索引を保持した。研究原本・raw・失敗・STOPは保持Gitまたは対応するprivate保全に残る。既存コピーは独立障害域バックアップではない。詳細は[PRESERVATION_AUDIT.md](PRESERVATION_AUDIT.md)。

B3保護5件と既存KEEP_REQUIRED59件はすべて登録・物理directoryを保持した。B3状態調査・process/service/runtime/科学SOURCEへの操作、科学計算・数値再解析、SERVER_A/LOCAL操作は0件。詳細は[B3_AND_UNTARGETED_RETENTION.md](B3_AND_UNTARGETED_RETENTION.md)。

## 公開と終了

整理専用branch `codex/pr-environment-inventory-20261026`の新規directoryに本6資料だけを追加する。公開直前の最新remoteを親にした通常fast-forward更新を行い、GitHub読み戻しと独立公開cloneのlocal/remote SHA・内容照合結果は本人専用実行記録と最終回答へ記録する。過去の報告と研究branchは上書きしない。

対象ID別結果は[ADDITIONAL146_REMOVAL_RESULTS.csv](ADDITIONAL146_REMOVAL_RESULTS.csv)、研究単位集計は[RESEARCH_UNIT_RESULTS.csv](RESEARCH_UNIT_RESULTS.csv)、方法と保全条件は[REMOVAL_METHODS.csv](REMOVAL_METHODS.csv)。正確な実path・branch・完全HEAD・復旧binding・remove journalは本人専用領域に保持した。

次にユーザーが判断すべき未解決事項は今回の146件についてなし。保持64件は現状保持し、今回の許可で整理を拡張しない。公開・読み戻し照合後にこの整理作業をmandatory STOPとする。次の作業なし。
