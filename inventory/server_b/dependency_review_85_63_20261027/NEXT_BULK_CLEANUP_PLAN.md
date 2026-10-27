# 次期146件の研究単位別一括削除準備

**調査・候補確定・計画のみ。今回removeもforce removeも実行していない。利用終了判断と実削除の許可は未取得。旧83/86件やLOCAL/SERVER_Aの許可を転用しない。**

対象は[NEW_CLEANUP_CANDIDATES.csv](NEW_CLEANUP_CANDIDATES.csv)の正確な146 IDだけ。元85件から84、元63件から62。0128・0130は除外し、旧B3直接保護3件、他のKEEP_REQUIRED59件、common Gitと外部原本を対象外に保つ。研究単位は運用上の整理単位であり、同一SOURCE/cohort/科学結果という認定ではない。

| 研究単位・利用終了判断が必要な単位 | 件数 | 正確なID集合 |
| --- | ---: | --- |
| BASELINE_OR_OTHER_UNRESOLVED | 3 | SERVER_B-WT-0035, SERVER_B-WT-0148, SERVER_B-WT-0184 |
| DF_H2 | 16 | SERVER_B-WT-0036, SERVER_B-WT-0038, SERVER_B-WT-0039, SERVER_B-WT-0040, SERVER_B-WT-0041, SERVER_B-WT-0042, SERVER_B-WT-0043, SERVER_B-WT-0044, SERVER_B-WT-0045, SERVER_B-WT-0046, SERVER_B-WT-0047, SERVER_B-WT-0049, SERVER_B-WT-0050, SERVER_B-WT-0051, SERVER_B-WT-0052, SERVER_B-WT-0053 |
| DF_H246_HIGHER_PF | 1 | SERVER_B-WT-0060 |
| DF_H4 | 70 | SERVER_B-WT-0065, SERVER_B-WT-0066, SERVER_B-WT-0069, SERVER_B-WT-0075, SERVER_B-WT-0076, SERVER_B-WT-0077, SERVER_B-WT-0078, SERVER_B-WT-0080, SERVER_B-WT-0084, SERVER_B-WT-0085, SERVER_B-WT-0086, SERVER_B-WT-0087, SERVER_B-WT-0088, SERVER_B-WT-0091, SERVER_B-WT-0095, SERVER_B-WT-0096, SERVER_B-WT-0099, SERVER_B-WT-0101, SERVER_B-WT-0102, SERVER_B-WT-0103, SERVER_B-WT-0104, SERVER_B-WT-0105, SERVER_B-WT-0106, SERVER_B-WT-0107, SERVER_B-WT-0111, SERVER_B-WT-0112, SERVER_B-WT-0113, SERVER_B-WT-0114, SERVER_B-WT-0115, SERVER_B-WT-0116, SERVER_B-WT-0117, SERVER_B-WT-0119, SERVER_B-WT-0120, SERVER_B-WT-0262, SERVER_B-WT-0263, SERVER_B-WT-0264, SERVER_B-WT-0265, SERVER_B-WT-0266, SERVER_B-WT-0267, SERVER_B-WT-0268, SERVER_B-WT-0269, SERVER_B-WT-0270, SERVER_B-WT-0271, SERVER_B-WT-0272, SERVER_B-WT-0273, SERVER_B-WT-0274, SERVER_B-WT-0276, SERVER_B-WT-0277, SERVER_B-WT-0278, SERVER_B-WT-0279, SERVER_B-WT-0281, SERVER_B-WT-0282, SERVER_B-WT-0283, SERVER_B-WT-0284, SERVER_B-WT-0285, SERVER_B-WT-0290, SERVER_B-WT-0292, SERVER_B-WT-0293, SERVER_B-WT-0295, SERVER_B-WT-0296, SERVER_B-WT-0297, SERVER_B-WT-0298, SERVER_B-WT-0303, SERVER_B-WT-0304, SERVER_B-WT-0307, SERVER_B-WT-0308, SERVER_B-WT-0309, SERVER_B-WT-0310, SERVER_B-WT-0311, SERVER_B-WT-0312 |
| DF_H46_PF_PR | 3 | SERVER_B-WT-0129, SERVER_B-WT-0137, SERVER_B-WT-0138 |
| DF_H6 | 14 | SERVER_B-WT-0149, SERVER_B-WT-0150, SERVER_B-WT-0151, SERVER_B-WT-0157, SERVER_B-WT-0158, SERVER_B-WT-0161, SERVER_B-WT-0167, SERVER_B-WT-0170, SERVER_B-WT-0171, SERVER_B-WT-0176, SERVER_B-WT-0185, SERVER_B-WT-0188, SERVER_B-WT-0189, SERVER_B-WT-0376 |
| DF_H8 | 4 | SERVER_B-WT-0074, SERVER_B-WT-0211, SERVER_B-WT-0243, SERVER_B-WT-0244 |
| DF_NATIVE_TRACK_A | 5 | SERVER_B-WT-0251, SERVER_B-WT-0323, SERVER_B-WT-0325, SERVER_B-WT-0326, SERVER_B-WT-0356 |
| PAPER_EVIDENCE_PRESENTATION | 3 | SERVER_B-WT-0328, SERVER_B-WT-0334, SERVER_B-WT-0339 |
| PAULI_AND_REPRESENTATION | 7 | SERVER_B-WT-0013, SERVER_B-WT-0318, SERVER_B-WT-0321, SERVER_B-WT-0350, SERVER_B-WT-0351, SERVER_B-WT-0352, SERVER_B-WT-0353 |
| TRACK_B_G6_HISTORY | 10 | SERVER_B-WT-0015, SERVER_B-WT-0016, SERVER_B-WT-0018, SERVER_B-WT-0019, SERVER_B-WT-0022, SERVER_B-WT-0026, SERVER_B-WT-0031, SERVER_B-WT-0032, SERVER_B-WT-0033, SERVER_B-WT-0034 |
| TRACK_B_OTHER_HISTORY | 10 | SERVER_B-WT-0362, SERVER_B-WT-0363, SERVER_B-WT-0364, SERVER_B-WT-0365, SERVER_B-WT-0366, SERVER_B-WT-0367, SERVER_B-WT-0368, SERVER_B-WT-0369, SERVER_B-WT-0370, SERVER_B-WT-0371 |

## 保全と物理参照の分離

候補84件の管理外資料は既存94 archive・3 Git bundle・小型copy・stage/working/index復元資料を再利用して保全条件を確認した。元63件のうち候補62件はfreshにtracked clean・untracked0・ignored0で、保持Git ref/objectによる取得条件を確認した。全候補のSOURCE・重要commit・結果/失敗/STOPを保持Gitまたは対応する検証済み保全から辿れる。非公開ID/path/branch/完全HEAD/保持ref/復元bindingを本人専用領域に保持する。

絶対path参照は保存されたmanifest・実行観測・失敗trace・draft・未実行PLAN・終了した実行契約に分かれた。RUNNING観測には後続terminal/STOPを照合し、古い観測を現在のprocess状態として扱わない。旧PLAN・SOURCE・contract・grant・ledger・marker・STOPの本文と科学条件は変更しない。

物理checkoutを将来削除すると、旧絶対pathを持つ保存PLANやconfigがそのまま即時実行可能とは限らない。将来その版を使う場合、保持commitとprivate archiveから必要な旧workspace/pathを正確に復元してから、その研究固有の科学実行許可・one-shot・retry/STOP条件に従う。別path/configへ変えるなら別の具体的許可が必要。今回の候補化は科学契約の解除・新科学実行・旧STOP再開の認定ではない。

## 実削除前に必要な判断と条件

1. ユーザー/研究担当が上表の各研究単位について、SERVER_Bでの今後の利用・再開予定の終了を判断し、正確なID集合へ実削除を明示指示する。今回全146件はPENDING_OWNER_DECISIONである。
2. Codexが指示後に実path/branch/HEAD/登録/backlink/common Git/保持refと重要objectをfresh確認する。現在利用するprocess/job/lock、新しいSOURCE/input/control/runtime依存、具体的な他環境bindingがあればそのIDを保留する。
3. 現在の93件はCLEANで通常removeの候補。残り53件はLOCAL_ASSETS_REMAINで、保全完了と通常remove可否は別である。新しいdirty/untracked/ignored leafを含めて保全を更新し、具体的な削除方法の許可を受ける。旧86件に限る単一force許可はこの53件へ適用しない。今回forceの実行許可は0。
4. 新しいHEAD/差分・保全欠損・具体的依存・利用/lock・通常Git拒否は該当IDを保留する。common Git/object/alternateの異常、ID/path対応の共通異常、B3保護領域との交差はバッチ全体をSTOPする。無関係なCodex作業の進行だけで全worktreeを凍結しない。
5. 許可された方法と固定ID順序で1つの論理batchを組み、同じcommon Gitへの操作を競合させない計画とする。branch削除、force push、main merge、履歴書換え、clean/reset/prune/gc/手動再帰削除へ拡張しない。
6. 実行後は成功IDのdirectory/登録消失、保留ID・対象外・重要ref/object・private原本・archive/bundle・B3資産の保持を監査する。容量を未計測の旧数字から推定しない。

復旧情報とarchive/member原本は非公開領域、公開の履歴参照は[既存保全manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50612696d8ef7db322729fd3946b2033d3d39e01/inventory/server_b/bulk83_preservation171_20261027/PRESERVATION_MANIFEST.csv)とGitの固定版を用いる。同じfilesystem上のcopyであり独立障害域バックアップとは表現しない。

**今回は削除0。調査・計画・匿名化公開を確認してmandatory STOP。**
