# 固定83件削除後の保全・B3非干渉監査

2026-10-27 JST。対象は[50612696d8efの固定83件](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50612696d8ef7db322729fd3946b2033d3d39e01/inventory/server_b/bulk83_preservation171_20261027/READY83_IDS.csv)。今回のユーザー指示はこの集合の物理linked checkout削除だけを明示許可し、旧CSVのactual_removal_authorized=NOは以前の記録のまま保持した。承認証明書・grant・認証基盤は作成していない。

## 実行と登録

83件の実path/branch/完全HEAD/common Gitを本人専用表から照合し、ID集合・順序を完全一致させた。各HEAD/branch/登録backlink、tracked clean・untracked0・ignored0、lockなし、本人UIDのcwd/argv/fdによる対象path利用なしを確認。既存の限定依存監査を再利用し、既知のB3/runtime controlファイルに新しい対象絶対path参照がないことを静的確認した。数値データの再解析、科学SOURCE実行、全379 checkoutの再監査は行わない。

全83件は通常git worktree removeがexit0で終了し、IDごとに物理directoryとGit登録の消失を確認。3 common Gitに対する削除は固定sequenceで逐次実行し、force/prune/clean/reset/gc/手動再帰削除/lock解除は0。branch削除・merge・rebase・履歴書換えも0。

削除前379登録、削除後296登録。差分83は成功した83 pathと完全一致し、予期しない登録消失0・並行追加0だった。対象外296件は全て登録と物理directoryが保持され、登録metadataにも変化なし。83件以外のindexや研究ファイルに書込みを行っていない。無関係なchatの作業進行を禁止・凍結する措置は取らない。

## Git branch・科学SOURCE・結果

全83 HEADは削除後も保持local refから取得可能で、commit/treeの存在を確認。関連する3 object storeで延べ30,833個のcommit/blobを削除後にbatch-checkし、全て取得可能だった。削除前後の全local branch/tag refに変化0。tracked SOURCE/code/文書/結果/図表/失敗/STOPは保持Git treeと履歴から引き続き取得できる。削除前のlive GitHub tip保持を利用し、branchや重要commitを削除していない。

main checkout、4 common Gitとobject store、既知alternateを保持した。削除対象にcommon Git ownerは含めない。物理checkoutの展開copyが消えたことと、Git履歴・科学SOURCE自体の喪失を区別する。科学的に異なるcohortや結果の統合はしていない。

## private原本・復旧資料

前回検証済み94 archive・3 full-ancestry Git bundleをすべて保持し、削除前後の存在・サイズ・mtime/inodeに変化なし。94 member manifestとprivate ID/path/branch/HEAD対応・復旧情報も保持。前回のSHA/stream読戻し・別bare repositoryからの到達性検証を再利用し、無関係な大容量原本の再hash・再コピーはしていない。

前回47小型copyと3件のstage/working/index復旧用12 fileも存在・サイズ一致を確認。既知外部provider/input/private raw/result/checkpoint/runtime/cacheは削除対象外に原位置保持し、保護資源の実path/resolved path/device/inode・file sizeに変化を検出しなかった。今回の書込みは許可された83 checkout/登録の通常remove、新しい本人専用実行記録、独立公開cloneだけである。

**保全copy/archive/bundleは同じfilesystem上にあり、独立障害域バックアップではない。** 元の研究STOP・grant・ledger・markerは変更/移動せず、復元時の参照を非公開領域に保持した。

## B3とその他の保護群

B3保護対象SERVER_B-WT-0003・SERVER_B-WT-0123・SERVER_B-WT-0124・SERVER_B-WT-0128の4件を全て登録・物理directoryの両方で保持。SOURCE/baseline/provider/input/runtime/venv/cache、共通Git/object/alternate、private raw/result/checkpoint、旧grant/ledger/marker/STOPと復旧資料は対象外とした。B3の稼働や科学完了の判定、status/service/journal調査、再開/retry/修復、process/scheduler/cgroup/共有設定の操作、科学callは0。

前回の171件は全件保持。その内訳である追加候補86件と依存保留85件は、全件で登録・物理directoryが存在し、登録metadataも削除前後で一致した。旧他環境待ち63件、既存KEEP_REQUIRED59件、旧B3保護ラベル3件も保持。この4分類の合計は296件で、0128は171件内に含まれる。重複カウントで保護数を水増ししない。

他環境へのSSH/変更、他ユーザー・他jobへのprocess/file操作は0。本監査は今回操作の非干渉と限定scopeの保持確認であり、全環境・全依存不存在の証明ではない。実行中の研究結果は通常進行し得るので、rawの全域凍結や数値同一性の再監査は行っていない。
