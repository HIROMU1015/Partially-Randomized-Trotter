# B3・対象外worktreeの保持と非干渉

B3保護IDのSERVER_B-WT-0003、SERVER_B-WT-0123、SERVER_B-WT-0124、SERVER_B-WT-0128は全て登録・物理directoryを保持。既知のSOURCE/baseline/provider/input/runtime/venv/cache・common Git/object/alternate・private raw/result/checkpoint・旧grant/ledger/marker/STOPと復旧情報は今回の削除範囲外とした。保護資源の実path/resolved path/device/inodeとfile sizeに削除前後の変化を検出しなかった。

B3の現在稼働・失敗原因・科学的完了を判定していない。status/service/journal調査、再開/retry/修復、controller/worker/service/scheduler/runtime/科学SOURCEの操作、科学callは0。保護ラベルを現在稼働の認定には使わない。

対象外210件は全て登録・物理directoryを保持し、登録metadataの変化0。内訳は具体的依存が残る85件、旧他環境待ち63件、既存KEEP_REQUIRED59件、旧B3保護ラベル3件。0128は85件の中に含み、保護4件を別加算して重複計上しない。

4 common Git/object storeと既知alternateを保持。主checkoutとGit ownerは削除しない。他環境のworktree、共有runtime・symlinkの外部リンク先へ削除を波及させていない。symlinkはliteralで保全し、file列挙・copyは外部リンク先を追わず、Gitによるcheckoutの削除だけを行った。対象内mountがないことも確認した。

書込みは許可された固定86件のcheckout/登録削除、新しい本人専用実行記録・必要時の追加保全、独立公開cloneに限定した。対象外研究branch/index/HEAD/原本の編集、共有設定変更、他ユーザー・他jobのprocess操作は0。他環境へ無断SSH接続していない。

非干渉は今回の操作scopeと限定した保持確認に基づく。全環境依存の不存在証明や研究rawの全域凍結ではない。無関係なCodex作業の通常進行を止める措置は取っていない。
