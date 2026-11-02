# Storage・STOP・最終受入

固定SOURCE：[78c571005fe766c61b32dc867e006531099ff608](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/78c571005fe766c61b32dc867e006531099ff608)。直接基点 `d24cccbecdf0928fe977080dd7b28c591df0089e`。ローカル独立受入 `6bffb5b5819918698b522270ff19b255766d3072` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。

固定Global BundleのRAM intent、physical statefile、trusted ordered trace prefix、journal、summary、receipt、diagnostic、marker、deliveryとcaller observationを既存offline checkerで結合した。marker単独、receipt-before-close、close/caller失敗、ACK欠損、partial write、RAM/physical mismatchを成功にしない。保存されなかった証拠をsnapshotから補完しない。

単一global claim/call/byte budget、親arrival順、STOP後ENTER禁止、STOP前inflightの合法drain、START_RESERVEDとbody entryの区別、UNKNOWN非返却、保存先別STOP成否を変更しない。S失敗後P1開始禁止、P1失敗時S保存も固定43-case matrixで検証。障害の26件はINCOMPLETE_EVIDENCEのままで、testの期待分類一致と完全成功を混同しない。

新port側はpair stream close→report close→caller witnessを照合し、F01失敗のinflight sidecarは完了prefixへ数えない。これは固定BundleのStorage V2証明の代用品ではない。port interface gridはphase単位で停止し同phase workerが完了し得る。global permit/STOP後ENTERの保証は別の固定Bundle試験に限定し、新port入りglobal schedulerを検証したとは表示しない。

caller witness自身のclose/power failure、fsync durability、実trusted storeのpower-loss履歴は未確立。原STOP/marker/旧one-shotを解除・再使用しない。
