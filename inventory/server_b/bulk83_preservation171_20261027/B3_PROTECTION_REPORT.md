# DF-native B3保護と今回の非干渉

B3は未完了の保護対象として扱う。前回公開の09:36停止観測は当時の履歴であり、今回B3の現在稼働、停止原因、科学取得完了、再開可否について新しい断定をしない。今回B3 status/service/journalの再調査、再開、retry、修復、科学的終了判定は行っていない。

固定の保護IDはSERVER_B-WT-0003、SERVER_B-WT-0123、SERVER_B-WT-0124。既知のB側SOURCE/input索引からSERVER_B-WT-0128にも具体的bindingがあるため、元171件の行を保持したまま今回判断をKEEP_REQUIREDとし、新候補へ追加しない。保護ラベルを稼働認定には使わない。

controller/worker、SOURCE/baseline、provider/input/raw/result/checkpoint、Python/runtime/venv/cache、Git/object/alternate、grant/ledger/marker/STOP、既存の復旧資料を保持する。本作業の書込みは新しい本人専用保全領域と既存公開cloneの新directory・公開Gitだけ。B3関連treeのstatus/ignored/raw走査、SOURCE実行・科学call、process/service/scheduler/cgroup/共有設定変更は0。

Track B/G6/W2Cの履歴資料にあるB3は別研究である。今回その対象内の管理外資料を保全したことを、DF-native B3の計算操作と混同しない。

非干渉の確認は実行した操作のscope、対象254件のHEAD/index、保全前後のsource metadata、本人UIDの対象path使用に基づく。B3全raw再hashや全環境の凍結を行ったという意味ではない。
