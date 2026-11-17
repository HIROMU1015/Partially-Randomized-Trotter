# Romulaptopが起動中の受付

currentは `romutail-pbs-current-95a74e448-cpu100-qg230-laptop`。
ローカル0の受付フラグ `accepting_new_jobs=true` を実機で確認した。
通常探索の中央設定では `local0-wsl` を任意参加として許可する。
接続できて端末側がONであり、workerから新しい空き枠の報告があるときに割り当てる。
既存の優先ホストに直ちに使える空き枠がある場合は優先する。Romulaptopの参加を待つ猶予は0秒。
端末側の受付をOFFにした場合や切断で空き枠の報告が失効した場合は新規割当を止める。
復帰後の回収は同じattemptを使う。

CPU追加上限はRomulaptop・RomuTailともなし。qg2の30 CPU、PBSの100 CPUは従来どおり。
qg1は停止中。新規探索には新currentを選び、既存runは固定済み設定を使う。
受領済み結果や科学予算は同じ。新規科学実行・PBS投入は各0件。

実機のWindows→WSL中継で、空の検査用spoolを使い既存の受付フラグON・空き1枠を確認した。
任意hostの参加・Pause・切断回収・中継の既存人工テスト21件が通った。
通常WindowsとRomuTailの明示版・指定なしPreviewは同じrun `search-a235d071bc903650c5be` を返した。
[release manifestの差分](RELEASE_ROMULAPTOP.json)
