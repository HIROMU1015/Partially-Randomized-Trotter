# B0Iの未確定Q3と後継run

後継run `search-d50d9a24133580d78e25` はRomuTailの `.rpe-central/state/runs` に `PREPARED` として登録した。固定版は `romutail-b0i-hold-b89818df0`、controller SOURCEは `b89818df07f99c337b77a5aa060576565a0b51ab`。後継runを操作対象に設定した。新規探索のcurrentは `romutail-work-conserving-5c8ced4b2-laptop4x4`。

元run `search-0edbfa5f4df9cb2257f5` は `V2_RESULT_REQUIRES_REVIEW` のSTOPを保持する。B0I p8・4th(m5_best)・8hのjob `bf164759c491f8d2c68b`、Romulaptop WSLのattempt `5062201d1f98428b8486f0fbd4d693d4` は `UNKNOWN` のまま保全した。元REQUEST・台帳と照合したshot配分1件を後継runで保留し、その配分を割当・採用から外す。同じ構成の別配分は通常の評価対象に残る。元attemptの状態が変わった場合は開始前に照合し直す。

保存入力による§10.5照合はPASS。再利用可能な評価はB0I 23件、PR-Pauli 41件、PR-DF 31件。B0Iの登録36構成と、従来の探索範囲・Q3件数上限なしを保持した。元runの固定記録10ファイルのhashが一致した。人工・回帰テスト50件を通過し、RomuTail、server2、Romulaptop WSL、PBS gatewayの固定配置を照合した。新たな実科学計算、キュー投入、controller起動は各0件。

Romulaptopの通常入口で後継runを確認する。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId search-d50d9a24133580d78e25
```

利用者が探索開始を指示したら、同じrun IDを `-Action Start` で指定する。中央は開始時に§10.5を再照合する。
