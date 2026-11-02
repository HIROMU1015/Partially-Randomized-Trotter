# E/T v7：継続認可に基づくauthorization-only直接子

利用者原文：

> 次の作業に進んでよい。
> 次も停止した場合についてもそのまま修正作業に入って、計算完了まで進めて

再開指示：

> 続きを行って

S7 `74682c40d0cdcca92d244a9edf4bb2852361eef5`で新namespace v4の一回実行を認可する。
v6で原因特定した未実施W位相置換の会計を修正。strict raw phase/誤差保証/2e-7/全science条件/capsを保持。
unsupported scalar phaseのT費用はunknownのまま。新phase keysやfixture/precision変更はない。
S7の直接子A7のdiffは本receiptとauthorization JSONのみ。
旧source/結果/consumed markerを保持、各run retry0、成功・失敗とも記録を保存する。
既定保証内の純技術修正なら利用者の継続認可で次versionへ進み、比較完了時はmandatory STOP。
science条件/保証/capsの変更が必要なら判断を戻す。
