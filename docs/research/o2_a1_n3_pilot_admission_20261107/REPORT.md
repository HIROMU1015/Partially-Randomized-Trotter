# 752-shot pilot attempt02：技術修正・SOURCE固定

CPU資源判定を、own affinity内の連続観測・raw tick保存・実入力前の有限待機へ変更した。[実行仕様](EXECUTION.md)参照。旧attempt01/結果/STOPを保全し、同じ752-shot条件・seed20261108・stream順で一回再投入する。
現在は人工検証・SOURCE固定段階。公開本文/blob/treeを照合してから持続managerから開始する。rank/range等の科学的STOP条件は維持し、752を超える測定や科学retryをしない。

人工52件PASS。固定入力identity・認証runtimeを静的に照合し、PROFILE_LOCK/INPUT_BINDINGを保存。旧main3,087ファイル、旧停止実行1,086ファイル、前pilot29ファイルを保全。科学入力前に公開照合する。
