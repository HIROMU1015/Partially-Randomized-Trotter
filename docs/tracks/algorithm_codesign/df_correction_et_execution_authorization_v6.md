# E/T v6 authorization-only directly after S6

利用者原文：

> 次の作業に進んでよい。
> 次も停止した場合についてもそのまま修正作業に入って、計算完了まで進めて

中断後の指示：

> 続きを行って

source S6 `90ce36e3fc383fcb4988c4df86328a08b0fef789`。guard式、2e-7、全scientific条件とcapsは維持。
保存だけを観測強化したv6を新namespace v3で一回実行する。runs1/retry0、旧run/marker不変。
A6はS6の直接子で、変更はこのreceiptと新authorization JSONだけ。
各run終了時に停止・証拠保存し、純技術修正なら利用者の継続認可で次versionへ進める。
科学保証・比較条件・資源上限の変更が必要なら判断を戻す。比較完了時はmandatory STOP。
