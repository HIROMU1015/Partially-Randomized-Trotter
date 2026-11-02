# 2026-10-16 H8 B0 anchorの実行実装・人工検証

利用者の指示で既存4構成8軸のadapter/launcherを実装した。TRAIN p14q1/p21q1/p14q8、HOLDOUT p21q8、各cosine/sine、最大4 worker/2 wave、6物理core・176.25 GiB条件を維持した。旧計画・旧B1 STOPを改変していない。

復元Gateの再記述によるbasis照合、original scalar一回、左右branch/X/Yのcontrol検証後にwaveを投入する。SOURCE/profile/inputとcost taskの証拠、監視親identity、fresh gate更新、資源/PSI/OOM監視、owned worker回収と予約非返却を接続した。fresh gateはconfig/grantや既存receiptとは別の新規ファイルだけに書く。

17新unittest＋旧計画10 methods PASS、2 qubit provider fixture8項目PASS。人工4軸compileでcompiler接続を確認し、実H8科学計算/compile/fitは0。実行SOURCE `572a337d0dc3fa78b2fd6b25ace6a37845db384e` を固定し、14 source/265 provider/34,273 runtime/8 inputのbyte照合を行った。

無効grant案はSOURCE結合済みでもapproved=false、CPU[]、command=nullで起動できない。取得・B1 fit/holdout・B2・Cは別承認。旧673保護ファイルと114 worktree・旧V2未追跡scriptを保全し、準備をcommit/push・remote確認後mandatory STOP。

[現在の準備と承認範囲](../track_a_h8_b0_anchor_execution_readiness_20261016.md)。
