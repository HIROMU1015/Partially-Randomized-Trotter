# Track A H6：D・B・Aの限定並列検証

利用者の「DBAの検証をまずH6で行って」「できるだけ並行して」を実行承認として記録する。
linear H6/1Å/STO-3G、保存tol-only DF19/sector400、T=.8、P=0。
既存 SOURCE・結果・grant・ledger・markerを変更せず、専用cloneと新しい実行領域を使用する。

- D：B0+I p11/q4 S4、cosine判定後sineの最大2軸。保存予測は各1289041/5 RZ。
  相対誤差の分母は実測値、各軸1/50以上なら即STOP。再fit・追加compileなし。
- B：p5/q8/r2のK2 s2_multiとK4 s4、各n8、計16 trajectory/32軸。
  各群の先頭control合格後に残り7を並列化。条件付き抽選だけを変更し量子event重みは変更しない。
  K4 s2_multi、他q、他候補の母平均費用は未解決として残す。controlは別に計8 probe。
- A：p3/4×q2/4/8×r4×K2/4の12構成、p5/q4・8/r8/K4、p5/q16/r2/K2、
  p4/q8/r8/K4、計16構成の正式signal/normalization/経験的適格性。
  保存済みouter診断・preparationを再利用する。通常/非相殺、既存5精度、倍率1。
  このstageはsampling/compile/ランキング/費用再会計を行わない。

各stageは独立SOURCE closure/grant/one-shot lock/outputで実行し、それぞれmandatory STOPする。
正常jobを他stageのSTOPで一括終了しない。入力・compiler等の共有依存が破損した場合は関連stageを停止する。
一時的PSIやCPU競合は観測のみで、実OOMや科学的不一致に対する保護を維持する。
fresh affinity/RAMと既存類似taskの実測から並列度を選び、旧worker/RAM上限を恒常的に移さない。
共有環境や他利用者のprocessへ変更を行わない。AI subagentは使用しない。

実行前の実装検証は作者自身による人工検証であり、独立レビューではない。
実測・モデル推定・有限標本pilot・経験的allowanceを区別し、厳密bias/母平均保証を付けない。
実行結果は終了後に追記する。現時点は実行準備で、既存H8系列を再実行しない。

[機械可読契約・入力来歴](../../artifacts/resource_applicability/track_a_h6_dba/2026-10-18/)。
