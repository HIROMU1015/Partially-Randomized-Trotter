# 2026-10-17 H8 非相殺bias診断：実装・証拠照合・人工検証（実行未承認）

利用者指示（comment.md）に従い、全184決定論候補（旧B0+I復元12を含む）とPR確認union6の非相殺診断runnerを準備した。SOURCE `d567e6e1f3b1713b455a80da9b5c5f9e7848f2de`。
各real/imag軸で `abs(discard)+abs(PF)`（full PF `abs(PF)`、PR `abs(outer_PF)+abs(finite_RTE)`）を用い、shotは凍結D0 accountingを呼び出しbias量だけを置換する。5精度・倍率1/10・α.025・u/headroom≤.01を維持。
通常signed bias・経験的非相殺診断・厳密bias保証（null）を出力上で分ける。保存通常会計1,900行とPR paired費用の完全再現を実行前gateにした。
費用比較はB2全4軸PASSの結合時だけで、未検証S4 modelで補わない。旧32と184を別集合で判定し、未解決候補を含む集合の最良・PR比はnull。
PR union6の保存証拠をmetadataだけで照合し、6候補とも成分・端点・forward・normalization・u・32 paired trajectory・seedが揃うことを確認した。数値closure等は承認後gateに残した。確認4次Taylor層は0観測。
B2 recovery attempt02は16:15:03 JSTにMEMORY_PRESSURE（user.slice full PSI）でSTOP。本作業のcgroup PSI累積は不変。費用比較は実行不能。
人工54件（2 interpreter）＋回帰113件PASS。実H8の非相殺bias・shot・費用・ランキング・新signal/compile/sampling/fitは0。
次はユーザーがB2の扱いと、shot診断stageの一回実行を承認するかを判断する。mandatory STOP。
