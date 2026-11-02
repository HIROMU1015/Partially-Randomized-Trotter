# 2026-10-05 Track B：BM-0.5同値性監査

## 新しいreview指示

利用者から`REVISE_BM0_METHOD_DELTA_BEFORE_BM1`を受領した。
BM-0を維持し、science前にcompact BCHとの同値性とscore／reuse差を閉じる。
初回pilotのscopeはleading heuristic＋I2評価、primary count、人工costはsecondaryという修正指示。

## 技術監査の結果

base `3b2d624adde979f7d8f983bc7fdbbec88f90c500`から独立B branchで実施。
Maxwell v1のEq.(10)/(19)をnative nested列へ逐次適用するとK_floor+K_A/(4m^2)になる。
任意sizeの再帰導出を記し、抽象free words/Fractionの9 fixtureで、literal exp/log・compact・BMの
degree3係数がすべて完全一致した。内部係数の誤置換K_A/m^2は6非退化fixtureで検出した。

同backend・同aggregationならscoreと順位も同じ。norm前後のgroupingは差を作り得るが、
compact側も同じpolicyと同じfloor/internal cacheを取得できる。
現案にはその対照から得られないscore／reuseを示せず、new-method gateは不通過。
[監査本文](../../tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md)を根拠とする。

## 反映と停止

旧BM-0本文・artifactを保持し、[BM-1 v2 amendment](../../tracks/algorithm_codesign/bm1_pilot_scope_amendment_v2.md)と
[GPT packet](../../tracks/algorithm_codesign/bm05_review_packet_20261005.md)を追加した。
元science result/status/markersとTrack Aを変更しない。
形式algebra監査以外のscience/Hamiltonian/NPZ/signal/trajectory/compile/GPU/full suiteは0。
BM-1を新手法検証として実行しない。公開後STOPし、applicationへの縮小や別deltaの採否はGPTへ戻す。
