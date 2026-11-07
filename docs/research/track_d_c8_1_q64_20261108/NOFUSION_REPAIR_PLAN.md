# C8-1：保存4 wrapperのsimulation fusion数値修正・再検証

2026-11-08 JST。利用者の「修正して再実行してよい」に基づく修正版。旧SOURCE/失敗/STOPはそのまま保持する。[数値診断PLAN](NUMERICAL_REPAIR_PLAN.md)とSOURCE46ce57e1による固定4診断の完了を待ち、同じ保存sample00の4 wrapperだけを再検証する。残7 pathのraw生成・追加28 compileへの自動進行は含めない。

## 診断と最小修正の理由

保存strong X/Yの再診断ではCAR-observable Frobeniusが2.1474799974401975e−10 / 2.2578758683084535e−10、raw-observable差も2e−10を超えた。compiled unitaryのunitarity Frobeniusは約4.2813e−10。既存component1e−10は通過したがsignal2e−10で不成立。診断配列は非公開のまま保持する。

C7-4の固定人工block生成器を使う9qubit/1000反復の技術試験では、既存generic local matrix経路のunitarity Frobeniusがfusion有効1.4506072461498746e−11、無効3.407997940484205e−12。native gate経路も同値だった。これは数値伝播の改善候補を選ぶ技術証拠で、実H4の原因の完全配賦やq64 PASSではない。5qubitではdefault fusionが適用されないため同じ改善は観測しなかった。

**既存公式Aer double backendの`fusion_enable=False`だけを選択する。** SmallUnitaryとBatchedStateActionsを継承・再利用し、gate matrix/cache・phase・全256入力列・Reference全出力を維持する。新sampler/Gaussian/compiler/state simulatorを実装しない。17qubit全Operatorは禁止。数値出力のphase fit/renormalizationなし。

これは**simulation内部fusion**だけの変更であり、保存P0回路、strongのbasis共有/head/tail/63step境界融合・位相集約、Referenceの62内部融合/cut1境界損失は変更しない。

## 固定再実行範囲

H4/1Å/STO-3G/Nα=Nβ=2、元H/number-CDF8/LD2/S2/s1/r32/K0/h=.1、q64/cut32、厳密元H ground仮定・準備費除外。保存sample00 raw/referenceと4 P0のbytes/hashを継承する。新sampling/raw生成/compileはすべて0。元8event/seed/cohortを変更しない。

P0/Qiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/宣言位相保持/topology-freeは不変。component1e−10、signal Frobenius2e−10、rtol0、漏れ/χ/actual adjoint/AB/CAR/ordered path/X/Y全条件を維持する。

新namespaceは`run03_nofusion_repair`。保存strong X/Y・Reference X/Yを各一回、合計4 postcheckだけ修正版で行う。既存diagnostic runner/checkerを使い、失敗時もcheck/observableを保存する。診断完了後に4 taskを実affinity CPUで並列化し、十分な空きRAMを確認する。旧診断workerへsignal/killせず完了を待つ。共有環境/他ユーザーへ変更しない。

4件終了時にPASS/FAILと原本前後照合を保全しSTOPする。固定条件不一致や検証FAILをPASSへ変更しない。追加の設定探索/retry・閾値緩和なし。残7 path、新q、sampling、compile、fit、N、全13-round更新、ground/DF solve、詳細レビューは0。

旧SOURCE・raw・入力・結果・grant・ledger・marker・STOPと公開制限を保全。非公開event/QPY/NPZ/内部環境は公開せずscalar/check/hash/SOURCE/結果を公開する。remote SHA/tree/public bytes照合後mandatory STOP。次の研究判断はユーザー/GPTへ戻す。
