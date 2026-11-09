# Track D C7-4：保存価格モデル改訂・数値安定性・次段階準備

2026-11-07 JST。利用者指示による一括許可内。基点acea439の正確なSHAは `acea43987edae0d5eb0c186a89aa8cd7139fbd7b`。旧SOURCE/結果/STOPを変更せず、専用branch/worktree/namespaceで実施する。追加の承認証明・grantは作らない。

## 固定条件と入力

H4 linear1Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered S2/s1/r32/K0/h=δ=.1。元H厳密ground仮定・準備費除外。packet1d6087f6、endpoint a8813aca、law c347d7f2の完全identityとP0を既存price recordで照合する。P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initially_zero=false/topology-free/位相保持。旧component1e-10、observable2e-10、rtol0、phase fitなしを維持する。

C6a q2/q4各16、C6b q8の32、C7-1 q16の16、C7-3 q32の8、計88 whole-path/352 price records。X/Yは同pathであり176独立標本としない。C7-3はsampling/raw35bfd、compile/post94e94、最終acea439を区別する。旧C6a係数/共分散・holdout診断を不変に保つ。

## 実行順序と有限モデル候補

1. 保存352価格・correctness・event hash・feature・旧モデルのidentityを検査して専用成果へ統合。既存describe/統計規約を再利用し、q別標本数・方式/資源/axis共分散・旧残差を保存する。
2. CXは既存C7-2局所式/反例と88整数残差を対応させ、固定/step/boundary/P0残差を分離。未知のG0/全P0 bridgeを観測式から逆算して証明しない。
3. 探索的候補は固定C6a affine、全5qでのfixed+bulk affine（2係数）、[1,q,sqrt(q)]（3係数、max of two fluctuating lanesを動機とする仮説）、[1,q,basis runs]またはReferenceの[1,q,max(lane run)]（3係数）に限定。step/boundary/cut定数のdesign rankを確認し、共線項へ独立の因果係数を与えない。countsとdepthで同じ低次元候補を記録し、高次多項式/恣意的feature探索はしない。
4. 方式別絶対価格・paired差を別responseとしてfit。whole-path単位、q内のnを維持し、equal-q重みでOLS。leave-one-q-outの5fold（q32 foldは外挿）を全候補に保存し、モデル選択に使うため独立holdout/確認的coverageではない。原標本qごとに再標本化する固定seedの256 bootstrapは統計再解析で、新H4 trajectory samplingではない。方式/資源/XYを同じindexで再標本化し、係数・将来round間共分散を保持する。bias/law誤差を0とは置かない。
5. 既存eventのみからrun/support/左右lane特徴を抽出する。分子回路の再生成/compileは0。q32の既存compiled wire/DAG featureは価格の事後的再記述として比較し、そのまま未使用q予測へ入力しない。q8/q16も必要最小限の保存結果を利用する。E[max]とmax(E)を交換しない。
6. 保存correctness全件のq/label/方式別推移を記録し、q32の最大observable残差を示した1 pathを固定規則（最大、同値はID/method/axis辞書順）で選んで保存observable/参照NPZだけを限定再照合する。旧simulationやcompileを再実行しない。原本は既存hashへ照合し、不一致なら該当処理STOP。
7. 人工3/5qubit固定blockとactual inverse/宣言phase、cache keyを検査。長さ1024/8192/32768 blockで公式Aer generic-local-matrix経路と、native rz/sx/x/cxを保つ別revisionの経路を、longdoubleの短block行列累乗と比較する。全行列coverageを保ち、full17 Operatorなし。短い人工block2本のみ同P0をcompileして互換性を確認する。既存高速化・cache回帰も再利用。toyで閾値を超えれば記録し、将来長q numerical GOは出さない。別経路は事前に並列比較する候補であり失敗後fallbackではない。
8. 元凍結モデルと新q-only探索版のq64/128/256/512/1024/2048/4096予測・共有係数共分散を別保存。feature版には未使用qのfeature lawが不足する場合、数値予測を捏造せず条件付き仕様を示す。shot乗算/全13-round総費用は算定しない。
9. 次qは64を自動採択せず、モデル分岐・外挿・実行可能性・数値余裕を比較して条件案を作る。探索的pilotの科学call数を有限にし、確認的標本数は別のtail/variance/bias根拠が必要と明記する。

## 保全・資源・終了

新H4 sampling/build/compile、追加標本、旧run retry/resume、ground solve/DFfit、N/actual RPE/全13-round費用、routing/T/noise、詳細科学レビュー0。数値再解析・人工テスト・最小技術改善・予測・報告の範囲のみ。共有CPU/RAM等の実空きを確認して独立処理を並列化し、他job/共有OS/venvを変更しない。SOURCE/input/law不一致、本質的correctness不成立/精度緩和の必要、保護資料の破損、scope外計算が必要なら依存処理をSTOPし、独立な許可済み作業は保全して完了する。

旧event/QPY/NPZと内部CPU/RAM/pathは非公開。SOURCE/PLAN/統合scalar/探索fit/共分散/数値検証/次段階案/STOPのみ公開。専用branchへcommit/push、remote SHA/tree/bytes照合後mandatory STOP。次担当ユーザー/GPTが研究判断・次の科学実行範囲を採択する。
