# C8-1：承認済み数値修正・保存回路の再検証PLAN

2026-11-08 JST。利用者の「修正して再実行してよい」により、公開1e88074fの数値preflight FAILを対象に修正・再実行する。これは旧SOURCE/STOPを書き換える許可ではない。新branch・新namespaceで来歴を分ける。

## 固定条件と最初の修正

旧[PLAN](PLAN.md)のH4/1Å/STO-3G/Nα=Nβ=2、元H/number-CDF8/LD2/S2/s1/r32/K0/h=.1、q64/cut32、元H厳密ground仮定・準備費除外を維持する。全8保存event/seed、sample00のraw PASS、4 P0回路をそのまま利用する。SOURCE6b1d613d/8a6bbdac、旧結果・失敗・STOPを保全する。

現checkerはFrobenius閾値で例外を出した場合、途中checkとobservableを保存しない。今回この診断欠落を修正する。既存checkerを変更せず、worker内のcallbackで各check・observable・9qubit以下のmatrixを例外前に保存する。callbackは必ず復元する。失敗をPASSへ置換せず、失敗後も新samplingや後続pathを投入しない。

## 固定診断batch

新SOURCEを固定した後、保存sample00のstrong X/Y・Reference X/Yを各一回、合計4 postcheck再実行する。既存C7-3 exact cache、公式Aer double unitary≤9qubit、既存BatchedStateActions/公式Aer double statevector・全256入力列と全Reference出力を使う。17qubit全Operatorは作らない。

P0/回路/位相/入力/対照/閾値は不変。component abs1e−10、observable Frobenius/signal2e−10、rtol0、global phase fitting/renormalizationなし。旧raw/compileの再実行0、sampling0。4 taskは実affinity CPUを分配して並列化し、小BLAS nested並列だけを抑える。十分な実RAMを確認する。共有環境/他jobへ変更しない。

診断出力は`run02_diagnostics`。partial check、CAR/raw両observable差、例外、SOURCE/input hash、旧原本の前後照合、STOPを保存する。array/matrixはlocalのみ、公開はscalar/check/hashのみ。

## 原因に基づく修正と境界

今回のユーザー許可は数値不具合の修正・再検証を含む。診断で原因を特定できた場合、既存資産を優先し必要最小限の数値修正を別のSOURCEで固定し、人工検証後に同じ保存回路を検証する。科学条件/判定基準/比較回路を変更する修正は採用しない。新たな高精度経路を必要とする場合は、その技術的必要性・精度・適用範囲を実行前に記載する。診断batchの自動無制限retryは行わない。

本診断runnerは4件終了時にSTOPを記録し、残7 pathを自動投入しない。修正が成立しない場合は不足を明記して停止する。sample00の全raw/P0/pairedχ gateが成立するまで残7を解放しない。既存4compileは再compileせず、原campaign総32の上限を維持する。

q≥128、追加sampling、ground/DF solve、モデルfit、N再算定、全13-round更新、重要科学レビューは対象外。新旧のSOURCE/input/raw/result/grant/ledger/marker/STOPと公開制限を保全する。成果公開・remote SHA/tree/bytes照合後mandatory STOP。
