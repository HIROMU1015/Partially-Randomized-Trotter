# 固定Pauli LD1/LD2のZ1＋T1/Q3診断

2026-11-14 JST。ユーザーが明示許可した2評価だけを実行するnamespace。
正本入力は[INPUTS.json](INPUTS.json)、固定Bは `eecbcdaed88b626b00858c7e94b15f598258d5ec`。
新しいrunnerは固定kernelを変更せず、既存APIを各構成1回呼ぶ。

- 元Pauli packet `714247619afb4714fd663eb15c0470ef32bdc7af` と全26roundをSOURCE照合済み。
- canonical正本はBの`CANONICAL_BINDING_METADATA.json`。ユーザー記載の元LD2 IDに重複した文字列があるため、明示された正本A/B artifactの64桁IDを使用する。物理候補の変更ではない。
- X/Yともcontrol vector `[64,60,56,52,47,43,38,34,30,25,21,17,12]`、USER_SUPPLIED。
- 保存solver energyを条件付きproxy、ideal finite-K IID、EXACT_AFFINE、全事象二乗energy loss。正式ground/実law/数値certificate/十分N資格は付与しない。
- rho=1、count4225、verification55,043,173、total frontier8192、total transition141,559,036を固定。旧部分DPは使わない。
- 元Minkowski・元signal・旧DPは不変。δE加算、目標精度filter、追加配分、N探索、MC、価格/Gは行わない。

実行は2 independent worker。CPU/RAM/threadの独自capなし。既存のresource monitorで自身のworkerだけを観測する。既存total kernelの`step`は返り値を変えない観測wrapperで進捗を記録し、`_total/lemma/DP`の戻り値を境界で保存する。数値SOURCE本体・演算順・allowanceは不変。

rho閉包不成立、非有限、mass不保存、上下界不整合、正当な残余lossがない場合はUNKNOWN/STOPし、再試行しない。有限追跡がkernelの明示scopeを越えることも拒否する。scope内で捨てた状態は既存kernelのstate-dependent残余上界を必ず会計し、partial statusを保つ。追跡範囲の拡張はしない。

人工caller3 test PASS（[RUNNER_TESTS](RUNNER_TESTS.txt)）。これは科学性能や閉包成立の証拠ではない。
run01は排他的に新設し、同じ出力先への再実行・上書きを拒否する。

結果保存・commit/push・remote照合後はmandatory STOP。
