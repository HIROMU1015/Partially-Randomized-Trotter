# T0人工・回帰検証

2026-11-09 JST。**29件PASS、FAIL／skipなし。** 保存H4値の再解析、科学性能・P0互換性の検証ではない。

| Suite | 件数 | 実施内容・保存出力 |
|---|---:|---|
| NativeTests | 18 | stdlibの小native word。既存C7-4 feature回帰1件を含む。[出力](../../../artifacts/track_d_t0_native_resources/2026-11-09/technical/native_unit_tests.txt) |
| QiskitNativeTests | 4 | Qiskit1.3.0の既存metricsとの照合、reader/mapping/拒否条件。[出力](../../../artifacts/track_d_t0_native_resources/2026-11-09/technical/qiskit_reader_tests.txt) |
| 固定設計mock | 7 | 元a2b7099dの人工mockを無変更で再実行。[出力](../../../artifacts/track_d_t0_native_resources/2026-11-09/technical/design_mock_regression.txt) |

18件に含まれるC7-4回帰は重複計上しない。三suiteは独立に並行実行した。T0 coreのimportと空word計数がQiskit/numpyをimportしないことも別途確認した。共有環境・Python環境の変更はない。

## 確認した性質

| 性質 | 結果・適用範囲 |
|---|---|
| RZ/CX/count/size/width | PASS。RZ(0)、idle qubit、measurement、global phase metadataを区別 |
| Serial／parallel | PASS。独立laneを余分に直列化せず、共通gateの後だけ依存を伝播 |
| 非一様arrival | PASS。三指標別の入力・出力と10^24級整数を保持。float化による丸めなし |
| Filtered同期 | PASS。CX/measurementを削除せず、対象外の重み0操作としてwireを同期 |
| Classical依存 | PASS。異なるqubitから同じclassical bitへのmeasurementも同期 |
| 全chunk cut | PASS。4種alphabetの長さ0〜3、85 word、全313 cutで一括counterとtemplate連結が一致。非一様入力でも確認 |
| Lane embedding | PASS。独立registerの順序交換、shared ancilla後続、qubit/classical permutationを保持 |
| Empty／identity | PASS。空word、ゼロwire、template前後へのidentity連結 |
| Actual inverse受取 | PASS。callerが明示した逆順・逆角・3SX展開列を計数。max-plus逆行列は提供しない |
| 一回限りstream | PASS。各yield後にcounterの処理済みcountを検査するproducerで、先読みlist化・二度読みを排除 |
| Unsupported | PASS。CRZ/RZZ/CSWAP等、custom gate、symbolic angle、directive、条件、arity・mapping不正を拒否 |
| 旧counter照合 | PASS。C7-2関数本体の零入力wire exit、既存metricsの9項目と一致 |
| 既存feature回帰 | PASS。`Models.test_feature_max_is_pathwise`をそのまま限定実行、lane-runの意味を保持 |

小native wordのalphabetはRZ(0)、X(1)、CX(0,1)、Measure(1,c0)。word数は各長さ0〜3の完全列挙であり、PR sampling／whole-path標本ではない。人工gate列の実行・unitary計算も行っていない。

Qiskit比較では小さい人工native circuitを作り、`transpile`に呼出し禁止patchを設定した上でcount/depthのみ照合した。既存の `track_d_minimal_reference.metrics`、`weighted_depth`、C7-4 featureと回帰methodはASTで関数本体だけを読み、科学runner・fit・simulationを含む既存module一括実行を避けた。既存回帰module全件をPASSと呼ばない。

## 再実行入口

repository rootから、stdlib suiteと旧mockは次の入口を使う。

```bash
python3 -B tests/check_track_d_native_resources.py NativeTests
python3 -B docs/research/track_d_longq_structural_resource_design_20261109/check_maxplus_mock.py
```

Qiskit1.3.0が既存導入済みのプロジェクト環境では次を実行する。新しい環境導入は今回していない。

```bash
python -B tests/check_track_d_native_resources.py QiskitNativeTests
```

Qiskit未導入環境でこの4件だけを実行するとskipとなる仕様だが、今回の記録はQiskit1.3.0導入済み環境で全4件を実行しPASSしている。

## 残る制約・既知の反例

入力wordのCX対は2本のまま数える。P0が0へ消す場合も、T0がP0互換を保証していないことと整合する。C7-2の局所非加法性・CSWAP shell・shell＋max(lane)反例は保持し、補正定数で消していない。

このテストはP0 interface閉包、分子AB=V／χ／測定同値性、全support、長q精度、全law期待値やactual RPEを確認していない。global phase metadataの浮動小数点和は整数資源の厳密性と別である。C6a〜C8-1の適格price record・A/F/Sを更新しない。
