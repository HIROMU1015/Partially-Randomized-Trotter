# Canonical circuit fingerprintの分離計測・省メモリ化

2026-11-07 JST / SERVER 2 / Codex主エージェントによる人工技術検証。

旧policyのJSON bytes/SHA256を保つopt-in実装を追加した。既存の科学builder・既定fingerprint・compilerと、
実H4 Q2のSOURCE/入力/結果/STOPは変更していない。**実H4の新規build/compileは0回**。
人工PASSを実H4・m17のメモリ/時間/価格の保証にはしない。

[新実装][MODULE] / [人工互換性テスト][TEST] / [分離計測runner][RUNNER] / [集約JSON][RAW] / [テスト記録][TESTRESULT]。

## SOURCEと再利用

最初の6条件のSOURCEは[`1401bca826bd9fa3fcd4ec27a0f5b6243f8ed2da`][SOURCE]。
人工入力の自動gate名を修正した限界ケースのSOURCEは[`c3cdb08ef223e30b77e7f01d54ead62c1a3a85ce`][REPAIRSOURCE]。
fingerprint実装と旧serializerは両SOURCEで同一bytesのまま。
基点はQ2限定検証成果`ce2e6736ca1178a4adafc4af344852d958951ea6`。
それぞれテストしたlocal準備commitと同一treeを公開・取得・照合後、benchmarkを実行した。
人工入力はコードに記載した数値だけから生成し、保存分子データ・private state/HDF5/MO積分/provenanceを読まない。

`rte_compiled_cost.py`の旧`_operation_payload`と`_numeric_parameter_payload`を不変更で再利用した。
新実装はtop-levelの固定5 fieldを旧`sort_keys=True,separators=(",",":")`の順序で出力し、
各instructionを旧serializerで展開・JSON化して逐次SHAへ渡す。
policy名`qiskit_recursive_numeric_circuit_v2`、float hex/array hash、operation型・再帰定義、
control state、clbit/qubitの配置、global phaseを含む旧表現は変えていない。
独自のoperation cacheや永続cacheは追加していない。

公開関数は`streamed_canonical_qiskit_circuit_fingerprint`と`iter_canonical_qiskit_circuit_bytes`。
byte iteratorは完全消費し、hash中に入力を変更しないこと。既定の科学pipelineへの置換は行っていない。

## 互換性検証

**人工テストは42 distinct項目PASS**。初回41/41（28.10秒：新規互換性24件＋既存人工compiled-cost/cache回帰17件）、
人工入力修正後に新規互換性25/25（23.11秒）を実行した。変更しなかった旧価格回帰17件は初回結果を再利用。
新規テストは、空回路、primitive、open/closed control、再帰custom gate、Unicode、array/complex/numpy値、
条件付きgate、および人工二mode B2の3長さとX/Y wrapperを旧JSON **bytes単位**で比較した。
同時にSHA一致を確認し、phase/parameter/order/対象qubit/measurement/definition/ctrl_stateの変更を識別する。
別call間の入力変更にも追従し、stale cacheは使わない。

symbolic/layout/calibration/非有限scalar/object array/循環definitionは旧方式と同様に拒否する。
Qiskit1.3.0のcontrol-flowには旧方式も`QuantumCircuit` parameterのTypeErrorを返す。
初回テストでは対応済みと仮定したケースが失敗したため、旧policyの対応範囲を拡張せず、同じ拒否を確認するテストへ修正した。
新規人工テストから分子solverや実H4信号評価は呼ばない。既存回帰内の人工compileと、実H4の実行は別である。
local検証であり、別reviewerやCIによる独立再現ではない。

## 計測protocolと読み方

人工回路は3 qubit/1 clbit、固定数値で、flat（rz/sx/cx/x）とcontrolled（crz/controlled RZZ/controlled XXPlusYY/p）の
反復64/256/1024、最後のH/測定からなる。さらにcontrolled列256反復を単一custom gateへ包む限界ケースを1条件用意した。
H4 trajectoryやroundの代用ではなく、m17へ外挿しない。

各観測はimport後・回路生成前のparentから新しいfork childを作り、入力を独立生成する。
方式ごと3回、実行順をlegacy→stream/stream→legacy/legacy→streamと交替。
自分の計測worker同士のCPU/メモリ競合を避けるため順次実行した。CPU affinity・thread/RAM上限や共有設定は変更していない。

- buildとfingerprintを別に測定。各stage前にGCを行い、tracemallocを開始する。
- full測定はlazy gate definition生成を含む。Python peakはそのstageで新たに追跡したallocationの最大値で、入力全体のRAMではない。
- 別childで旧処理をpayload生成、JSON+UTF8、SHAへ分離。後段のtraceには既に保持しているpayload/bytesを含めない。
  stageのpeakを単純加算してfull peakとはしない。
- full測定後の別callでwarm・tracemallocなし時間も測る。coldのtraced時間と混ぜない。
- RSSはchildの累積process high-waterと前後の現在値。stage単独のpeak RSSは未測定/null。
  fork前のimport状態とQiskitのlazy cacheの影響を含む。

最初の[run01][RUN01]はflat/controlledの6条件・42観測を完了後、1024反復の単一custom gateで入力identityが異なることを検出した。
Qiskitの自動circuit名がfork childのPIDを含み、`to_gate`でoperation名になっていたためで、極小診断の同一入力内では両方式が一致した。
[STOPと極小診断][STOP]を保存し、rawを変更せず計測を停止。該当3観測は比較結論から除外した。
gate名を明示固定する回帰テストを追加し、限界の確認に十分な256反復だけを新SOURCEの[run02][RUN02]で測定した。
先行6条件は再計測せず、SOURCEが不変更のfingerprint実装に対する有効な観測として集約した。

依存環境を用意した上での人工検証入口（出力先は未使用directoryを指定する）：

```sh
PYTHONPATH=src:tests:scripts python -m pytest -q tests/test_streamed_circuit_fingerprint.py tests/test_rte_compiled_cost.py
PYTHONPATH=src python scripts/benchmark_streamed_circuit_fingerprint.py --output artifacts/streamed_circuit_fingerprint/new-artificial-run
```

## 観測結果

**各条件内で両方式のSHAが一致（7条件・49観測）。** 初回の入力identity不一致3観測は除外し、STOP/rawを保全した。
以下は方式ごと3回の中央値。MBは10^6 bytes、時間はwarm・tracemallocなしの別call。min/maxとcoldのtraced時間はJSONに保存。

| 人工case | 反復 | top-level instructions | 旧Python peak MB | stream peak MB | 旧warm秒 | stream warm秒 |
|---|---:|---:|---:|---:|---:|---:|
| flat | 64 | 258 | 3.522 | 0.443 | 0.036 | 0.039 |
| flat | 256 | 1026 | 13.737 | 1.173 | 0.239 | 0.157 |
| flat | 1024 | 4098 | 52.192 | 3.627 | 0.832 | 0.618 |
| controlled | 64 | 258 | 86.769 | 8.421 | 1.789 | 1.357 |
| controlled | 256 | 1026 | 334.955 | 12.329 | 6.716 | 5.424 |
| controlled | 1024 | 4098 | 1348.614 | 25.626 | 31.489 | 22.292 |
| one_large_definition | 256 | 2 | 332.888 | 332.887 | 7.102 | 7.075 |

flat/controlled列ではPython peakの大幅な削減を観測した。小さいflat caseでは時間がわずかに増えており、一般の速度優位は主張しない。
単一の大きなdefinitionでは両方式とも約333MBで、top-level逐次化だけではその内部保持を解消できない。

最大controlled（4098 instructions）の別childでの分離計測：

| Stage | tracemalloc付き秒 | そのstageのPython traced peak |
|---|---:|---:|
| input build | 10.739 | 17.581 MB |
| recursive_payload | 65.971 | 948.439 MB |
| JSON_UTF8 | 29.165 | 400.178 MB |
| SHA256 | 0.124 | 137 bytes |

canonical JSONは200,085,292 bytes。SHA自体よりpayload生成とJSON化が主要負担だった。
最大controlledの累積process peak RSS中央値は旧2,866,428 KiB、stream 300,472 KiB。
これらはimport/build/lazy definitionの履歴を含み、fingerprint単独のRSS peakや実H4の削減量ではない。

## 残る制約と次段階

新実装が捨てるのは**全top-level instructionのpayload/JSONの同時保持**である。
1個のinstruction内の再帰definition/配列serializerは旧実装のまま展開するため、非常に大きな個別gateは依然として大きい。
Qiskitが入力gateへ保存するlazy definition、元flat circuit、wrapper copy、compiled DAG、event tupleも別に残る。
全process RAMがO(1)になったという主張はしない。同じSHAを保つため全canonical bytesの処理は必要で、JSON量そのものは減らない。
旧Q2のbuild/fingerprint込みpeakを今回分解し直したものではなく、その実H4 peakの内訳や削減後RAMは未測定のまま。

次は、人工pipelineでこのopt-in実装を接続し、flat circuit/wrapper copy/whole DAGの保持を切り分ける作業が候補。
実H4へ適用して再検証する場合は、新SOURCE・入力・round・固定trajectory・compiler・build/compile回数・STOPを指定した実行承認が必要。
この成果だけで実H4の旧測定値を書き換えたり、m17 build/compile、期待価格MC、強PF価格、総RPE費用へ進んだりしない。

**Codex**：人工結果とREPORTを専用branchへ公開し、remote SHA/tree/公開内容を照合して本作業を終了する。
**ユーザー**：次の技術接続または実H4限定検証の範囲を判断する。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/1401bca826bd9fa3fcd4ec27a0f5b6243f8ed2da
[REPAIRSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/c3cdb08ef223e30b77e7f01d54ead62c1a3a85ce
[MODULE]: ../../../src/trotterlib/streamed_circuit_fingerprint.py
[TEST]: ../../../tests/test_streamed_circuit_fingerprint.py
[RUNNER]: ../../../scripts/benchmark_streamed_circuit_fingerprint.py
[RAW]: ../../../artifacts/streamed_circuit_fingerprint/2026-11-07/summary.json
[RUN01]: ../../../artifacts/streamed_circuit_fingerprint/2026-11-07/run01/result.json
[STOP]: ../../../artifacts/streamed_circuit_fingerprint/2026-11-07/run01/STOP.json
[RUN02]: ../../../artifacts/streamed_circuit_fingerprint/2026-11-07/run02/result.json
[TESTRESULT]: ../../../artifacts/streamed_circuit_fingerprint/2026-11-07/tests.json
