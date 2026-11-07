# B2人工pipelineのfingerprint接続とflat/wrapper/DAG分離

2026-11-07 JST / SERVER 2 / Codex主エージェントによる人工技術検証。

**人工Q=1・2・8で限定PASS。** 省メモリfingerprintをnative最終hash、wrapper改変チェック、既存pair compilerへ接続し、
旧方式とnative/X/Y wrapper/compiled fingerprint・全6価格指標が一致した。既定動作は旧関数のまま。
**今回の実H4 build/compileは0回**。過去のSOURCE commit・raw・STOPは保全し、既存結果を書き換えていない。

[全結果][RESULT] / [summary][SUMMARY] / [人工20 test][TESTRESULT] / [runner][RUNNER] / [test source][TEST]。

## SOURCE・入力・再利用

実行SOURCEは[`fcfe3b6732ba795fbfa06ef677c2a141bd36ccff`][SOURCE]。
基点は前回のfingerprint人工検証成果`e75e1b217ec4a74214b1a1e395de80e8e48aeb4b`。
同一treeのlocal準備commitを保全し、公開SHA/treeを取得・照合したSOURCEで計測した。
実行が消費する9ファイルは開始時にcommit内容、終了時にSHA256不変を確認した。

既存の人工二mode B2 rank4、LD2、partial-S2/r4/K2、delta=.02、ordinary controlを使用。
`tests/b2_cost_method_cases.py`の`artificial_round`と`event_pattern('mixed')`を再利用し、
Q1/Q2/Q8に対する4/8/32個の明示canonical eventを使用した。benchmarkでRNG samplingはしていない。
system2＋ancilla1、元のsigned factor/scalar/identity/bit order、X=Re/Y=Imの規約を維持。
保存分子入力、private state/HDF5/MO積分/provenance、実H4 solver・信号は使用していない。

compilerはQiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend/targetなし・論理all-to-all。
backend fingerprintは`no_backend`、scopeは`single_hadamard_interrogation_without_state_preparation`。
6観測で人工whole compileは12回（各観測のX/Y各1回）。unittest内の人工compileとは分けて記録する。
期待価格MC・強PF価格・shot・総RPE費用の計算ではない。

## 接続した最小API

変更したlibraryは3ファイル。新しい`fingerprint_function=None`は、未指定時に旧関数を選ぶ。

- `stream_native`からnative `_finish`へ、最終circuitのhasherを渡す。
- `build_native_hadamard_wrapper`で同じhasherを改変チェックに使う。検査自体は省略しない。
- `o1_m1_resource_bridge._compile_pair`で同じhasherをwrapper identityに使い、既存compilerへ渡す。

gate emitter、sampler、wrapper構成、compiler、6指標計算は再実装していない。
`streamed_canonical_qiskit_circuit_fingerprint`と旧recursive serializerは前回から不変更で、policy/SHAも変更していない。
任意hasherは同じcanonical identity policyを実装する必要がある。placeholder hashや事後phase fittingは使っていない。
既定の`MatchedNoPrepCostProvider`/MC経路を新方式へ自動切替する設定は追加していない。

**人工20/20 test PASS、21.23秒**（新規7＋既存streaming13）。
Q1/Q2/Q8の従来nativeとのgate列・global phase・wrapper identity、独立CAR oracleによるcontrolled意味、
両hasherの改変検出、失敗の伝播、既存pair compilerでのX/Y compiled identityと6指標を確認した。
localの機能test後にSOURCEを固定し、公開SOURCEで下記pipelineを実行した。独立reviewerやCIによる再現ではない。

## 段階計測のprotocol

旧/新hasher×人工Q1/Q2/Q8の6条件を、各1回・それぞれ新しいfork childで実行した。
worker競合が時間に混ざらないよう順次実行し、方式の先行順を条件間で交替した。
CPU affinity・thread/RAM上限や共有環境・他processは変更していない。
単回pilotであり、一般の速度改善率・有意差は主張しない。

前回の`profile`/RSS helperを再利用。nativeの実際のhash callback直前でflat emissionを閉じ、hashを別stageにした。
wrapperはprecheck、native再検証hash、compose/H/Sdg/H/measureを分ける。
pair compilerの入力hash、whole compile＋6指標計測、compiled hashと、前後のDAG変換も分けた。
計測のためにhash検証を飛ばしたり、gate/wrapperを別実装に置換したりしていない。

- 各stage直前にGC・tracemalloc開始。Python peakはそのstageで新たに追跡した割当で、既に生存する入力全体を含まない。
- 回路copyはhash後の診断で、直後に解放。`circuit_to_dag(copy_operations=True)`の診断DAGも記録後に解放し、次のcompileへ重ねて保持しない。
- 最後はnative、X/Y wrapper、X/Y compiled回路、明示event tupleを保持してRSSを記録する。
- RSSは前後の現在値とchild累積high-water。各stageのRSS peakは未測定/null。
  変換器・compilerのRust/C側割当、allocator再利用・未返却領域もあるため、Python peakだけからDAG全体のRAMを判定しない。
- 時間はtracemalloc付き。GC等も含む全wall時間はJSONに別記。stageのpeakを合算して全体peakにはしない。

## 結果

全3条件でnative、両wrapper、compiled identity、compiler/backend/scope/measurement数が一致し、6指標の差は全て0。
表の価格は今回の人工回路だけの値で、各条件のX/Y・旧/新で一致した。

| 人工Q | native instructions | CX count | RZ count | CX depth | RZ depth | total depth | circuit size |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 86 | 140 | 243 | 140 | 153 | 340 | 490 |
| 2 | 171 | 276 | 465 | 276 | 296 | 659 | 940 |
| 8 | 657 | 1076 | 1761 | 1076 | 1134 | 2521 | 3556 |

native構築と最終fingerprintのPython traced peak（MB=10^6 bytes）：

| 人工Q | flat旧 MB | flat新 MB | native hash旧 MB | native hash新 MB |
|---:|---:|---:|---:|---:|
| 1 | 0.324 | 0.323 | 7.437 | 1.958 |
| 2 | 0.427 | 0.428 | 14.540 | 3.406 |
| 8 | 0.869 | 0.868 | 53.543 | 10.464 |

最大Q8の段階別観測。X軸を示し、Y軸の全stage値はJSONへ保存した。

| Q8/X stage | 旧秒 | 新秒 | 旧Python peak MB | 新Python peak MB |
|---|---:|---:|---:|---:|
| flat emission | 0.1978 | 0.1977 | 0.8692 | 0.8684 |
| native hash | 3.1606 | 2.8496 | 53.5429 | 10.4643 |
| native copy | 0.0259 | 0.0258 | 0.2828 | 0.2828 |
| wrapper native検証hash | 2.6329 | 2.3622 | 47.1530 | 4.7598 |
| wrapper compose＋測定 | 0.0521 | 0.0520 | 0.5808 | 0.5807 |
| wrapper入力hash | 3.3926 | 3.1003 | 51.5238 | 10.7764 |
| 入力DAG変換 | 0.0270 | 0.0262 | 0.2815 | 0.2815 |
| whole compile＋6指標 | 0.4752 | 0.4725 | 1.7036 | 1.7033 |
| compiled hash | 3.0058 | 2.7639 | 47.1051 | 6.2643 |
| compiled DAG変換 | 0.0014 | 0.0011 | 0.0032 | 0.0032 |

Q8終了時の現在RSSは旧342,544 KiB、新274,836 KiB、累積process peakは旧347,960 KiB、新274,836 KiB。
これらにはimport、native/両wrapper/両compiled回路、lazy定義、allocatorの履歴を含む。stage単独のRAM増分や実H4の削減量として扱わない。

## 限界と次の作業

このpilotではhashのpeak低下を実pipelineで確認できた。一方でflat circuit生成、copy、wrapper compose、whole compile/DAGは
同じ処理のままで、最終回路・2 wrapper・2 compiled回路の保持も残る。
Qiskitのlazy definitionと個別instruction内の再帰展開も残り、pipeline全体が定数メモリになったとは主張しない。
繰り返しhash/検証の時間も残る。改変チェックを無条件に省略したり、object IDだけのcacheに置換したりしていない。

この小さい人工ケースでDAGの現在RSS差が0に見えても、DAGが無メモリ・長roundで安価という意味ではない。
実H4・m17の時間/RAM/価格・compiler実行可能性は未測定/nullのままで、Q8から外挿しない。

**次の候補（未実行）**：新方式を実H4 Q2の保存済み固定trajectory `[0,4,4,0,4,0,0,4]`へ限定適用する。
Linear H4/1Å/STO-3G・保存DF12・H_phys target・generation LD3・partial-S2/r4/K2・delta=.02・T=.04・ordinary control、
元H基底state仮定・prep費用除外を維持。compiler/backend/scopeも前回Q2と同じにする。
保存済みQ2成果`ce2e6736ca1178a4adafc4af344852d958951ea6`のnative/wrapper/compiled identity・全6価格をreferenceとして再利用し、
candidateのstream nativeを1回、X/Yを各1回compileする案（合計2回、旧reference再build/compileなし）。
実行前に最小runnerの新SOURCE・入力・trajectory・回数・不一致時STOPを固定する必要がある。この追加科学実行は今回の人工作業の範囲外で、未実行。

**ユーザー**：上記の実H4 Q2限定検証を次に承認するか判断する。
**Codex**：今回の人工成果を公開しremote SHA/tree/全変更ファイルを照合して作業終了。
次の明示指示があれば新runner/SOURCEを用意し、その固定範囲内で進める。m17・MC・強PF・総費用へ自動進行しない。

再現入口（依存環境が必要、出力先は未使用directory）：

```sh
PYTHONPATH=src:tests:scripts python -m pytest -q tests/test_b2_fingerprint_pipeline.py tests/test_b2_streamed_native_cost.py
PYTHONPATH=src:tests:scripts python scripts/benchmark_b2_fingerprint_pipeline.py --output artifacts/b2_fingerprint_pipeline/new-artificial-run
```

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/fcfe3b6732ba795fbfa06ef677c2a141bd36ccff
[RESULT]: ../../../artifacts/b2_fingerprint_pipeline/2026-11-07/run01/result.json
[SUMMARY]: ../../../artifacts/b2_fingerprint_pipeline/2026-11-07/summary.json
[TESTRESULT]: ../../../artifacts/b2_fingerprint_pipeline/2026-11-07/tests.json
[RUNNER]: ../../../scripts/benchmark_b2_fingerprint_pipeline.py
[TEST]: ../../../tests/test_b2_fingerprint_pipeline.py
