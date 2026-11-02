# F01の原因と限定修正

[入口](README.md) / [保存境界](failure_provenance_contract.md)。原指摘は[F01][F01]、対象原SOURCEは[OLD][OLD]。

## 1. Stale resultとaxis遷移

旧`run_pair`はsample正常return時だけ`result`へ代入し、失敗時には`'result' in locals()`でeventを取得した。次sampleがreturn前に例外を出しても、前trialのresultが残る。Xを終えた後、Yのstate guard/bits factory/sampleで失敗する場合も、前axisのeventを誤帰属し得た。

修正版はpair開始・axis開始・trial開始で`current_event`/結果状態/候補prefixを初期化する。`current_attempt`はaxis、trial ID/index、stage、result取得、event観測、record構築、candidate完成、emit開始/return、trial完了を持つ。axis guardやbits factoryではtrial IDはnull、sample return前ならresult未取得・inflight_event=nullと記録する。

sampleがreturnしてからevent参照やvalidationが失敗した場合も、取得段階を分ける。angle/evidence取得後の例外には**現在trialのeventだけ**を残す。last_completed_trialは別fieldで保持し、失敗trialへ前eventを流用しない。

## 2. Candidateとcommitted prefix

旧digestは`stream.emit`前、counterは後で更新され、保存失敗時のtraceとcounterのscopeがずれた。

修正版はpair/axis/prefix digestとcounterをcopyした候補に更新する。この候補は`candidate_ready`まで未確定であり、`emit`が正常returnした後だけ`committed`参照を候補へ切り替える。完了trace/counter/last trialは同じ参照に含める。hash更新やrecord構築・emitが例外になっても、前の完了prefixを変更しない。

正常完了の`trace_sha256`・axis trace・最初のcorrectness prefix・counterの意味は維持した。失敗時の`trace_sha256`は明示的に`COMPLETED_TRIALS_AFTER_NORMAL_EMIT_RETURN`とする。`candidate_trace_sha256`は未commitの別fieldであり、成功済みtraceではない。

memory observer等がtrial commit後に失敗した場合は、そのtrialは既に完了済みである。`current_attempt.trial_completed`とlast_completed_trialでこの境界も区別する。全bundleとしては引き続き失敗・STOPであり、全N成功へ格上げしない。

## 3. Streamの部分失敗

`Stream.emit`はunbuffered writeの返却byte数を受け取った時点で記録し、その後file hashを更新する。hash更新中の失敗では`file_hash_covers_written_bytes=false`、対応hashはnullにする。旧順序のように、writeが返ったのにbytesが未加算のままになることを避けた。

失敗stageをSERIALIZATION／CHAIN_HASH／WRITE／FILE_HASH／FLUSHで記録する。streamの完全row/chainの確定は従来通り全write/hash処理後、flushより前である。従ってflush失敗では「完全rowが存在し得るがemit未return、trial未完了」が成立する。単にdigest更新を後へ移しただけでpartial writeがなくなるとは扱わない。

## 4. Failure記録自体の失敗

原例外に`b3_failure_provenance`を付け、typed failureのstream保存が失敗しても、原例外を再raiseする。secondary failureは別fieldへ残す。frozen例外の`to_dict()`が失敗しても原原因を置き換えない。

child側の`.failure.json`はこのtrial provenanceを既存のexact canonical文字列として引き継ぐ。control log書込みやcleanup closeのsecondary例外も、原trial例外を隠さない。書込不能ならメモリ上の例外・保存済みprefixに限られ、強制kill後のRAM回収を保証する仕組みではない。

## 5. 科学・計測scopeを変えていないこと

既存のendpoint.sampleの呼出、bit factory/消費順、raw/coin/event分類、event phase/order/weight、Full F3・canonical angle/requeryを変更していない。N、16 cells、Layer A/B、二pass、setup/cache/observer、confidence/error、one-shot/retry/STOPの定数と仕様も不変。

新しいtiming stageやscience callは追加していない。failure provenanceのbookkeeping/hash copyは実装上の追加作業であり、将来の実費には含まれる。**実CPU費用が旧版と同一・高速になったとは主張しない。** fake clockの8ケースでは既存粗stage記録と正常出力bytesまで一致した。実G7 correctness・性能は未検証である。

変更はStream、run_pair、executeの失敗転送/cleanupに限定。外部atomic admissionの実装、C1/G9価格接続、科学条件の採択を行っていない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[F01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6576e606d4edcc6c53f095ee5bdf750defa99a42/docs/research/track_b_g7_g9_p1_final_driver_delta_20261019/technical_findings.json
[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py
