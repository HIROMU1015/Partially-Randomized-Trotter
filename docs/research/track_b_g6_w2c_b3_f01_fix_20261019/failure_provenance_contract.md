# F01 failure record：生成・受理・完了・永続化の境界

[入口](README.md) / [原因](f01_root_cause_and_fix.md)。これは失敗metadataの技術仕様であり、科学実行契約・数値保証・authorityではない。既存のstream envelope schemaは変更していない。

## 1. 境界とfield

| 境界 | 記録・意味 |
|---|---|
| 結果未取得 | `current_attempt.result_acquired=false`、`inflight_event=null`。前trialのeventを示さない |
| 結果取得／event観測 | result_acquired=true。event_presentは現在のevent参照が得られたか。ZEROならfalseで正常。RESULT_ACCESS失敗ならeventは未観測 |
| Record構築 | `record_constructed`。event/evidenceのin-memory記録構築を示し、serialization/保存成功を示さない |
| Candidate trace | `candidate_ready`と`candidate_trace_sha256`。copy上で構築した未commit候補。hash途中失敗ならready=false・hash=null |
| Emit開始／正常return | `stream_emit_started/stream_emit_returned`。正常returnは必要なwrite/hash/指定flush処理を通ったというdriver上の境界 |
| Trial完了 | `committed`参照の公開。completed_prefix、trace_sha256、last_completed_trialが同じ完了集合を示す。trial_completedで現在trialとの一致を記録 |
| Streamの書込みprefix | `stream_prefix_before_failure_record`。typed failure自体を保存する前の観測snapshotであり、event trial counterとは別 |
| 再読・電源断耐性 | `readback_verified=false`、`fsync_durability_verified=false`。productionでreread/fsyncを実施済みとはしない |

`current_attempt.stage`はPAIR_IDENTITY、AXIS_IDENTITY、BITS_FACTORY、TRIAL_BUDGET、SAMPLING、RESULT_ACCESS、RESULT_VALIDATION、EVENT_VALIDATION、RECORD_CONSTRUCTION、ANGLE_AND_REQUERY、EVIDENCE_CONSTRUCTION、CANDIDATE_DIGEST、STREAM_EMIT、TRIAL_COMPLETED、MEMORY_OBSERVATION、AXIS_COMPLETION、FINAL_IDENTITYを区別する。

trial開始前のaxis guard/bits factoryではaxisだけを示し、trial ID/indexはnull。pairの最初・最後のguardではaxisもnull。最後の正常trialはlast_completed_trialへ保存する。axis切替後もinflightには残さない。

## 2. Digestのdomain

- `trace_sha256`：emit正常return後に完了が確定したtrialの、従来と同じkind/event canonical bytes列。
- `candidate_trace_sha256`：現在trialを追加しようとした候補。完了済みtraceと別で、失敗時にのみ提示する。
- `last_complete_chain_sha256`：Streamが確定した完全rowのchain。memory/failure record等も含み得るため、trial traceと同じdigestではない。
- `written_bytes_sha256`：writeが返したbyte列の累積hash。partial tailも含み得る。file hash更新が失敗した場合はnull。productionのdisk reread検証hashではない。

`complete_rows`はStreamの`rows`、すなわちwriterがrow metadataを確定した数である。失敗writeが完全newlineを出した後にfile hash処理で止まった場合など、**それより先のbyteがfileに存在しないという保証ではない**。flush失敗ではrowsに現在rowが含まれてもemit未return・trial未完了となる。

`written_bytes_including_possible_partial_tail`は成功したwrite呼出が返したbyte数を追跡する。例外を出したI/Oに未報告の副作用がある場合まで計測済みとはしない。stream/hash fieldだけから、未完了trialを完了として回復・resumeしない。

## 3. Secondary failureと保存限界

failure recordの保存も失敗した場合、原例外に付けたprovenanceに`failure_record_write_error`と失敗後stream snapshotを残す。これらは書けなかったfailure record自体へ遡って保存されるわけではない。child control logが書ければ、そのexact canonical文書を`trial_failure_provenance`として残す。

既存codecに従い、外部streamのpayloadはcanonical JSON文字列で、int/Fractionはhexタグ（例：`{"int":"0x1"}`）を使う。既存`.failure.json`では追加provenanceもこの文字列として格納する。旧stream envelopeのprevious/SHA/payload構造は不変。消費側は新`failure_provenance_schema=g6_w2c_b3_f01_failure_v1`を確認し、null/未観測を0へ変換しない。

OS kill、電源断、OOM、全保存先のI/O不能では、メモリ上の例外属性・最終receiptの保存まで保証しない。保存できた完全prefixと部分末尾を区別して扱い、全N成功・独立再現・fsync耐性を認定しない。今回の人工partial-write/flush testsはprivate fileの検証であり、実storageの耐障害試験ではない。

失敗はtyped failureでありrandom zeroではない。元のone-shot、retry0、resumeなし、mandatory STOPを維持し、本仕様でclaimを発行・回復・再利用しない。
