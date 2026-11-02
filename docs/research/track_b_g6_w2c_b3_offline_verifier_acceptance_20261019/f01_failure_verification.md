# F01：可視行と完了trialの区別

基準は07f65831f045bf1fdbd4b370e44c7f59a9901dddのfailure_provenance_contract.md、実装は基点のF01統合driver。

| 境界 | 検証 |
|---|---|
| Sample結果未取得 | result_acquired=false、inflight_event=null。前trial eventの誤帰属を拒否 |
| Event取得 | event_presentと現在inflightの有無。ZEROのeventなしを許す |
| Record構築 | record_constructed。まだserialization成功を意味しない |
| Candidate完成 | candidate_readyとcandidate_trace_sha256の形・依存順 |
| Write開始 | stream_emit_started。candidate完成後のみ |
| Emit正常return | stream_emit_returned。write/hash/指定flushの通過 |
| Trial完了 | trial_completed、last_completed_trialとcompleted_prefixの同じ連続集合 |
| 再読/耐久性 | producerのreadback_verified=false/fsync_durability_verified=falseをそのまま記録 |

`check_failure`はschema、STOP/retry0/resumeなし/is_random_zero=false、現在attemptのID/index、stage flagの順序、最後の完了trial、count/traceとの結合を確認する。
完了prefixはlast_completed_trialから連続上限を決め、実際の行から再計数。inflightを加算しない。取得前の古いevent、未完了trialを完了へ加算する記録を拒否する。
保存eventが全文ならcandidateを実際の未commit行から再計算できる。行が未保存またはcompactなら候補traceの独立確認はINSUFFICIENT_INDEPENDENT_EVIDENCE。inflight内容から欠落trialを回復したとはしない。

Stream snapshotのcomplete_rows・chain・written bytes・file hash coverageを検査する。snapshotのbytes分を現在fileから読みhashを照合するが、これは**検証時の再読**でありproducerが当時readback/fsyncしたとはしない。
embedded failure record前のsnapshotはその直前の実bytes/chain/rowsへ結合する。control logのtrial_failure_provenanceはexact canonical stringとして安全にdecodeする。
FILE_HASH失敗では書込み済み完全行がwriter metadataより1行先にあり得る。coverage=falseならwriter hashはnull。検証器が取得する現在file SHAとは別に保つ。
FLUSH失敗では完全rowとwriter rowsが増えていてもemit未returnなので完了trialは増えない。
WRITEの13byte部分末尾は正常JSONL行にもtrialにも数えない。生存するcontrolにprovenanceがなければfailure自体はTECHNICAL_FAILURE、完了境界は独立証拠不足とする。

[保存した4ケースの実検証](failure_verification_examples.json)：sample failureは可視1/完了1、partial writeは可視1/完了1＋13bytes、flushとfile hash failureは可視2/完了1。
[人工fixture](failure_fixtures/)は既存Stream/run_pairとprivate fault I/Oで生成。元科学結果・実storageの耐電源断試験ではない。
file SHA、event連結trace SHA、envelope chain SHAは別domain。OS kill/OOM/全保存先不能ではfailure記録の存在自体を保証できない。
retry/resume・append・claim回復は実装していない。
