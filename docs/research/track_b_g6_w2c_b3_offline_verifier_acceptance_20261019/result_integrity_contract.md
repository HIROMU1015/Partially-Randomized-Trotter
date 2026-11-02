# R1 pairとserial globalの検証契約

適用SOURCEは基点564409487a35e15d73f1d0f0c2611824b400ab02のStream/run_pair/execute。別schemaを推測して通さない。

`verify_pair(path, spec, summary=None, failure_path=None)`は独立pair API。specはphase/pass_name/arm/layer/N。Xの0..N-1→Yの0..N-1、correctnessは必ず16/axis。
Readerはrbで行を逐次読む。envelopeはprevious/sha256/payloadのみ、重複JSON key・NaN・float/untagged number・非canonical hex/rationalを拒否する。
chainは保存された**元payload文字列**へのSHA256(previous+text)。検証用canonical再生成をchain入力には使わない。file SHAはnewline・部分末尾を含む実bytes全体、rowsは完全newline行のみ。
正整数・負整数・Fractionはdriverのint/rat hex codec、bool/string/nullと配列/mapを区別する。F3独自Evidence文書のstring内容をdriver codecとして展開しない。eval等はない。

第一走査でbytes/chain/可視trial行・failure記録を検査し、F01の完了上限を得る。第二走査で順序・完了counts・trace/prefixを確認する。二走査のfile metadataが変化した場合は拒否する。
R/C/A以外をsampling outcomeにしない。memory/failure rowはcallに含めない。trial IDの全fieldと連続indexを照合し、axis/pair counts、予定N、correctness EVENT coverage、summaryのhash/rows/bytes/countsを結びつける。
メモリrowは該当axisの4096完了位置を確認する。未知recordはUNSUPPORTED_SCHEMA。失敗後のtrialは拒否する。

## 証拠水準

| 検査 | 証拠水準 |
|---|---|
| envelope/chain/file SHA、ID/count、全文kind/eventのhash | INDEPENDENTLY_RECOMPUTED_FROM_RAW_BYTES |
| A/B・両passのevent hash比較 | COMPARED_SAVED_EVENT_HASHES |
| compactを含むfull traceのsummary値同士 | COMPARED_SUMMARY_CLAIMS |
| compact以後の全event bytesと連結traceの独立再計算 | INSUFFICIENT_INDEPENDENT_EVIDENCE |

全文はcorrectness全16、profile先頭16/axis。全文区間のevent bytesを連結したtrace/prefixは独立再計算できる。
compactはevent/evidence/angle SHAの形とchain保護を確認し、別fileとのevent SHA一致を比較できる。**単体SHAの列から元event bytesやその連結SHAを復元できない**。
compactを含むpairではtrace_sha256を独立取得値としてnullとし、summary_full_traceの証拠をCLAIMとする。証拠不足を「hash不一致」と呼ばない。
保存hash比較はハッシュ衝突耐性を仮定した技術整合性で、sampling law/bit消費や数学的angleの実行をraw streamだけで再証明するものではない。

## 状態

| 状態 | 定義 |
|---|---|
| VERIFIED_COMPLETE | 予定全trial、整合したpair summary。globalでは24pair・cross照合・summary envelope・completion receiptまで整合。独立再計算不能なtrace範囲は別fieldで残す |
| VERIFIED_PARTIAL_PREFIX | 完全行の整合prefixがあり、予定未達または最終不完全行。完了とはしない |
| TECHNICAL_FAILURE | 整合したF01/child control/parent failureが存在。可視行を復旧して成功としない |
| INTEGRITY_MISMATCH | chain/file/count/identity/summary/provenanceに明示的矛盾、または不正な入力 |
| INSUFFICIENT_EVIDENCE | 必須file/summary/receipt欠落。pair単独で全行があってもsummaryなしでは完了認定しない |
| UNSUPPORTED_SCHEMA | 未対応schema・record・truncated failure descriptor等。並列版を逐次版と推測しない |

不完全fileをfull completion summaryが指す場合は、summaryとの矛盾としてINTEGRITY_MISMATCHになる。
receiptはsummary chainへ結合するが、producer自身はsummary file SHAをreceiptへ記録しない。検証器が読み取ったsummary file SHAと、receiptが申告するchainを区別する。receipt metadataはauthorityの独立検証ではない。
全N/receipt整合は科学的正確性・物理RZ精度・母期待費用・方式優位の認定ではない。

## Cross/global

`cross_pair`はpair検査と分離した2-way trial iterator。axisごとに1rowずつ進め、A/B、correctness先頭16、baseline/instrumentedのevent hashを照合する。全trialのRAM setはない。
serial adapterは固定24pairのidentity/order/contractを照合し、完了callをphase/pass別に報告する。親failure_prefixが指定された場合は完了pair prefixと各保存pairを検査しTECHNICAL_FAILURE。費用比較へ入れない。
CLIはexplicit人工manifestのみを受け、出力は入力と包含関係のない新規directoryへexclusive保存する。
計算量はfileの線形走査数回、メモリは最大行＋固定24pair summary/検査metadata。巨大な単一行自体はその行サイズのRAMを要する。64GiB file全体を保持しない。
