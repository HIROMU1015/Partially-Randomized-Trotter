# 独立照合契約と証拠境界

旧coreと旧P1を変更せず、v3専用に明示変換した。全trialをRAMへ保持しない。archiveはsafe streamingで専用tempへ展開し、入力はread-only。リンク・special file・path traversal・重複member・出力衝突を拒否する。

## Schemaとidentity

固定 `protocol.json` と7 schemaの全assertion語彙を標準libraryで検査する。任意JSON Schema実装を称していない。local urn `$ref` のみ。exact int/bool、const/enum、required/additionalProperties、pattern/minimum、items/anyOfを処理する。canonical pair/summary envelopeは既存Readerの独立codec・chain検査を再利用する。

BUNDLE_BINDINGはPLAN直後の一件、成功receiptは最後の一件。17個の完全contractはformal/candidate/runtime/source pin digestを再計算する。SOURCE bytesとschema bytesを固定copyへ照合する。runtimeには6役割、import場所、entrypoint、8 schema pin、Python executable/libraryの保存identity形状を照合する。過去server filesystemへはアクセスしない。Python bytes未公開なのでhistorical interpreterの独立bytes認証ではない。

外部 `run_id` は公開metadataのcase label。raw trial IDはbundle-localであり、BUNDLE_BINDING formal digestがbundle境界となる。manifest labelをraw内部の独立UUIDと称さない。47 Host reportは原公開reportsから抽出した同一JSON値、正常12のsettingsは公開benchmark metadataとも一致する（metadata_binding_verification.json）。60性能観測を再実行・独立検証したとは称さない。

## Global accounting

3 phase ×8 pair、各X→Y、arm/layer/pass/offsetを独立planから照合。planned/reserved/state counts/call cap、statefile完全長・SHA、pair順とworker完了順を分ける。stateは最終snapshotであり、過去の重複reserveやENTERを全部証明する履歴ではない。

v3 protocolはSTART_RESERVEDからFAILEDも許す。このためv3 adapterはcoreのstate集計を再利用し、startedの許容範囲にFAILEDの未観測entryを残す。failed slotを理由にstarted observedを増やさない。現物47件はこの範囲と整合する。実際のstart/transitionの未保存履歴はNOT_ESTABLISHED。

## Control/STOP/F01

PLANだけ旧core用名へ変換し、BUNDLE_BINDINGを別検査する。v3 receipt schemaと原byte offsetは別途保持する。log失敗時にはfailure通知が先頭連続でなくsubsequenceになることがある。原parent_arrivalとHostの該当通知との完全一致・単調順を検査したうえで旧control validatorへ正規化する。原arrival・欠落範囲は結果に残し、欠落通知を生成しない。WORKER_END不足・失敗log欠損はP06 pending。

FAIL/ENDが同じtrialを二度通知しても両通知を保持し、callはstateから一度数える。physical例外時刻の全順序とはしない。first failureのin-flight snapshot identity/offset/capを照合する。

既存F01 verifierがcurrent attempt、last completed、candidate/committed trace、in-flight event、stream prefix/partial tailを検査する。flush-in-emitで完全JSONL行が存在しても、emit return前ならstate COMPLETEDへ足さない。logical committed、visible row、保存SHAの有無を分ける。compact event hash同士は比較できるが、非保存のfull concatenated traceを復元したとはしない。

controlには全trialごとのordered RESERVE/ENTER履歴がない。P08はpending。別のnormalized人工操作fixtureではSTOP後ENTER、重複reserve/complete、ACK不正、STOP前in-flightのdrainを検査する。このtestを実rawの歴史証明へ昇格しない。

## 完了と保存

正常はplanned全COMPLETED、24 pair OK、failureなし、summary/receipt/hash/contract binding一致、ACK＝物理＝予約、fake STOPPED申告、log/STOP保存エラーなしを要求。STOP保存失敗や親crashの偽completeを拒否する。

summaryはfinalization前snapshot。pair bytes＋summary後bytes、state retained bytes＋receipt前後control bytesを分ける。Host reportの予算はfinalization後。Reserved/ACK observed/physical/unknown creditを別表示し、不明creditは返却しない。上限64 GiB/256 MiB、障害fixtureの縮小予算も保存settingsに一致させる。

親crashはsummary/receiptを推定生成しない。raw nonterminal reserved/started/in-flightはunknownへ分類する。completed state観測と欠けたfinal completed報告を分ける。保存worker EXITED/join宣言は現時点のOS監視・durability証明ではない。実fsync/power failure耐性や実atomic authorityの独立証明は対象外。
