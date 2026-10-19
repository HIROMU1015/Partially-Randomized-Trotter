# Q1–Q10 独立技術受入の引継ぎ

**結論:** 技術監査は完了。公開rawの保存範囲は整合したが、新しい人工writer故障で予約会計のproducer不整合を再現した。正式production bridgeの前に技術修正を要する。科学実行は閉鎖。

**Q1.** 選択209公開入力のbytes/SHAを独立照合。主要SOURCE/schemaは測定b9cabcfc・content f3ba0cd0・公開76ca3c40で一致。inventory25項目は測定commitにも存在し、2管理scriptはその後の追加。Git blob identity、SOURCE/input pinsを区別して記録。過去Python3.12実体を新Python3.10で代用していない。

**Q2.** 公開18件はVERIFIED_COMPLETE1、TECHNICAL_FAILURE13、INSUFFICIENT_EVIDENCE4。正常は768 calls/全24pair/receipt/trace/予算が整合。4件はbinding_recheck・commit_ack_uncertain・parent_crash・raceで、snapshotは照合できたが原transition履歴は不足。別の新S projectionは768 fake callsで正常、別のwriter故障はINTEGRITY_MISMATCHを検出。

**Q3.** FAKE SQLiteのstore/scope/claim/digest/spent/STOP snapshotを独立確認。通常raw13件のordered trace、STOP/drain/unknown/call/byte会計を再計算。固定producer31人工testも別枠で再実行し新race/spent/reuse拒否を確認。実atomic authorityの証明ではない。writer故障はstate START_RESERVED1、report reserved0、started0、completefalse、fake STOPPEDとなり、修正必須。

**Q4.** 旧47件pendingは保持。新18件・新ローカルreplay・旧47件を別directory/identityで記録。旧P07/P08/P11等の履歴を遡及復元していない。

**Q5.** 16cell、3phase、一条件8448、S/P1二条件16896は算術整合。n64/1024も4608/66048。prefix同一とidentity domain分離の提案を静的確認。新namespace・正式Nは未採択。科学二条件one-shot bridgeは未実装。

**Q6.** 公開report13件＋別Sでhost parent+worker/outer wall、pair/setup/online/angle/evidence/output/IPC/alternative observer partition/memory/IOを照合。二重加算なし、unknownはnull、failure prefix除外。coarse generator/auditの未分離分や単独observer overheadを既知としない。S/P1両fusedではfused単独効果を識別できない。

**Q7.** n256の明示synthetic二条件traceは41,899,756 bytes。256MiBより小さいが実G7のevent/bit長/保存・journal・二条件bridgeを含む全量上界は未取得。Full-N31,710,848の同粒度traceはformat下界だけで18,265,448,448 bytes以上となり、256MiBを超える。粒度変更は別契約。保存障害で予約が失われたように見える今回の会計不整合は要修正。

**Q8.** 独立回帰623 PASS=旧540＋新83、FAIL/ERROR/SKIP各0、禁止科学import0。固定producer31件も別枠PASSで加算しない。既存統合125/過去696/481も今回のlocal countへ加算しない。確認したproducer不整合1件と、guardのtrusted-caller保証限界・4 snapshot履歴不足を明示。

**Q9.** ユーザー/GPTがN、条件、coverage/CPU精度/順序、seed、trace/予算、STOP/unknown、CPU estimand、trusted store/callerを決める。次のCodex技術作業はまず予約会計の修正版と保存境界故障検査。その後、別指示で採択scopeのproduction global bundle bridge/独立adapterを作成・人工固定する。現在のguard解除・正式releaseは今回実装しない。

**Q10.** 実G7/F3 sampling、新科学計算、保存科学値の数値再解析、正式科学SOURCE採択、実authorization/grant/claim発行・消費はすべて0。科学GOなし。

## 次の担当とSTOP

ユーザーとGPTが上記の契約判断と技術修正の次指示を決める。今回のCodexはEvidence Packetをcommit/push、remote HEAD/公開bytes/SHA照合、旧SOURCE/raw/grant/ledger/marker/STOPと親dirty/untracked保全確認後mandatory STOP。production bridge・実correctness・pilot sampling・Full-Nへ自動移行しない。

公開receiptはcontent commitを固定し、そのreceiptだけを後続管理commitへ保存する。最終HEADの全公開bytesは別途/tmpと最終chatで再照合し、自己hash循環を避ける。
