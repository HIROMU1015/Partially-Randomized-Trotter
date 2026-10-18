# 並列adapter：PENDING_SERVER_CONTRACT

実装済みの共通pair verifierとserial global adapterを境界にし、`PendingParallelAdapter.verify()`は常にPENDING_SERVER_CONTRACT/implemented=falseを返す。
未知のworkers=8等から具体的な並列schema・scheduler・STOP・費用配分を推測しない。

公開済み初回人工pilot `75ba5b944c56669a2136393924a8bb8d22e40fd1` の記録のみ読んだ。進行中の最終8pair版は未公開・未取得。待たずに今回の人工基盤を完成した。
その初回pilotも並列STOPは既存逐次R1と同一でないと記録している。trace一致だけでR1契約を据え置かない。

後続公開時に必要な検査はmachineのP01–P11に実装したチェックリストとして登録済み：

1. run/phase/pass/arm/layer/axis/indexを結ぶglobal trial identityとpair内順序。
2. worker別current attempt/inflight/failure、複数失敗、started/completed/failed/unknownの全call会計。
3. 全workerで共有するone-shot call/output/log budget、出力先衝突、worker異常exit。
4. correctness→profile等のphase barrierとguard/admission/atomic claim。reserve後は失敗してもspent。
5. STOP通知後の新trial開始0と、通知前に開始したinflightの区別。通知時刻・観測boundaryをschemaで固定。
6. 部分保存・failure prefix・global/pair summary/receiptの結合。欠落workerを成功・ZERO・0費用にしない。
7. worker CPUと重複wall、parent dispatch、共有setup/最終outputの配分、fine observer pathの変更。
8. STOP保存失敗、複数worker失敗、claim非再利用、retry0/resumeなし。

serverの固定commit・公開schema・契約変更案・SOURCE/hash・人工logsを受領した後、Codexが明示的な新adapterを別versionで実装し、global accounting/STOP fault注入を追加するのが次の提案。
それまでは11項目全てPENDING_EVIDENCE。実並列実行テストPASSや正式採択は今回主張しない。最終science release/正式digest/実claimは別承認。
