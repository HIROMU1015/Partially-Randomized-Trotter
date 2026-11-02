# Parent allocation one-shot execution authorization

利用者の明示指示原文：

> 次の作業に進んで大丈夫です

直前に、固定source Sの準備完了・別途one-shot承認待ちを報告したことに対する実行承認として記録する。

source S：`8dfec590a8d760eb15932d862698b270649e9bcf`。contract SHA256：`42187e9c8cb387ea72ce0d93b45e2e4c40ee002a1170e6d940b386e13facf5c6`。

Sのauthorization-only直接子Aから実行する。変更はauthorization JSONと本receiptのみ。

固定3-qubit provider、p=(1/5,3/10,1/2)、x=5/7、r1、同じfinite P5。最大15構成/75 law、旧unique guard最大36、新root synthesis最大18各一回、root controlled診断最大72。固定2 worker/numeric threads1とCPU/RSS/output/log上限を維持する。

候補freeze後に新価格を取得する。retry0。失敗prefixの科学利用は禁止する。成功/失敗にかかわらずmandatory STOP。旧source/結果/authorization/markerを保持し、新namespaceだけへ記録する。

今回の承認は次stageや研究採択を含まない。結果・read-only監査・GPT引き継ぎをcommit/pushし、科学判断はGPTへ返す。
