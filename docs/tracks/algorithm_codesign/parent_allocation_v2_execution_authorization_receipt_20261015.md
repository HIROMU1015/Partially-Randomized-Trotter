# Parent allocation v2 one-shot実行承認記録

2026-10-15 JST。利用者の原文指示：

> 次の作業に進んで大丈夫です

直前のv2準備完了報告で「次は固定S2に対する別途one-shot実行承認」と提示した後の利用者承認。固定S2の一回実行・保存結果/provenance監査・GitHub公開・mandatory STOPに限定する。

固定source S2：`a874f443411e03ad4d9d329834f09073a64a6621`。公開確認HEAD `0cf6a90735e6e8d304d1c37ad79898d2e6921d9b`は実行sourceに用いない。
このcommit A2はS2のauthorization-only直接子であり、変更は承認JSONと本receiptのみ。
固定contract：`artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v2/execution_contract_v2.json`。SHA256：`a197f8e92056fa8f6c33c8a2d4d4d653d711343a221c97e21ac78c7a686de896`。

science execution authorized=true、runs=1、retries=0、mandatory STOP=true。
既存v1のsource／authorization／結果／消費済みmarker／STOPを保持し、新v2 namespaceのみを使用する。
新価格前の候補freeze、最大15構成／75 law、最大18新key・各一回、2 worker・numeric threads1、全科学条件・guard・資源上限を維持する。
partial failure・resource failureでも再実行しない。終了後は保存結果のread-only監査・証拠公開のみ行い、次の科学的採否判断はGPTへ返す。
